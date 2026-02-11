"""Tests for feature management tools (create, delete, add_value, add_value_record)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.conftest import assert_error, assert_success, parse_json_result


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_feature(
    target_table_name: str = "Image",
    schema_name: str = "test_schema",
    feature_table_name: str = "Image_Diagnosis",
    term_col_names: list[str] | None = None,
    asset_col_names: list[str] | None = None,
    value_col_names: list[str] | None = None,
) -> MagicMock:
    """Build a mock feature object returned by ml.lookup_feature."""
    mock_feature = MagicMock()
    mock_feature.target_table.name = target_table_name
    mock_feature.feature_table.schema.name = schema_name
    mock_feature.feature_table.name = feature_table_name

    def _make_cols(names: list[str] | None) -> list[MagicMock]:
        if not names:
            return []
        cols = []
        for n in names:
            c = MagicMock()
            c.name = n
            cols.append(c)
        return cols

    mock_feature.term_columns = _make_cols(term_col_names)
    mock_feature.asset_columns = _make_cols(asset_col_names)
    mock_feature.value_columns = _make_cols(value_col_names)
    return mock_feature


def _mock_path_insert(mock_ml: MagicMock, inserted_rid: str = "RID-001") -> MagicMock:
    """Wire up mock_ml.pathBuilder() so that .schemas[...].tables[...].insert([...]) works.

    Returns the mock insert callable so callers can assert on it.
    """
    mock_table_path = MagicMock()
    mock_table_path.insert.return_value = [{"RID": inserted_rid}]

    mock_tables = MagicMock()
    mock_tables.__getitem__ = MagicMock(return_value=mock_table_path)

    mock_schema = MagicMock()
    mock_schema.tables = mock_tables

    mock_schemas = MagicMock()
    mock_schemas.__getitem__ = MagicMock(return_value=mock_schema)

    mock_pb = MagicMock()
    mock_pb.schemas = mock_schemas
    mock_ml.pathBuilder.return_value = mock_pb

    return mock_table_path.insert


# =============================================================================
# create_feature
# =============================================================================


class TestCreateFeature:
    """Tests for the create_feature tool."""

    @pytest.mark.asyncio
    async def test_create_feature_success(self, feature_tools, mock_ml):
        """create_feature returns status='created' with correct fields."""
        result = await feature_tools["create_feature"](
            table_name="Image",
            feature_name="Diagnosis",
            comment="A clinical diagnosis label",
            terms=["Diagnosis_Type"],
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert data["feature_name"] == "Diagnosis"
        assert data["target_table"] == "Image"

        mock_ml.create_feature.assert_called_once_with(
            target_table="Image",
            feature_name="Diagnosis",
            terms=["Diagnosis_Type"],
            assets=[],
            metadata=[],
            comment="A clinical diagnosis label",
        )

    @pytest.mark.asyncio
    async def test_create_feature_defaults(self, feature_tools, mock_ml):
        """create_feature uses empty defaults for optional list params."""
        await feature_tools["create_feature"](
            table_name="Subject",
            feature_name="Status",
        )
        mock_ml.create_feature.assert_called_once_with(
            target_table="Subject",
            feature_name="Status",
            terms=[],
            assets=[],
            metadata=[],
            comment="",
        )

    @pytest.mark.asyncio
    async def test_create_feature_no_connection(self, feature_tools_disconnected):
        """create_feature returns error when not connected."""
        result = await feature_tools_disconnected["create_feature"](
            table_name="Image",
            feature_name="Diagnosis",
        )
        data = assert_error(result, "No active catalog connection")
        assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_create_feature_exception(self, feature_tools, mock_ml):
        """create_feature returns error when DerivaML raises an exception."""
        mock_ml.create_feature.side_effect = RuntimeError("Table already exists")
        result = await feature_tools["create_feature"](
            table_name="Image",
            feature_name="Diagnosis",
        )
        data = assert_error(result, "Table already exists")
        assert data["status"] == "error"


# =============================================================================
# delete_feature
# =============================================================================


class TestDeleteFeature:
    """Tests for the delete_feature tool."""

    @pytest.mark.asyncio
    async def test_delete_feature_success(self, feature_tools, mock_ml):
        """delete_feature returns status='deleted' when the feature exists."""
        mock_ml.delete_feature.return_value = True
        result = await feature_tools["delete_feature"](
            table_name="Image",
            feature_name="Diagnosis",
        )
        data = assert_success(result)
        assert data["status"] == "deleted"
        assert data["feature_name"] == "Diagnosis"
        assert data["table_name"] == "Image"

        mock_ml.delete_feature.assert_called_once_with("Image", "Diagnosis")

    @pytest.mark.asyncio
    async def test_delete_feature_not_found(self, feature_tools, mock_ml):
        """delete_feature returns status='not_found' when the feature does not exist."""
        mock_ml.delete_feature.return_value = False
        result = await feature_tools["delete_feature"](
            table_name="Image",
            feature_name="NonExistent",
        )
        data = parse_json_result(result)
        assert data["status"] == "not_found"
        assert "NonExistent" in data["message"]

    @pytest.mark.asyncio
    async def test_delete_feature_no_connection(self, feature_tools_disconnected):
        """delete_feature returns error when not connected."""
        result = await feature_tools_disconnected["delete_feature"](
            table_name="Image",
            feature_name="Diagnosis",
        )
        data = assert_error(result, "No active catalog connection")

    @pytest.mark.asyncio
    async def test_delete_feature_exception(self, feature_tools, mock_ml):
        """delete_feature returns error when DerivaML raises."""
        mock_ml.delete_feature.side_effect = RuntimeError("Permission denied")
        result = await feature_tools["delete_feature"](
            table_name="Image",
            feature_name="Diagnosis",
        )
        data = assert_error(result, "Permission denied")


# =============================================================================
# add_feature_value
# =============================================================================


class TestAddFeatureValue:
    """Tests for the add_feature_value tool."""

    @pytest.mark.asyncio
    async def test_add_feature_value_success_term_column(self, feature_tools, mock_ml):
        """add_feature_value inserts a record using the first term column."""
        mock_feature = _make_mock_feature(term_col_names=["Diagnosis_Type"])
        mock_ml.lookup_feature.return_value = mock_feature
        mock_insert = _mock_path_insert(mock_ml, inserted_rid="RID-100")

        result = await feature_tools["add_feature_value"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            value="Normal",
        )
        data = assert_success(result)
        assert data["status"] == "added"
        assert data["target_rid"] == "1-ABC"
        assert data["feature_name"] == "Diagnosis"
        assert data["Diagnosis_Type"] == "Normal"
        assert data["execution_rid"] == "EXE-TEST"
        assert data["rid"] == "RID-100"

        # Verify the record dict passed to insert
        mock_insert.assert_called_once()
        inserted_records = mock_insert.call_args[0][0]
        assert len(inserted_records) == 1
        record = inserted_records[0]
        assert record["Image"] == "1-ABC"
        assert record["Feature_Name"] == "Diagnosis"
        assert record["Execution"] == "EXE-TEST"
        assert record["Diagnosis_Type"] == "Normal"

    @pytest.mark.asyncio
    async def test_add_feature_value_success_asset_column(self, feature_tools, mock_ml):
        """add_feature_value falls back to asset column when no term columns."""
        mock_feature = _make_mock_feature(
            feature_table_name="Image_Segmentation",
            asset_col_names=["Mask"],
        )
        mock_ml.lookup_feature.return_value = mock_feature
        _mock_path_insert(mock_ml, inserted_rid="RID-200")

        result = await feature_tools["add_feature_value"](
            table_name="Image",
            feature_name="Segmentation",
            target_rid="1-DEF",
            value="ASSET-RID-1",
        )
        data = assert_success(result)
        assert data["Mask"] == "ASSET-RID-1"

    @pytest.mark.asyncio
    async def test_add_feature_value_success_value_column(self, feature_tools, mock_ml):
        """add_feature_value falls back to value column when no term or asset columns."""
        mock_feature = _make_mock_feature(
            feature_table_name="Image_Score",
            value_col_names=["confidence"],
        )
        mock_ml.lookup_feature.return_value = mock_feature
        _mock_path_insert(mock_ml, inserted_rid="RID-300")

        result = await feature_tools["add_feature_value"](
            table_name="Image",
            feature_name="Score",
            target_rid="1-GHI",
            value="0.95",
        )
        data = assert_success(result)
        assert data["confidence"] == "0.95"

    @pytest.mark.asyncio
    async def test_add_feature_value_no_value_columns(self, feature_tools, mock_ml):
        """add_feature_value errors when the feature has no value columns at all."""
        mock_feature = _make_mock_feature()  # all column lists empty by default
        mock_ml.lookup_feature.return_value = mock_feature

        result = await feature_tools["add_feature_value"](
            table_name="Image",
            feature_name="Empty",
            target_rid="1-XYZ",
            value="anything",
        )
        data = assert_error(result, "no value columns")

    @pytest.mark.asyncio
    async def test_add_feature_value_explicit_execution_rid(
        self, feature_tools, mock_ml
    ):
        """add_feature_value uses an explicitly provided execution_rid."""
        mock_feature = _make_mock_feature(term_col_names=["Diagnosis_Type"])
        mock_ml.lookup_feature.return_value = mock_feature
        mock_insert = _mock_path_insert(mock_ml)

        result = await feature_tools["add_feature_value"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            value="Normal",
            execution_rid="EXE-CUSTOM",
        )
        data = assert_success(result)
        assert data["execution_rid"] == "EXE-CUSTOM"

        inserted_record = mock_insert.call_args[0][0][0]
        assert inserted_record["Execution"] == "EXE-CUSTOM"

    @pytest.mark.asyncio
    async def test_add_feature_value_uses_active_tool_execution(
        self, feature_tools, mock_ml, mock_conn_manager
    ):
        """add_feature_value prefers active_tool_execution over MCP execution."""
        # Set up an active_tool_execution on the connection info
        tool_exe = MagicMock()
        tool_exe.execution_rid = "EXE-TOOL"
        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = tool_exe

        mock_feature = _make_mock_feature(term_col_names=["Diagnosis_Type"])
        mock_ml.lookup_feature.return_value = mock_feature
        mock_insert = _mock_path_insert(mock_ml)

        result = await feature_tools["add_feature_value"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            value="Normal",
        )
        data = assert_success(result)
        assert data["execution_rid"] == "EXE-TOOL"

        # Clean up so other tests are not affected
        conn_info.active_tool_execution = None

    @pytest.mark.asyncio
    async def test_add_feature_value_no_execution(
        self, feature_tools, mock_ml, mock_conn_manager
    ):
        """add_feature_value errors when no execution is available at all."""
        # Temporarily remove all execution sources
        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = None
        mock_conn_manager.get_active_execution.return_value = None

        result = await feature_tools["add_feature_value"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            value="Normal",
        )
        data = assert_error(result, "No active execution")

        # Restore for other tests
        mock_execution = MagicMock()
        mock_execution.execution_rid = "EXE-TEST"
        mock_conn_manager.get_active_execution.return_value = mock_execution

    @pytest.mark.asyncio
    async def test_add_feature_value_no_connection(self, feature_tools_disconnected):
        """add_feature_value returns error when not connected."""
        result = await feature_tools_disconnected["add_feature_value"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            value="Normal",
        )
        data = assert_error(result, "No active catalog connection")

    @pytest.mark.asyncio
    async def test_add_feature_value_insert_exception(self, feature_tools, mock_ml):
        """add_feature_value returns error when the insert call fails."""
        mock_feature = _make_mock_feature(term_col_names=["Diagnosis_Type"])
        mock_ml.lookup_feature.return_value = mock_feature

        # Make pathBuilder chain raise during insert
        mock_table_path = MagicMock()
        mock_table_path.insert.side_effect = RuntimeError("Insert failed")
        mock_tables = MagicMock()
        mock_tables.__getitem__ = MagicMock(return_value=mock_table_path)
        mock_schema = MagicMock()
        mock_schema.tables = mock_tables
        mock_schemas = MagicMock()
        mock_schemas.__getitem__ = MagicMock(return_value=mock_schema)
        mock_pb = MagicMock()
        mock_pb.schemas = mock_schemas
        mock_ml.pathBuilder.return_value = mock_pb

        result = await feature_tools["add_feature_value"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            value="Normal",
        )
        data = assert_error(result, "Insert failed")


# =============================================================================
# add_feature_value_record
# =============================================================================


class TestAddFeatureValueRecord:
    """Tests for the add_feature_value_record tool."""

    @pytest.mark.asyncio
    async def test_add_feature_value_record_success(self, feature_tools, mock_ml):
        """add_feature_value_record inserts a record with all provided values."""
        mock_feature = _make_mock_feature(
            term_col_names=["Diagnosis_Type"],
            value_col_names=["confidence"],
        )
        mock_ml.lookup_feature.return_value = mock_feature
        mock_insert = _mock_path_insert(mock_ml, inserted_rid="RID-500")

        values = {"Diagnosis_Type": "Normal", "confidence": 0.95}
        result = await feature_tools["add_feature_value_record"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            values=values,
        )
        data = assert_success(result)
        assert data["status"] == "added"
        assert data["target_rid"] == "1-ABC"
        assert data["feature_name"] == "Diagnosis"
        assert data["values"] == values
        assert data["execution_rid"] == "EXE-TEST"
        assert data["rid"] == "RID-500"

        # Verify the record dict passed to insert
        mock_insert.assert_called_once()
        inserted_records = mock_insert.call_args[0][0]
        assert len(inserted_records) == 1
        record = inserted_records[0]
        assert record["Image"] == "1-ABC"
        assert record["Feature_Name"] == "Diagnosis"
        assert record["Execution"] == "EXE-TEST"
        assert record["Diagnosis_Type"] == "Normal"
        assert record["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_add_feature_value_record_explicit_execution_rid(
        self, feature_tools, mock_ml
    ):
        """add_feature_value_record uses an explicit execution_rid."""
        mock_feature = _make_mock_feature(term_col_names=["Diagnosis_Type"])
        mock_ml.lookup_feature.return_value = mock_feature
        mock_insert = _mock_path_insert(mock_ml)

        result = await feature_tools["add_feature_value_record"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            values={"Diagnosis_Type": "Abnormal"},
            execution_rid="EXE-EXPLICIT",
        )
        data = assert_success(result)
        assert data["execution_rid"] == "EXE-EXPLICIT"

        inserted_record = mock_insert.call_args[0][0][0]
        assert inserted_record["Execution"] == "EXE-EXPLICIT"

    @pytest.mark.asyncio
    async def test_add_feature_value_record_no_execution(
        self, feature_tools, mock_ml, mock_conn_manager
    ):
        """add_feature_value_record errors when no execution is available."""
        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = None
        mock_conn_manager.get_active_execution.return_value = None

        result = await feature_tools["add_feature_value_record"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            values={"Diagnosis_Type": "Normal"},
        )
        data = assert_error(result, "No active execution")

        # Restore for other tests
        mock_execution = MagicMock()
        mock_execution.execution_rid = "EXE-TEST"
        mock_conn_manager.get_active_execution.return_value = mock_execution

    @pytest.mark.asyncio
    async def test_add_feature_value_record_no_connection(
        self, feature_tools_disconnected
    ):
        """add_feature_value_record returns error when not connected."""
        result = await feature_tools_disconnected["add_feature_value_record"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            values={"Diagnosis_Type": "Normal"},
        )
        data = assert_error(result, "No active catalog connection")

    @pytest.mark.asyncio
    async def test_add_feature_value_record_insert_exception(
        self, feature_tools, mock_ml
    ):
        """add_feature_value_record returns error when insert fails."""
        mock_feature = _make_mock_feature(term_col_names=["Diagnosis_Type"])
        mock_ml.lookup_feature.return_value = mock_feature

        mock_table_path = MagicMock()
        mock_table_path.insert.side_effect = RuntimeError("Conflict")
        mock_tables = MagicMock()
        mock_tables.__getitem__ = MagicMock(return_value=mock_table_path)
        mock_schema = MagicMock()
        mock_schema.tables = mock_tables
        mock_schemas = MagicMock()
        mock_schemas.__getitem__ = MagicMock(return_value=mock_schema)
        mock_pb = MagicMock()
        mock_pb.schemas = mock_schemas
        mock_ml.pathBuilder.return_value = mock_pb

        result = await feature_tools["add_feature_value_record"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            values={"Diagnosis_Type": "Normal"},
        )
        data = assert_error(result, "Conflict")

    @pytest.mark.asyncio
    async def test_add_feature_value_record_empty_values(self, feature_tools, mock_ml):
        """add_feature_value_record works with an empty values dict (only base fields)."""
        mock_feature = _make_mock_feature(term_col_names=["Diagnosis_Type"])
        mock_ml.lookup_feature.return_value = mock_feature
        mock_insert = _mock_path_insert(mock_ml, inserted_rid="RID-EMPTY")

        result = await feature_tools["add_feature_value_record"](
            table_name="Image",
            feature_name="Diagnosis",
            target_rid="1-ABC",
            values={},
        )
        data = assert_success(result)
        assert data["values"] == {}

        inserted_record = mock_insert.call_args[0][0][0]
        # Should still have the base fields
        assert inserted_record["Image"] == "1-ABC"
        assert inserted_record["Feature_Name"] == "Diagnosis"
        assert inserted_record["Execution"] == "EXE-TEST"
