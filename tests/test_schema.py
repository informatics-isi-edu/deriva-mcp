"""Unit tests for schema manipulation tools.

Tests cover schema tools:
- create_table: create a standard table with columns and foreign keys
- create_asset_table: create an asset table with file management columns
- list_asset_executions: list executions associated with an asset
- add_asset_type: add a term to the Asset_Type vocabulary
- add_asset_type_to_asset: associate an asset type with an asset
- remove_asset_type_from_asset: remove an asset type from an asset
- set_table_description: update a table's comment/description
- set_table_display_name: update a table's UI display name
- set_row_name_pattern: set Handlebars row name pattern
- add_column: add a new column to an existing table
- set_column_description: update a column's comment/description
- set_column_display_name: update a column's UI display name
- set_column_nullok: set whether a column allows NULL values

Note: get_schema_description and get_table_columns were moved to resources.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import assert_error, assert_success, parse_json_result

# Patch target for TableHandle -- the tools do
# ``from deriva_ml.model.handles import TableHandle`` inside the function body,
# so we must patch at the *source* module.
_TABLE_HANDLE_PATCH = "deriva_ml.model.handles.TableHandle"

# Patch target for BuiltinTypes used by add_column (imported from
# ``deriva_ml.core.enums``).
_BUILTIN_TYPES_ENUMS_PATCH = "deriva_ml.core.enums.BuiltinTypes"


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_table(name="TestTable", schema_name="test_schema", column_names=None):
    """Create a mock table object returned by ml.create_table / ml.create_asset.

    The mock has .name, .schema.name, and .columns (iterable of objects with .name).
    """
    if column_names is None:
        column_names = ["RID", "Name"]

    mock_table = MagicMock()
    mock_table.name = name
    mock_table.schema.name = schema_name

    mock_columns = []
    for cname in column_names:
        col = MagicMock()
        col.name = cname
        mock_columns.append(col)
    mock_table.columns = mock_columns

    return mock_table


def _make_mock_column(
    name="Col1",
    typename="text",
    nullok=True,
    description="",
    display_name="Col 1",
    is_system=False,
):
    """Create a mock column handle for TableHandle.user_columns / all_columns."""
    col = MagicMock()
    col.name = name
    col._column.type.typename = typename
    col.nullok = nullok
    col.description = description
    col.get_display_name.return_value = display_name
    col.is_system_column = is_system
    return col


def _setup_table_handle(mock_ml, table_name="TestTable", schema_name="test_schema"):
    """Configure mock_ml.model.name_to_table() to return a mock table.

    Returns the mock table object so tests can further customise it.
    """
    mock_table = MagicMock()
    mock_table.name = table_name
    mock_table.schema.name = schema_name
    mock_ml.model.name_to_table.return_value = mock_table
    return mock_table


# =============================================================================
# TestCreateTable
# =============================================================================


class TestCreateTable:
    """Tests for the create_table tool."""

    @pytest.mark.asyncio
    async def test_create_simple_table(self, schema_tools, mock_ml):
        """Creating a table with columns returns status=created."""
        mock_table = _make_mock_table(
            "Subject", "test_schema", ["RID", "Name", "Age"]
        )
        mock_ml.create_table.return_value = mock_table

        with patch("deriva_ml.ColumnDefinition") as MockColDef, \
             patch("deriva_ml.ForeignKeyDefinition"), \
             patch("deriva_ml.TableDefinition") as MockTableDef:
            MockColDef.side_effect = lambda **kw: MagicMock(**kw)
            MockTableDef.side_effect = lambda **kw: MagicMock(**kw)

            result = await schema_tools["create_table"](
                table_name="Subject",
                columns=[
                    {"name": "Name", "type": "text", "nullok": False},
                    {"name": "Age", "type": "int4"},
                ],
                comment="Subject data",
            )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["table_name"] == "Subject"
        assert data["schema"] == "test_schema"
        assert "Name" in data["columns"]
        assert "Age" in data["columns"]
        mock_ml.create_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_table_no_columns(self, schema_tools, mock_ml):
        """Creating a table with no extra columns succeeds."""
        mock_table = _make_mock_table("EmptyTable", "test_schema", ["RID"])
        mock_ml.create_table.return_value = mock_table

        with patch("deriva_ml.ColumnDefinition"), \
             patch("deriva_ml.ForeignKeyDefinition"), \
             patch("deriva_ml.TableDefinition") as MockTableDef:
            MockTableDef.side_effect = lambda **kw: MagicMock(**kw)

            result = await schema_tools["create_table"](table_name="EmptyTable")

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["table_name"] == "EmptyTable"

    @pytest.mark.asyncio
    async def test_create_table_with_foreign_keys(self, schema_tools, mock_ml):
        """Creating a table with foreign keys passes fkey_defs to create_table."""
        mock_table = _make_mock_table(
            "Sample", "test_schema", ["RID", "Name", "Subject"]
        )
        mock_ml.create_table.return_value = mock_table
        mock_ml.default_schema = "test_schema"

        with patch("deriva_ml.ColumnDefinition") as MockColDef, \
             patch("deriva_ml.ForeignKeyDefinition") as MockFKDef, \
             patch("deriva_ml.TableDefinition") as MockTableDef:
            MockColDef.side_effect = lambda **kw: MagicMock(**kw)
            MockFKDef.side_effect = lambda **kw: MagicMock(**kw)
            MockTableDef.side_effect = lambda **kw: MagicMock(**kw)

            result = await schema_tools["create_table"](
                table_name="Sample",
                columns=[
                    {"name": "Name", "type": "text", "nullok": False},
                    {"name": "Subject", "type": "text", "nullok": False},
                ],
                foreign_keys=[
                    {
                        "column": "Subject",
                        "referenced_table": "Subject",
                        "on_delete": "CASCADE",
                    }
                ],
                comment="Samples linked to subjects",
            )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["table_name"] == "Sample"
        mock_ml.create_table.assert_called_once()
        # Verify FK definition was constructed
        MockFKDef.assert_called_once()
        fk_kwargs = MockFKDef.call_args.kwargs
        assert fk_kwargs["colnames"] == ["Subject"]
        assert fk_kwargs["pk_tname"] == "Subject"
        assert fk_kwargs["on_delete"] == "CASCADE"

    @pytest.mark.asyncio
    async def test_create_table_default_type(self, schema_tools, mock_ml):
        """Columns without an explicit type default to text."""
        mock_table = _make_mock_table("T", "test_schema", ["RID", "Notes"])
        mock_ml.create_table.return_value = mock_table

        with patch("deriva_ml.ColumnDefinition") as MockColDef, \
             patch("deriva_ml.ForeignKeyDefinition"), \
             patch("deriva_ml.TableDefinition") as MockTableDef:
            MockColDef.side_effect = lambda **kw: MagicMock(**kw)
            MockTableDef.side_effect = lambda **kw: MagicMock(**kw)

            result = await schema_tools["create_table"](
                table_name="T",
                columns=[{"name": "Notes"}],
            )

        data = assert_success(result)
        assert data["status"] == "created"

    @pytest.mark.asyncio
    async def test_create_table_unknown_type_defaults_text(self, schema_tools, mock_ml):
        """An unrecognised column type falls back to text."""
        mock_table = _make_mock_table("T", "test_schema", ["RID", "Data"])
        mock_ml.create_table.return_value = mock_table

        with patch("deriva_ml.ColumnDefinition") as MockColDef, \
             patch("deriva_ml.ForeignKeyDefinition"), \
             patch("deriva_ml.TableDefinition") as MockTableDef:
            MockColDef.side_effect = lambda **kw: MagicMock(**kw)
            MockTableDef.side_effect = lambda **kw: MagicMock(**kw)

            result = await schema_tools["create_table"](
                table_name="T",
                columns=[{"name": "Data", "type": "unknown_type"}],
            )

        data = assert_success(result)
        assert data["status"] == "created"

    @pytest.mark.asyncio
    async def test_create_table_all_column_types(self, schema_tools, mock_ml):
        """Creating a table with every supported column type succeeds."""
        all_types = [
            "text", "int2", "int4", "int8", "float4", "float8",
            "boolean", "date", "timestamp", "timestamptz", "json", "jsonb", "markdown",
        ]
        col_names = ["RID"] + [f"col_{t}" for t in all_types]
        mock_table = _make_mock_table("AllTypes", "test_schema", col_names)
        mock_ml.create_table.return_value = mock_table

        columns = [{"name": f"col_{t}", "type": t} for t in all_types]

        with patch("deriva_ml.ColumnDefinition") as MockColDef, \
             patch("deriva_ml.ForeignKeyDefinition"), \
             patch("deriva_ml.TableDefinition") as MockTableDef:
            MockColDef.side_effect = lambda **kw: MagicMock(**kw)
            MockTableDef.side_effect = lambda **kw: MagicMock(**kw)

            result = await schema_tools["create_table"](
                table_name="AllTypes",
                columns=columns,
            )

        data = assert_success(result)
        assert data["status"] == "created"
        assert len(data["columns"]) == len(col_names)
        assert MockColDef.call_count == len(all_types)

    @pytest.mark.asyncio
    async def test_create_table_fk_default_referenced_column(self, schema_tools, mock_ml):
        """Foreign keys default to referencing RID when no referenced_column given."""
        mock_table = _make_mock_table("T", "test_schema", ["RID", "Ref"])
        mock_ml.create_table.return_value = mock_table
        mock_ml.default_schema = "test_schema"

        with patch("deriva_ml.ColumnDefinition") as MockColDef, \
             patch("deriva_ml.ForeignKeyDefinition") as MockFKDef, \
             patch("deriva_ml.TableDefinition") as MockTableDef:
            MockColDef.side_effect = lambda **kw: MagicMock(**kw)
            MockFKDef.side_effect = lambda **kw: MagicMock(**kw)
            MockTableDef.side_effect = lambda **kw: MagicMock(**kw)

            result = await schema_tools["create_table"](
                table_name="T",
                columns=[{"name": "Ref", "type": "text"}],
                foreign_keys=[{"column": "Ref", "referenced_table": "Other"}],
            )

        data = assert_success(result)
        assert data["status"] == "created"
        fk_kwargs = MockFKDef.call_args.kwargs
        assert fk_kwargs["pk_colnames"] == ["RID"]
        assert fk_kwargs["on_delete"] == "NO ACTION"

    @pytest.mark.asyncio
    async def test_create_table_error(self, schema_tools, mock_ml):
        """When create_table raises, the tool returns an error."""
        mock_ml.create_table.side_effect = RuntimeError("Table already exists")

        with patch("deriva_ml.ColumnDefinition"), \
             patch("deriva_ml.ForeignKeyDefinition"), \
             patch("deriva_ml.TableDefinition") as MockTableDef:
            MockTableDef.side_effect = lambda **kw: MagicMock(**kw)

            result = await schema_tools["create_table"](table_name="Duplicate")

        data = assert_error(result, "Table already exists")

    @pytest.mark.asyncio
    async def test_create_table_no_connection(self, schema_tools_disconnected):
        """When disconnected, create_table returns a connection error."""
        result = await schema_tools_disconnected["create_table"](table_name="X")
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestCreateAssetTable
# =============================================================================


class TestCreateAssetTable:
    """Tests for the create_asset_table tool."""

    @pytest.mark.asyncio
    async def test_create_asset_table_basic(self, schema_tools, mock_ml):
        """Creating an asset table returns status=created with asset columns."""
        mock_table = _make_mock_table(
            "Image",
            "test_schema",
            ["RID", "URL", "Filename", "Length", "MD5", "Description"],
        )
        mock_ml.create_asset.return_value = mock_table

        result = await schema_tools["create_asset_table"](
            asset_name="Image",
            comment="Medical images for analysis",
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["table_name"] == "Image"
        assert data["schema"] == "test_schema"
        assert "URL" in data["columns"]
        assert "Filename" in data["columns"]
        mock_ml.create_asset.assert_called_once_with(
            asset_name="Image",
            column_defs=None,
            referenced_tables=None,
            comment="Medical images for analysis",
        )

    @pytest.mark.asyncio
    async def test_create_asset_table_with_extra_columns(self, schema_tools, mock_ml):
        """Creating an asset table with additional columns passes column_defs."""
        mock_table = _make_mock_table(
            "Image",
            "test_schema",
            ["RID", "URL", "Filename", "Width", "Height"],
        )
        mock_ml.create_asset.return_value = mock_table

        result = await schema_tools["create_asset_table"](
            asset_name="Image",
            columns=[
                {"name": "Width", "type": "int4"},
                {"name": "Height", "type": "int4"},
            ],
        )

        data = assert_success(result)
        assert data["status"] == "created"
        # column_defs should be a list (not None)
        call_kwargs = mock_ml.create_asset.call_args.kwargs
        assert call_kwargs["column_defs"] is not None
        assert len(call_kwargs["column_defs"]) == 2

    @pytest.mark.asyncio
    async def test_create_asset_table_with_referenced_tables(self, schema_tools, mock_ml):
        """Creating an asset table with referenced_tables resolves table objects."""
        mock_table = _make_mock_table("Image", "test_schema", ["RID", "URL"])
        mock_ml.create_asset.return_value = mock_table

        ref_table = MagicMock()
        mock_ml.model.name_to_table.return_value = ref_table

        result = await schema_tools["create_asset_table"](
            asset_name="Image",
            referenced_tables=["Subject"],
        )

        data = assert_success(result)
        assert data["status"] == "created"
        mock_ml.model.name_to_table.assert_called_once_with("Subject")
        call_kwargs = mock_ml.create_asset.call_args.kwargs
        assert call_kwargs["referenced_tables"] is not None
        assert len(call_kwargs["referenced_tables"]) == 1

    @pytest.mark.asyncio
    async def test_create_asset_table_unknown_column_type(self, schema_tools, mock_ml):
        """An unrecognised column type in asset table defaults to text."""
        mock_table = _make_mock_table("Model", "test_schema", ["RID", "URL", "Extra"])
        mock_ml.create_asset.return_value = mock_table

        result = await schema_tools["create_asset_table"](
            asset_name="Model",
            columns=[{"name": "Extra", "type": "nonexistent_type"}],
        )

        data = assert_success(result)
        assert data["status"] == "created"

    @pytest.mark.asyncio
    async def test_create_asset_table_error(self, schema_tools, mock_ml):
        """When create_asset raises, the tool returns an error."""
        mock_ml.create_asset.side_effect = RuntimeError("Asset name taken")

        result = await schema_tools["create_asset_table"](asset_name="Duplicate")

        data = assert_error(result, "Asset name taken")

    @pytest.mark.asyncio
    async def test_create_asset_table_no_connection(self, schema_tools_disconnected):
        """When disconnected, create_asset_table returns a connection error."""
        result = await schema_tools_disconnected["create_asset_table"](
            asset_name="Image"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestListAssetExecutions
# =============================================================================


class TestListAssetExecutions:
    """Tests for the list_asset_executions tool."""

    @pytest.mark.asyncio
    async def test_list_executions_returns_records(self, schema_tools, mock_ml):
        """list_asset_executions returns formatted execution records."""
        mock_exe1 = MagicMock()
        mock_exe1.execution_rid = "EXE-001"
        mock_exe1.workflow_rid = "WF-001"
        mock_exe1.status.value = "Complete"
        mock_exe1.description = "Training run"

        mock_exe2 = MagicMock()
        mock_exe2.execution_rid = "EXE-002"
        mock_exe2.workflow_rid = "WF-002"
        mock_exe2.status.value = "Running"
        mock_exe2.description = "Inference run"

        mock_ml.list_asset_executions.return_value = [mock_exe1, mock_exe2]

        result = await schema_tools["list_asset_executions"](asset_rid="3JSE")

        data = parse_json_result(result)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["execution_rid"] == "EXE-001"
        assert data[0]["status"] == "Complete"
        assert data[0]["workflow_rid"] == "WF-001"
        assert data[0]["description"] == "Training run"
        assert data[1]["execution_rid"] == "EXE-002"
        assert data[1]["status"] == "Running"
        mock_ml.list_asset_executions.assert_called_once_with("3JSE", asset_role=None)

    @pytest.mark.asyncio
    async def test_list_executions_with_role_filter(self, schema_tools, mock_ml):
        """list_asset_executions passes asset_role filter."""
        mock_exe = MagicMock()
        mock_exe.execution_rid = "EXE-001"
        mock_exe.workflow_rid = "WF-001"
        mock_exe.status.value = "Complete"
        mock_exe.description = "Output asset"

        mock_ml.list_asset_executions.return_value = [mock_exe]

        result = await schema_tools["list_asset_executions"](
            asset_rid="3JSE", asset_role="Output"
        )

        data = parse_json_result(result)
        assert len(data) == 1
        assert data[0]["execution_rid"] == "EXE-001"
        mock_ml.list_asset_executions.assert_called_once_with(
            "3JSE", asset_role="Output"
        )

    @pytest.mark.asyncio
    async def test_list_executions_empty(self, schema_tools, mock_ml):
        """list_asset_executions returns empty list when no executions found."""
        mock_ml.list_asset_executions.return_value = []

        result = await schema_tools["list_asset_executions"](asset_rid="NONE")

        data = parse_json_result(result)
        assert isinstance(data, list)
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_list_executions_status_without_value(self, schema_tools, mock_ml):
        """Status objects without .value fall back to str()."""
        mock_exe = MagicMock(spec=[])  # empty spec so hasattr(status, 'value') is False
        mock_exe.execution_rid = "EXE-003"
        mock_exe.workflow_rid = "WF-003"
        mock_exe.status = "Pending"
        mock_exe.description = "Pending execution"

        mock_ml.list_asset_executions.return_value = [mock_exe]

        result = await schema_tools["list_asset_executions"](asset_rid="ABC")

        data = parse_json_result(result)
        assert data[0]["status"] == "Pending"

    @pytest.mark.asyncio
    async def test_list_executions_error(self, schema_tools, mock_ml):
        """When list_asset_executions raises, the tool returns an error."""
        mock_ml.list_asset_executions.side_effect = RuntimeError("Asset not found")

        result = await schema_tools["list_asset_executions"](asset_rid="BAD")

        data = assert_error(result, "Asset not found")

    @pytest.mark.asyncio
    async def test_list_executions_no_connection(self, schema_tools_disconnected):
        """When disconnected, list_asset_executions returns a connection error."""
        result = await schema_tools_disconnected["list_asset_executions"](
            asset_rid="X"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestAddAssetType
# =============================================================================


class TestAddAssetType:
    """Tests for the add_asset_type tool."""

    @pytest.mark.asyncio
    async def test_add_asset_type_success(self, schema_tools, mock_ml):
        """Adding an asset type returns status=created with term details."""
        mock_term = MagicMock()
        mock_term.name = "Segmentation Mask"
        mock_term.description = "Binary mask images for segmentation"
        mock_term.rid = "AT-001"
        mock_ml.add_term.return_value = mock_term

        result = await schema_tools["add_asset_type"](
            type_name="Segmentation Mask",
            description="Binary mask images for segmentation",
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "Segmentation Mask"
        assert data["description"] == "Binary mask images for segmentation"
        assert data["rid"] == "AT-001"

        call_kwargs = mock_ml.add_term.call_args.kwargs
        assert call_kwargs["term_name"] == "Segmentation Mask"
        assert call_kwargs["description"] == "Binary mask images for segmentation"
        assert call_kwargs["exists_ok"] is True

    @pytest.mark.asyncio
    async def test_add_asset_type_error(self, schema_tools, mock_ml):
        """When add_term raises, the tool returns an error."""
        mock_ml.add_term.side_effect = RuntimeError("Permission denied")

        result = await schema_tools["add_asset_type"](
            type_name="Forbidden", description="No access"
        )

        data = assert_error(result, "Permission denied")

    @pytest.mark.asyncio
    async def test_add_asset_type_no_connection(self, schema_tools_disconnected):
        """When disconnected, add_asset_type returns a connection error."""
        result = await schema_tools_disconnected["add_asset_type"](
            type_name="X", description="Y"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestAddAssetTypeToAsset
# =============================================================================


class TestAddAssetTypeToAsset:
    """Tests for the add_asset_type_to_asset tool."""

    @pytest.mark.asyncio
    async def test_add_type_to_asset_success(self, schema_tools, mock_ml):
        """Adding an asset type to an asset returns status=added."""
        mock_asset = MagicMock()
        mock_asset.asset_types = ["Training_Data", "Raw_Data"]
        mock_ml.lookup_asset.return_value = mock_asset

        result = await schema_tools["add_asset_type_to_asset"](
            asset_rid="3JSE", type_name="Training_Data"
        )

        data = assert_success(result)
        assert data["status"] == "added"
        assert data["asset_rid"] == "3JSE"
        assert data["type_name"] == "Training_Data"
        assert data["asset_types"] == ["Training_Data", "Raw_Data"]
        mock_ml.lookup_asset.assert_called_once_with("3JSE")
        mock_asset.add_asset_type.assert_called_once_with("Training_Data")

    @pytest.mark.asyncio
    async def test_add_type_to_asset_error(self, schema_tools, mock_ml):
        """When lookup_asset raises, the tool returns an error."""
        mock_ml.lookup_asset.side_effect = RuntimeError("Asset not found")

        result = await schema_tools["add_asset_type_to_asset"](
            asset_rid="BAD", type_name="X"
        )

        data = assert_error(result, "Asset not found")

    @pytest.mark.asyncio
    async def test_add_type_to_asset_no_connection(self, schema_tools_disconnected):
        """When disconnected, add_asset_type_to_asset returns a connection error."""
        result = await schema_tools_disconnected["add_asset_type_to_asset"](
            asset_rid="X", type_name="Y"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestRemoveAssetTypeFromAsset
# =============================================================================


class TestRemoveAssetTypeFromAsset:
    """Tests for the remove_asset_type_from_asset tool."""

    @pytest.mark.asyncio
    async def test_remove_type_from_asset_success(self, schema_tools, mock_ml):
        """Removing an asset type returns status=removed with updated types."""
        mock_asset = MagicMock()
        mock_asset.asset_types = ["Raw_Data"]
        mock_ml.lookup_asset.return_value = mock_asset

        result = await schema_tools["remove_asset_type_from_asset"](
            asset_rid="3JSE", type_name="Training_Data"
        )

        data = assert_success(result)
        assert data["status"] == "removed"
        assert data["asset_rid"] == "3JSE"
        assert data["type_name"] == "Training_Data"
        assert data["asset_types"] == ["Raw_Data"]
        mock_ml.lookup_asset.assert_called_once_with("3JSE")
        mock_asset.remove_asset_type.assert_called_once_with("Training_Data")

    @pytest.mark.asyncio
    async def test_remove_type_from_asset_error(self, schema_tools, mock_ml):
        """When remove_asset_type raises, the tool returns an error."""
        mock_asset = MagicMock()
        mock_asset.remove_asset_type.side_effect = RuntimeError("Type not found on asset")
        mock_ml.lookup_asset.return_value = mock_asset

        result = await schema_tools["remove_asset_type_from_asset"](
            asset_rid="3JSE", type_name="NonExistent"
        )

        data = assert_error(result, "Type not found on asset")

    @pytest.mark.asyncio
    async def test_remove_type_from_asset_no_connection(self, schema_tools_disconnected):
        """When disconnected, remove_asset_type_from_asset returns a connection error."""
        result = await schema_tools_disconnected["remove_asset_type_from_asset"](
            asset_rid="X", type_name="Y"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestSetTableDescription
# =============================================================================


class TestSetTableDescription:
    """Tests for the set_table_description tool."""

    @pytest.mark.asyncio
    async def test_set_table_description_success(self, schema_tools, mock_ml):
        """Setting a table description returns status=updated."""
        _setup_table_handle(mock_ml, "Image")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_table_description"](
                table_name="Image",
                description="Medical images for analysis",
            )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["table_name"] == "Image"
        assert data["description"] == "Medical images for analysis"
        mock_ml.model.name_to_table.assert_called_once_with("Image")
        assert mock_handle.description == "Medical images for analysis"

    @pytest.mark.asyncio
    async def test_set_table_description_error(self, schema_tools, mock_ml):
        """When name_to_table raises, the tool returns an error."""
        mock_ml.model.name_to_table.side_effect = RuntimeError("Table not found")

        result = await schema_tools["set_table_description"](
            table_name="Nonexistent", description="desc"
        )

        data = assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_set_table_description_no_connection(
        self, schema_tools_disconnected
    ):
        """When disconnected, set_table_description returns a connection error."""
        result = await schema_tools_disconnected["set_table_description"](
            table_name="X", description="Y"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestSetTableDisplayName
# =============================================================================


class TestSetTableDisplayName:
    """Tests for the set_table_display_name tool."""

    @pytest.mark.asyncio
    async def test_set_table_display_name_success(self, schema_tools, mock_ml):
        """Setting a display name returns status=updated."""
        _setup_table_handle(mock_ml, "Image")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_table_display_name"](
                table_name="Image",
                display_name="Medical Images",
            )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["table_name"] == "Image"
        assert data["display_name"] == "Medical Images"
        mock_handle.set_display_name.assert_called_once_with("Medical Images")

    @pytest.mark.asyncio
    async def test_set_table_display_name_error(self, schema_tools, mock_ml):
        """When the handle raises, the tool returns an error."""
        mock_ml.model.name_to_table.side_effect = RuntimeError("Table not found")

        result = await schema_tools["set_table_display_name"](
            table_name="Bad", display_name="X"
        )

        data = assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_set_table_display_name_no_connection(
        self, schema_tools_disconnected
    ):
        """When disconnected, set_table_display_name returns a connection error."""
        result = await schema_tools_disconnected["set_table_display_name"](
            table_name="X", display_name="Y"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestSetRowNamePattern
# =============================================================================


class TestSetRowNamePattern:
    """Tests for the set_row_name_pattern tool."""

    @pytest.mark.asyncio
    async def test_set_row_name_pattern_success(self, schema_tools, mock_ml):
        """Setting a row name pattern returns status=updated."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_row_name_pattern"](
                table_name="Subject",
                pattern="{{{Name}}}",
            )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["table_name"] == "Subject"
        assert data["pattern"] == "{{{Name}}}"
        mock_handle.set_row_name_pattern.assert_called_once_with("{{{Name}}}")

    @pytest.mark.asyncio
    async def test_set_row_name_pattern_compound(self, schema_tools, mock_ml):
        """Setting a compound Handlebars pattern succeeds."""
        _setup_table_handle(mock_ml, "Image")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_row_name_pattern"](
                table_name="Image",
                pattern="{{{Filename}}} ({{{RID}}})",
            )

        data = assert_success(result)
        assert data["pattern"] == "{{{Filename}}} ({{{RID}}})"

    @pytest.mark.asyncio
    async def test_set_row_name_pattern_error(self, schema_tools, mock_ml):
        """When name_to_table raises, the tool returns an error."""
        mock_ml.model.name_to_table.side_effect = RuntimeError("Table not found")

        result = await schema_tools["set_row_name_pattern"](
            table_name="Bad", pattern="x"
        )

        data = assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_set_row_name_pattern_no_connection(
        self, schema_tools_disconnected
    ):
        """When disconnected, set_row_name_pattern returns a connection error."""
        result = await schema_tools_disconnected["set_row_name_pattern"](
            table_name="X", pattern="Y"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestAddColumn
# =============================================================================


class TestAddColumn:
    """Tests for the add_column tool."""

    @pytest.mark.asyncio
    async def test_add_column_success(self, schema_tools, mock_ml):
        """Adding a column returns status=created with details."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_col.name = "Age"
            mock_handle.add_column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["add_column"](
                table_name="Subject",
                column_name="Age",
                column_type="int4",
                nullok=True,
                comment="Subject age in years",
            )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["table_name"] == "Subject"
        assert data["column_name"] == "Age"
        assert data["column_type"] == "int4"

    @pytest.mark.asyncio
    async def test_add_column_default_type(self, schema_tools, mock_ml):
        """Adding a column without specifying type defaults to text."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_col.name = "Notes"
            mock_handle.add_column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["add_column"](
                table_name="Subject",
                column_name="Notes",
            )

        data = assert_success(result)
        assert data["column_type"] == "text"

    @pytest.mark.asyncio
    async def test_add_column_with_default_value(self, schema_tools, mock_ml):
        """Adding a column with a default value passes it to add_column."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_col.name = "Status"
            mock_handle.add_column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["add_column"](
                table_name="Subject",
                column_name="Status",
                column_type="text",
                default="Active",
            )

        data = assert_success(result)
        assert data["status"] == "created"
        call_kwargs = mock_handle.add_column.call_args.kwargs
        assert call_kwargs["default"] == "Active"

    @pytest.mark.asyncio
    async def test_add_column_not_null(self, schema_tools, mock_ml):
        """Adding a not-null column passes nullok=False."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_col.name = "Name"
            mock_handle.add_column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["add_column"](
                table_name="Subject",
                column_name="Name",
                nullok=False,
            )

        data = assert_success(result)
        call_kwargs = mock_handle.add_column.call_args.kwargs
        assert call_kwargs["nullok"] is False

    @pytest.mark.asyncio
    async def test_add_column_unknown_type(self, schema_tools, mock_ml):
        """An unrecognised column type falls back to text."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_col.name = "Data"
            mock_handle.add_column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["add_column"](
                table_name="Subject",
                column_name="Data",
                column_type="unknown_type",
            )

        data = assert_success(result)
        assert data["status"] == "created"
        # column_type in response reflects the input string
        assert data["column_type"] == "unknown_type"

    @pytest.mark.asyncio
    async def test_add_column_error(self, schema_tools, mock_ml):
        """When add_column raises, the tool returns an error."""
        mock_ml.model.name_to_table.side_effect = RuntimeError("Table not found")

        result = await schema_tools["add_column"](
            table_name="Nonexistent", column_name="Col"
        )

        data = assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_add_column_no_connection(self, schema_tools_disconnected):
        """When disconnected, add_column returns a connection error."""
        result = await schema_tools_disconnected["add_column"](
            table_name="X", column_name="Y"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestSetColumnDescription
# =============================================================================


class TestSetColumnDescription:
    """Tests for the set_column_description tool."""

    @pytest.mark.asyncio
    async def test_set_column_description_success(self, schema_tools, mock_ml):
        """Setting a column description returns status=updated."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_handle.column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_column_description"](
                table_name="Subject",
                column_name="Age",
                description="Subject age in years at enrollment",
            )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["table_name"] == "Subject"
        assert data["column_name"] == "Age"
        assert data["description"] == "Subject age in years at enrollment"
        mock_handle.column.assert_called_once_with("Age")
        assert mock_col.description == "Subject age in years at enrollment"

    @pytest.mark.asyncio
    async def test_set_column_description_error(self, schema_tools, mock_ml):
        """When table lookup fails, the tool returns an error."""
        mock_ml.model.name_to_table.side_effect = RuntimeError("Table not found")

        result = await schema_tools["set_column_description"](
            table_name="Bad", column_name="Col", description="desc"
        )

        data = assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_set_column_description_column_not_found(
        self, schema_tools, mock_ml
    ):
        """When column lookup fails, the tool returns an error."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_handle.column.side_effect = KeyError("Column 'Bad' not found")
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_column_description"](
                table_name="Subject",
                column_name="Bad",
                description="desc",
            )

        data = assert_error(result)

    @pytest.mark.asyncio
    async def test_set_column_description_no_connection(
        self, schema_tools_disconnected
    ):
        """When disconnected, set_column_description returns a connection error."""
        result = await schema_tools_disconnected["set_column_description"](
            table_name="X", column_name="Y", description="Z"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestSetColumnDisplayName
# =============================================================================


class TestSetColumnDisplayName:
    """Tests for the set_column_display_name tool."""

    @pytest.mark.asyncio
    async def test_set_column_display_name_success(self, schema_tools, mock_ml):
        """Setting a column display name returns status=updated."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_handle.column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_column_display_name"](
                table_name="Subject",
                column_name="DOB",
                display_name="Date of Birth",
            )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["table_name"] == "Subject"
        assert data["column_name"] == "DOB"
        assert data["display_name"] == "Date of Birth"
        mock_handle.column.assert_called_once_with("DOB")
        mock_col.set_display_name.assert_called_once_with("Date of Birth")

    @pytest.mark.asyncio
    async def test_set_column_display_name_error(self, schema_tools, mock_ml):
        """When table lookup fails, the tool returns an error."""
        mock_ml.model.name_to_table.side_effect = RuntimeError("Table not found")

        result = await schema_tools["set_column_display_name"](
            table_name="Bad", column_name="Col", display_name="Name"
        )

        data = assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_set_column_display_name_no_connection(
        self, schema_tools_disconnected
    ):
        """When disconnected, set_column_display_name returns a connection error."""
        result = await schema_tools_disconnected["set_column_display_name"](
            table_name="X", column_name="Y", display_name="Z"
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestSetColumnNullok
# =============================================================================


class TestSetColumnNullok:
    """Tests for the set_column_nullok tool."""

    @pytest.mark.asyncio
    async def test_set_column_nullok_true(self, schema_tools, mock_ml):
        """Setting nullok=True returns status=updated."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_handle.column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_column_nullok"](
                table_name="Subject",
                column_name="Age",
                nullok=True,
            )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["table_name"] == "Subject"
        assert data["column_name"] == "Age"
        assert data["nullok"] is True
        mock_col.set_nullok.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_set_column_nullok_false(self, schema_tools, mock_ml):
        """Setting nullok=False returns status=updated."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_handle.column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_column_nullok"](
                table_name="Subject",
                column_name="Name",
                nullok=False,
            )

        data = assert_success(result)
        assert data["nullok"] is False
        mock_col.set_nullok.assert_called_once_with(False)

    @pytest.mark.asyncio
    async def test_set_column_nullok_error(self, schema_tools, mock_ml):
        """When set_nullok raises (e.g. column has NULLs), the tool returns an error."""
        _setup_table_handle(mock_ml, "Subject")

        with patch(_TABLE_HANDLE_PATCH) as MockTableHandle:
            mock_handle = MagicMock()
            mock_col = MagicMock()
            mock_col.set_nullok.side_effect = RuntimeError("Column contains NULL values")
            mock_handle.column.return_value = mock_col
            MockTableHandle.return_value = mock_handle

            result = await schema_tools["set_column_nullok"](
                table_name="Subject",
                column_name="Name",
                nullok=False,
            )

        data = assert_error(result, "Column contains NULL values")

    @pytest.mark.asyncio
    async def test_set_column_nullok_no_connection(self, schema_tools_disconnected):
        """When disconnected, set_column_nullok returns a connection error."""
        result = await schema_tools_disconnected["set_column_nullok"](
            table_name="X", column_name="Y", nullok=True
        )
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestToolRegistration
# =============================================================================


class TestToolRegistration:
    """Verify that all schema tools are registered."""

    def test_all_tools_registered(self, schema_tools):
        """All schema tools should be captured by the fixture."""
        expected_tools = [
            "create_table",
            "create_asset_table",
            "list_asset_executions",
            "add_asset_type",
            "add_asset_type_to_asset",
            "remove_asset_type_from_asset",
            "set_table_description",
            "set_table_display_name",
            "set_row_name_pattern",
            "add_column",
            "set_column_description",
            "set_column_display_name",
            "set_column_nullok",
        ]
        for tool_name in expected_tools:
            assert tool_name in schema_tools, f"Missing tool: {tool_name}"

    def test_tool_count(self, schema_tools):
        """There should be exactly 13 schema tools."""
        assert len(schema_tools) == 13
