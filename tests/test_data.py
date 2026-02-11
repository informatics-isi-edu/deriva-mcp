"""Tests for data query and manipulation tools.

Tests cover all six data tools:
- get_table: retrieve all records from a table
- query_table: filtered/paginated queries
- count_table: count records with optional filters
- insert_records: insert new records with safety checks
- get_record: fetch a single record by RID
- update_record: update fields on an existing record
"""

import pytest
from unittest.mock import MagicMock

from tests.conftest import parse_json_result, assert_success, assert_error


# =============================================================================
# Helpers
# =============================================================================


def _setup_path_builder(mock_ml, fetch_data=None, table_schema="test_schema"):
    """Configure mock_ml with a realistic pathBuilder chain.

    Sets up: ml.model.name_to_table(), ml.pathBuilder() -> pb.schemas[...].tables[...]
    with .filter(), .entities(), .fetch(), .insert(), .update() support.

    Returns the mock_path object for further customisation.
    """
    if fetch_data is None:
        fetch_data = []

    # Table model object
    mock_table = MagicMock()
    mock_table.schema.name = table_schema
    mock_ml.model.name_to_table.return_value = mock_table
    mock_ml.model.is_vocabulary.return_value = False
    mock_ml.model.is_asset.return_value = False

    # pathBuilder chain
    mock_pb = MagicMock()
    mock_path = MagicMock()
    mock_entities = MagicMock()
    mock_entities.fetch.return_value = fetch_data

    mock_path.entities.return_value = mock_entities
    mock_path.filter.return_value = mock_path
    mock_path.insert.return_value = fetch_data  # insert returns inserted rows
    mock_pb.schemas.__getitem__.return_value.tables.__getitem__.return_value = mock_path

    mock_ml.pathBuilder.return_value = mock_pb

    return mock_path


# =============================================================================
# TestGetTable
# =============================================================================


class TestGetTable:
    """Tests for the get_table tool."""

    @pytest.mark.asyncio
    async def test_get_table_returns_records(self, data_tools, mock_ml):
        """get_table returns all rows from the table."""
        rows = [
            {"RID": "1-AAA", "Name": "Alpha"},
            {"RID": "1-BBB", "Name": "Beta"},
        ]
        mock_ml.get_table_as_dict.return_value = iter(rows)

        result = await data_tools["get_table"]("Subject")
        data = assert_success(result)

        assert data["table"] == "Subject"
        assert data["count"] == 2
        assert data["records"] == rows
        assert data["limit"] == 1000
        mock_ml.get_table_as_dict.assert_called_once_with("Subject")

    @pytest.mark.asyncio
    async def test_get_table_respects_limit(self, data_tools, mock_ml):
        """get_table stops after 'limit' records."""
        rows = [{"RID": f"1-{i}", "Val": i} for i in range(10)]
        mock_ml.get_table_as_dict.return_value = iter(rows)

        result = await data_tools["get_table"]("BigTable", limit=3)
        data = assert_success(result)

        assert data["count"] == 3
        assert data["limit"] == 3

    @pytest.mark.asyncio
    async def test_get_table_empty(self, data_tools, mock_ml):
        """get_table on an empty table returns zero records."""
        mock_ml.get_table_as_dict.return_value = iter([])

        result = await data_tools["get_table"]("Empty")
        data = assert_success(result)

        assert data["count"] == 0
        assert data["records"] == []

    @pytest.mark.asyncio
    async def test_get_table_disconnected(self, data_tools_disconnected):
        """get_table returns an error when no connection is active."""
        result = await data_tools_disconnected["get_table"]("X")
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestQueryTable
# =============================================================================


class TestQueryTable:
    """Tests for the query_table tool."""

    @pytest.mark.asyncio
    async def test_query_table_basic(self, data_tools, mock_ml):
        """query_table fetches records through the pathBuilder chain."""
        rows = [{"RID": "1-AAA", "Name": "Alpha"}]
        _setup_path_builder(mock_ml, fetch_data=rows)

        result = await data_tools["query_table"]("Image")
        data = assert_success(result)

        assert data["table"] == "Image"
        assert data["records"] == rows
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_query_table_with_filters(self, data_tools, mock_ml):
        """query_table applies equality filters via path.filter()."""
        rows = [{"RID": "1-CCC", "Species": "Human"}]
        mock_path = _setup_path_builder(mock_ml, fetch_data=rows)

        result = await data_tools["query_table"](
            "Subject", filters={"Species": "Human"}
        )
        data = assert_success(result)

        assert data["count"] == 1
        mock_path.filter.assert_called()

    @pytest.mark.asyncio
    async def test_query_table_with_columns(self, data_tools, mock_ml):
        """query_table selects only requested columns."""
        rows = [{"RID": "1-AAA", "Name": "Alpha", "Age": 30, "Status": "Active"}]
        _setup_path_builder(mock_ml, fetch_data=rows)

        result = await data_tools["query_table"](
            "Subject", columns=["RID", "Name"]
        )
        data = assert_success(result)

        record = data["records"][0]
        assert set(record.keys()) == {"RID", "Name"}

    @pytest.mark.asyncio
    async def test_query_table_caps_limit_at_1000(self, data_tools, mock_ml):
        """query_table caps the limit to 1000 even if a higher value is passed."""
        _setup_path_builder(mock_ml, fetch_data=[])

        result = await data_tools["query_table"]("Image", limit=5000)
        data = assert_success(result)

        assert data["limit"] == 1000

    @pytest.mark.asyncio
    async def test_query_table_with_offset(self, data_tools, mock_ml):
        """query_table applies offset by slicing fetched records.

        The implementation fetches (offset + limit) rows and slices from
        offset:, so with 5 available rows, offset=2, limit=2 fetches 4 rows
        and returns the last 2 after slicing (indices 2 and 3).
        """
        rows = [{"RID": f"1-{i}"} for i in range(5)]
        mock_path = _setup_path_builder(mock_ml, fetch_data=rows)
        mock_entities = mock_path.entities.return_value

        result = await data_tools["query_table"]("Image", limit=2, offset=2)
        data = assert_success(result)

        assert data["offset"] == 2
        # fetch is called with limit = offset + limit = 4
        mock_entities.fetch.assert_called_once_with(limit=4)
        # The source returns all records after slicing at offset, which is
        # rows[2:] from the 5 fetched -> 3 records (indices 2, 3, 4) since
        # all 5 rows are returned by the mock even though fetch(limit=4).
        # Mock returns all 5 rows regardless. After slicing [2:] -> 3 records.
        assert data["count"] == 3
        assert data["records"] == [{"RID": "1-2"}, {"RID": "1-3"}, {"RID": "1-4"}]

    @pytest.mark.asyncio
    async def test_query_table_disconnected(self, data_tools_disconnected):
        """query_table returns an error when no connection is active."""
        result = await data_tools_disconnected["query_table"]("X")
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestCountTable
# =============================================================================


class TestCountTable:
    """Tests for the count_table tool."""

    @pytest.mark.asyncio
    async def test_count_table_basic(self, data_tools, mock_ml):
        """count_table returns the number of rows."""
        rows = [{"RID": "1"}, {"RID": "2"}, {"RID": "3"}]
        _setup_path_builder(mock_ml, fetch_data=rows)

        result = await data_tools["count_table"]("Image")
        data = assert_success(result)

        assert data["table"] == "Image"
        assert data["count"] == 3

    @pytest.mark.asyncio
    async def test_count_table_with_filters(self, data_tools, mock_ml):
        """count_table applies filters before counting."""
        rows = [{"RID": "1-X"}]
        mock_path = _setup_path_builder(mock_ml, fetch_data=rows)

        result = await data_tools["count_table"](
            "Subject", filters={"Status": "Active"}
        )
        data = assert_success(result)

        assert data["count"] == 1
        assert data["filters"] == {"Status": "Active"}
        mock_path.filter.assert_called()

    @pytest.mark.asyncio
    async def test_count_table_empty(self, data_tools, mock_ml):
        """count_table returns 0 for an empty table."""
        _setup_path_builder(mock_ml, fetch_data=[])

        result = await data_tools["count_table"]("Empty")
        data = assert_success(result)

        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_count_table_disconnected(self, data_tools_disconnected):
        """count_table returns an error when no connection is active."""
        result = await data_tools_disconnected["count_table"]("X")
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestInsertRecords
# =============================================================================


class TestInsertRecords:
    """Tests for the insert_records tool."""

    @pytest.mark.asyncio
    async def test_insert_records_success(self, data_tools, mock_ml):
        """insert_records inserts into a domain table and returns RIDs."""
        inserted = [{"RID": "2-NEW1", "Name": "New"}, {"RID": "2-NEW2", "Name": "Also New"}]
        mock_path = _setup_path_builder(mock_ml, fetch_data=inserted)

        records_to_insert = [{"Name": "New"}, {"Name": "Also New"}]
        result = await data_tools["insert_records"]("Subject", records_to_insert)
        data = assert_success(result)

        assert data["status"] == "inserted"
        assert data["table"] == "Subject"
        assert data["inserted_count"] == 2
        assert data["rids"] == ["2-NEW1", "2-NEW2"]
        mock_path.insert.assert_called_once_with(records_to_insert)

    @pytest.mark.asyncio
    async def test_insert_records_blocks_managed_table_dataset(self, data_tools, mock_ml):
        """insert_records rejects inserts into the Dataset managed table."""
        _setup_path_builder(mock_ml)

        result = await data_tools["insert_records"]("Dataset", [{"Name": "x"}])
        data = assert_error(result, "create_dataset")

        assert data["table"] == "Dataset"

    @pytest.mark.asyncio
    async def test_insert_records_blocks_managed_table_execution(self, data_tools, mock_ml):
        """insert_records rejects inserts into the Execution managed table."""
        _setup_path_builder(mock_ml)

        result = await data_tools["insert_records"]("Execution", [{"Name": "x"}])
        data = assert_error(result, "create_execution")

    @pytest.mark.asyncio
    async def test_insert_records_blocks_managed_table_workflow(self, data_tools, mock_ml):
        """insert_records rejects inserts into the Workflow managed table."""
        _setup_path_builder(mock_ml)

        result = await data_tools["insert_records"]("Workflow", [{"Name": "x"}])
        data = assert_error(result, "create_workflow")

    @pytest.mark.asyncio
    async def test_insert_records_blocks_managed_pattern_dataset_member(self, data_tools, mock_ml):
        """insert_records rejects inserts into dataset member tables (pattern match)."""
        _setup_path_builder(mock_ml)

        result = await data_tools["insert_records"]("Image_Dataset_Image", [{"x": 1}])
        data = assert_error(result, "add_dataset_members")

    @pytest.mark.asyncio
    async def test_insert_records_blocks_managed_pattern_feature(self, data_tools, mock_ml):
        """insert_records rejects inserts into feature tables (pattern match)."""
        _setup_path_builder(mock_ml)

        result = await data_tools["insert_records"]("Execution_Image_Feature", [{"x": 1}])
        data = assert_error(result, "add_feature_value")

    @pytest.mark.asyncio
    async def test_insert_records_blocks_vocabulary_table(self, data_tools, mock_ml):
        """insert_records rejects inserts into vocabulary tables."""
        mock_table = MagicMock()
        mock_table.schema.name = "test_schema"
        mock_ml.model.name_to_table.return_value = mock_table
        mock_ml.model.is_vocabulary.return_value = True
        mock_ml.model.is_asset.return_value = False

        result = await data_tools["insert_records"]("MyVocab", [{"Name": "x"}])
        data = assert_error(result, "add_term")

    @pytest.mark.asyncio
    async def test_insert_records_blocks_asset_table(self, data_tools, mock_ml):
        """insert_records rejects inserts into asset tables."""
        mock_table = MagicMock()
        mock_table.schema.name = "test_schema"
        mock_ml.model.name_to_table.return_value = mock_table
        mock_ml.model.is_vocabulary.return_value = False
        mock_ml.model.is_asset.return_value = True

        result = await data_tools["insert_records"]("MyAsset", [{"URL": "x"}])
        data = assert_error(result, "asset_file_path")

    @pytest.mark.asyncio
    async def test_insert_records_blocks_ml_schema_table(self, data_tools, mock_ml):
        """insert_records rejects inserts into ML schema tables."""
        mock_table = MagicMock()
        mock_table.schema.name = "deriva-ml"  # matches mock_ml.ml_schema
        mock_ml.model.name_to_table.return_value = mock_table
        mock_ml.model.is_vocabulary.return_value = False
        mock_ml.model.is_asset.return_value = False

        result = await data_tools["insert_records"]("SomeMLTable", [{"x": 1}])
        data = assert_error(result, "dedicated tool")

    @pytest.mark.asyncio
    async def test_insert_records_disconnected(self, data_tools_disconnected):
        """insert_records returns an error when no connection is active."""
        result = await data_tools_disconnected["insert_records"]("X", [{}])
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestGetRecord
# =============================================================================


class TestGetRecord:
    """Tests for the get_record tool."""

    @pytest.mark.asyncio
    async def test_get_record_found(self, data_tools, mock_ml):
        """get_record returns the matching record when found."""
        record = {"RID": "1-ABC", "Name": "Test", "Age": 42}
        _setup_path_builder(mock_ml, fetch_data=[record])

        result = await data_tools["get_record"]("Subject", "1-ABC")
        data = assert_success(result)

        assert data["table"] == "Subject"
        assert data["rid"] == "1-ABC"
        assert data["record"] == record

    @pytest.mark.asyncio
    async def test_get_record_not_found(self, data_tools, mock_ml):
        """get_record returns not_found when no record matches."""
        _setup_path_builder(mock_ml, fetch_data=[])

        result = await data_tools["get_record"]("Subject", "NOPE")
        data = parse_json_result(result)

        assert data["status"] == "not_found"
        assert "NOPE" in data["message"]

    @pytest.mark.asyncio
    async def test_get_record_disconnected(self, data_tools_disconnected):
        """get_record returns an error when no connection is active."""
        result = await data_tools_disconnected["get_record"]("X", "1-X")
        assert_error(result, "No active catalog connection")


# =============================================================================
# TestUpdateRecord
# =============================================================================


class TestUpdateRecord:
    """Tests for the update_record tool."""

    @pytest.mark.asyncio
    async def test_update_record_success(self, data_tools, mock_ml):
        """update_record applies updates and returns success."""
        existing = {"RID": "1-ABC", "Name": "Old", "Age": 30}
        mock_path = _setup_path_builder(mock_ml, fetch_data=[existing])

        result = await data_tools["update_record"](
            "Subject", "1-ABC", {"Name": "New", "Age": 31}
        )
        data = assert_success(result)

        assert data["status"] == "updated"
        assert data["table"] == "Subject"
        assert data["rid"] == "1-ABC"
        assert set(data["updated_fields"]) == {"Name", "Age"}
        mock_path.update.assert_called_once()

        # Verify the record passed to update contains the merged values
        updated_record = mock_path.update.call_args[0][0][0]
        assert updated_record["Name"] == "New"
        assert updated_record["Age"] == 31
        assert updated_record["RID"] == "1-ABC"

    @pytest.mark.asyncio
    async def test_update_record_not_found(self, data_tools, mock_ml):
        """update_record returns not_found when the RID does not exist."""
        _setup_path_builder(mock_ml, fetch_data=[])

        result = await data_tools["update_record"](
            "Subject", "NOPE", {"Name": "X"}
        )
        data = parse_json_result(result)

        assert data["status"] == "not_found"
        assert "NOPE" in data["message"]

    @pytest.mark.asyncio
    async def test_update_record_disconnected(self, data_tools_disconnected):
        """update_record returns an error when no connection is active."""
        result = await data_tools_disconnected["update_record"](
            "X", "1-X", {"a": 1}
        )
        assert_error(result, "No active catalog connection")
