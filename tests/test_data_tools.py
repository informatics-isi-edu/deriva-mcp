"""Integration tests for data query and manipulation tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.tools.data import register_data_tools
from deriva_ml_mcp.tools.catalog import register_catalog_tools
from deriva_ml_mcp.tools.schema import register_schema_tools

if TYPE_CHECKING:
    from tests.conftest import CatalogManager

from tests.conftest import parse_json_result


def setup_tools(conn_manager: ConnectionManager) -> dict:
    """Helper to register tools and capture them."""
    mcp = MagicMock()
    captured_tools = {}

    def capture_tool():
        def decorator(func):
            captured_tools[func.__name__] = func
            return func
        return decorator

    mcp.tool = capture_tool
    register_catalog_tools(mcp, conn_manager)
    register_schema_tools(mcp, conn_manager)
    register_data_tools(mcp, conn_manager)
    return captured_tools


class TestGetTable:
    """Tests for the get_table tool."""

    @pytest.mark.asyncio
    async def test_get_table_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting all records from a table."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["get_table"](
            table_name="Subject",
        )

        data = parse_json_result(result)
        assert data["table"] == "Subject"
        assert "records" in data
        assert "count" in data
        assert isinstance(data["records"], list)

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_get_table_with_limit(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting records with a limit."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["get_table"](
            table_name="Subject",
            limit=5,
        )

        data = parse_json_result(result)
        assert data["limit"] == 5
        assert data["count"] <= 5

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_get_table_no_connection(self):
        """Test getting table without connection fails."""
        conn_manager = ConnectionManager()
        tools = setup_tools(conn_manager)

        result = await tools["get_table"](
            table_name="Subject",
        )

        data = parse_json_result(result)
        assert data["status"] == "error"


class TestQueryTable:
    """Tests for the query_table tool."""

    @pytest.mark.asyncio
    async def test_query_table_all(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test querying all records from a table."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["query_table"](
            table_name="Subject",
        )

        data = parse_json_result(result)
        assert data["table"] == "Subject"
        assert "records" in data
        assert isinstance(data["records"], list)

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_query_table_with_columns(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test querying specific columns."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["query_table"](
            table_name="Subject",
            columns=["RID", "Name"],
        )

        data = parse_json_result(result)
        if data["count"] > 0:
            # Each record should only have the requested columns
            record = data["records"][0]
            assert "RID" in record
            assert "Name" in record

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_query_table_with_limit_offset(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test querying with pagination."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["query_table"](
            table_name="Subject",
            limit=2,
            offset=0,
        )

        data = parse_json_result(result)
        assert data["limit"] == 2
        assert data["offset"] == 0

        conn_manager.disconnect()


class TestCountTable:
    """Tests for the count_table tool."""

    @pytest.mark.asyncio
    async def test_count_table(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test counting records in a table."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["count_table"](
            table_name="Subject",
        )

        data = parse_json_result(result)
        assert data["table"] == "Subject"
        assert "count" in data
        assert isinstance(data["count"], int)
        assert data["count"] >= 0

        conn_manager.disconnect()


class TestInsertRecords:
    """Tests for the insert_records tool."""

    @pytest.mark.asyncio
    async def test_insert_records_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test inserting records into a domain table."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["insert_records"](
            table_name="Subject",
            records=[
                {"Name": "Test Subject 1"},
                {"Name": "Test Subject 2"},
            ],
        )

        data = parse_json_result(result)
        assert data["status"] == "inserted"
        assert data["inserted_count"] == 2
        assert len(data["rids"]) == 2

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_insert_records_managed_table_fails(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test that inserting into managed tables fails with guidance."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Try to insert into Dataset table (managed)
        result = await tools["insert_records"](
            table_name="Dataset",
            records=[{"Name": "Test"}],
        )

        data = parse_json_result(result)
        assert data["status"] == "error"
        assert "create_dataset" in data["message"]

        conn_manager.disconnect()


class TestGetRecord:
    """Tests for the get_record tool."""

    @pytest.mark.asyncio
    async def test_get_record_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting a single record by RID."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # First insert a record to get its RID
        insert_result = await tools["insert_records"](
            table_name="Subject",
            records=[{"Name": "Record To Fetch"}],
        )
        insert_data = parse_json_result(insert_result)
        rid = insert_data["rids"][0]

        # Get the record
        result = await tools["get_record"](
            table_name="Subject",
            rid=rid,
        )

        data = parse_json_result(result)
        assert data["rid"] == rid
        assert "record" in data
        assert data["record"]["Name"] == "Record To Fetch"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_get_record_not_found(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting a record that doesn't exist."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["get_record"](
            table_name="Subject",
            rid="NONEXISTENT-RID",
        )

        data = parse_json_result(result)
        assert data["status"] == "not_found"

        conn_manager.disconnect()


class TestUpdateRecord:
    """Tests for the update_record tool."""

    @pytest.mark.asyncio
    async def test_update_record_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test updating a record."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # First insert a record
        insert_result = await tools["insert_records"](
            table_name="Subject",
            records=[{"Name": "Original Name"}],
        )
        insert_data = parse_json_result(insert_result)
        rid = insert_data["rids"][0]

        # Update the record
        result = await tools["update_record"](
            table_name="Subject",
            rid=rid,
            updates={"Name": "Updated Name"},
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["rid"] == rid
        assert "Name" in data["updated_fields"]

        # Verify the update
        get_result = await tools["get_record"](
            table_name="Subject",
            rid=rid,
        )
        get_data = parse_json_result(get_result)
        assert get_data["record"]["Name"] == "Updated Name"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_update_record_not_found(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test updating a record that doesn't exist."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["update_record"](
            table_name="Subject",
            rid="NONEXISTENT-RID",
            updates={"Name": "New Name"},
        )

        data = parse_json_result(result)
        assert data["status"] == "not_found"

        conn_manager.disconnect()
