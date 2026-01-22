"""Integration tests for schema manipulation tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.tools.schema import register_schema_tools
from deriva_ml_mcp.tools.catalog import register_catalog_tools

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
    return captured_tools


class TestCreateTable:
    """Tests for the create_table tool."""

    @pytest.mark.asyncio
    async def test_create_simple_table(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating a simple table."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["create_table"](
            table_name="TestSubject",
            columns=[
                {"name": "Name", "type": "text", "nullok": False},
                {"name": "Age", "type": "int4"},
                {"name": "Notes", "type": "markdown"},
            ],
            comment="Test subject table",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["table_name"] == "TestSubject"
        assert "Name" in data["columns"]
        assert "Age" in data["columns"]

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_create_table_with_foreign_key(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating a table with foreign key."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # First create the referenced table
        await tools["create_table"](
            table_name="Project",
            columns=[{"name": "Name", "type": "text"}],
        )

        # Then create table with foreign key
        result = await tools["create_table"](
            table_name="Experiment",
            columns=[
                {"name": "Name", "type": "text"},
                {"name": "Project", "type": "text"},
            ],
            foreign_keys=[
                {"column": "Project", "referenced_table": "Project"},
            ],
            comment="Experiment table linked to Project",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["table_name"] == "Experiment"

        conn_manager.disconnect()


class TestCreateAssetTable:
    """Tests for the create_asset_table tool."""

    @pytest.mark.asyncio
    async def test_create_asset_table(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating an asset table."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["create_asset_table"](
            asset_name="TestDocument",
            columns=[{"name": "Title", "type": "text"}],
            comment="Test document asset table",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["table_name"] == "TestDocument"
        # Asset tables should have URL, Filename, etc.
        assert "URL" in data["columns"]
        assert "Filename" in data["columns"]

        conn_manager.disconnect()


class TestGetSchemaDescription:
    """Tests for the get_schema_description tool."""

    @pytest.mark.asyncio
    async def test_get_schema_description(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting schema description."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["get_schema_description"]()

        data = parse_json_result(result)
        assert "domain_schema" in data
        assert "ml_schema" in data
        assert "schemas" in data

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_get_schema_with_system_columns(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting schema with system columns included."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["get_schema_description"](
            include_system_columns=True,
        )

        data = parse_json_result(result)
        assert "schemas" in data

        conn_manager.disconnect()


class TestTableManipulation:
    """Tests for table manipulation tools."""

    @pytest.mark.asyncio
    async def test_set_table_description(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test setting table description."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["set_table_description"](
            table_name="Subject",
            description="Updated description for Subject table",
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["description"] == "Updated description for Subject table"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_set_table_display_name(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test setting table display name."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["set_table_display_name"](
            table_name="Subject",
            display_name="Study Subjects",
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["display_name"] == "Study Subjects"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_set_row_name_pattern(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test setting row name pattern."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["set_row_name_pattern"](
            table_name="Subject",
            pattern="{{{Name}}} ({{{RID}}})",
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["pattern"] == "{{{Name}}} ({{{RID}}})"

        conn_manager.disconnect()


class TestColumnManipulation:
    """Tests for column manipulation tools."""

    @pytest.mark.asyncio
    async def test_add_column(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a column to a table."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["add_column"](
            table_name="Subject",
            column_name="Height",
            column_type="float4",
            nullok=True,
            comment="Subject height in cm",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["column_name"] == "Height"
        assert data["column_type"] == "float4"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_set_column_description(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test setting column description."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["set_column_description"](
            table_name="Subject",
            column_name="Name",
            description="Full name of the subject",
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["description"] == "Full name of the subject"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_set_column_display_name(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test setting column display name."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["set_column_display_name"](
            table_name="Subject",
            column_name="Name",
            display_name="Subject Name",
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["display_name"] == "Subject Name"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_get_table_columns(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting table columns."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["get_table_columns"](
            table_name="Subject",
        )

        data = parse_json_result(result)
        assert isinstance(data, list)
        # Should have at least Name column
        column_names = [c["name"] for c in data]
        assert "Name" in column_names

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_get_table_columns_with_system(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting table columns including system columns."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["get_table_columns"](
            table_name="Subject",
            include_system=True,
        )

        data = parse_json_result(result)
        assert isinstance(data, list)
        column_names = [c["name"] for c in data]
        # Should have system columns
        assert "RID" in column_names

        conn_manager.disconnect()


class TestAssetTypes:
    """Tests for asset type tools."""

    @pytest.mark.asyncio
    async def test_add_asset_type(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a new asset type."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["add_asset_type"](
            type_name="Test_Asset_Type",
            description="A test asset type for testing",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["name"] == "Test_Asset_Type"

        conn_manager.disconnect()
