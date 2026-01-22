"""Integration tests for catalog management tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, AsyncMock

import pytest

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.tools.catalog import register_catalog_tools

if TYPE_CHECKING:
    from tests.conftest import CatalogManager

from tests.conftest import parse_json_result, assert_success, assert_error


class TestConnectCatalog:
    """Tests for the connect_catalog tool."""

    @pytest.mark.asyncio
    async def test_connect_success(self, catalog_manager: "CatalogManager"):
        """Test successful connection to a catalog."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()
        register_catalog_tools(mcp, conn_manager)

        # Get the registered tool function
        connect_catalog = None
        for call in mcp.tool.return_value.call_args_list:
            pass
        # The tool decorator returns a function, find it
        connect_catalog = mcp.tool.return_value

        # Since we're testing the actual function, extract it differently
        # Re-register to capture the function
        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        # Test connect
        result = await captured_tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
            domain_schema=catalog_manager.domain_schema,
        )

        data = parse_json_result(result)
        assert data["status"] == "connected"
        assert data["hostname"] == catalog_manager.hostname
        assert data["catalog_id"] == str(catalog_manager.catalog_id)
        assert data["domain_schema"] == catalog_manager.domain_schema
        assert "workflow_rid" in data
        assert "execution_rid" in data

        # Clean up
        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_connect_invalid_catalog(self, catalog_host: str):
        """Test connection to non-existent catalog returns error."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        result = await captured_tools["connect_catalog"](
            hostname=catalog_host,
            catalog_id="99999999",  # Non-existent catalog
        )

        data = parse_json_result(result)
        assert data["status"] == "error"


class TestDisconnectCatalog:
    """Tests for the disconnect_catalog tool."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, catalog_manager: "CatalogManager"):
        """Test successful disconnection."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        # First connect
        await captured_tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Then disconnect
        result = await captured_tools["disconnect_catalog"]()
        data = parse_json_result(result)
        assert data["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_disconnect_no_connection(self):
        """Test disconnect when no connection exists."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        result = await captured_tools["disconnect_catalog"]()
        data = parse_json_result(result)
        assert data["status"] == "no_active_connection"


class TestSetActiveCatalog:
    """Tests for the set_active_catalog tool."""

    @pytest.mark.asyncio
    async def test_set_active_success(self, catalog_manager: "CatalogManager"):
        """Test setting active catalog."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        # Connect first
        await captured_tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Set active
        result = await captured_tools["set_active_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        data = parse_json_result(result)
        assert data["status"] == "success"

        # Clean up
        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_set_active_not_found(self):
        """Test setting active to non-existent connection."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        result = await captured_tools["set_active_catalog"](
            hostname="nonexistent.host",
            catalog_id="123",
        )

        data = parse_json_result(result)
        assert data["status"] == "error"


class TestCreateCatalog:
    """Tests for the create_catalog tool."""

    @pytest.mark.asyncio
    async def test_create_catalog_success(self, catalog_host: str):
        """Test creating a new catalog."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        result = await captured_tools["create_catalog"](
            hostname=catalog_host,
            project_name="test_mcp_create",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["hostname"] == catalog_host
        assert "catalog_id" in data
        assert data["project_name"] == "test_mcp_create"

        # Clean up - delete the created catalog
        catalog_id = data["catalog_id"]
        await captured_tools["delete_catalog"](
            hostname=catalog_host,
            catalog_id=catalog_id,
        )


class TestDeleteCatalog:
    """Tests for the delete_catalog tool."""

    @pytest.mark.asyncio
    async def test_delete_catalog_success(self, catalog_host: str):
        """Test deleting a catalog."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        # First create a catalog to delete
        create_result = await captured_tools["create_catalog"](
            hostname=catalog_host,
            project_name="test_mcp_delete",
        )
        create_data = parse_json_result(create_result)
        catalog_id = create_data["catalog_id"]

        # Delete it
        result = await captured_tools["delete_catalog"](
            hostname=catalog_host,
            catalog_id=catalog_id,
        )

        data = parse_json_result(result)
        assert data["status"] == "deleted"
        assert data["catalog_id"] == catalog_id


class TestValidateRids:
    """Tests for the validate_rids tool."""

    @pytest.mark.asyncio
    async def test_validate_rids_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test validating existing RIDs."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        # First populate the catalog with datasets
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        # Connect
        await captured_tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Validate the dataset RID
        result = await captured_tools["validate_rids"](
            dataset_rids=[dataset_desc.dataset.dataset_rid],
        )

        data = parse_json_result(result)
        assert data["is_valid"] is True
        assert len(data["errors"]) == 0

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_validate_rids_missing(self, mcp_connection_manager: ConnectionManager):
        """Test validating non-existent RIDs."""
        mcp = MagicMock()

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, mcp_connection_manager)

        result = await captured_tools["validate_rids"](
            dataset_rids=["NONEXISTENT-RID"],
        )

        data = parse_json_result(result)
        assert data["is_valid"] is False
        assert len(data["errors"]) > 0


class TestCite:
    """Tests for the cite tool."""

    @pytest.mark.asyncio
    async def test_cite_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test generating citation URL."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        # Populate with datasets
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        # Connect
        await captured_tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Generate citation
        result = await captured_tools["cite"](
            rid=dataset_desc.dataset.dataset_rid,
        )

        data = parse_json_result(result)
        assert "url" in data
        assert data["rid"] == dataset_desc.dataset.dataset_rid
        assert data["is_snapshot"] is True

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_cite_current(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test generating citation URL without snapshot."""
        mcp = MagicMock()
        conn_manager = ConnectionManager()

        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        captured_tools = {}

        def capture_tool():
            def decorator(func):
                captured_tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_catalog_tools(mcp, conn_manager)

        await captured_tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await captured_tools["cite"](
            rid=dataset_desc.dataset.dataset_rid,
            current=True,
        )

        data = parse_json_result(result)
        assert "url" in data
        assert data["is_snapshot"] is False

        conn_manager.disconnect()
