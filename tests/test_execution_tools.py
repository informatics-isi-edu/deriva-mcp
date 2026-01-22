"""Integration tests for execution management tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.tools.execution import register_execution_tools, register_storage_tools, _active_executions
from deriva_ml_mcp.tools.catalog import register_catalog_tools
from deriva_ml_mcp.tools.workflow import register_workflow_tools
from deriva_ml_mcp.tools.dataset import register_dataset_tools

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
    register_workflow_tools(mcp, conn_manager)
    register_execution_tools(mcp, conn_manager)
    register_storage_tools(mcp, conn_manager)
    register_dataset_tools(mcp, conn_manager)
    return captured_tools


class TestCreateExecution:
    """Tests for the create_execution tool."""

    @pytest.mark.asyncio
    async def test_create_execution_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating a new execution."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Add workflow type first
        await tools["add_workflow_type"](
            type_name="Training",
            description="Training workflows",
        )

        # Create execution
        result = await tools["create_execution"](
            workflow_name="Test Execution",
            workflow_type="Training",
            description="Testing execution creation",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert "execution_rid" in data
        assert "workflow_rid" in data

        # Clean up active execution
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_create_execution_with_datasets(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating execution with input datasets."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="Processing",
            description="Data processing workflows",
        )

        result = await tools["create_execution"](
            workflow_name="Dataset Processing",
            workflow_type="Processing",
            description="Process dataset",
            dataset_rids=[dataset_desc.dataset.dataset_rid],
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["dataset_count"] == 1

        # Clean up
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        conn_manager.disconnect()


class TestExecutionLifecycle:
    """Tests for execution lifecycle tools."""

    @pytest.mark.asyncio
    async def test_start_stop_execution(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test starting and stopping an execution."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="Test",
            description="Test workflows",
        )

        # Create execution
        await tools["create_execution"](
            workflow_name="Lifecycle Test",
            workflow_type="Test",
        )

        # Start execution
        start_result = await tools["start_execution"]()
        start_data = parse_json_result(start_result)
        assert start_data["status"] == "started"

        # Stop execution
        stop_result = await tools["stop_execution"]()
        stop_data = parse_json_result(stop_result)
        assert stop_data["status"] == "completed"

        # Clean up
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_start_execution_no_active(self):
        """Test starting execution when none exists."""
        conn_manager = ConnectionManager()
        tools = setup_tools(conn_manager)

        # Don't connect or create execution
        result = await tools["start_execution"]()

        data = parse_json_result(result)
        assert data["status"] == "error"


class TestUpdateExecutionStatus:
    """Tests for the update_execution_status tool."""

    @pytest.mark.asyncio
    async def test_update_status(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test updating execution status."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="StatusTest",
            description="Status test workflows",
        )

        await tools["create_execution"](
            workflow_name="Status Test",
            workflow_type="StatusTest",
        )

        result = await tools["update_execution_status"](
            status="running",
            message="Processing step 1 of 5",
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["new_status"] == "running"
        assert data["message"] == "Processing step 1 of 5"

        # Clean up
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        conn_manager.disconnect()


class TestGetExecutionInfo:
    """Tests for the get_execution_info tool."""

    @pytest.mark.asyncio
    async def test_get_execution_info(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting execution info."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="InfoTest",
            description="Info test workflows",
        )

        await tools["create_execution"](
            workflow_name="Info Test",
            workflow_type="InfoTest",
        )

        result = await tools["get_execution_info"]()

        data = parse_json_result(result)
        assert "execution_rid" in data
        assert "workflow_rid" in data
        assert "working_dir" in data

        # Clean up
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_get_execution_info_no_active(self):
        """Test getting info when no execution exists."""
        conn_manager = ConnectionManager()
        tools = setup_tools(conn_manager)

        result = await tools["get_execution_info"]()

        data = parse_json_result(result)
        assert data["status"] == "no_active_execution"


class TestGetExecutionWorkingDir:
    """Tests for the get_execution_working_dir tool."""

    @pytest.mark.asyncio
    async def test_get_working_dir(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test getting execution working directory."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="DirTest",
            description="Directory test workflows",
        )

        await tools["create_execution"](
            workflow_name="Dir Test",
            workflow_type="DirTest",
        )

        result = await tools["get_execution_working_dir"]()

        data = parse_json_result(result)
        assert "working_dir" in data
        assert "execution_rid" in data

        # Clean up
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        conn_manager.disconnect()


class TestCreateExecutionDataset:
    """Tests for the create_execution_dataset tool."""

    @pytest.mark.asyncio
    async def test_create_execution_dataset(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating a dataset from execution."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="DatasetCreation",
            description="Dataset creation workflows",
        )

        await tools["create_execution"](
            workflow_name="Dataset Creator",
            workflow_type="DatasetCreation",
        )

        result = await tools["create_execution_dataset"](
            description="Created from execution",
            dataset_types=["Complete"],
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert "dataset_rid" in data
        assert "execution_rid" in data

        # Clean up
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        conn_manager.disconnect()


class TestSetExecutionDescription:
    """Tests for the set_execution_description tool."""

    @pytest.mark.asyncio
    async def test_set_execution_description(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test setting execution description."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="DescTest",
            description="Description test workflows",
        )

        create_result = await tools["create_execution"](
            workflow_name="Desc Test",
            workflow_type="DescTest",
            description="Original description",
        )
        create_data = parse_json_result(create_result)
        execution_rid = create_data["execution_rid"]

        new_description = "Updated execution description with results"
        result = await tools["set_execution_description"](
            execution_rid=execution_rid,
            description=new_description,
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["description"] == new_description

        # Clean up
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        conn_manager.disconnect()


class TestExecutionNesting:
    """Tests for execution nesting tools."""

    @pytest.mark.asyncio
    async def test_add_nested_execution(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a nested execution."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="Sweep",
            description="Parameter sweep workflows",
        )

        # Create parent execution
        parent_result = await tools["create_execution"](
            workflow_name="Parent Sweep",
            workflow_type="Sweep",
            description="Parent execution",
        )
        parent_data = parse_json_result(parent_result)
        parent_rid = parent_data["execution_rid"]

        # Clean up parent from active
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        # Create child execution
        child_result = await tools["create_execution"](
            workflow_name="Child Run",
            workflow_type="Sweep",
            description="Child execution",
        )
        child_data = parse_json_result(child_result)
        child_rid = child_data["execution_rid"]

        # Clean up child from active
        if key in _active_executions:
            del _active_executions[key]

        # Add child to parent
        result = await tools["add_nested_execution"](
            parent_execution_rid=parent_rid,
            child_execution_rid=child_rid,
            sequence=0,
        )

        data = parse_json_result(result)
        assert data["status"] == "added"
        assert data["parent_rid"] == parent_rid
        assert data["child_rid"] == child_rid

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_list_nested_executions(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test listing nested executions."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="Pipeline",
            description="Pipeline workflows",
        )

        # Create parent
        parent_result = await tools["create_execution"](
            workflow_name="Pipeline Parent",
            workflow_type="Pipeline",
        )
        parent_data = parse_json_result(parent_result)
        parent_rid = parent_data["execution_rid"]

        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        # Create and nest child
        child_result = await tools["create_execution"](
            workflow_name="Pipeline Child",
            workflow_type="Pipeline",
        )
        child_data = parse_json_result(child_result)
        child_rid = child_data["execution_rid"]

        if key in _active_executions:
            del _active_executions[key]

        await tools["add_nested_execution"](
            parent_execution_rid=parent_rid,
            child_execution_rid=child_rid,
        )

        # List children
        result = await tools["list_nested_executions"](
            execution_rid=parent_rid,
        )

        data = parse_json_result(result)
        assert data["count"] >= 1
        assert len(data["children"]) >= 1

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_list_parent_executions(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test listing parent executions."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        await tools["add_workflow_type"](
            type_name="Nested",
            description="Nested workflows",
        )

        # Create parent
        parent_result = await tools["create_execution"](
            workflow_name="Nested Parent",
            workflow_type="Nested",
        )
        parent_data = parse_json_result(parent_result)
        parent_rid = parent_data["execution_rid"]

        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        # Create and nest child
        child_result = await tools["create_execution"](
            workflow_name="Nested Child",
            workflow_type="Nested",
        )
        child_data = parse_json_result(child_result)
        child_rid = child_data["execution_rid"]

        if key in _active_executions:
            del _active_executions[key]

        await tools["add_nested_execution"](
            parent_execution_rid=parent_rid,
            child_execution_rid=child_rid,
        )

        # List parents of child
        result = await tools["list_parent_executions"](
            execution_rid=child_rid,
        )

        data = parse_json_result(result)
        assert data["count"] >= 1
        assert len(data["parents"]) >= 1

        conn_manager.disconnect()


class TestStorageTools:
    """Tests for storage management tools."""

    @pytest.mark.asyncio
    async def test_clear_cache(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test clearing the cache."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # This should work even with empty cache
        result = tools["clear_cache"]()

        data = parse_json_result(result)
        assert data["status"] == "success"
        assert "files_removed" in data
        assert "bytes_freed" in data

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_clean_execution_dirs(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test cleaning execution directories."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # This should work even with no execution dirs
        result = tools["clean_execution_dirs"]()

        data = parse_json_result(result)
        assert data["status"] == "success"
        assert "dirs_removed" in data
        assert "bytes_freed" in data

        conn_manager.disconnect()
