"""Integration tests for workflow management tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.tools.workflow import register_workflow_tools
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
    register_workflow_tools(mcp, conn_manager)
    return captured_tools


class TestCreateWorkflow:
    """Tests for the create_workflow tool."""

    @pytest.mark.asyncio
    async def test_create_workflow_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating a new workflow."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # First add the workflow type
        await tools["add_workflow_type"](
            type_name="Training",
            description="ML training workflows",
        )

        # Create workflow
        result = await tools["create_workflow"](
            name="Test Training Workflow",
            workflow_type="Training",
            description="A test workflow for training models",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["name"] == "Test Training Workflow"
        assert data["workflow_type"] == "Training"
        assert "workflow_rid" in data

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_create_workflow_no_connection(self):
        """Test creating workflow without connection fails."""
        conn_manager = ConnectionManager()
        tools = setup_tools(conn_manager)

        result = await tools["create_workflow"](
            name="Test Workflow",
            workflow_type="Training",
        )

        data = parse_json_result(result)
        assert data["status"] == "error"


class TestAddWorkflowType:
    """Tests for the add_workflow_type tool."""

    @pytest.mark.asyncio
    async def test_add_workflow_type_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a new workflow type."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["add_workflow_type"](
            type_name="Data_Preprocessing",
            description="Workflows for preprocessing data",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["name"] == "Data_Preprocessing"
        assert "rid" in data

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_add_workflow_type_exists_ok(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a workflow type that already exists (should succeed)."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Add type twice
        await tools["add_workflow_type"](
            type_name="Inference",
            description="Inference workflows",
        )

        result = await tools["add_workflow_type"](
            type_name="Inference",
            description="Inference workflows again",
        )

        data = parse_json_result(result)
        # Should succeed with exists_ok=True
        assert data["status"] == "created"

        conn_manager.disconnect()


class TestLookupWorkflowByUrl:
    """Tests for the lookup_workflow_by_url tool."""

    @pytest.mark.asyncio
    async def test_lookup_workflow_not_found(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test looking up a workflow by URL that doesn't exist."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["lookup_workflow_by_url"](
            url="https://github.com/nonexistent/repo/blob/main/train.py",
        )

        data = parse_json_result(result)
        assert data["found"] is False
        assert data["workflow"] is None

        conn_manager.disconnect()


class TestSetWorkflowDescription:
    """Tests for the set_workflow_description tool."""

    @pytest.mark.asyncio
    async def test_set_workflow_description(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test setting workflow description."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create a workflow type and workflow
        await tools["add_workflow_type"](
            type_name="Evaluation",
            description="Model evaluation workflows",
        )

        create_result = await tools["create_workflow"](
            name="Eval Workflow",
            workflow_type="Evaluation",
            description="Original description",
        )
        create_data = parse_json_result(create_result)
        workflow_rid = create_data["workflow_rid"]

        # Update description
        new_description = "Updated workflow description with more details"
        result = await tools["set_workflow_description"](
            workflow_rid=workflow_rid,
            description=new_description,
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["description"] == new_description

        conn_manager.disconnect()
