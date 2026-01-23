"""Integration tests for dataset management tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.tools.dataset import register_dataset_tools
from deriva_ml_mcp.tools.catalog import register_catalog_tools
from deriva_ml_mcp.tools.execution import register_execution_tools

if TYPE_CHECKING:
    from deriva_ml.demo_catalog import DatasetDescription
    from tests.conftest import CatalogManager

from tests.conftest import parse_json_result, assert_success, assert_error


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
    register_dataset_tools(mcp, conn_manager)
    register_execution_tools(mcp, conn_manager)
    return captured_tools


class TestCreateDataset:
    """Tests for the create_dataset tool."""

    @pytest.mark.asyncio
    async def test_create_dataset_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating a new dataset."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        # Connect
        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create dataset (uses connection's MCP execution)
        result = await tools["create_dataset"](
            description="Test dataset from MCP",
            dataset_types=["Complete"],
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert "dataset_rid" in data
        assert data["description"] == "Test dataset from MCP"
        assert "Complete" in data["dataset_types"]

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_create_dataset_no_connection(self):
        """Test creating dataset without connection fails."""
        conn_manager = ConnectionManager()
        tools = setup_tools(conn_manager)

        result = await tools["create_dataset"](
            description="Test dataset",
        )

        data = parse_json_result(result)
        assert data["status"] == "error"
        assert "No active" in data["message"]


class TestListDatasetMembers:
    """Tests for the list_dataset_members tool."""

    @pytest.mark.asyncio
    async def test_list_dataset_members(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test listing dataset members."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["list_dataset_members"](
            dataset_rid=dataset_desc.dataset.dataset_rid,
        )

        data = parse_json_result(result)
        # Should have members - either Image, Subject, or Dataset
        assert isinstance(data, dict)
        # Check at least one table has members
        total_members = sum(len(members) for members in data.values() if isinstance(members, list))
        assert total_members > 0 or "status" not in data  # Either has members or no error

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_list_dataset_members_with_version(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test listing dataset members at specific version."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Get current version
        version = str(dataset_desc.dataset.current_version) if dataset_desc.dataset.current_version else "0.1.0"

        result = await tools["list_dataset_members"](
            dataset_rid=dataset_desc.dataset.dataset_rid,
            version=version,
        )

        data = parse_json_result(result)
        assert "status" not in data or data.get("status") != "error"

        conn_manager.disconnect()


class TestAddDatasetMembers:
    """Tests for the add_dataset_members tool."""

    @pytest.mark.asyncio
    async def test_add_dataset_members(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding members to a dataset."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        ml = catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create a dataset first
        create_result = await tools["create_dataset"](
            description="Test dataset for adding members",
        )
        create_data = parse_json_result(create_result)
        dataset_rid = create_data["dataset_rid"]

        # Get a subject RID to add
        pb = ml.pathBuilder()
        subjects = list(pb.schemas[ml.default_schema].Subject.path.entities())
        if subjects:
            subject_rid = subjects[0]["RID"]

            result = await tools["add_dataset_members"](
                dataset_rid=dataset_rid,
                member_rids=[subject_rid],
            )

            data = parse_json_result(result)
            assert data["status"] == "success"
            assert data["added_count"] == 1

        conn_manager.disconnect()


class TestDeleteDatasetMembers:
    """Tests for the delete_dataset_members tool."""

    @pytest.mark.asyncio
    async def test_delete_dataset_members(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test removing members from a dataset."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        ml = catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create a dataset and add a member
        create_result = await tools["create_dataset"](
            description="Test dataset for removing members",
        )
        create_data = parse_json_result(create_result)
        dataset_rid = create_data["dataset_rid"]

        # Get a subject RID
        pb = ml.pathBuilder()
        subjects = list(pb.schemas[ml.default_schema].Subject.path.entities())
        if subjects:
            subject_rid = subjects[0]["RID"]

            # Add member
            await tools["add_dataset_members"](
                dataset_rid=dataset_rid,
                member_rids=[subject_rid],
            )

            # Remove member
            result = await tools["delete_dataset_members"](
                dataset_rid=dataset_rid,
                member_rids=[subject_rid],
            )

            data = parse_json_result(result)
            assert data["status"] == "success"
            assert data["removed_count"] == 1

        conn_manager.disconnect()


class TestIncrementDatasetVersion:
    """Tests for the increment_dataset_version tool."""

    @pytest.mark.asyncio
    async def test_increment_version_minor(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test incrementing minor version."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["increment_dataset_version"](
            dataset_rid=dataset_desc.dataset.dataset_rid,
            description="Test version increment",
            component="minor",
        )

        data = parse_json_result(result)
        assert data["status"] == "success"
        assert data["new_version"] is not None
        assert data["component"] == "minor"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_increment_version_patch(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test incrementing patch version."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["increment_dataset_version"](
            dataset_rid=dataset_desc.dataset.dataset_rid,
            description="Bug fix",
            component="patch",
        )

        data = parse_json_result(result)
        assert data["status"] == "success"

        conn_manager.disconnect()


class TestSetDatasetDescription:
    """Tests for the set_dataset_description tool."""

    @pytest.mark.asyncio
    async def test_set_description(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test setting dataset description."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        new_description = "Updated description for testing"
        result = await tools["set_dataset_description"](
            dataset_rid=dataset_desc.dataset.dataset_rid,
            description=new_description,
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["description"] == new_description

        conn_manager.disconnect()


class TestDatasetTypes:
    """Tests for add_dataset_type and remove_dataset_type tools."""

    @pytest.mark.asyncio
    async def test_add_dataset_type(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a type to a dataset."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create dataset without types
        create_result = await tools["create_dataset"](
            description="Dataset for type testing",
        )
        create_data = parse_json_result(create_result)
        dataset_rid = create_data["dataset_rid"]

        # Add type
        result = await tools["add_dataset_type"](
            dataset_rid=dataset_rid,
            dataset_type="Training",
        )

        data = parse_json_result(result)
        assert data["status"] == "added"
        assert "Training" in data["dataset_types"]

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_remove_dataset_type(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test removing a type from a dataset."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create dataset with a type
        create_result = await tools["create_dataset"](
            description="Dataset for type removal testing",
            dataset_types=["Training", "Complete"],
        )
        create_data = parse_json_result(create_result)
        dataset_rid = create_data["dataset_rid"]

        # Remove type
        result = await tools["remove_dataset_type"](
            dataset_rid=dataset_rid,
            dataset_type="Training",
        )

        data = parse_json_result(result)
        assert data["status"] == "removed"
        assert "Training" not in data["dataset_types"]

        conn_manager.disconnect()


class TestDatasetHierarchy:
    """Tests for dataset hierarchy tools."""

    @pytest.mark.asyncio
    async def test_add_dataset_child(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a child dataset."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create parent dataset
        parent_result = await tools["create_dataset"](
            description="Parent dataset",
            dataset_types=["Complete"],
        )
        parent_data = parse_json_result(parent_result)
        parent_rid = parent_data["dataset_rid"]

        # Create child dataset
        child_result = await tools["create_dataset"](
            description="Child dataset",
            dataset_types=["Training"],
        )
        child_data = parse_json_result(child_result)
        child_rid = child_data["dataset_rid"]

        # Add child to parent
        result = await tools["add_dataset_child"](
            parent_rid=parent_rid,
            child_rid=child_rid,
        )

        data = parse_json_result(result)
        assert data["status"] == "added"
        assert data["parent_rid"] == parent_rid
        assert data["child_rid"] == child_rid

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_list_dataset_children(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test listing child datasets."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # The demo datasets have a hierarchy - list children of root
        result = await tools["list_dataset_children"](
            dataset_rid=dataset_desc.dataset.dataset_rid,
        )

        data = parse_json_result(result)
        assert isinstance(data, list)
        # Demo datasets should have children
        if len(data) > 0:
            assert "dataset_rid" in data[0]

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_list_dataset_parents(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test listing parent datasets."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Get a child dataset RID from the hierarchy
        children = dataset_desc.members.get("Dataset", [])
        if children:
            child_rid = children[0].dataset.dataset_rid

            result = await tools["list_dataset_parents"](
                dataset_rid=child_rid,
            )

            data = parse_json_result(result)
            assert isinstance(data, list)
            # Should have at least the parent
            assert len(data) >= 1

        conn_manager.disconnect()


class TestGetDatasetSpec:
    """Tests for the get_dataset_spec tool."""

    @pytest.mark.asyncio
    async def test_get_dataset_spec(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test generating DatasetSpecConfig string."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["get_dataset_spec"](
            dataset_rid=dataset_desc.dataset.dataset_rid,
        )

        data = parse_json_result(result)
        assert "spec" in data
        assert "DatasetSpecConfig" in data["spec"]
        assert dataset_desc.dataset.dataset_rid in data["spec"]
        assert data["dataset_rid"] == dataset_desc.dataset.dataset_rid

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_get_dataset_spec_with_version(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test generating DatasetSpecConfig with specific version."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["get_dataset_spec"](
            dataset_rid=dataset_desc.dataset.dataset_rid,
            version="0.1.0",
        )

        data = parse_json_result(result)
        assert "spec" in data
        assert "0.1.0" in data["spec"]
        # Should not have warning when explicit version provided
        assert "warning" not in data

        conn_manager.disconnect()


class TestDatasetTypeTerm:
    """Tests for dataset type vocabulary tools."""

    @pytest.mark.asyncio
    async def test_create_dataset_type_term(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating a new dataset type term."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["create_dataset_type_term"](
            type_name="Custom_Test_Type",
            description="A custom type for testing",
            synonyms=["test", "custom"],
        )

        data = parse_json_result(result)
        assert data["status"] in ["created", "exists"]
        assert data["name"] == "Custom_Test_Type"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_delete_dataset_type_term(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test deleting a dataset type term."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # First create a term to delete
        await tools["create_dataset_type_term"](
            type_name="Type_To_Delete",
            description="This will be deleted",
        )

        # Delete it
        result = await tools["delete_dataset_type_term"](
            type_name="Type_To_Delete",
        )

        data = parse_json_result(result)
        assert data["status"] == "deleted"

        conn_manager.disconnect()
