"""Unit tests for dataset management tools.

Tests cover the dataset tools registered by register_dataset_tools:
- create_dataset: Create datasets within execution context
- get_dataset_spec: Generate DatasetSpecConfig strings
- list_dataset_members: List members grouped by table
- add_dataset_members: Add RIDs to dataset
- delete_dataset_members: Remove RIDs from dataset
- increment_dataset_version: Increment semantic version
- delete_dataset: Soft-delete dataset
- set_dataset_description: Update description
- add_dataset_type: Add type to dataset
- remove_dataset_type: Remove type from dataset
- add_dataset_element_type: Register table as element type
- add_dataset_child: Nest datasets
- list_dataset_parents: List parent datasets
- estimate_bag_size: Estimate bag download size
- preview_denormalized_dataset: Join tables into wide format
- create_dataset_type_term: Create Dataset_Type vocab term
- delete_dataset_type_term: Delete Dataset_Type vocab term
- split_dataset: Split dataset into train/test
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import assert_error, assert_success, parse_json_result


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_dataset(
    dataset_rid="DS-001",
    description="Test dataset",
    dataset_types=None,
    current_version="0.1.0",
):
    """Create a mock dataset object with standard attributes."""
    ds = MagicMock()
    ds.dataset_rid = dataset_rid
    ds.description = description
    ds.dataset_types = list(dataset_types) if dataset_types is not None else ["Training"]
    ds.current_version = current_version
    return ds


def _make_mock_execution(execution_rid="EXE-001", workflow_rid="WF-001"):
    """Create a mock execution object for create_dataset tests."""
    exe = MagicMock()
    exe.execution_rid = execution_rid
    exe.workflow_rid = workflow_rid
    return exe


# =============================================================================
# TestCreateDataset
# =============================================================================


class TestCreateDataset:
    """Tests for the create_dataset tool."""

    @pytest.mark.asyncio
    async def test_create_with_active_tool_execution(self, dataset_tools, mock_conn_manager):
        """When an active tool execution exists, use it to create the dataset."""
        mock_execution = _make_mock_execution()
        mock_dataset = _make_mock_dataset()
        mock_execution.create_dataset.return_value = mock_dataset

        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = mock_execution

        result = await dataset_tools["create_dataset"](
            description="Training images",
            dataset_types=["Training"],
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["dataset_rid"] == "DS-001"
        assert data["description"] == "Test dataset"
        assert data["dataset_types"] == ["Training"]
        assert data["execution_rid"] == "EXE-001"
        mock_execution.create_dataset.assert_called_once_with(
            description="Training images",
            dataset_types=["Training"],
        )

    @pytest.mark.asyncio
    async def test_create_with_mcp_connection_execution(self, dataset_tools, mock_conn_manager):
        """When no active tool execution, fall back to the MCP connection execution."""
        mock_execution = mock_conn_manager.get_active_execution()
        mock_dataset = _make_mock_dataset(dataset_rid="DS-002")
        mock_execution.create_dataset.return_value = mock_dataset

        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = None

        result = await dataset_tools["create_dataset"](
            description="Test",
            dataset_types=["Testing"],
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["dataset_rid"] == "DS-002"

    @pytest.mark.asyncio
    async def test_create_no_execution_returns_error(self, dataset_tools, mock_conn_manager):
        """When no execution context at all, return an error."""
        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = None
        mock_conn_manager.get_active_execution.return_value = None

        result = await dataset_tools["create_dataset"](description="Test")

        data = assert_error(result)
        assert "No active execution context" in data["message"]

    @pytest.mark.asyncio
    async def test_create_default_dataset_types(self, dataset_tools, mock_conn_manager):
        """When dataset_types is None, pass empty list."""
        mock_execution = mock_conn_manager.get_active_execution()
        mock_dataset = _make_mock_dataset(dataset_types=[])
        mock_execution.create_dataset.return_value = mock_dataset

        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = None

        result = await dataset_tools["create_dataset"](description="Empty types")

        data = assert_success(result)
        mock_execution.create_dataset.assert_called_once_with(
            description="Empty types",
            dataset_types=[],
        )

    @pytest.mark.asyncio
    async def test_create_version_from_dataset(self, dataset_tools, mock_conn_manager):
        """Version in response comes from the created dataset."""
        mock_execution = mock_conn_manager.get_active_execution()
        mock_dataset = _make_mock_dataset(current_version="1.0.0")
        mock_execution.create_dataset.return_value = mock_dataset

        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = None

        result = await dataset_tools["create_dataset"](description="Versioned")

        data = assert_success(result)
        assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_create_version_fallback_to_default(self, dataset_tools, mock_conn_manager):
        """When dataset has no current_version, use parameter or default."""
        mock_execution = mock_conn_manager.get_active_execution()
        mock_dataset = _make_mock_dataset(current_version=None)
        mock_execution.create_dataset.return_value = mock_dataset

        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = None

        result = await dataset_tools["create_dataset"](description="No version")

        data = assert_success(result)
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_create_exception(self, dataset_tools, mock_conn_manager):
        """When create_dataset raises, return an error."""
        mock_execution = mock_conn_manager.get_active_execution()
        mock_execution.create_dataset.side_effect = RuntimeError("catalog unavailable")

        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.active_tool_execution = None

        result = await dataset_tools["create_dataset"](description="Bad")

        assert_error(result, expected_message="catalog unavailable")

    @pytest.mark.asyncio
    async def test_create_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["create_dataset"](description="Test")

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestGetDatasetSpec
# =============================================================================


class TestGetDatasetSpec:
    """Tests for the get_dataset_spec tool."""

    @pytest.mark.asyncio
    async def test_get_spec_with_version(self, dataset_tools, mock_ml):
        """When version is provided, use it without a warning."""
        mock_dataset = _make_mock_dataset(current_version="0.21.0")
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["get_dataset_spec"](
            dataset_rid="28CT",
            version="0.20.0",
        )

        data = parse_json_result(result)
        assert data["spec"] == 'DatasetSpecConfig(rid="28CT", version="0.20.0")'
        assert data["dataset_rid"] == "28CT"
        assert data["version"] == "0.20.0"
        assert data["description"] == "Test dataset"
        assert data["dataset_types"] == ["Training"]
        assert "warning" not in data

    @pytest.mark.asyncio
    async def test_get_spec_without_version_uses_current(self, dataset_tools, mock_ml):
        """When no version, use current_version with a warning."""
        mock_dataset = _make_mock_dataset(current_version="0.21.0")
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["get_dataset_spec"](dataset_rid="28CT")

        data = parse_json_result(result)
        assert data["version"] == "0.21.0"
        assert "warning" in data
        assert "reproducibility" in data["warning"].lower()

    @pytest.mark.asyncio
    async def test_get_spec_no_current_version(self, dataset_tools, mock_ml):
        """When dataset has no current_version and no version provided, default to 0.1.0."""
        mock_dataset = _make_mock_dataset(current_version=None)
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["get_dataset_spec"](dataset_rid="28CT")

        data = parse_json_result(result)
        assert data["version"] == "0.1.0"
        assert "warning" in data

    @pytest.mark.asyncio
    async def test_get_spec_exception(self, dataset_tools, mock_ml):
        """When lookup_dataset raises, return an error."""
        mock_ml.lookup_dataset.side_effect = ValueError("Dataset not found")

        result = await dataset_tools["get_dataset_spec"](dataset_rid="INVALID")

        assert_error(result, expected_message="Dataset not found")

    @pytest.mark.asyncio
    async def test_get_spec_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["get_dataset_spec"](dataset_rid="28CT")

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestListDatasetMembers
# =============================================================================




# =============================================================================
# TestAddDatasetMembers
# =============================================================================


class TestAddDatasetMembers:
    """Tests for the add_dataset_members tool."""

    @pytest.mark.asyncio
    async def test_add_members_success(self, dataset_tools, mock_ml):
        """Adding members returns status=success with added_count."""
        mock_dataset = _make_mock_dataset()
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["add_dataset_members"](
            dataset_rid="DS-001",
            member_rids=["IMG-1", "IMG-2", "IMG-3"],
        )

        data = assert_success(result)
        assert data["added_count"] == 3
        assert data["dataset_rid"] == "DS-001"
        mock_dataset.add_dataset_members.assert_called_once_with(
            members=["IMG-1", "IMG-2", "IMG-3"],
        )

    @pytest.mark.asyncio
    async def test_add_members_empty_list(self, dataset_tools, mock_ml):
        """Adding empty list succeeds with count 0."""
        mock_dataset = _make_mock_dataset()
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["add_dataset_members"](
            dataset_rid="DS-001",
            member_rids=[],
        )

        data = assert_success(result)
        assert data["added_count"] == 0

    @pytest.mark.asyncio
    async def test_add_members_exception(self, dataset_tools, mock_ml):
        """When add_dataset_members raises, return an error."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.add_dataset_members.side_effect = ValueError("Invalid RID")
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["add_dataset_members"](
            dataset_rid="DS-001",
            member_rids=["INVALID"],
        )

        assert_error(result, expected_message="Invalid RID")

    @pytest.mark.asyncio
    async def test_add_members_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["add_dataset_members"](
            dataset_rid="DS-001",
            member_rids=["IMG-1"],
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestDeleteDatasetMembers
# =============================================================================


class TestDeleteDatasetMembers:
    """Tests for the delete_dataset_members tool."""

    @pytest.mark.asyncio
    async def test_delete_members_success(self, dataset_tools, mock_ml):
        """Removing members returns status=success with removed_count."""
        mock_dataset = _make_mock_dataset()
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["delete_dataset_members"](
            dataset_rid="DS-001",
            member_rids=["IMG-1", "IMG-2"],
        )

        data = assert_success(result)
        assert data["removed_count"] == 2
        assert data["dataset_rid"] == "DS-001"
        mock_dataset.delete_dataset_members.assert_called_once_with(
            members=["IMG-1", "IMG-2"],
        )

    @pytest.mark.asyncio
    async def test_delete_members_empty_list(self, dataset_tools, mock_ml):
        """Removing empty list succeeds with count 0."""
        mock_dataset = _make_mock_dataset()
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["delete_dataset_members"](
            dataset_rid="DS-001",
            member_rids=[],
        )

        data = assert_success(result)
        assert data["removed_count"] == 0

    @pytest.mark.asyncio
    async def test_delete_members_exception(self, dataset_tools, mock_ml):
        """When delete_dataset_members raises, return an error."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.delete_dataset_members.side_effect = RuntimeError("Member not found")
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["delete_dataset_members"](
            dataset_rid="DS-001",
            member_rids=["NOTFOUND"],
        )

        assert_error(result, expected_message="Member not found")

    @pytest.mark.asyncio
    async def test_delete_members_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["delete_dataset_members"](
            dataset_rid="DS-001",
            member_rids=["IMG-1"],
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestIncrementDatasetVersion
# =============================================================================


class TestIncrementDatasetVersion:
    """Tests for the increment_dataset_version tool."""

    @pytest.mark.asyncio
    async def test_increment_minor(self, dataset_tools, mock_ml):
        """Incrementing minor version returns new and previous versions."""
        mock_dataset = _make_mock_dataset(current_version="0.5.0")
        mock_dataset.increment_dataset_version.return_value = "0.6.0"
        mock_ml.lookup_dataset.return_value = mock_dataset

        with patch("deriva_mcp.tools.dataset.VersionPart", create=True):
            result = await dataset_tools["increment_dataset_version"](
                dataset_rid="DS-001",
                description="Added new images",
                component="minor",
            )

        data = assert_success(result)
        assert data["new_version"] == "0.6.0"
        assert data["previous_version"] == "0.5.0"
        assert data["dataset_rid"] == "DS-001"
        assert data["description"] == "Added new images"
        assert data["component"] == "minor"

    @pytest.mark.asyncio
    async def test_increment_major(self, dataset_tools, mock_ml):
        """Incrementing major version works correctly."""
        mock_dataset = _make_mock_dataset(current_version="1.2.3")
        mock_dataset.increment_dataset_version.return_value = "2.0.0"
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["increment_dataset_version"](
            dataset_rid="DS-001",
            description="Schema change",
            component="major",
        )

        data = assert_success(result)
        assert data["new_version"] == "2.0.0"
        assert data["previous_version"] == "1.2.3"
        assert data["component"] == "major"

    @pytest.mark.asyncio
    async def test_increment_patch(self, dataset_tools, mock_ml):
        """Incrementing patch version works correctly."""
        mock_dataset = _make_mock_dataset(current_version="1.2.3")
        mock_dataset.increment_dataset_version.return_value = "1.2.4"
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["increment_dataset_version"](
            dataset_rid="DS-001",
            description="Fixed labels",
            component="patch",
        )

        data = assert_success(result)
        assert data["new_version"] == "1.2.4"
        assert data["component"] == "patch"

    @pytest.mark.asyncio
    async def test_increment_default_component(self, dataset_tools, mock_ml):
        """Default component is 'minor'."""
        mock_dataset = _make_mock_dataset(current_version="0.1.0")
        mock_dataset.increment_dataset_version.return_value = "0.2.0"
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["increment_dataset_version"](
            dataset_rid="DS-001",
        )

        data = assert_success(result)
        assert data["component"] == "minor"

    @pytest.mark.asyncio
    async def test_increment_null_versions(self, dataset_tools, mock_ml):
        """When versions are None, null is returned in the response."""
        mock_dataset = _make_mock_dataset(current_version=None)
        mock_dataset.increment_dataset_version.return_value = None
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["increment_dataset_version"](
            dataset_rid="DS-001",
        )

        data = assert_success(result)
        assert data["new_version"] is None
        assert data["previous_version"] is None

    @pytest.mark.asyncio
    async def test_increment_exception(self, dataset_tools, mock_ml):
        """When increment_dataset_version raises, return an error."""
        mock_ml.lookup_dataset.side_effect = RuntimeError("Version conflict")

        result = await dataset_tools["increment_dataset_version"](
            dataset_rid="DS-001",
        )

        assert_error(result, expected_message="Version conflict")

    @pytest.mark.asyncio
    async def test_increment_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["increment_dataset_version"](
            dataset_rid="DS-001",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestDeleteDataset
# =============================================================================


class TestDeleteDataset:
    """Tests for the delete_dataset tool."""

    @pytest.mark.asyncio
    async def test_delete_success(self, dataset_tools, mock_ml):
        """Deleting a dataset returns status=deleted."""
        mock_dataset = _make_mock_dataset()
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["delete_dataset"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert data["status"] == "deleted"
        assert data["dataset_rid"] == "DS-001"
        assert data["recursive"] is False
        mock_ml.delete_dataset.assert_called_once_with(mock_dataset, recurse=False)

    @pytest.mark.asyncio
    async def test_delete_recursive(self, dataset_tools, mock_ml):
        """Recursive delete passes recurse=True."""
        mock_dataset = _make_mock_dataset()
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["delete_dataset"](
            dataset_rid="DS-001",
            recurse=True,
        )

        data = parse_json_result(result)
        assert data["recursive"] is True
        mock_ml.delete_dataset.assert_called_once_with(mock_dataset, recurse=True)

    @pytest.mark.asyncio
    async def test_delete_exception(self, dataset_tools, mock_ml):
        """When delete_dataset raises, return an error."""
        mock_ml.lookup_dataset.side_effect = ValueError("Not found")

        result = await dataset_tools["delete_dataset"](dataset_rid="INVALID")

        assert_error(result, expected_message="Not found")

    @pytest.mark.asyncio
    async def test_delete_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["delete_dataset"](dataset_rid="DS-001")

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestSetDatasetDescription
# =============================================================================


class TestSetDatasetDescription:
    """Tests for the set_dataset_description tool."""

    @pytest.mark.asyncio
    async def test_set_description_success(self, dataset_tools, mock_ml):
        """Setting a description returns status=updated."""
        mock_dataset = _make_mock_dataset()
        mock_ml.lookup_dataset.return_value = mock_dataset

        # Set up pathBuilder chain
        mock_pb = MagicMock()
        mock_dataset_path = MagicMock()
        mock_pb.schemas.__getitem__.return_value.Dataset = mock_dataset_path
        mock_ml.pathBuilder.return_value = mock_pb

        result = await dataset_tools["set_dataset_description"](
            dataset_rid="DS-001",
            description="Updated description",
        )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["dataset_rid"] == "DS-001"
        assert data["description"] == "Updated description"
        mock_dataset_path.update.assert_called_once_with(
            [{"RID": "DS-001", "Description": "Updated description"}],
        )
        assert mock_dataset.description == "Updated description"

    @pytest.mark.asyncio
    async def test_set_description_exception(self, dataset_tools, mock_ml):
        """When update raises, return an error."""
        mock_ml.lookup_dataset.side_effect = RuntimeError("Update failed")

        result = await dataset_tools["set_dataset_description"](
            dataset_rid="DS-001",
            description="New",
        )

        assert_error(result, expected_message="Update failed")

    @pytest.mark.asyncio
    async def test_set_description_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["set_dataset_description"](
            dataset_rid="DS-001",
            description="New",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestAddDatasetType
# =============================================================================


class TestAddDatasetType:
    """Tests for the add_dataset_type tool."""

    @pytest.mark.asyncio
    async def test_add_type_success(self, dataset_tools, mock_ml):
        """Adding a type returns status=added with updated types list."""
        mock_dataset = _make_mock_dataset(dataset_types=["Training", "Image"])
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["add_dataset_type"](
            dataset_rid="DS-001",
            dataset_type="Image",
        )

        data = assert_success(result)
        assert data["status"] == "added"
        assert data["dataset_rid"] == "DS-001"
        assert data["dataset_types"] == ["Training", "Image"]
        mock_dataset.add_dataset_type.assert_called_once_with("Image")

    @pytest.mark.asyncio
    async def test_add_type_exception(self, dataset_tools, mock_ml):
        """When add_dataset_type raises, return an error."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.add_dataset_type.side_effect = ValueError("Type not in vocabulary")
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["add_dataset_type"](
            dataset_rid="DS-001",
            dataset_type="NonExistent",
        )

        assert_error(result, expected_message="Type not in vocabulary")

    @pytest.mark.asyncio
    async def test_add_type_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["add_dataset_type"](
            dataset_rid="DS-001",
            dataset_type="Training",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestRemoveDatasetType
# =============================================================================


class TestRemoveDatasetType:
    """Tests for the remove_dataset_type tool."""

    @pytest.mark.asyncio
    async def test_remove_type_success(self, dataset_tools, mock_ml):
        """Removing a type returns status=removed with updated types list."""
        mock_dataset = _make_mock_dataset(dataset_types=["Testing"])
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["remove_dataset_type"](
            dataset_rid="DS-001",
            dataset_type="Training",
        )

        data = assert_success(result)
        assert data["status"] == "removed"
        assert data["dataset_rid"] == "DS-001"
        assert data["dataset_types"] == ["Testing"]
        mock_dataset.remove_dataset_type.assert_called_once_with("Training")

    @pytest.mark.asyncio
    async def test_remove_type_exception(self, dataset_tools, mock_ml):
        """When remove_dataset_type raises, return an error."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.remove_dataset_type.side_effect = ValueError("Type not on dataset")
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["remove_dataset_type"](
            dataset_rid="DS-001",
            dataset_type="Missing",
        )

        assert_error(result, expected_message="Type not on dataset")

    @pytest.mark.asyncio
    async def test_remove_type_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["remove_dataset_type"](
            dataset_rid="DS-001",
            dataset_type="Training",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestAddDatasetElementType
# =============================================================================


class TestAddDatasetElementType:
    """Tests for the add_dataset_element_type tool."""

    @pytest.mark.asyncio
    async def test_add_element_type_success(self, dataset_tools, mock_ml):
        """Registering a table returns status=success with association table name."""
        mock_table = MagicMock()
        mock_table.name = "Dataset_Subject"
        mock_ml.add_dataset_element_type.return_value = mock_table

        result = await dataset_tools["add_dataset_element_type"](table_name="Subject")

        data = assert_success(result)
        assert data["table_name"] == "Subject"
        assert data["association_table"] == "Dataset_Subject"
        mock_ml.add_dataset_element_type.assert_called_once_with("Subject")

    @pytest.mark.asyncio
    async def test_add_element_type_exception(self, dataset_tools, mock_ml):
        """When add_dataset_element_type raises, return an error."""
        mock_ml.add_dataset_element_type.side_effect = ValueError("Table not found")

        result = await dataset_tools["add_dataset_element_type"](table_name="Missing")

        assert_error(result, expected_message="Table not found")

    @pytest.mark.asyncio
    async def test_add_element_type_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["add_dataset_element_type"](
            table_name="Subject",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestAddDatasetChild
# =============================================================================


class TestAddDatasetChild:
    """Tests for the add_dataset_child tool."""

    @pytest.mark.asyncio
    async def test_add_child_success(self, dataset_tools, mock_ml):
        """Nesting a child dataset returns status=added."""
        mock_parent = _make_mock_dataset(dataset_rid="PARENT-001")
        mock_ml.lookup_dataset.return_value = mock_parent

        result = await dataset_tools["add_dataset_child"](
            parent_rid="PARENT-001",
            child_rid="CHILD-001",
        )

        data = assert_success(result)
        assert data["status"] == "added"
        assert data["parent_rid"] == "PARENT-001"
        assert data["child_rid"] == "CHILD-001"
        mock_parent.add_dataset_members.assert_called_once_with(members=["CHILD-001"])

    @pytest.mark.asyncio
    async def test_add_child_exception(self, dataset_tools, mock_ml):
        """When add_dataset_members raises, return an error."""
        mock_parent = _make_mock_dataset()
        mock_parent.add_dataset_members.side_effect = ValueError("Cannot nest")
        mock_ml.lookup_dataset.return_value = mock_parent

        result = await dataset_tools["add_dataset_child"](
            parent_rid="PARENT-001",
            child_rid="INVALID",
        )

        assert_error(result, expected_message="Cannot nest")

    @pytest.mark.asyncio
    async def test_add_child_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["add_dataset_child"](
            parent_rid="P-001",
            child_rid="C-001",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestListDatasetParents
# =============================================================================


class TestListDatasetParents:
    """Tests for the list_dataset_parents tool."""

    @pytest.mark.asyncio
    async def test_list_parents_success(self, dataset_tools, mock_ml):
        """Listing parents returns serialized parent datasets."""
        mock_child = _make_mock_dataset()
        parent1 = _make_mock_dataset(
            dataset_rid="PARENT-1", description="Complete dataset",
            dataset_types=["Complete"], current_version="1.0.0",
        )
        mock_child.list_dataset_parents.return_value = [parent1]
        mock_ml.lookup_dataset.return_value = mock_child

        result = await dataset_tools["list_dataset_parents"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert len(data) == 1
        assert data[0]["dataset_rid"] == "PARENT-1"
        assert data[0]["description"] == "Complete dataset"
        assert data[0]["dataset_types"] == ["Complete"]
        assert data[0]["current_version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_list_parents_empty(self, dataset_tools, mock_ml):
        """When no parents, return empty array."""
        mock_child = _make_mock_dataset()
        mock_child.list_dataset_parents.return_value = []
        mock_ml.lookup_dataset.return_value = mock_child

        result = await dataset_tools["list_dataset_parents"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert data == []

    @pytest.mark.asyncio
    async def test_list_parents_with_params(self, dataset_tools, mock_ml):
        """Recurse and version parameters are passed through."""
        mock_child = _make_mock_dataset()
        mock_child.list_dataset_parents.return_value = []
        mock_ml.lookup_dataset.return_value = mock_child

        await dataset_tools["list_dataset_parents"](
            dataset_rid="DS-001",
            recurse=True,
            version="2.0.0",
        )

        mock_child.list_dataset_parents.assert_called_once_with(
            recurse=True, version="2.0.0",
        )

    @pytest.mark.asyncio
    async def test_list_parents_exception(self, dataset_tools, mock_ml):
        """When list_dataset_parents raises, return an error."""
        mock_ml.lookup_dataset.side_effect = RuntimeError("Lookup failed")

        result = await dataset_tools["list_dataset_parents"](dataset_rid="INVALID")

        assert_error(result, expected_message="Lookup failed")

    @pytest.mark.asyncio
    async def test_list_parents_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["list_dataset_parents"](
            dataset_rid="DS-001",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestEstimateBagSize
# =============================================================================


class TestEstimateBagSize:
    """Tests for the estimate_bag_size tool."""

    @pytest.mark.asyncio
    async def test_estimate_success(self, dataset_tools, mock_ml):
        """Estimating bag size returns table breakdown and totals."""
        mock_estimate = {
            "tables": {
                "Image": {"row_count": 100, "is_asset": True, "asset_bytes": 500_000_000},
                "Subject": {"row_count": 20, "is_asset": False, "asset_bytes": 0},
            },
            "total_rows": 120,
            "total_asset_bytes": 500_000_000,
            "total_asset_size": "476.8 MB",
        }
        mock_ml.estimate_bag_size.return_value = mock_estimate

        mock_spec_cls = MagicMock()
        mock_spec_instance = MagicMock()
        mock_spec_cls.return_value = mock_spec_instance

        with patch("deriva_ml.dataset.aux_classes.DatasetSpec", mock_spec_cls):
            result = await dataset_tools["estimate_bag_size"](
                dataset_rid="DS-001",
                version="1.0.0",
            )

        data = assert_success(result)
        assert data["total_rows"] == 120
        assert data["total_asset_bytes"] == 500_000_000
        assert data["total_asset_size"] == "476.8 MB"
        assert "Image" in data["tables"]
        assert data["tables"]["Image"]["is_asset"] is True
        assert data["tables"]["Image"]["asset_bytes"] == 500_000_000
        assert data["tables"]["Subject"]["row_count"] == 20
        mock_spec_cls.assert_called_once_with(
            rid="DS-001", version="1.0.0", exclude_tables=None,
        )
        mock_ml.estimate_bag_size.assert_called_once_with(mock_spec_instance)

    @pytest.mark.asyncio
    async def test_estimate_with_exclude_tables(self, dataset_tools, mock_ml):
        """Exclude tables are passed through to DatasetSpec."""
        mock_ml.estimate_bag_size.return_value = {
            "tables": {},
            "total_rows": 0,
            "total_asset_bytes": 0,
            "total_asset_size": "0 B",
        }

        mock_spec_cls = MagicMock()
        mock_spec_instance = MagicMock()
        mock_spec_cls.return_value = mock_spec_instance

        with patch("deriva_ml.dataset.aux_classes.DatasetSpec", mock_spec_cls):
            result = await dataset_tools["estimate_bag_size"](
                dataset_rid="DS-001",
                version="1.0.0",
                exclude_tables=["Study", "Protocol"],
            )

        data = assert_success(result)
        mock_spec_cls.assert_called_once_with(
            rid="DS-001", version="1.0.0", exclude_tables={"Study", "Protocol"},
        )

    @pytest.mark.asyncio
    async def test_estimate_error(self, dataset_tools, mock_ml):
        """Errors are returned as error status."""
        mock_ml.estimate_bag_size.side_effect = Exception("Version not found")

        mock_spec_cls = MagicMock()
        mock_spec_cls.return_value = MagicMock()

        with patch("deriva_ml.dataset.aux_classes.DatasetSpec", mock_spec_cls):
            result = await dataset_tools["estimate_bag_size"](
                dataset_rid="DS-001",
                version="99.0.0",
            )

        assert_error(result, expected_message="Version not found")

    @pytest.mark.asyncio
    async def test_estimate_not_connected(self, dataset_tools_disconnected):
        """Returns error when no catalog is connected."""
        result = await dataset_tools_disconnected["estimate_bag_size"](
            dataset_rid="DS-001",
            version="1.0.0",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestDenormalizeDataset
# =============================================================================


class TestDenormalizeDataset:
    """Tests for the preview_denormalized_dataset tool."""

    @pytest.mark.asyncio
    async def test_schema_only_no_rid(self, dataset_tools, mock_ml):
        """When no dataset_rid, returns schema shape and global size estimates."""
        mock_ml.denormalize_info.return_value = {
            "columns": [("Image.RID", "ermrest_rid"), ("Image.Filename", "text")],
            "join_path": ["Image"],
            "tables": {"Image": {"row_count": 100, "is_asset": True, "asset_bytes": 5000}},
            "total_rows": 100,
            "total_asset_bytes": 5000,
            "total_asset_size": "4.9 KB",
        }

        result = await dataset_tools["preview_denormalized_dataset"](
            include_tables=["Image"],
        )

        data = parse_json_result(result)
        assert "columns" in data
        assert "join_path" in data
        assert "tables" in data
        assert data["total_rows"] == 100
        assert "rows" not in data
        mock_ml.denormalize_info.assert_called_once_with(["Image"])

    @pytest.mark.asyncio
    async def test_dataset_scoped_no_rows(self, dataset_tools, mock_ml):
        """With dataset_rid but limit=0, returns scoped estimates without rows."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.denormalize_info.return_value = {
            "columns": [("Image.RID", "ermrest_rid")],
            "join_path": ["Image"],
            "tables": {"Image": {"row_count": 50, "is_asset": False, "asset_bytes": 0}},
            "total_rows": 50,
            "total_asset_bytes": 0,
            "total_asset_size": "0 B",
        }
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["preview_denormalized_dataset"](
            include_tables=["Image"],
            dataset_rid="DS-001",
            limit=0,
        )

        data = parse_json_result(result)
        assert data["total_rows"] == 50
        assert data["dataset_rid"] == "DS-001"
        assert "rows" not in data
        mock_dataset.denormalize_info.assert_called_once_with(["Image"], version=None)

    @pytest.mark.asyncio
    async def test_dataset_with_rows(self, dataset_tools, mock_ml):
        """With dataset_rid and limit>0, returns estimates plus row preview."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.denormalize_info.return_value = {
            "columns": [("Image.RID", "ermrest_rid"), ("Image.Filename", "text")],
            "join_path": ["Image"],
            "tables": {"Image": {"row_count": 50, "is_asset": False, "asset_bytes": 0}},
            "total_rows": 50,
            "total_asset_bytes": 0,
            "total_asset_size": "0 B",
        }
        mock_dataset.denormalize_as_dict.return_value = iter([
            {"Image.RID": "IMG-1", "Image.Filename": "a.jpg"},
            {"Image.RID": "IMG-2", "Image.Filename": "b.jpg"},
        ])
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["preview_denormalized_dataset"](
            include_tables=["Image"],
            dataset_rid="DS-001",
            limit=25,
        )

        data = parse_json_result(result)
        assert data["total_rows"] == 50
        assert "rows" in data
        assert data["count"] == 2
        assert data["rows"][0]["Image.RID"] == "IMG-1"

    @pytest.mark.asyncio
    async def test_dataset_with_version(self, dataset_tools, mock_ml):
        """Version parameter is passed through to denormalize_info and denormalize_as_dict."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.denormalize_info.return_value = {
            "columns": [("col", "text")],
            "join_path": ["Image"],
            "tables": {"Image": {"row_count": 1, "is_asset": False, "asset_bytes": 0}},
            "total_rows": 1,
            "total_asset_bytes": 0,
            "total_asset_size": "0 B",
        }
        mock_dataset.denormalize_as_dict.return_value = iter([{"col": "val"}])
        mock_ml.lookup_dataset.return_value = mock_dataset

        await dataset_tools["preview_denormalized_dataset"](
            include_tables=["Image"],
            dataset_rid="DS-001",
            version="1.0.0",
            limit=25,
        )

        mock_dataset.denormalize_info.assert_called_once_with(["Image"], version="1.0.0")
        mock_dataset.denormalize_as_dict.assert_called_once_with(["Image"], version="1.0.0")

    @pytest.mark.asyncio
    async def test_respects_limit(self, dataset_tools, mock_ml):
        """Limit parameter truncates rows."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.denormalize_info.return_value = {
            "columns": [("col", "text")],
            "join_path": ["Image"],
            "tables": {"Image": {"row_count": 100, "is_asset": False, "asset_bytes": 0}},
            "total_rows": 100,
            "total_asset_bytes": 0,
            "total_asset_size": "0 B",
        }
        mock_dataset.denormalize_as_dict.return_value = iter(
            [{"col": f"val{i}"} for i in range(100)]
        )
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["preview_denormalized_dataset"](
            include_tables=["Image"],
            dataset_rid="DS-001",
            limit=5,
        )

        data = parse_json_result(result)
        assert data["count"] == 5
        assert data["limit"] == 5

    @pytest.mark.asyncio
    async def test_exception(self, dataset_tools, mock_ml):
        """When lookup_dataset raises, return an error."""
        mock_ml.lookup_dataset.side_effect = RuntimeError("Bad table")

        result = await dataset_tools["preview_denormalized_dataset"](
            include_tables=["NonExistent"],
            dataset_rid="DS-001",
        )

        assert_error(result, expected_message="Bad table")

    @pytest.mark.asyncio
    async def test_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["preview_denormalized_dataset"](
            include_tables=["Image"],
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestCreateDatasetTypeTerm
# =============================================================================


class TestCreateDatasetTypeTerm:
    """Tests for the create_dataset_type_term tool."""

    @pytest.mark.asyncio
    async def test_create_term_success(self, dataset_tools, mock_ml):
        """Creating a type term returns status=created."""
        mock_term = MagicMock()
        mock_term.name = "Validation"
        mock_term.description = "Held-out data"
        mock_term.synonyms = ["val", "valid"]
        mock_term.rid = "DT-001"
        mock_ml.add_term.return_value = mock_term

        result = await dataset_tools["create_dataset_type_term"](
            type_name="Validation",
            description="Held-out data",
            synonyms=["val", "valid"],
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "Validation"
        assert data["description"] == "Held-out data"
        assert data["synonyms"] == ["val", "valid"]
        assert data["rid"] == "DT-001"
        mock_ml.add_term.assert_called_once_with(
            table="Dataset_Type",
            term_name="Validation",
            description="Held-out data",
            synonyms=["val", "valid"],
            exists_ok=False,
        )

    @pytest.mark.asyncio
    async def test_create_term_no_synonyms(self, dataset_tools, mock_ml):
        """When synonyms is None, pass empty list."""
        mock_term = MagicMock()
        mock_term.name = "Custom"
        mock_term.description = "Custom type"
        mock_term.synonyms = []
        mock_term.rid = "DT-002"
        mock_ml.add_term.return_value = mock_term

        result = await dataset_tools["create_dataset_type_term"](
            type_name="Custom",
            description="Custom type",
        )

        data = assert_success(result)
        assert data["synonyms"] == []
        mock_ml.add_term.assert_called_once_with(
            table="Dataset_Type",
            term_name="Custom",
            description="Custom type",
            synonyms=[],
            exists_ok=False,
        )

    @pytest.mark.asyncio
    async def test_create_term_already_exists(self, dataset_tools, mock_ml):
        """When term already exists, return status=exists with existing info."""
        mock_ml.add_term.side_effect = ValueError("Term already exists")
        mock_existing = MagicMock()
        mock_existing.name = "Training"
        mock_existing.description = "Training data"
        mock_existing.synonyms = ["train"]
        mock_existing.rid = "DT-003"
        mock_ml.lookup_term.return_value = mock_existing

        result = await dataset_tools["create_dataset_type_term"](
            type_name="Training",
            description="Training data",
        )

        data = parse_json_result(result)
        assert data["status"] == "exists"
        assert data["name"] == "Training"
        assert data["rid"] == "DT-003"

    @pytest.mark.asyncio
    async def test_create_term_already_exists_lookup_fails(self, dataset_tools, mock_ml):
        """When term exists but lookup also fails, return original error."""
        mock_ml.add_term.side_effect = ValueError("Term already exists in catalog")
        mock_ml.lookup_term.side_effect = RuntimeError("Lookup failed too")

        result = await dataset_tools["create_dataset_type_term"](
            type_name="Bad",
            description="desc",
        )

        assert_error(result, expected_message="already exists")

    @pytest.mark.asyncio
    async def test_create_term_other_error(self, dataset_tools, mock_ml):
        """When add_term raises a non-exists error, return it directly."""
        mock_ml.add_term.side_effect = RuntimeError("Connection lost")

        result = await dataset_tools["create_dataset_type_term"](
            type_name="Bad",
            description="desc",
        )

        assert_error(result, expected_message="Connection lost")

    @pytest.mark.asyncio
    async def test_create_term_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["create_dataset_type_term"](
            type_name="Test",
            description="desc",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestDeleteDatasetTypeTerm
# =============================================================================


class TestDeleteDatasetTypeTerm:
    """Tests for the delete_dataset_type_term tool."""

    @pytest.mark.asyncio
    async def test_delete_term_success(self, dataset_tools, mock_ml):
        """Deleting a type term returns status=deleted."""
        result = await dataset_tools["delete_dataset_type_term"](type_name="Obsolete")

        data = parse_json_result(result)
        assert data["status"] == "deleted"
        assert data["name"] == "Obsolete"
        mock_ml.delete_term.assert_called_once_with("Dataset_Type", "Obsolete")

    @pytest.mark.asyncio
    async def test_delete_term_exception(self, dataset_tools, mock_ml):
        """When delete_term raises, return an error."""
        mock_ml.delete_term.side_effect = ValueError("Foreign key constraint")

        result = await dataset_tools["delete_dataset_type_term"](type_name="InUse")

        assert_error(result, expected_message="Foreign key constraint")

    @pytest.mark.asyncio
    async def test_delete_term_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["delete_dataset_type_term"](
            type_name="Test",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestSplitDataset
# =============================================================================


class TestSplitDataset:
    """Tests for the split_dataset tool.

    The split_dataset tool does a ``from deriva_ml.dataset.split import split_dataset``
    inside the function body.  That module may not exist in the installed version of
    deriva_ml, so we inject a fake module into ``sys.modules`` before each test.
    """

    @staticmethod
    def _patch_split_module(mock_fn):
        """Create and register a fake ``deriva_ml.dataset.split`` module.

        Returns a context manager that removes it on exit.
        """
        fake_mod = types.ModuleType("deriva_ml.dataset.split")
        fake_mod.split_dataset = mock_fn  # type: ignore[attr-defined]

        class _Ctx:
            def __enter__(self_ctx):
                sys.modules["deriva_ml.dataset.split"] = fake_mod
                return mock_fn

            def __exit__(self_ctx, *exc):
                sys.modules.pop("deriva_ml.dataset.split", None)

        return _Ctx()

    @pytest.mark.asyncio
    async def test_split_success(self, dataset_tools, mock_ml):
        """Splitting a dataset returns status=success with split info."""
        split_result = {
            "split": {"rid": "SPLIT-001", "version": "0.1.0"},
            "training": {"rid": "TRAIN-001", "version": "0.1.0", "member_count": 80},
            "testing": {"rid": "TEST-001", "version": "0.1.0", "member_count": 20},
            "source": {"rid": "DS-001"},
        }

        mock_split = MagicMock(return_value=split_result)
        with self._patch_split_module(mock_split):
            result = await dataset_tools["split_dataset"](
                source_dataset_rid="DS-001",
                test_size=0.2,
                seed=42,
            )

            data = assert_success(result)
            assert data["split"]["rid"] == "SPLIT-001"
            assert data["training"]["member_count"] == 80
            assert data["testing"]["member_count"] == 20
            assert data["source"]["rid"] == "DS-001"

            mock_split.assert_called_once_with(
                ml=mock_ml,
                source_dataset_rid="DS-001",
                test_size=0.2,
                train_size=None,
                seed=42,
                shuffle=True,
                stratify_by_column=None,
                element_table=None,
                include_tables=None,
                training_types=None,
                testing_types=None,
                split_description="",
                dry_run=False,
            )

    @pytest.mark.asyncio
    async def test_split_with_all_params(self, dataset_tools, mock_ml):
        """All split parameters are passed through."""
        split_result = {
            "split": {"rid": "SPLIT-002"},
            "training": {"rid": "TRAIN-002"},
            "testing": {"rid": "TEST-002"},
        }

        mock_split = MagicMock(return_value=split_result)
        with self._patch_split_module(mock_split):
            await dataset_tools["split_dataset"](
                source_dataset_rid="DS-001",
                test_size=0.3,
                train_size=0.7,
                seed=123,
                shuffle=False,
                stratify_by_column="Image_Classification_Image_Class",
                element_table="Image",
                include_tables=["Image", "Image_Classification"],
                training_types=["Labeled"],
                testing_types=["Labeled"],
                split_description="Stratified split",
                dry_run=True,
            )

            mock_split.assert_called_once_with(
                ml=mock_ml,
                source_dataset_rid="DS-001",
                test_size=0.3,
                train_size=0.7,
                seed=123,
                shuffle=False,
                stratify_by_column="Image_Classification_Image_Class",
                element_table="Image",
                include_tables=["Image", "Image_Classification"],
                training_types=["Labeled"],
                testing_types=["Labeled"],
                split_description="Stratified split",
                dry_run=True,
            )

    @pytest.mark.asyncio
    async def test_split_dry_run(self, dataset_tools, mock_ml):
        """Dry run returns preview without modifying catalog."""
        dry_result = {
            "dry_run": True,
            "source_rid": "DS-001",
            "training_count": 80,
            "testing_count": 20,
        }

        mock_split = MagicMock(return_value=dry_result)
        with self._patch_split_module(mock_split):
            result = await dataset_tools["split_dataset"](
                source_dataset_rid="DS-001",
                dry_run=True,
            )

            data = assert_success(result)
            assert data["dry_run"] is True

    @pytest.mark.asyncio
    async def test_split_exception(self, dataset_tools, mock_ml):
        """When split_dataset raises, return an error."""
        mock_split = MagicMock(side_effect=ValueError("Not enough samples"))
        with self._patch_split_module(mock_split):
            result = await dataset_tools["split_dataset"](
                source_dataset_rid="DS-001",
            )

            assert_error(result, expected_message="Not enough samples")

    @pytest.mark.asyncio
    async def test_split_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        # The disconnected manager raises before the import, so we still need
        # the fake module available in case the import happens first.
        mock_split = MagicMock()
        with self._patch_split_module(mock_split):
            result = await dataset_tools_disconnected["split_dataset"](
                source_dataset_rid="DS-001",
            )

            assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestSerializeDataset (unit test for the internal helper)
# =============================================================================


class TestSerializeDataset:
    """Tests for the _serialize_dataset helper function."""

    def test_serialize_basic(self):
        """Basic serialization extracts expected fields."""
        from deriva_mcp.tools.dataset import _serialize_dataset

        mock_ds = _make_mock_dataset(
            dataset_rid="DS-100",
            description="A dataset",
            dataset_types=["Training", "Image"],
            current_version="1.2.3",
        )

        result = _serialize_dataset(mock_ds)

        assert result["dataset_rid"] == "DS-100"
        assert result["description"] == "A dataset"
        assert result["dataset_types"] == ["Training", "Image"]
        assert result["current_version"] == "1.2.3"

    def test_serialize_null_version(self):
        """When current_version is None, serializes as None."""
        from deriva_mcp.tools.dataset import _serialize_dataset

        mock_ds = _make_mock_dataset(current_version=None)

        result = _serialize_dataset(mock_ds)

        assert result["current_version"] is None

    def test_serialize_empty_types(self):
        """Empty dataset_types serializes as empty list."""
        from deriva_mcp.tools.dataset import _serialize_dataset

        mock_ds = _make_mock_dataset(dataset_types=[])

        result = _serialize_dataset(mock_ds)

        assert result["dataset_types"] == []


# =============================================================================
# TestToolRegistration
# =============================================================================


class TestToolRegistration:
    """Verify that all expected dataset tools are registered."""

    def test_all_tools_registered(self, dataset_tools):
        """All expected dataset tools should be registered."""
        expected_tools = [
            "create_dataset",
            "get_dataset_spec",
            "add_dataset_members",
            "delete_dataset_members",
            "increment_dataset_version",
            "delete_dataset",
            "set_dataset_description",
            "add_dataset_type",
            "remove_dataset_type",
            "add_dataset_element_type",
            "add_dataset_child",
            "list_dataset_parents",
            "estimate_bag_size",
            "preview_denormalized_dataset",
            "create_dataset_type_term",
            "delete_dataset_type_term",
            "split_dataset",
        ]
        for tool_name in expected_tools:
            assert tool_name in dataset_tools, f"Tool '{tool_name}' not registered"

    def test_no_extra_tools(self, dataset_tools):
        """Only the expected tools should be registered."""
        expected_count = 19
        assert len(dataset_tools) == expected_count, (
            f"Expected {expected_count} tools, got {len(dataset_tools)}: "
            f"{sorted(dataset_tools.keys())}"
        )
