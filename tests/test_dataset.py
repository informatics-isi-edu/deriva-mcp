"""Unit tests for dataset management tools.

Tests cover all 21 dataset tools registered by register_dataset_tools:
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
- list_dataset_children: List child datasets
- list_dataset_parents: List parent datasets
- list_dataset_executions: List executions that used a dataset
- download_dataset: Download as BDBag
- denormalize_dataset: Join tables into wide format
- create_dataset_type_term: Create Dataset_Type vocab term
- delete_dataset_type_term: Delete Dataset_Type vocab term
- restructure_assets: Restructure assets into directory hierarchy
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


class TestListDatasetMembers:
    """Tests for the list_dataset_members tool."""

    @pytest.mark.asyncio
    async def test_list_members_basic(self, dataset_tools, mock_ml):
        """List members grouped by table name."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.list_dataset_members.return_value = {
            "Image": [{"RID": "IMG-1"}, {"RID": "IMG-2"}],
            "Subject": [{"RID": "SUBJ-1"}],
        }
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["list_dataset_members"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert len(data["Image"]) == 2
        assert data["Image"][0]["RID"] == "IMG-1"
        assert len(data["Subject"]) == 1
        assert data["Subject"][0]["RID"] == "SUBJ-1"

    @pytest.mark.asyncio
    async def test_list_members_with_version(self, dataset_tools, mock_ml):
        """Version is passed through to the underlying method."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.list_dataset_members.return_value = {}
        mock_ml.lookup_dataset.return_value = mock_dataset

        await dataset_tools["list_dataset_members"](
            dataset_rid="DS-001",
            version="1.0.0",
        )

        mock_dataset.list_dataset_members.assert_called_once_with(
            version="1.0.0", recurse=False, limit=None,
        )

    @pytest.mark.asyncio
    async def test_list_members_recurse(self, dataset_tools, mock_ml):
        """Recurse parameter is passed through."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.list_dataset_members.return_value = {}
        mock_ml.lookup_dataset.return_value = mock_dataset

        await dataset_tools["list_dataset_members"](
            dataset_rid="DS-001",
            recurse=True,
        )

        mock_dataset.list_dataset_members.assert_called_once_with(
            version=None, recurse=True, limit=None,
        )

    @pytest.mark.asyncio
    async def test_list_members_with_limit(self, dataset_tools, mock_ml):
        """Limit parameter is passed through."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.list_dataset_members.return_value = {}
        mock_ml.lookup_dataset.return_value = mock_dataset

        await dataset_tools["list_dataset_members"](
            dataset_rid="DS-001",
            limit=10,
        )

        mock_dataset.list_dataset_members.assert_called_once_with(
            version=None, recurse=False, limit=10,
        )

    @pytest.mark.asyncio
    async def test_list_members_empty(self, dataset_tools, mock_ml):
        """Empty dataset returns empty dict."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.list_dataset_members.return_value = {}
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["list_dataset_members"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert data == {}

    @pytest.mark.asyncio
    async def test_list_members_only_rid_returned(self, dataset_tools, mock_ml):
        """Only the RID field is returned from each member (extra fields stripped)."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.list_dataset_members.return_value = {
            "Image": [{"RID": "IMG-1", "Filename": "img1.jpg", "Size": 1024}],
        }
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["list_dataset_members"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert data["Image"] == [{"RID": "IMG-1"}]

    @pytest.mark.asyncio
    async def test_list_members_exception(self, dataset_tools, mock_ml):
        """When list_dataset_members raises, return an error."""
        mock_ml.lookup_dataset.side_effect = RuntimeError("Dataset not found")

        result = await dataset_tools["list_dataset_members"](dataset_rid="INVALID")

        assert_error(result, expected_message="Dataset not found")

    @pytest.mark.asyncio
    async def test_list_members_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["list_dataset_members"](dataset_rid="DS-001")

        assert_error(result, expected_message="No active catalog connection")


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

        with patch("deriva_ml_mcp.tools.dataset.VersionPart", create=True):
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
# TestListDatasetChildren
# =============================================================================


class TestListDatasetChildren:
    """Tests for the list_dataset_children tool."""

    @pytest.mark.asyncio
    async def test_list_children_success(self, dataset_tools, mock_ml):
        """Listing children returns serialized child datasets."""
        mock_parent = _make_mock_dataset()
        child1 = _make_mock_dataset(
            dataset_rid="CHILD-1", description="Training set",
            dataset_types=["Training"], current_version="0.2.0",
        )
        child2 = _make_mock_dataset(
            dataset_rid="CHILD-2", description="Testing set",
            dataset_types=["Testing"], current_version="0.3.0",
        )
        mock_parent.list_dataset_children.return_value = [child1, child2]
        mock_ml.lookup_dataset.return_value = mock_parent

        result = await dataset_tools["list_dataset_children"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert len(data) == 2
        assert data[0]["dataset_rid"] == "CHILD-1"
        assert data[0]["description"] == "Training set"
        assert data[0]["dataset_types"] == ["Training"]
        assert data[0]["current_version"] == "0.2.0"
        assert data[1]["dataset_rid"] == "CHILD-2"

    @pytest.mark.asyncio
    async def test_list_children_empty(self, dataset_tools, mock_ml):
        """When no children, return empty array."""
        mock_parent = _make_mock_dataset()
        mock_parent.list_dataset_children.return_value = []
        mock_ml.lookup_dataset.return_value = mock_parent

        result = await dataset_tools["list_dataset_children"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert data == []

    @pytest.mark.asyncio
    async def test_list_children_with_params(self, dataset_tools, mock_ml):
        """Recurse and version parameters are passed through."""
        mock_parent = _make_mock_dataset()
        mock_parent.list_dataset_children.return_value = []
        mock_ml.lookup_dataset.return_value = mock_parent

        await dataset_tools["list_dataset_children"](
            dataset_rid="DS-001",
            recurse=True,
            version="1.0.0",
        )

        mock_parent.list_dataset_children.assert_called_once_with(
            recurse=True, version="1.0.0",
        )

    @pytest.mark.asyncio
    async def test_list_children_null_version(self, dataset_tools, mock_ml):
        """Child with null current_version serializes as None."""
        mock_parent = _make_mock_dataset()
        child = _make_mock_dataset(dataset_rid="CHILD-1", current_version=None)
        mock_parent.list_dataset_children.return_value = [child]
        mock_ml.lookup_dataset.return_value = mock_parent

        result = await dataset_tools["list_dataset_children"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert data[0]["current_version"] is None

    @pytest.mark.asyncio
    async def test_list_children_exception(self, dataset_tools, mock_ml):
        """When list_dataset_children raises, return an error."""
        mock_ml.lookup_dataset.side_effect = RuntimeError("Lookup failed")

        result = await dataset_tools["list_dataset_children"](dataset_rid="INVALID")

        assert_error(result, expected_message="Lookup failed")

    @pytest.mark.asyncio
    async def test_list_children_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["list_dataset_children"](
            dataset_rid="DS-001",
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
# TestListDatasetExecutions
# =============================================================================


class TestListDatasetExecutions:
    """Tests for the list_dataset_executions tool."""

    @pytest.mark.asyncio
    async def test_list_executions_success(self, dataset_tools, mock_ml):
        """Listing executions returns execution details."""
        mock_dataset = _make_mock_dataset()

        mock_exe1 = MagicMock()
        mock_exe1.execution_rid = "EXE-1"
        mock_exe1.configuration.description = "Training run"
        mock_exe1.status.value = "complete"
        mock_exe1.workflow_rid = "WF-1"

        mock_exe2 = MagicMock()
        mock_exe2.execution_rid = "EXE-2"
        mock_exe2.configuration = None
        mock_exe2.status = None
        mock_exe2.workflow_rid = "WF-2"

        mock_dataset.list_executions.return_value = [mock_exe1, mock_exe2]
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["list_dataset_executions"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert len(data) == 2
        assert data[0]["execution_rid"] == "EXE-1"
        assert data[0]["description"] == "Training run"
        assert data[0]["status"] == "complete"
        assert data[0]["workflow_rid"] == "WF-1"
        assert data[1]["execution_rid"] == "EXE-2"
        assert data[1]["description"] is None
        assert data[1]["status"] is None

    @pytest.mark.asyncio
    async def test_list_executions_empty(self, dataset_tools, mock_ml):
        """When no executions, return empty array."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.list_executions.return_value = []
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["list_dataset_executions"](dataset_rid="DS-001")

        data = parse_json_result(result)
        assert data == []

    @pytest.mark.asyncio
    async def test_list_executions_exception(self, dataset_tools, mock_ml):
        """When list_executions raises, return an error."""
        mock_ml.lookup_dataset.side_effect = RuntimeError("Not found")

        result = await dataset_tools["list_dataset_executions"](dataset_rid="INVALID")

        assert_error(result, expected_message="Not found")

    @pytest.mark.asyncio
    async def test_list_executions_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["list_dataset_executions"](
            dataset_rid="DS-001",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestDownloadDataset
# =============================================================================


class TestDownloadDataset:
    """Tests for the download_dataset tool."""

    @pytest.mark.asyncio
    async def test_download_success(self, dataset_tools, mock_ml):
        """Downloading a dataset returns bag metadata."""
        mock_bag = MagicMock()
        mock_bag.dataset_rid = "DS-001"
        mock_bag.current_version = "1.0.0"
        mock_bag.description = "Training data"
        mock_bag.dataset_types = ["Training"]
        mock_bag.execution_rid = "EXE-001"
        mock_bag.model.bag_path = "/tmp/bags/DS-001"
        mock_ml.download_dataset_bag.return_value = mock_bag

        mock_spec_cls = MagicMock()
        mock_spec_instance = MagicMock()
        mock_spec_cls.return_value = mock_spec_instance

        with patch("deriva_ml.dataset.aux_classes.DatasetSpec", mock_spec_cls):
            result = await dataset_tools["download_dataset"](
                dataset_rid="DS-001",
                version="1.0.0",
            )

        data = assert_success(result)
        assert data["status"] == "downloaded"
        assert data["dataset_rid"] == "DS-001"
        assert data["version"] == "1.0.0"
        assert data["description"] == "Training data"
        assert data["dataset_types"] == ["Training"]
        assert data["execution_rid"] == "EXE-001"
        assert data["bag_path"] == "/tmp/bags/DS-001"
        mock_spec_cls.assert_called_once_with(
            rid="DS-001", version="1.0.0", materialize=True,
        )
        mock_ml.download_dataset_bag.assert_called_once_with(mock_spec_instance)

    @pytest.mark.asyncio
    async def test_download_no_materialize(self, dataset_tools, mock_ml):
        """Downloading with materialize=False passes the flag through."""
        mock_bag = MagicMock()
        mock_bag.dataset_rid = "DS-001"
        mock_bag.current_version = "1.0.0"
        mock_bag.description = "Metadata only"
        mock_bag.dataset_types = []
        mock_bag.execution_rid = None
        mock_bag.model.bag_path = "/tmp/bags/DS-001"
        mock_ml.download_dataset_bag.return_value = mock_bag

        mock_spec_cls = MagicMock()
        mock_spec_instance = MagicMock()
        mock_spec_cls.return_value = mock_spec_instance

        with patch("deriva_ml.dataset.aux_classes.DatasetSpec", mock_spec_cls):
            result = await dataset_tools["download_dataset"](
                dataset_rid="DS-001",
                version="1.0.0",
                materialize=False,
            )

        data = assert_success(result)
        assert data["status"] == "downloaded"
        mock_spec_cls.assert_called_once_with(
            rid="DS-001", version="1.0.0", materialize=False,
        )

    @pytest.mark.asyncio
    async def test_download_null_version(self, dataset_tools, mock_ml):
        """When bag has no current_version, version is None."""
        mock_bag = MagicMock()
        mock_bag.dataset_rid = "DS-001"
        mock_bag.current_version = None
        mock_bag.description = "No version"
        mock_bag.dataset_types = []
        mock_bag.execution_rid = None
        mock_bag.model.bag_path = "/tmp/bags/DS-001"
        mock_ml.download_dataset_bag.return_value = mock_bag

        mock_spec_cls = MagicMock()
        mock_spec_cls.return_value = MagicMock()

        with patch("deriva_ml.dataset.aux_classes.DatasetSpec", mock_spec_cls):
            result = await dataset_tools["download_dataset"](
                dataset_rid="DS-001",
                version="1.0.0",
            )

        data = assert_success(result)
        assert data["version"] is None

    @pytest.mark.asyncio
    async def test_download_exception(self, dataset_tools, mock_ml):
        """When download_dataset_bag raises, return an error."""
        mock_ml.download_dataset_bag.side_effect = RuntimeError("Download failed")

        mock_spec_cls = MagicMock()
        mock_spec_cls.return_value = MagicMock()

        with patch("deriva_ml.dataset.aux_classes.DatasetSpec", mock_spec_cls):
            result = await dataset_tools["download_dataset"](
                dataset_rid="DS-001",
                version="1.0.0",
            )

        assert_error(result, expected_message="Download failed")

    @pytest.mark.asyncio
    async def test_download_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["download_dataset"](
            dataset_rid="DS-001",
            version="1.0.0",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# TestDenormalizeDataset
# =============================================================================


class TestDenormalizeDataset:
    """Tests for the denormalize_dataset tool."""

    @pytest.mark.asyncio
    async def test_denormalize_success(self, dataset_tools, mock_ml):
        """Denormalizing returns columns and rows."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.denormalize_as_dict.return_value = iter([
            {"Image.RID": "IMG-1", "Image.Filename": "a.jpg", "Diagnosis.Label": "Normal"},
            {"Image.RID": "IMG-2", "Image.Filename": "b.jpg", "Diagnosis.Label": "Abnormal"},
        ])
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["denormalize_dataset"](
            dataset_rid="DS-001",
            include_tables=["Image", "Diagnosis"],
        )

        data = parse_json_result(result)
        assert data["dataset_rid"] == "DS-001"
        assert data["include_tables"] == ["Image", "Diagnosis"]
        assert data["columns"] == ["Image.RID", "Image.Filename", "Diagnosis.Label"]
        assert data["count"] == 2
        assert data["rows"][0]["Image.RID"] == "IMG-1"
        assert data["rows"][1]["Diagnosis.Label"] == "Abnormal"

    @pytest.mark.asyncio
    async def test_denormalize_with_version(self, dataset_tools, mock_ml):
        """Version parameter is passed through."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.denormalize_as_dict.return_value = iter([
            {"col": "val"},
        ])
        mock_ml.lookup_dataset.return_value = mock_dataset

        await dataset_tools["denormalize_dataset"](
            dataset_rid="DS-001",
            include_tables=["Image"],
            version="1.0.0",
        )

        mock_dataset.denormalize_as_dict.assert_called_once_with(
            ["Image"], version="1.0.0",
        )

    @pytest.mark.asyncio
    async def test_denormalize_respects_limit(self, dataset_tools, mock_ml):
        """Limit parameter truncates rows."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.denormalize_as_dict.return_value = iter([
            {"col": f"val{i}"} for i in range(100)
        ])
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["denormalize_dataset"](
            dataset_rid="DS-001",
            include_tables=["Image"],
            limit=5,
        )

        data = parse_json_result(result)
        assert data["count"] == 5
        assert data["limit"] == 5

    @pytest.mark.asyncio
    async def test_denormalize_empty(self, dataset_tools, mock_ml):
        """Empty dataset returns empty columns and rows."""
        mock_dataset = _make_mock_dataset()
        mock_dataset.denormalize_as_dict.return_value = iter([])
        mock_ml.lookup_dataset.return_value = mock_dataset

        result = await dataset_tools["denormalize_dataset"](
            dataset_rid="DS-001",
            include_tables=["Image"],
        )

        data = parse_json_result(result)
        assert data["columns"] == []
        assert data["rows"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_denormalize_exception(self, dataset_tools, mock_ml):
        """When denormalize_as_dict raises, return an error."""
        mock_ml.lookup_dataset.side_effect = RuntimeError("Bad table")

        result = await dataset_tools["denormalize_dataset"](
            dataset_rid="DS-001",
            include_tables=["NonExistent"],
        )

        assert_error(result, expected_message="Bad table")

    @pytest.mark.asyncio
    async def test_denormalize_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["denormalize_dataset"](
            dataset_rid="DS-001",
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
# TestRestructureAssets
# =============================================================================


class TestRestructureAssets:
    """Tests for the restructure_assets tool."""

    @pytest.mark.asyncio
    async def test_restructure_success(self, dataset_tools, mock_ml, tmp_path):
        """Restructuring assets returns status=success with file count."""
        mock_dataset = _make_mock_dataset(current_version="1.0.0")
        mock_ml.lookup_dataset.return_value = mock_dataset

        # Create mock bag
        mock_bag = MagicMock()
        mock_dataset.to_bag.return_value = mock_bag

        # Create actual files in tmp_path for rglob to count
        result_path = tmp_path / "output"
        result_path.mkdir()
        (result_path / "training").mkdir()
        (result_path / "training" / "Normal").mkdir()
        (result_path / "training" / "Normal" / "img1.jpg").touch()
        (result_path / "training" / "Normal" / "img2.jpg").touch()
        mock_bag.restructure_assets.return_value = result_path

        result = await dataset_tools["restructure_assets"](
            dataset_rid="DS-001",
            asset_table="Image",
            output_dir=str(tmp_path / "output"),
            group_by=["Diagnosis"],
        )

        data = assert_success(result)
        assert data["dataset_rid"] == "DS-001"
        assert data["asset_table"] == "Image"
        assert data["group_by"] == ["Diagnosis"]
        assert data["file_count"] == 2
        mock_dataset.to_bag.assert_called_once_with(materialize=True)

    @pytest.mark.asyncio
    async def test_restructure_with_version(self, dataset_tools, mock_ml, tmp_path):
        """When version is provided, set_version is called first."""
        mock_dataset = _make_mock_dataset()
        mock_versioned = _make_mock_dataset(current_version="2.0.0")
        mock_dataset.set_version.return_value = mock_versioned
        mock_ml.lookup_dataset.return_value = mock_dataset

        mock_bag = MagicMock()
        mock_versioned.to_bag.return_value = mock_bag
        result_path = tmp_path / "out"
        result_path.mkdir()
        mock_bag.restructure_assets.return_value = result_path

        result = await dataset_tools["restructure_assets"](
            dataset_rid="DS-001",
            asset_table="Image",
            output_dir=str(tmp_path / "out"),
            version="2.0.0",
        )

        data = assert_success(result)
        mock_dataset.set_version.assert_called_once_with("2.0.0")

    @pytest.mark.asyncio
    async def test_restructure_default_group_by(self, dataset_tools, mock_ml, tmp_path):
        """When group_by is None, pass empty list."""
        mock_dataset = _make_mock_dataset()
        mock_ml.lookup_dataset.return_value = mock_dataset

        mock_bag = MagicMock()
        mock_dataset.to_bag.return_value = mock_bag
        result_path = tmp_path / "out"
        result_path.mkdir()
        mock_bag.restructure_assets.return_value = result_path

        await dataset_tools["restructure_assets"](
            dataset_rid="DS-001",
            asset_table="Image",
            output_dir=str(tmp_path / "out"),
        )

        call_kwargs = mock_bag.restructure_assets.call_args.kwargs
        assert call_kwargs["group_by"] == []

    @pytest.mark.asyncio
    async def test_restructure_no_materialize(self, dataset_tools, mock_ml, tmp_path):
        """When materialize=False, pass it through to to_bag."""
        mock_dataset = _make_mock_dataset()
        mock_ml.lookup_dataset.return_value = mock_dataset

        mock_bag = MagicMock()
        mock_dataset.to_bag.return_value = mock_bag
        result_path = tmp_path / "out"
        result_path.mkdir()
        mock_bag.restructure_assets.return_value = result_path

        await dataset_tools["restructure_assets"](
            dataset_rid="DS-001",
            asset_table="Image",
            output_dir=str(tmp_path / "out"),
            materialize=False,
        )

        mock_dataset.to_bag.assert_called_once_with(materialize=False)

    @pytest.mark.asyncio
    async def test_restructure_exception(self, dataset_tools, mock_ml):
        """When restructure_assets raises, return an error."""
        mock_ml.lookup_dataset.side_effect = RuntimeError("Download failed")

        result = await dataset_tools["restructure_assets"](
            dataset_rid="DS-001",
            asset_table="Image",
            output_dir="/tmp/out",
        )

        assert_error(result, expected_message="Download failed")

    @pytest.mark.asyncio
    async def test_restructure_no_connection(self, dataset_tools_disconnected):
        """When not connected, return an error."""
        result = await dataset_tools_disconnected["restructure_assets"](
            dataset_rid="DS-001",
            asset_table="Image",
            output_dir="/tmp/out",
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
        from deriva_ml_mcp.tools.dataset import _serialize_dataset

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
        from deriva_ml_mcp.tools.dataset import _serialize_dataset

        mock_ds = _make_mock_dataset(current_version=None)

        result = _serialize_dataset(mock_ds)

        assert result["current_version"] is None

    def test_serialize_empty_types(self):
        """Empty dataset_types serializes as empty list."""
        from deriva_ml_mcp.tools.dataset import _serialize_dataset

        mock_ds = _make_mock_dataset(dataset_types=[])

        result = _serialize_dataset(mock_ds)

        assert result["dataset_types"] == []


# =============================================================================
# TestToolRegistration
# =============================================================================


class TestToolRegistration:
    """Verify that all expected dataset tools are registered."""

    def test_all_tools_registered(self, dataset_tools):
        """All 21 dataset tools should be registered."""
        expected_tools = [
            "create_dataset",
            "get_dataset_spec",
            "list_dataset_members",
            "add_dataset_members",
            "delete_dataset_members",
            "increment_dataset_version",
            "delete_dataset",
            "set_dataset_description",
            "add_dataset_type",
            "remove_dataset_type",
            "add_dataset_element_type",
            "add_dataset_child",
            "list_dataset_children",
            "list_dataset_parents",
            "list_dataset_executions",
            "download_dataset",
            "denormalize_dataset",
            "create_dataset_type_term",
            "delete_dataset_type_term",
            "restructure_assets",
            "split_dataset",
        ]
        for tool_name in expected_tools:
            assert tool_name in dataset_tools, f"Tool '{tool_name}' not registered"

    def test_no_extra_tools(self, dataset_tools):
        """Only the expected tools should be registered."""
        expected_count = 21
        assert len(dataset_tools) == expected_count, (
            f"Expected {expected_count} tools, got {len(dataset_tools)}: "
            f"{sorted(dataset_tools.keys())}"
        )
