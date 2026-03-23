"""Unit tests for execution management tools.

Tests cover execution tools (async):
    - create_execution
    - start_execution
    - stop_execution
    - update_execution_status
    - set_execution_description
    - restore_execution
    - create_execution_dataset
    - add_nested_execution
    - list_nested_executions
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    assert_error,
    assert_success,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_execution(
    execution_rid: str = "EXE-TOOL-1",
    workflow_rid: str = "WF-1",
    status: str = "running",
    datasets: list | None = None,
    dataset_rids: list | None = None,
    working_dir: str = "/tmp/exec",
    uploaded_assets: object | None = None,
    description: str = "Test execution",
) -> MagicMock:
    """Create a mock execution object with standard attributes."""
    mock_execution = MagicMock()
    mock_execution.execution_rid = execution_rid
    mock_execution.workflow_rid = workflow_rid
    mock_execution.status = MagicMock(value=status)
    mock_execution.datasets = datasets if datasets is not None else []
    mock_execution.dataset_rids = dataset_rids if dataset_rids is not None else []
    mock_execution.working_dir = Path(working_dir)
    mock_execution.uploaded_assets = uploaded_assets
    mock_execution.description = description
    return mock_execution


def _set_active_execution(mock_conn_manager, execution):
    """Set the active tool execution on the mock connection manager."""
    conn_info = mock_conn_manager.get_active_connection_info_or_raise()
    conn_info.active_tool_execution = execution


def _clear_active_execution(mock_conn_manager):
    """Clear the active tool execution on the mock connection manager."""
    conn_info = mock_conn_manager.get_active_connection_info_or_raise()
    conn_info.active_tool_execution = None


# =============================================================================
# TestCreateExecution
# =============================================================================


class TestCreateExecution:
    """Tests for the create_execution tool."""

    @pytest.mark.asyncio
    @patch("deriva_ml.execution.execution_configuration.ExecutionConfiguration")
    @patch("deriva_ml.dataset.aux_classes.DatasetSpec")
    async def test_create_basic(
        self, mock_ds_cls, mock_ec_cls, execution_tools, mock_ml, mock_conn_manager
    ):
        """create_execution creates an execution and returns its details."""
        mock_workflow = MagicMock()
        mock_ml.create_workflow.return_value = mock_workflow

        mock_execution = _make_mock_execution()
        mock_ml.create_execution.return_value = mock_execution

        result = await execution_tools["create_execution"](
            workflow_name="CIFAR Training",
            workflow_type="Training",
            description="Train ResNet on CIFAR-10",
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["execution_rid"] == "EXE-TOOL-1"
        assert data["workflow_rid"] == "WF-1"
        assert data["description"] == "Train ResNet on CIFAR-10"
        assert data["dataset_count"] == 0
        assert data["asset_count"] == 0
        assert data["dry_run"] is False

        mock_ml.create_workflow.assert_called_once_with(
            name="CIFAR Training",
            workflow_type="Training",
            description="Train ResNet on CIFAR-10",
        )

    @pytest.mark.asyncio
    @patch("deriva_ml.execution.execution_configuration.ExecutionConfiguration")
    @patch("deriva_ml.dataset.aux_classes.DatasetSpec")
    async def test_create_with_datasets_and_assets(
        self, mock_ds_cls, mock_ec_cls, execution_tools, mock_ml, mock_conn_manager
    ):
        """create_execution with dataset and asset RIDs reports correct counts."""
        mock_workflow = MagicMock()
        mock_ml.create_workflow.return_value = mock_workflow
        mock_execution = _make_mock_execution()
        mock_ml.create_execution.return_value = mock_execution

        result = await execution_tools["create_execution"](
            workflow_name="Pipeline",
            workflow_type="Training",
            description="Full pipeline",
            dataset_rids=["1-AAA", "1-BBB"],
            asset_rids=["2-CCC"],
        )

        data = assert_success(result)
        assert data["dataset_count"] == 2
        assert data["asset_count"] == 1

    @pytest.mark.asyncio
    @patch("deriva_ml.execution.execution_configuration.ExecutionConfiguration")
    @patch("deriva_ml.dataset.aux_classes.DatasetSpec")
    async def test_create_dry_run(
        self, mock_ds_cls, mock_ec_cls, execution_tools, mock_ml, mock_conn_manager
    ):
        """create_execution with dry_run=True passes the flag through."""
        mock_workflow = MagicMock()
        mock_ml.create_workflow.return_value = mock_workflow
        mock_execution = _make_mock_execution()
        mock_ml.create_execution.return_value = mock_execution

        result = await execution_tools["create_execution"](
            workflow_name="Test Run",
            workflow_type="Training",
            description="Debug data loading",
            dry_run=True,
        )

        data = assert_success(result)
        assert data["dry_run"] is True
        mock_ml.create_execution.assert_called_once()
        call_kwargs = mock_ml.create_execution.call_args
        assert call_kwargs.kwargs["dry_run"] is True

    @pytest.mark.asyncio
    @patch("deriva_ml.execution.execution_configuration.ExecutionConfiguration")
    @patch("deriva_ml.dataset.aux_classes.DatasetSpec")
    async def test_create_sets_active_execution(
        self, mock_ds_cls, mock_ec_cls, execution_tools, mock_ml, mock_conn_manager
    ):
        """create_execution stores the execution as the active tool execution."""
        mock_workflow = MagicMock()
        mock_ml.create_workflow.return_value = mock_workflow
        mock_execution = _make_mock_execution(execution_rid="EXE-NEW")
        mock_ml.create_execution.return_value = mock_execution

        await execution_tools["create_execution"](
            workflow_name="Test",
            workflow_type="Training",
        )

        conn_info = mock_conn_manager.get_active_connection_info_or_raise()
        assert conn_info.active_tool_execution == mock_execution

    @pytest.mark.asyncio
    async def test_create_exception(self, execution_tools, mock_ml):
        """create_execution returns error when workflow creation fails."""
        mock_ml.create_workflow.side_effect = Exception("Workflow type not found")

        result = await execution_tools["create_execution"](
            workflow_name="Bad",
            workflow_type="NonExistent",
        )

        assert_error(result, "Workflow type not found")

    @pytest.mark.asyncio
    async def test_create_no_connection(self, execution_tools_disconnected):
        """create_execution returns error when not connected."""
        result = await execution_tools_disconnected["create_execution"](
            workflow_name="Test",
            workflow_type="Training",
        )

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestStartExecution
# =============================================================================


class TestStartExecution:
    """Tests for the start_execution tool."""

    @pytest.mark.asyncio
    async def test_start_success(self, execution_tools, mock_conn_manager):
        """start_execution starts timing and returns the execution RID."""
        mock_execution = _make_mock_execution(execution_rid="EXE-START")
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["start_execution"]()

        data = assert_success(result)
        assert data["status"] == "started"
        assert data["execution_rid"] == "EXE-START"
        mock_execution.execution_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_no_active_execution(self, execution_tools, mock_conn_manager):
        """start_execution returns error when no execution is active."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["start_execution"]()

        data = assert_error(result, "No active execution")

    @pytest.mark.asyncio
    async def test_start_exception(self, execution_tools, mock_conn_manager):
        """start_execution returns error when execution_start raises."""
        mock_execution = _make_mock_execution()
        mock_execution.execution_start.side_effect = Exception("Already started")
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["start_execution"]()

        assert_error(result, "Already started")

    @pytest.mark.asyncio
    async def test_start_no_connection(self, execution_tools_disconnected):
        """start_execution returns error when not connected to a catalog."""
        result = await execution_tools_disconnected["start_execution"]()

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestStopExecution
# =============================================================================


class TestStopExecution:
    """Tests for the stop_execution tool."""

    @pytest.mark.asyncio
    async def test_stop_success(self, execution_tools, mock_conn_manager):
        """stop_execution stops timing and returns completion status."""
        mock_execution = _make_mock_execution(execution_rid="EXE-STOP")
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["stop_execution"]()

        data = assert_success(result)
        assert data["status"] == "completed"
        assert data["execution_rid"] == "EXE-STOP"
        mock_execution.execution_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_no_active_execution(self, execution_tools, mock_conn_manager):
        """stop_execution returns error when no execution is active."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["stop_execution"]()

        data = assert_error(result, "No active execution")

    @pytest.mark.asyncio
    async def test_stop_exception(self, execution_tools, mock_conn_manager):
        """stop_execution returns error when execution_stop raises."""
        mock_execution = _make_mock_execution()
        mock_execution.execution_stop.side_effect = Exception("Not started")
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["stop_execution"]()

        assert_error(result, "Not started")

    @pytest.mark.asyncio
    async def test_stop_no_connection(self, execution_tools_disconnected):
        """stop_execution returns error when not connected."""
        result = await execution_tools_disconnected["stop_execution"]()

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestUpdateExecutionStatus
# =============================================================================


class TestUpdateExecutionStatus:
    """Tests for the update_execution_status tool."""

    @pytest.mark.asyncio
    async def test_update_status_running(self, execution_tools, mock_conn_manager):
        """update_execution_status with 'running' calls update_status correctly."""
        mock_execution = _make_mock_execution(execution_rid="EXE-STATUS")
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["update_execution_status"](
            status="running",
            message="Processing batch 5/10",
        )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["execution_rid"] == "EXE-STATUS"
        assert data["new_status"] == "running"
        assert data["message"] == "Processing batch 5/10"
        mock_execution.update_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_failed(self, execution_tools, mock_conn_manager):
        """update_execution_status with 'failed' status works correctly."""
        mock_execution = _make_mock_execution()
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["update_execution_status"](
            status="failed",
            message="OOM error during training",
        )

        data = assert_success(result)
        assert data["new_status"] == "failed"
        assert data["message"] == "OOM error during training"

    @pytest.mark.asyncio
    async def test_update_status_completed(self, execution_tools, mock_conn_manager):
        """update_execution_status with 'completed' status works correctly."""
        mock_execution = _make_mock_execution()
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["update_execution_status"](
            status="completed",
            message="Training finished with 95% accuracy",
        )

        data = assert_success(result)
        assert data["new_status"] == "completed"

    @pytest.mark.asyncio
    async def test_update_status_pending(self, execution_tools, mock_conn_manager):
        """update_execution_status with 'pending' status works correctly."""
        mock_execution = _make_mock_execution()
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["update_execution_status"](
            status="pending",
            message="Waiting for resources",
        )

        data = assert_success(result)
        assert data["new_status"] == "pending"

    @pytest.mark.asyncio
    async def test_update_status_unknown_defaults_to_running(
        self, execution_tools, mock_conn_manager
    ):
        """update_execution_status with unknown status defaults to running enum."""
        mock_execution = _make_mock_execution()
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["update_execution_status"](
            status="unknown_status",
            message="Some message",
        )

        data = assert_success(result)
        # The tool still returns what was passed
        assert data["new_status"] == "unknown_status"
        # But internally it maps to Status.running
        mock_execution.update_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_case_insensitive(
        self, execution_tools, mock_conn_manager
    ):
        """update_execution_status handles case-insensitive status values."""
        mock_execution = _make_mock_execution()
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["update_execution_status"](
            status="COMPLETED",
            message="Done",
        )

        data = assert_success(result)
        assert data["status"] == "updated"

    @pytest.mark.asyncio
    async def test_update_status_no_active_execution(
        self, execution_tools, mock_conn_manager
    ):
        """update_execution_status returns error when no execution is active."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["update_execution_status"](
            status="running",
            message="Some update",
        )

        assert_error(result, "No active execution")

    @pytest.mark.asyncio
    async def test_update_status_exception(self, execution_tools, mock_conn_manager):
        """update_execution_status returns error when update_status raises."""
        mock_execution = _make_mock_execution()
        mock_execution.update_status.side_effect = Exception("Server error")
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["update_execution_status"](
            status="running",
            message="Some update",
        )

        assert_error(result, "Server error")

    @pytest.mark.asyncio
    async def test_update_status_no_connection(self, execution_tools_disconnected):
        """update_execution_status returns error when not connected."""
        result = await execution_tools_disconnected["update_execution_status"](
            status="running",
            message="Update",
        )

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestSetExecutionDescription
# =============================================================================


class TestSetExecutionDescription:
    """Tests for the set_execution_description tool."""

    @pytest.mark.asyncio
    async def test_set_description_success(self, execution_tools, mock_ml):
        """set_execution_description updates the description and returns confirmation."""
        mock_record = MagicMock()
        mock_ml.lookup_execution.return_value = mock_record

        result = await execution_tools["set_execution_description"](
            execution_rid="2-XYZ",
            description="Training run with lr=0.001, achieved 95% accuracy",
        )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["execution_rid"] == "2-XYZ"
        assert data["description"] == "Training run with lr=0.001, achieved 95% accuracy"
        mock_ml.lookup_execution.assert_called_once_with("2-XYZ")
        assert mock_record.description == "Training run with lr=0.001, achieved 95% accuracy"

    @pytest.mark.asyncio
    async def test_set_description_empty(self, execution_tools, mock_ml):
        """set_execution_description works with empty description."""
        mock_record = MagicMock()
        mock_ml.lookup_execution.return_value = mock_record

        result = await execution_tools["set_execution_description"](
            execution_rid="2-XYZ",
            description="",
        )

        data = assert_success(result)
        assert data["description"] == ""

    @pytest.mark.asyncio
    async def test_set_description_not_found(self, execution_tools, mock_ml):
        """set_execution_description returns error when execution not found."""
        mock_ml.lookup_execution.side_effect = Exception("Execution not found: NOPE")

        result = await execution_tools["set_execution_description"](
            execution_rid="NOPE",
            description="Test",
        )

        assert_error(result, "Execution not found")

    @pytest.mark.asyncio
    async def test_set_description_no_connection(self, execution_tools_disconnected):
        """set_execution_description returns error when not connected."""
        result = await execution_tools_disconnected["set_execution_description"](
            execution_rid="2-XYZ",
            description="Test",
        )

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestRestoreExecution
# =============================================================================


class TestRestoreExecution:
    """Tests for the restore_execution tool."""

    @pytest.mark.asyncio
    async def test_restore_success(
        self, execution_tools, mock_ml, mock_conn_manager
    ):
        """restore_execution restores and sets the active execution."""
        mock_execution = _make_mock_execution(
            execution_rid="EXE-RESTORED",
            workflow_rid="WF-RESTORED",
            datasets=[MagicMock(), MagicMock(), MagicMock()],
        )
        mock_ml.restore_execution.return_value = mock_execution

        result = await execution_tools["restore_execution"](execution_rid="EXE-RESTORED")

        data = assert_success(result)
        assert data["status"] == "restored"
        assert data["execution_rid"] == "EXE-RESTORED"
        assert data["workflow_rid"] == "WF-RESTORED"
        assert data["dataset_count"] == 3
        mock_ml.restore_execution.assert_called_once_with("EXE-RESTORED")

        # Verify it was set as active
        conn_info = mock_conn_manager.get_active_connection_info_or_raise()
        assert conn_info.active_tool_execution == mock_execution

    @pytest.mark.asyncio
    async def test_restore_not_found(self, execution_tools, mock_ml):
        """restore_execution returns error when execution doesn't exist."""
        mock_ml.restore_execution.side_effect = Exception(
            "Execution NOPE not found"
        )

        result = await execution_tools["restore_execution"](execution_rid="NOPE")

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_restore_no_connection(self, execution_tools_disconnected):
        """restore_execution returns error when not connected."""
        result = await execution_tools_disconnected["restore_execution"](
            execution_rid="EXE-1"
        )

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestCreateExecutionDataset
# =============================================================================


class TestCreateExecutionDataset:
    """Tests for the create_execution_dataset tool."""

    @pytest.mark.asyncio
    async def test_create_dataset_success(self, execution_tools, mock_conn_manager):
        """create_execution_dataset creates a dataset and returns its RID."""
        mock_execution = _make_mock_execution(execution_rid="EXE-DS")
        mock_dataset = MagicMock()
        mock_dataset.dataset_rid = "DS-NEW"
        mock_execution.create_dataset.return_value = mock_dataset
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["create_execution_dataset"](
            description="Augmented training data",
            dataset_types=["Training", "Augmented"],
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["dataset_rid"] == "DS-NEW"
        assert data["execution_rid"] == "EXE-DS"
        assert data["description"] == "Augmented training data"
        assert data["dataset_types"] == ["Training", "Augmented"]
        mock_execution.create_dataset.assert_called_once_with(
            description="Augmented training data",
            dataset_types=["Training", "Augmented"],
        )

    @pytest.mark.asyncio
    async def test_create_dataset_defaults(self, execution_tools, mock_conn_manager):
        """create_execution_dataset works with default arguments."""
        mock_execution = _make_mock_execution()
        mock_dataset = MagicMock()
        mock_dataset.dataset_rid = "DS-DEFAULT"
        mock_execution.create_dataset.return_value = mock_dataset
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["create_execution_dataset"]()

        data = assert_success(result)
        assert data["description"] == ""
        assert data["dataset_types"] == []
        mock_execution.create_dataset.assert_called_once_with(
            description="",
            dataset_types=[],
        )

    @pytest.mark.asyncio
    async def test_create_dataset_no_active_execution(
        self, execution_tools, mock_conn_manager
    ):
        """create_execution_dataset returns error when no execution is active."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["create_execution_dataset"]()

        assert_error(result, "No active execution")

    @pytest.mark.asyncio
    async def test_create_dataset_exception(self, execution_tools, mock_conn_manager):
        """create_execution_dataset returns error when creation fails."""
        mock_execution = _make_mock_execution()
        mock_execution.create_dataset.side_effect = Exception("Permission denied")
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["create_execution_dataset"](
            description="Test",
        )

        assert_error(result, "Permission denied")

    @pytest.mark.asyncio
    async def test_create_dataset_no_connection(self, execution_tools_disconnected):
        """create_execution_dataset returns error when not connected."""
        result = await execution_tools_disconnected["create_execution_dataset"]()

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestAddNestedExecution
# =============================================================================


class TestAddNestedExecution:
    """Tests for the add_nested_execution tool."""

    @pytest.mark.asyncio
    async def test_add_nested_success(self, execution_tools, mock_ml):
        """add_nested_execution creates a parent-child relationship."""
        mock_parent = MagicMock()
        mock_ml.lookup_execution.return_value = mock_parent

        result = await execution_tools["add_nested_execution"](
            parent_execution_rid="1-PARENT",
            child_execution_rid="1-CHILD",
            sequence=0,
        )

        data = assert_success(result)
        assert data["status"] == "added"
        assert data["parent_rid"] == "1-PARENT"
        assert data["child_rid"] == "1-CHILD"
        assert data["sequence"] == 0
        mock_ml.lookup_execution.assert_called_once_with("1-PARENT")
        mock_parent.add_nested_execution.assert_called_once_with(
            "1-CHILD", sequence=0
        )

    @pytest.mark.asyncio
    async def test_add_nested_no_sequence(self, execution_tools, mock_ml):
        """add_nested_execution works without sequence (parallel executions)."""
        mock_parent = MagicMock()
        mock_ml.lookup_execution.return_value = mock_parent

        result = await execution_tools["add_nested_execution"](
            parent_execution_rid="1-PARENT",
            child_execution_rid="1-CHILD",
        )

        data = assert_success(result)
        assert data["sequence"] is None
        mock_parent.add_nested_execution.assert_called_once_with(
            "1-CHILD", sequence=None
        )

    @pytest.mark.asyncio
    async def test_add_nested_parent_not_found(self, execution_tools, mock_ml):
        """add_nested_execution returns error when parent doesn't exist."""
        mock_ml.lookup_execution.side_effect = Exception("Execution not found")

        result = await execution_tools["add_nested_execution"](
            parent_execution_rid="NOPE",
            child_execution_rid="1-CHILD",
        )

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_add_nested_exception(self, execution_tools, mock_ml):
        """add_nested_execution returns error when nesting fails."""
        mock_parent = MagicMock()
        mock_parent.add_nested_execution.side_effect = Exception("Cycle detected")
        mock_ml.lookup_execution.return_value = mock_parent

        result = await execution_tools["add_nested_execution"](
            parent_execution_rid="1-PARENT",
            child_execution_rid="1-PARENT",
        )

        assert_error(result, "Cycle detected")

    @pytest.mark.asyncio
    async def test_add_nested_no_connection(self, execution_tools_disconnected):
        """add_nested_execution returns error when not connected."""
        result = await execution_tools_disconnected["add_nested_execution"](
            parent_execution_rid="1-P",
            child_execution_rid="1-C",
        )

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestListNestedExecutions
# =============================================================================


class TestListNestedExecutions:
    """Tests for the list_nested_executions tool."""

    @pytest.mark.asyncio
    async def test_list_nested_success(self, execution_tools, mock_ml):
        """list_nested_executions returns child executions."""
        mock_parent = MagicMock()
        child1 = _make_mock_execution(
            execution_rid="CHILD-1",
            workflow_rid="WF-1",
            status="completed",
            description="First fold",
        )
        child2 = _make_mock_execution(
            execution_rid="CHILD-2",
            workflow_rid="WF-1",
            status="running",
            description="Second fold",
        )
        mock_parent.list_nested_executions.return_value = [child1, child2]
        mock_ml.lookup_execution.return_value = mock_parent

        result = await execution_tools["list_nested_executions"](
            execution_rid="1-PARENT",
        )

        data = assert_success(result)
        assert data["parent_rid"] == "1-PARENT"
        assert data["recurse"] is False
        assert data["count"] == 2
        assert len(data["children"]) == 2
        assert data["children"][0]["execution_rid"] == "CHILD-1"
        assert data["children"][0]["status"] == "completed"
        assert data["children"][0]["description"] == "First fold"
        assert data["children"][1]["execution_rid"] == "CHILD-2"
        assert data["children"][1]["status"] == "running"

    @pytest.mark.asyncio
    async def test_list_nested_recurse(self, execution_tools, mock_ml):
        """list_nested_executions passes recurse=True."""
        mock_parent = MagicMock()
        mock_parent.list_nested_executions.return_value = []
        mock_ml.lookup_execution.return_value = mock_parent

        result = await execution_tools["list_nested_executions"](
            execution_rid="1-PARENT",
            recurse=True,
        )

        data = assert_success(result)
        assert data["recurse"] is True
        mock_parent.list_nested_executions.assert_called_once_with(recurse=True)

    @pytest.mark.asyncio
    async def test_list_nested_empty(self, execution_tools, mock_ml):
        """list_nested_executions returns empty list for leaf executions."""
        mock_parent = MagicMock()
        mock_parent.list_nested_executions.return_value = []
        mock_ml.lookup_execution.return_value = mock_parent

        result = await execution_tools["list_nested_executions"](
            execution_rid="1-LEAF",
        )

        data = assert_success(result)
        assert data["count"] == 0
        assert data["children"] == []

    @pytest.mark.asyncio
    async def test_list_nested_status_plain_string(self, execution_tools, mock_ml):
        """list_nested_executions handles status without .value attribute."""
        mock_parent = MagicMock()
        child = _make_mock_execution()
        child.status = "completed"  # plain string instead of enum
        mock_parent.list_nested_executions.return_value = [child]
        mock_ml.lookup_execution.return_value = mock_parent

        result = await execution_tools["list_nested_executions"](
            execution_rid="1-PARENT",
        )

        data = assert_success(result)
        assert data["children"][0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_nested_not_found(self, execution_tools, mock_ml):
        """list_nested_executions returns error when execution not found."""
        mock_ml.lookup_execution.side_effect = Exception("Execution not found")

        result = await execution_tools["list_nested_executions"](
            execution_rid="NOPE",
        )

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_list_nested_no_connection(self, execution_tools_disconnected):
        """list_nested_executions returns error when not connected."""
        result = await execution_tools_disconnected["list_nested_executions"](
            execution_rid="1-P",
        )

        assert_error(result, "No active catalog connection")
