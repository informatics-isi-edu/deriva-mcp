"""Unit tests for execution management and storage tools.

Tests cover all 17 execution tools and 2 storage tools:

Execution tools (async):
    - create_execution
    - start_execution
    - stop_execution
    - update_execution_status
    - set_execution_description
    - get_execution_info
    - restore_execution
    - asset_file_path
    - upload_execution_outputs
    - download_asset
    - create_execution_dataset
    - download_execution_dataset
    - get_execution_working_dir
    - add_nested_execution
    - list_nested_executions
    - list_parent_executions

Storage tools (sync):
    - clear_cache
    - clean_execution_dirs
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    _create_tool_capture,
    assert_error,
    assert_success,
    parse_json_result,
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
# Fixtures
# =============================================================================


@pytest.fixture
def storage_tools_disconnected(disconnected_conn_manager):
    """Capture storage tools with no connection."""
    from deriva_ml_mcp.tools.execution import register_storage_tools

    mcp, tools = _create_tool_capture()
    register_storage_tools(mcp, disconnected_conn_manager)
    return tools


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
# TestGetExecutionInfo
# =============================================================================


class TestGetExecutionInfo:
    """Tests for the get_execution_info tool."""

    @pytest.mark.asyncio
    async def test_get_info_success(self, execution_tools, mock_conn_manager):
        """get_execution_info returns detailed execution information."""
        mock_execution = _make_mock_execution(
            execution_rid="EXE-INFO",
            workflow_rid="WF-INFO",
            status="running",
            dataset_rids=["1-AAA", "1-BBB"],
            datasets=[MagicMock(), MagicMock()],
            working_dir="/tmp/exec_info",
            uploaded_assets=None,  # pending uploads
        )
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["get_execution_info"]()

        data = parse_json_result(result)
        assert data["execution_rid"] == "EXE-INFO"
        assert data["workflow_rid"] == "WF-INFO"
        assert data["status"] == "running"
        assert data["dataset_rids"] == ["1-AAA", "1-BBB"]
        assert data["dataset_count"] == 2
        assert data["working_dir"] == "/tmp/exec_info"
        assert data["upload_pending"] is True
        assert data["upload_reminder"] is not None

    @pytest.mark.asyncio
    async def test_get_info_uploads_complete(self, execution_tools, mock_conn_manager):
        """get_execution_info shows upload_pending=False when assets are uploaded."""
        mock_execution = _make_mock_execution(
            uploaded_assets={"Model": ["/path/model.pt"]},
        )
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["get_execution_info"]()

        data = parse_json_result(result)
        assert data["upload_pending"] is False
        assert data["upload_reminder"] is None

    @pytest.mark.asyncio
    async def test_get_info_no_active_execution(
        self, execution_tools, mock_conn_manager
    ):
        """get_execution_info returns no_active_execution when none is set."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["get_execution_info"]()

        data = parse_json_result(result)
        assert data["status"] == "no_active_execution"
        assert "create_execution" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_get_info_status_without_value(
        self, execution_tools, mock_conn_manager
    ):
        """get_execution_info handles status as a plain string (no .value)."""
        mock_execution = _make_mock_execution()
        mock_execution.status = "completed"  # plain string, no .value attribute
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["get_execution_info"]()

        data = parse_json_result(result)
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_info_exception(self, execution_tools, mock_conn_manager):
        """get_execution_info returns error on unexpected exception."""
        mock_execution = _make_mock_execution()
        # Make .datasets raise
        type(mock_execution).datasets = property(
            lambda self: (_ for _ in ()).throw(Exception("Internal error"))
        )
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["get_execution_info"]()

        assert_error(result, "Internal error")

    @pytest.mark.asyncio
    async def test_get_info_no_connection(self, execution_tools_disconnected):
        """get_execution_info returns error when not connected."""
        result = await execution_tools_disconnected["get_execution_info"]()

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
# TestAssetFilePath
# =============================================================================


class TestAssetFilePath:
    """Tests for the asset_file_path tool."""

    @pytest.mark.asyncio
    async def test_register_asset_success(self, execution_tools, mock_conn_manager):
        """asset_file_path registers a file and returns its staged path."""
        mock_execution = _make_mock_execution(execution_rid="EXE-ASSET")
        mock_asset_path = MagicMock()
        mock_asset_path.__str__ = lambda self: "/tmp/exec/Model/model.pt"
        mock_asset_path.file_name = "model.pt"
        mock_asset_path.asset_types = ["Model"]
        mock_execution.asset_file_path.return_value = mock_asset_path
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["asset_file_path"](
            asset_name="Model",
            file_name="/tmp/trained_model.pt",
        )

        data = assert_success(result)
        assert data["status"] == "registered"
        assert data["execution_rid"] == "EXE-ASSET"
        assert data["asset_name"] == "Model"
        assert data["file_path"] == "/tmp/exec/Model/model.pt"
        assert data["file_name"] == "model.pt"
        assert data["asset_types"] == ["Model"]
        assert "upload_execution_outputs" in data["note"].lower()

    @pytest.mark.asyncio
    async def test_register_with_asset_types(self, execution_tools, mock_conn_manager):
        """asset_file_path passes asset_types to the execution."""
        mock_execution = _make_mock_execution()
        mock_asset_path = MagicMock()
        mock_asset_path.__str__ = lambda self: "/tmp/exec/Image/photo.jpg"
        mock_asset_path.file_name = "photo.jpg"
        mock_asset_path.asset_types = ["Training", "Augmented"]
        mock_execution.asset_file_path.return_value = mock_asset_path
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["asset_file_path"](
            asset_name="Image",
            file_name="photo.jpg",
            asset_types=["Training", "Augmented"],
        )

        data = assert_success(result)
        assert data["asset_types"] == ["Training", "Augmented"]
        mock_execution.asset_file_path.assert_called_once_with(
            asset_name="Image",
            file_name="photo.jpg",
            asset_types=["Training", "Augmented"],
            copy_file=False,
            rename_file=None,
        )

    @pytest.mark.asyncio
    async def test_register_with_copy_and_rename(
        self, execution_tools, mock_conn_manager
    ):
        """asset_file_path passes copy_file and rename_file arguments."""
        mock_execution = _make_mock_execution()
        mock_asset_path = MagicMock()
        mock_asset_path.__str__ = lambda self: "/tmp/exec/Model/best_model.pt"
        mock_asset_path.file_name = "best_model.pt"
        mock_asset_path.asset_types = ["Model"]
        mock_execution.asset_file_path.return_value = mock_asset_path
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["asset_file_path"](
            asset_name="Model",
            file_name="/tmp/model.pt",
            copy_file=True,
            rename_file="best_model.pt",
        )

        data = assert_success(result)
        mock_execution.asset_file_path.assert_called_once_with(
            asset_name="Model",
            file_name="/tmp/model.pt",
            asset_types=None,
            copy_file=True,
            rename_file="best_model.pt",
        )

    @pytest.mark.asyncio
    async def test_register_no_active_execution(
        self, execution_tools, mock_conn_manager
    ):
        """asset_file_path returns error when no execution is active."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["asset_file_path"](
            asset_name="Model",
            file_name="model.pt",
        )

        assert_error(result, "No active execution")

    @pytest.mark.asyncio
    async def test_register_exception(self, execution_tools, mock_conn_manager):
        """asset_file_path returns error when execution raises."""
        mock_execution = _make_mock_execution()
        mock_execution.asset_file_path.side_effect = Exception(
            "Asset table 'BadTable' not found"
        )
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["asset_file_path"](
            asset_name="BadTable",
            file_name="file.dat",
        )

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_register_no_connection(self, execution_tools_disconnected):
        """asset_file_path returns error when not connected."""
        result = await execution_tools_disconnected["asset_file_path"](
            asset_name="Model",
            file_name="model.pt",
        )

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestUploadExecutionOutputs
# =============================================================================


class TestUploadExecutionOutputs:
    """Tests for the upload_execution_outputs tool."""

    @pytest.mark.asyncio
    async def test_upload_success(self, execution_tools, mock_conn_manager):
        """upload_execution_outputs uploads assets and returns summary."""
        mock_execution = _make_mock_execution(execution_rid="EXE-UPLOAD")
        mock_execution.upload_execution_outputs.return_value = {
            "Model": ["/tmp/model.pt"],
            "Image": ["/tmp/img1.jpg", "/tmp/img2.jpg"],
        }
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["upload_execution_outputs"]()

        data = assert_success(result)
        assert data["status"] == "uploaded"
        assert data["execution_rid"] == "EXE-UPLOAD"
        assert data["assets_uploaded"]["Model"] == 1
        assert data["assets_uploaded"]["Image"] == 2
        mock_execution.upload_execution_outputs.assert_called_once_with(
            clean_folder=True
        )

    @pytest.mark.asyncio
    async def test_upload_no_clean(self, execution_tools, mock_conn_manager):
        """upload_execution_outputs passes clean_folder=False."""
        mock_execution = _make_mock_execution()
        mock_execution.upload_execution_outputs.return_value = {}
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["upload_execution_outputs"](clean_folder=False)

        data = assert_success(result)
        mock_execution.upload_execution_outputs.assert_called_once_with(
            clean_folder=False
        )

    @pytest.mark.asyncio
    async def test_upload_empty_results(self, execution_tools, mock_conn_manager):
        """upload_execution_outputs succeeds with no assets to upload."""
        mock_execution = _make_mock_execution()
        mock_execution.upload_execution_outputs.return_value = {}
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["upload_execution_outputs"]()

        data = assert_success(result)
        assert data["assets_uploaded"] == {}

    @pytest.mark.asyncio
    async def test_upload_no_active_execution(
        self, execution_tools, mock_conn_manager
    ):
        """upload_execution_outputs returns error when no execution is active."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["upload_execution_outputs"]()

        assert_error(result, "No active execution")

    @pytest.mark.asyncio
    async def test_upload_exception(self, execution_tools, mock_conn_manager):
        """upload_execution_outputs returns error when upload fails."""
        mock_execution = _make_mock_execution()
        mock_execution.upload_execution_outputs.side_effect = Exception(
            "Upload failed: network error"
        )
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["upload_execution_outputs"]()

        assert_error(result, "Upload failed")

    @pytest.mark.asyncio
    async def test_upload_no_connection(self, execution_tools_disconnected):
        """upload_execution_outputs returns error when not connected."""
        result = await execution_tools_disconnected["upload_execution_outputs"]()

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestDownloadAsset
# =============================================================================


class TestDownloadAsset:
    """Tests for the download_asset tool."""

    @pytest.mark.asyncio
    async def test_download_to_working_dir(self, execution_tools, mock_conn_manager):
        """download_asset downloads to execution working dir by default."""
        mock_execution = _make_mock_execution(
            execution_rid="EXE-DL", working_dir="/tmp/exec_dl"
        )
        mock_asset_path = MagicMock()
        mock_asset_path.asset_path = Path("/tmp/exec_dl/model.pt")
        mock_asset_path.file_name = "model.pt"
        mock_asset_path.asset_table = "Model"
        mock_asset_path.asset_types = ["Model"]
        mock_execution.download_asset.return_value = mock_asset_path
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["download_asset"](asset_rid="1-ABC")

        data = assert_success(result)
        assert data["status"] == "downloaded"
        assert data["execution_rid"] == "EXE-DL"
        assert data["asset_rid"] == "1-ABC"
        assert data["file_path"] == "/tmp/exec_dl/model.pt"
        assert data["filename"] == "model.pt"
        assert data["asset_table"] == "Model"
        assert data["asset_types"] == ["Model"]
        mock_execution.download_asset.assert_called_once_with(
            asset_rid="1-ABC",
            dest_dir=Path("/tmp/exec_dl"),
        )

    @pytest.mark.asyncio
    async def test_download_to_custom_dir(self, execution_tools, mock_conn_manager):
        """download_asset downloads to a specified directory."""
        mock_execution = _make_mock_execution()
        mock_asset_path = MagicMock()
        mock_asset_path.asset_path = Path("/custom/dir/image.jpg")
        mock_asset_path.file_name = "image.jpg"
        mock_asset_path.asset_table = "Image"
        mock_asset_path.asset_types = ["Training"]
        mock_execution.download_asset.return_value = mock_asset_path
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["download_asset"](
            asset_rid="2-DEF",
            dest_dir="/custom/dir",
        )

        data = assert_success(result)
        assert data["file_path"] == "/custom/dir/image.jpg"
        mock_execution.download_asset.assert_called_once_with(
            asset_rid="2-DEF",
            dest_dir=Path("/custom/dir"),
        )

    @pytest.mark.asyncio
    async def test_download_no_active_execution(
        self, execution_tools, mock_conn_manager
    ):
        """download_asset returns error when no execution is active."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["download_asset"](asset_rid="1-ABC")

        assert_error(result, "No active execution")

    @pytest.mark.asyncio
    async def test_download_exception(self, execution_tools, mock_conn_manager):
        """download_asset returns error when download fails."""
        mock_execution = _make_mock_execution()
        mock_execution.download_asset.side_effect = Exception("Asset 1-XYZ not found")
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["download_asset"](asset_rid="1-XYZ")

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_download_no_connection(self, execution_tools_disconnected):
        """download_asset returns error when not connected."""
        result = await execution_tools_disconnected["download_asset"](
            asset_rid="1-ABC"
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
# TestDownloadExecutionDataset
# =============================================================================


class TestDownloadExecutionDataset:
    """Tests for the download_execution_dataset tool."""

    @pytest.mark.asyncio
    @patch("deriva_ml.dataset.aux_classes.DatasetSpec")
    async def test_download_dataset_success(
        self, mock_ds_cls, execution_tools, mock_conn_manager
    ):
        """download_execution_dataset downloads a bag and returns its details."""
        mock_execution = _make_mock_execution(execution_rid="EXE-BAG")
        mock_bag = MagicMock()
        mock_bag.dataset_rid = "DS-DOWNLOAD"
        mock_bag.current_version = "1.0.0"
        mock_bag.description = "Test dataset"
        mock_bag.dataset_types = ["Training"]
        mock_bag.execution_rid = "EXE-BAG"
        mock_bag.model = MagicMock()
        mock_bag.model.bag_path = Path("/tmp/bags/DS-DOWNLOAD")
        mock_execution.download_dataset_bag.return_value = mock_bag
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["download_execution_dataset"](
            dataset_rid="DS-DOWNLOAD",
            version="1.0.0",
        )

        data = assert_success(result)
        assert data["status"] == "downloaded"
        assert data["dataset_rid"] == "DS-DOWNLOAD"
        assert data["version"] == "1.0.0"
        assert data["description"] == "Test dataset"
        assert data["dataset_types"] == ["Training"]
        assert data["execution_rid"] == "EXE-BAG"
        assert data["bag_path"] == "/tmp/bags/DS-DOWNLOAD"

    @pytest.mark.asyncio
    @patch("deriva_ml.dataset.aux_classes.DatasetSpec")
    async def test_download_dataset_no_materialize(
        self, mock_ds_cls, execution_tools, mock_conn_manager
    ):
        """download_execution_dataset passes materialize=False."""
        mock_execution = _make_mock_execution()
        mock_bag = MagicMock()
        mock_bag.dataset_rid = "DS-1"
        mock_bag.current_version = "2.0.0"
        mock_bag.description = ""
        mock_bag.dataset_types = []
        mock_bag.execution_rid = "EXE-1"
        mock_bag.model = MagicMock()
        mock_bag.model.bag_path = Path("/tmp/bags/DS-1")
        mock_execution.download_dataset_bag.return_value = mock_bag
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["download_execution_dataset"](
            dataset_rid="DS-1",
            version="2.0.0",
            materialize=False,
        )

        data = assert_success(result)
        mock_execution.download_dataset_bag.assert_called_once()
        # Verify DatasetSpec was constructed with materialize=False
        mock_ds_cls.assert_called_once_with(
            rid="DS-1", version="2.0.0", materialize=False
        )

    @pytest.mark.asyncio
    @patch("deriva_ml.dataset.aux_classes.DatasetSpec")
    async def test_download_dataset_version_none(
        self, mock_ds_cls, execution_tools, mock_conn_manager
    ):
        """download_execution_dataset handles None current_version."""
        mock_execution = _make_mock_execution()
        mock_bag = MagicMock()
        mock_bag.dataset_rid = "DS-1"
        mock_bag.current_version = None
        mock_bag.description = ""
        mock_bag.dataset_types = []
        mock_bag.execution_rid = "EXE-1"
        mock_bag.model = MagicMock()
        mock_bag.model.bag_path = Path("/tmp/bags/DS-1")
        mock_execution.download_dataset_bag.return_value = mock_bag
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["download_execution_dataset"](
            dataset_rid="DS-1",
            version="1.0.0",
        )

        data = assert_success(result)
        assert data["version"] is None

    @pytest.mark.asyncio
    async def test_download_dataset_no_active_execution(
        self, execution_tools, mock_conn_manager
    ):
        """download_execution_dataset returns error when no execution is active."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["download_execution_dataset"](
            dataset_rid="DS-1",
            version="1.0.0",
        )

        assert_error(result, "No active execution")

    @pytest.mark.asyncio
    @patch("deriva_ml.dataset.aux_classes.DatasetSpec")
    async def test_download_dataset_exception(
        self, mock_ds_cls, execution_tools, mock_conn_manager
    ):
        """download_execution_dataset returns error when download fails."""
        mock_execution = _make_mock_execution()
        mock_execution.download_dataset_bag.side_effect = Exception(
            "Dataset DS-BAD not found"
        )
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["download_execution_dataset"](
            dataset_rid="DS-BAD",
            version="1.0.0",
        )

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_download_dataset_no_connection(self, execution_tools_disconnected):
        """download_execution_dataset returns error when not connected."""
        result = await execution_tools_disconnected["download_execution_dataset"](
            dataset_rid="DS-1",
            version="1.0.0",
        )

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestGetExecutionWorkingDir
# =============================================================================


class TestGetExecutionWorkingDir:
    """Tests for the get_execution_working_dir tool."""

    @pytest.mark.asyncio
    async def test_get_working_dir_success(self, execution_tools, mock_conn_manager):
        """get_execution_working_dir returns the working directory path."""
        mock_execution = _make_mock_execution(
            execution_rid="EXE-WD",
            working_dir="/tmp/my_execution",
        )
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["get_execution_working_dir"]()

        data = parse_json_result(result)
        assert data["working_dir"] == "/tmp/my_execution"
        assert data["execution_rid"] == "EXE-WD"

    @pytest.mark.asyncio
    async def test_get_working_dir_no_active_execution(
        self, execution_tools, mock_conn_manager
    ):
        """get_execution_working_dir returns error when no execution is active."""
        _clear_active_execution(mock_conn_manager)

        result = await execution_tools["get_execution_working_dir"]()

        assert_error(result, "No active execution")

    @pytest.mark.asyncio
    async def test_get_working_dir_exception(
        self, execution_tools, mock_conn_manager
    ):
        """get_execution_working_dir returns error on unexpected exception."""
        mock_execution = _make_mock_execution()
        type(mock_execution).working_dir = property(
            lambda self: (_ for _ in ()).throw(Exception("Disk error"))
        )
        _set_active_execution(mock_conn_manager, mock_execution)

        result = await execution_tools["get_execution_working_dir"]()

        assert_error(result, "Disk error")

    @pytest.mark.asyncio
    async def test_get_working_dir_no_connection(self, execution_tools_disconnected):
        """get_execution_working_dir returns error when not connected."""
        result = await execution_tools_disconnected["get_execution_working_dir"]()

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


# =============================================================================
# TestListParentExecutions
# =============================================================================


class TestListParentExecutions:
    """Tests for the list_parent_executions tool."""

    @pytest.mark.asyncio
    async def test_list_parents_success(self, execution_tools, mock_ml):
        """list_parent_executions returns parent executions."""
        mock_child = MagicMock()
        parent1 = _make_mock_execution(
            execution_rid="PARENT-1",
            workflow_rid="WF-SWEEP",
            status="completed",
            description="Parameter sweep",
        )
        mock_child.list_parent_executions.return_value = [parent1]
        mock_ml.lookup_execution.return_value = mock_child

        result = await execution_tools["list_parent_executions"](
            execution_rid="1-CHILD",
        )

        data = assert_success(result)
        assert data["child_rid"] == "1-CHILD"
        assert data["recurse"] is False
        assert data["count"] == 1
        assert len(data["parents"]) == 1
        assert data["parents"][0]["execution_rid"] == "PARENT-1"
        assert data["parents"][0]["workflow_rid"] == "WF-SWEEP"
        assert data["parents"][0]["status"] == "completed"
        assert data["parents"][0]["description"] == "Parameter sweep"

    @pytest.mark.asyncio
    async def test_list_parents_recurse(self, execution_tools, mock_ml):
        """list_parent_executions passes recurse=True for all ancestors."""
        mock_child = MagicMock()
        grandparent = _make_mock_execution(execution_rid="GP-1")
        parent = _make_mock_execution(execution_rid="P-1")
        mock_child.list_parent_executions.return_value = [parent, grandparent]
        mock_ml.lookup_execution.return_value = mock_child

        result = await execution_tools["list_parent_executions"](
            execution_rid="1-CHILD",
            recurse=True,
        )

        data = assert_success(result)
        assert data["recurse"] is True
        assert data["count"] == 2
        mock_child.list_parent_executions.assert_called_once_with(recurse=True)

    @pytest.mark.asyncio
    async def test_list_parents_empty(self, execution_tools, mock_ml):
        """list_parent_executions returns empty list for root executions."""
        mock_child = MagicMock()
        mock_child.list_parent_executions.return_value = []
        mock_ml.lookup_execution.return_value = mock_child

        result = await execution_tools["list_parent_executions"](
            execution_rid="1-ROOT",
        )

        data = assert_success(result)
        assert data["count"] == 0
        assert data["parents"] == []

    @pytest.mark.asyncio
    async def test_list_parents_not_found(self, execution_tools, mock_ml):
        """list_parent_executions returns error when execution not found."""
        mock_ml.lookup_execution.side_effect = Exception("Execution not found")

        result = await execution_tools["list_parent_executions"](
            execution_rid="NOPE",
        )

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_list_parents_no_connection(self, execution_tools_disconnected):
        """list_parent_executions returns error when not connected."""
        result = await execution_tools_disconnected["list_parent_executions"](
            execution_rid="1-C",
        )

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestClearCache (Storage tool - SYNC)
# =============================================================================


class TestClearCache:
    """Tests for the clear_cache storage tool (sync)."""

    def test_clear_cache_all(self, storage_tools, mock_ml):
        """clear_cache removes all cache entries and returns summary."""
        mock_ml.clear_cache.return_value = {
            "files_removed": 15,
            "dirs_removed": 3,
            "bytes_freed": 1048576,
            "errors": 0,
        }

        result = storage_tools["clear_cache"]()

        data = assert_success(result)
        assert data["status"] == "success"
        assert data["older_than_days"] is None
        assert data["files_removed"] == 15
        assert data["dirs_removed"] == 3
        assert data["bytes_freed"] == 1048576
        assert data["errors"] == 0
        mock_ml.clear_cache.assert_called_once_with(older_than_days=None)

    def test_clear_cache_older_than(self, storage_tools, mock_ml):
        """clear_cache with older_than_days only removes old entries."""
        mock_ml.clear_cache.return_value = {
            "files_removed": 5,
            "dirs_removed": 1,
            "bytes_freed": 512000,
            "errors": 0,
        }

        result = storage_tools["clear_cache"](older_than_days=7)

        data = assert_success(result)
        assert data["older_than_days"] == 7
        assert data["files_removed"] == 5
        mock_ml.clear_cache.assert_called_once_with(older_than_days=7)

    def test_clear_cache_empty(self, storage_tools, mock_ml):
        """clear_cache succeeds when cache is already empty."""
        mock_ml.clear_cache.return_value = {
            "files_removed": 0,
            "dirs_removed": 0,
            "bytes_freed": 0,
            "errors": 0,
        }

        result = storage_tools["clear_cache"]()

        data = assert_success(result)
        assert data["files_removed"] == 0
        assert data["bytes_freed"] == 0

    def test_clear_cache_exception(self, storage_tools, mock_ml):
        """clear_cache returns error when cache clearing fails."""
        mock_ml.clear_cache.side_effect = Exception("Permission denied")

        result = storage_tools["clear_cache"]()

        assert_error(result, "Permission denied")

    def test_clear_cache_no_connection(self, storage_tools_disconnected):
        """clear_cache returns error when not connected."""
        result = storage_tools_disconnected["clear_cache"]()

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestCleanExecutionDirs (Storage tool - SYNC)
# =============================================================================


class TestCleanExecutionDirs:
    """Tests for the clean_execution_dirs storage tool (sync)."""

    def test_clean_dirs_all(self, storage_tools, mock_ml):
        """clean_execution_dirs removes all execution directories."""
        mock_ml.clean_execution_dirs.return_value = {
            "dirs_removed": 5,
            "bytes_freed": 2097152,
            "errors": 0,
        }

        result = storage_tools["clean_execution_dirs"]()

        data = assert_success(result)
        assert data["status"] == "success"
        assert data["older_than_days"] is None
        assert data["exclude_rids"] is None
        assert data["dirs_removed"] == 5
        assert data["bytes_freed"] == 2097152
        assert data["errors"] == 0
        mock_ml.clean_execution_dirs.assert_called_once_with(
            older_than_days=None,
            exclude_rids=None,
        )

    def test_clean_dirs_older_than(self, storage_tools, mock_ml):
        """clean_execution_dirs with older_than_days only removes old dirs."""
        mock_ml.clean_execution_dirs.return_value = {
            "dirs_removed": 2,
            "bytes_freed": 100000,
            "errors": 0,
        }

        result = storage_tools["clean_execution_dirs"](older_than_days=30)

        data = assert_success(result)
        assert data["older_than_days"] == 30
        mock_ml.clean_execution_dirs.assert_called_once_with(
            older_than_days=30,
            exclude_rids=None,
        )

    def test_clean_dirs_with_excludes(self, storage_tools, mock_ml):
        """clean_execution_dirs excludes specified RIDs."""
        mock_ml.clean_execution_dirs.return_value = {
            "dirs_removed": 3,
            "bytes_freed": 500000,
            "errors": 0,
        }

        result = storage_tools["clean_execution_dirs"](
            exclude_rids=["1-ABC", "1-DEF"]
        )

        data = assert_success(result)
        assert data["exclude_rids"] == ["1-ABC", "1-DEF"]
        mock_ml.clean_execution_dirs.assert_called_once_with(
            older_than_days=None,
            exclude_rids=["1-ABC", "1-DEF"],
        )

    def test_clean_dirs_with_both_params(self, storage_tools, mock_ml):
        """clean_execution_dirs with both older_than_days and exclude_rids."""
        mock_ml.clean_execution_dirs.return_value = {
            "dirs_removed": 1,
            "bytes_freed": 250000,
            "errors": 0,
        }

        result = storage_tools["clean_execution_dirs"](
            older_than_days=14,
            exclude_rids=["1-KEEP"],
        )

        data = assert_success(result)
        assert data["older_than_days"] == 14
        assert data["exclude_rids"] == ["1-KEEP"]
        mock_ml.clean_execution_dirs.assert_called_once_with(
            older_than_days=14,
            exclude_rids=["1-KEEP"],
        )

    def test_clean_dirs_empty(self, storage_tools, mock_ml):
        """clean_execution_dirs succeeds when no directories to clean."""
        mock_ml.clean_execution_dirs.return_value = {
            "dirs_removed": 0,
            "bytes_freed": 0,
            "errors": 0,
        }

        result = storage_tools["clean_execution_dirs"]()

        data = assert_success(result)
        assert data["dirs_removed"] == 0

    def test_clean_dirs_with_errors(self, storage_tools, mock_ml):
        """clean_execution_dirs reports partial cleanup with errors."""
        mock_ml.clean_execution_dirs.return_value = {
            "dirs_removed": 3,
            "bytes_freed": 750000,
            "errors": 2,
        }

        result = storage_tools["clean_execution_dirs"]()

        data = assert_success(result)
        assert data["dirs_removed"] == 3
        assert data["errors"] == 2

    def test_clean_dirs_exception(self, storage_tools, mock_ml):
        """clean_execution_dirs returns error when cleanup fails."""
        mock_ml.clean_execution_dirs.side_effect = Exception("Disk error")

        result = storage_tools["clean_execution_dirs"]()

        assert_error(result, "Disk error")

    def test_clean_dirs_no_connection(self, storage_tools_disconnected):
        """clean_execution_dirs returns error when not connected."""
        result = storage_tools_disconnected["clean_execution_dirs"]()

        assert_error(result, "No active catalog connection")


# =============================================================================
# TestExecutionLifecycle (integration-style tests across multiple tools)
# =============================================================================


class TestExecutionLifecycle:
    """Tests for the full execution lifecycle using multiple tools together."""

    @pytest.mark.asyncio
    @patch("deriva_ml.execution.execution_configuration.ExecutionConfiguration")
    @patch("deriva_ml.dataset.aux_classes.DatasetSpec")
    async def test_full_lifecycle(
        self, mock_ds_cls, mock_ec_cls, execution_tools, mock_ml, mock_conn_manager
    ):
        """Test complete execution lifecycle: create -> start -> stop -> upload."""
        # 1. Create execution
        mock_workflow = MagicMock()
        mock_ml.create_workflow.return_value = mock_workflow
        mock_execution = _make_mock_execution(execution_rid="EXE-LIFECYCLE")
        mock_execution.upload_execution_outputs.return_value = {
            "Model": ["/tmp/model.pt"]
        }
        mock_ml.create_execution.return_value = mock_execution

        create_result = await execution_tools["create_execution"](
            workflow_name="Lifecycle Test",
            workflow_type="Training",
            description="Full lifecycle test",
        )
        create_data = assert_success(create_result)
        assert create_data["status"] == "created"

        # 2. Start execution
        start_result = await execution_tools["start_execution"]()
        start_data = assert_success(start_result)
        assert start_data["status"] == "started"
        mock_execution.execution_start.assert_called_once()

        # 3. Get info during execution
        info_result = await execution_tools["get_execution_info"]()
        info_data = parse_json_result(info_result)
        assert info_data["execution_rid"] == "EXE-LIFECYCLE"

        # 4. Stop execution
        stop_result = await execution_tools["stop_execution"]()
        stop_data = assert_success(stop_result)
        assert stop_data["status"] == "completed"
        mock_execution.execution_stop.assert_called_once()

        # 5. Upload outputs
        upload_result = await execution_tools["upload_execution_outputs"]()
        upload_data = assert_success(upload_result)
        assert upload_data["status"] == "uploaded"
        assert upload_data["assets_uploaded"]["Model"] == 1
