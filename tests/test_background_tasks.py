"""Unit tests for background task management tools."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import assert_error, assert_success, parse_json_result


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def bg_task_tools_disconnected(disconnected_conn_manager):
    """Capture background task tools with no connection."""
    from tests.conftest import _create_tool_capture

    from deriva_ml_mcp.tools.background_tasks import register_background_task_tools

    mcp, tools = _create_tool_capture()
    register_background_task_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def mock_task_manager():
    """Create a mock BackgroundTaskManager."""
    tm = MagicMock()
    tm.create_task = MagicMock()
    tm.get_task_snapshot_async = AsyncMock()
    tm.list_tasks_snapshots_async = AsyncMock()
    tm.cancel_task = MagicMock()
    return tm


# =============================================================================
# clone_catalog_async
# =============================================================================


class TestCloneCatalogAsync:
    """Tests for the clone_catalog_async tool."""

    @pytest.mark.asyncio
    async def test_clone_success(self, bg_task_tools, mock_task_manager):
        """Starting a clone returns status=started with task_id."""
        from deriva_ml_mcp.tasks import TaskType

        mock_task = MagicMock()
        mock_task.task_id = "abc12345"
        mock_task.task_type = TaskType.CLONE_CATALOG
        mock_task_manager.create_task.return_value = mock_task

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["clone_catalog_async"](
                source_hostname="www.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
            )

        data = assert_success(result)
        assert data["status"] == "started"
        assert data["task_id"] == "abc12345"
        assert data["task_type"] == "clone_catalog"
        assert "get_task_status" in data["message"]
        assert data["parameters"]["source"] == "www.example.org:1"
        assert data["parameters"]["root_rid"] == "3-HXMC"

        # Verify create_task was called with correct args
        call_kwargs = mock_task_manager.create_task.call_args
        assert call_kwargs.kwargs["user_id"] == "test_user"
        assert call_kwargs.kwargs["task_type"] == TaskType.CLONE_CATALOG
        params = call_kwargs.kwargs["parameters"]
        assert params["source_hostname"] == "www.example.org"
        assert params["source_catalog_id"] == "1"
        assert params["root_rid"] == "3-HXMC"

    @pytest.mark.asyncio
    async def test_clone_with_all_parameters(self, bg_task_tools, mock_task_manager):
        """Clone with all optional parameters passes them through correctly."""
        from deriva_ml_mcp.tasks import TaskType

        mock_task = MagicMock()
        mock_task.task_id = "task-full"
        mock_task.task_type = TaskType.CLONE_CATALOG
        mock_task_manager.create_task.return_value = mock_task

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["clone_catalog_async"](
                source_hostname="www.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
                dest_hostname="dest.example.org",
                alias="my-clone",
                add_ml_schema=False,
                asset_mode="full",
                copy_annotations=False,
                copy_policy=False,
                exclude_schemas=["public"],
                exclude_objects=["schema:table1"],
                reinitialize_dataset_versions=False,
                orphan_strategy="delete",
                prune_hidden_fkeys=True,
                truncate_oversized=True,
                include_tables=["schema:extra_table"],
                include_associations=False,
                include_vocabularies=False,
            )

        data = assert_success(result)
        assert data["status"] == "started"
        assert data["parameters"]["dest"] == "dest.example.org"
        assert data["parameters"]["alias"] == "my-clone"

        # Verify all parameters passed to create_task
        params = mock_task_manager.create_task.call_args.kwargs["parameters"]
        assert params["dest_hostname"] == "dest.example.org"
        assert params["alias"] == "my-clone"
        assert params["add_ml_schema"] is False
        assert params["asset_mode"] == "full"
        assert params["copy_annotations"] is False
        assert params["copy_policy"] is False
        assert params["exclude_schemas"] == ["public"]
        assert params["exclude_objects"] == ["schema:table1"]
        assert params["reinitialize_dataset_versions"] is False
        assert params["orphan_strategy"] == "delete"
        assert params["prune_hidden_fkeys"] is True
        assert params["truncate_oversized"] is True
        assert params["include_tables"] == ["schema:extra_table"]
        assert params["include_associations"] is False
        assert params["include_vocabularies"] is False

    @pytest.mark.asyncio
    async def test_clone_dest_defaults_to_source(self, bg_task_tools, mock_task_manager):
        """When dest_hostname is None, parameters show source as dest."""
        from deriva_ml_mcp.tasks import TaskType

        mock_task = MagicMock()
        mock_task.task_id = "task-default-dest"
        mock_task.task_type = TaskType.CLONE_CATALOG
        mock_task_manager.create_task.return_value = mock_task

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["clone_catalog_async"](
                source_hostname="www.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
            )

        data = parse_json_result(result)
        assert data["parameters"]["dest"] == "www.example.org"

    @pytest.mark.asyncio
    async def test_clone_uses_credential_user_when_disconnected(
        self, bg_task_tools_disconnected, mock_task_manager
    ):
        """When no active connection, user_id is derived from credentials."""
        from deriva_ml_mcp.tasks import TaskType

        mock_task = MagicMock()
        mock_task.task_id = "task-cred"
        mock_task.task_type = TaskType.CLONE_CATALOG
        mock_task_manager.create_task.return_value = mock_task

        with (
            patch(
                "deriva_ml_mcp.tools.background_tasks.get_task_manager",
                return_value=mock_task_manager,
            ),
            patch(
                "deriva_ml_mcp.tools.background_tasks.get_credential",
                return_value={"cookie": "test-cookie"},
            ),
            patch(
                "deriva_ml_mcp.tools.background_tasks.derive_user_id",
                return_value="credential_user_123",
            ),
        ):
            result = await bg_task_tools_disconnected["clone_catalog_async"](
                source_hostname="www.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
            )

        data = assert_success(result)
        assert data["status"] == "started"

        # Verify user_id came from credential
        call_kwargs = mock_task_manager.create_task.call_args.kwargs
        assert call_kwargs["user_id"] == "credential_user_123"

    @pytest.mark.asyncio
    async def test_clone_defaults_to_default_user_on_credential_failure(
        self, bg_task_tools_disconnected, mock_task_manager
    ):
        """When credential lookup fails, falls back to 'default_user'."""
        from deriva_ml_mcp.tasks import TaskType

        mock_task = MagicMock()
        mock_task.task_id = "task-default"
        mock_task.task_type = TaskType.CLONE_CATALOG
        mock_task_manager.create_task.return_value = mock_task

        with (
            patch(
                "deriva_ml_mcp.tools.background_tasks.get_task_manager",
                return_value=mock_task_manager,
            ),
            patch(
                "deriva_ml_mcp.tools.background_tasks.get_credential",
                side_effect=Exception("No credentials"),
            ),
        ):
            result = await bg_task_tools_disconnected["clone_catalog_async"](
                source_hostname="www.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
            )

        data = assert_success(result)
        call_kwargs = mock_task_manager.create_task.call_args.kwargs
        assert call_kwargs["user_id"] == "default_user"

    @pytest.mark.asyncio
    async def test_clone_error_returns_error_status(self, bg_task_tools):
        """When task creation fails, return status=error."""
        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            side_effect=RuntimeError("Task manager not available"),
        ):
            result = await bg_task_tools["clone_catalog_async"](
                source_hostname="www.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
            )

        data = assert_error(result, expected_message="Task manager not available")

    @pytest.mark.asyncio
    async def test_clone_create_task_exception(self, bg_task_tools, mock_task_manager):
        """When create_task raises an exception, return status=error."""
        mock_task_manager.create_task.side_effect = ValueError(
            "Max tasks per user exceeded"
        )

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["clone_catalog_async"](
                source_hostname="www.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
            )

        data = assert_error(result, expected_message="Max tasks per user exceeded")


# =============================================================================
# get_task_status
# =============================================================================


class TestGetTaskStatus:
    """Tests for the get_task_status tool."""

    @pytest.mark.asyncio
    async def test_get_status_running(self, bg_task_tools, mock_task_manager):
        """Getting status of a running task returns progress info."""
        snapshot = {
            "task_id": "task-run",
            "task_type": "clone_catalog",
            "status": "running",
            "created_at": "2025-01-01T00:00:00+00:00",
            "started_at": "2025-01-01T00:00:01+00:00",
            "completed_at": None,
            "parameters": {"source_hostname": "example.org"},
            "progress": {
                "current_step": "Copying data",
                "total_steps": 4,
                "current_step_number": 2,
                "percent_complete": 45.0,
                "message": "Copying table Subject...",
                "updated_at": "2025-01-01T00:00:30+00:00",
            },
            "error": None,
        }
        mock_task_manager.get_task_snapshot_async.return_value = snapshot

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["get_task_status"](task_id="task-run")

        data = parse_json_result(result)
        assert data["task_id"] == "task-run"
        assert data["status"] == "running"
        assert data["progress"]["percent_complete"] == 45.0
        assert data["progress"]["message"] == "Copying table Subject..."

    @pytest.mark.asyncio
    async def test_get_status_completed_with_result(
        self, bg_task_tools, mock_task_manager
    ):
        """Completed task includes result when include_result=True."""
        snapshot = {
            "task_id": "task-done",
            "task_type": "clone_catalog",
            "status": "completed",
            "created_at": "2025-01-01T00:00:00+00:00",
            "started_at": "2025-01-01T00:00:01+00:00",
            "completed_at": "2025-01-01T00:05:00+00:00",
            "parameters": {},
            "progress": {
                "current_step": "Completed",
                "percent_complete": 100.0,
                "message": "Completed",
            },
            "error": None,
            "result": {
                "status": "cloned",
                "dest_catalog_id": "42",
                "dest_hostname": "example.org",
            },
        }
        mock_task_manager.get_task_snapshot_async.return_value = snapshot

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["get_task_status"](
                task_id="task-done", include_result=True
            )

        data = parse_json_result(result)
        assert data["status"] == "completed"
        assert "result" in data
        assert data["result"]["dest_catalog_id"] == "42"

    @pytest.mark.asyncio
    async def test_get_status_exclude_result(self, bg_task_tools, mock_task_manager):
        """When include_result=False, result is removed from snapshot."""
        snapshot = {
            "task_id": "task-done",
            "status": "completed",
            "result": {"big": "data"},
            "progress": {},
            "error": None,
        }
        mock_task_manager.get_task_snapshot_async.return_value = snapshot

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["get_task_status"](
                task_id="task-done", include_result=False
            )

        data = parse_json_result(result)
        assert "result" not in data

    @pytest.mark.asyncio
    async def test_get_status_task_not_found(self, bg_task_tools, mock_task_manager):
        """When task is not found, return status=error."""
        mock_task_manager.get_task_snapshot_async.return_value = None

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["get_task_status"](task_id="nonexistent")

        data = assert_error(result, expected_message="not found or access denied")

    @pytest.mark.asyncio
    async def test_get_status_uses_connection_user_id(
        self, bg_task_tools, mock_task_manager
    ):
        """Task status uses user_id from active connection info."""
        snapshot = {"task_id": "t1", "status": "running", "progress": {}, "error": None}
        mock_task_manager.get_task_snapshot_async.return_value = snapshot

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools["get_task_status"](task_id="t1")

        mock_task_manager.get_task_snapshot_async.assert_called_once_with(
            "t1", "test_user"
        )

    @pytest.mark.asyncio
    async def test_get_status_default_user_when_disconnected(
        self, bg_task_tools_disconnected, mock_task_manager
    ):
        """When no active connection, uses 'default_user' for task lookup."""
        snapshot = {"task_id": "t1", "status": "running", "progress": {}, "error": None}
        mock_task_manager.get_task_snapshot_async.return_value = snapshot

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools_disconnected["get_task_status"](task_id="t1")

        mock_task_manager.get_task_snapshot_async.assert_called_once_with(
            "t1", "default_user"
        )

    @pytest.mark.asyncio
    async def test_get_status_failed_task(self, bg_task_tools, mock_task_manager):
        """Failed task includes error information."""
        snapshot = {
            "task_id": "task-fail",
            "status": "failed",
            "error": "Connection timeout",
            "progress": {"message": "Failed: Connection timeout"},
        }
        mock_task_manager.get_task_snapshot_async.return_value = snapshot

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["get_task_status"](task_id="task-fail")

        data = parse_json_result(result)
        assert data["status"] == "failed"
        assert data["error"] == "Connection timeout"

    @pytest.mark.asyncio
    async def test_get_status_exception(self, bg_task_tools):
        """When an exception occurs, return status=error."""
        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            side_effect=RuntimeError("Internal error"),
        ):
            result = await bg_task_tools["get_task_status"](task_id="t1")

        data = assert_error(result, expected_message="Internal error")


# =============================================================================
# list_tasks
# =============================================================================


class TestListTasks:
    """Tests for the list_tasks tool."""

    @pytest.mark.asyncio
    async def test_list_all_tasks(self, bg_task_tools, mock_task_manager):
        """Listing tasks with no filters returns all user tasks."""
        snapshots = [
            {
                "task_id": "t1",
                "task_type": "clone_catalog",
                "status": "running",
                "progress": {},
            },
            {
                "task_id": "t2",
                "task_type": "clone_catalog",
                "status": "completed",
                "progress": {},
            },
        ]
        mock_task_manager.list_tasks_snapshots_async.return_value = snapshots

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["list_tasks"]()

        data = parse_json_result(result)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["task_id"] == "t1"
        assert data[1]["task_id"] == "t2"

    @pytest.mark.asyncio
    async def test_list_tasks_filter_by_status(self, bg_task_tools, mock_task_manager):
        """Filtering by status passes the correct TaskStatus enum."""
        from deriva_ml_mcp.tasks import TaskStatus

        mock_task_manager.list_tasks_snapshots_async.return_value = []

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools["list_tasks"](status="running")

        call_kwargs = mock_task_manager.list_tasks_snapshots_async.call_args
        assert call_kwargs.kwargs["status_filter"] == TaskStatus.RUNNING
        assert call_kwargs.kwargs["task_type_filter"] is None

    @pytest.mark.asyncio
    async def test_list_tasks_filter_by_type(self, bg_task_tools, mock_task_manager):
        """Filtering by task_type passes the correct TaskType enum."""
        from deriva_ml_mcp.tasks import TaskType

        mock_task_manager.list_tasks_snapshots_async.return_value = []

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools["list_tasks"](task_type="clone_catalog")

        call_kwargs = mock_task_manager.list_tasks_snapshots_async.call_args
        assert call_kwargs.kwargs["task_type_filter"] == TaskType.CLONE_CATALOG
        assert call_kwargs.kwargs["status_filter"] is None

    @pytest.mark.asyncio
    async def test_list_tasks_filter_by_both(self, bg_task_tools, mock_task_manager):
        """Filtering by both status and task_type passes both enums."""
        from deriva_ml_mcp.tasks import TaskStatus, TaskType

        mock_task_manager.list_tasks_snapshots_async.return_value = []

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools["list_tasks"](
                status="completed", task_type="clone_catalog"
            )

        call_kwargs = mock_task_manager.list_tasks_snapshots_async.call_args
        assert call_kwargs.kwargs["status_filter"] == TaskStatus.COMPLETED
        assert call_kwargs.kwargs["task_type_filter"] == TaskType.CLONE_CATALOG

    @pytest.mark.asyncio
    async def test_list_tasks_no_filter(self, bg_task_tools, mock_task_manager):
        """No filters passes None for both status and type."""
        mock_task_manager.list_tasks_snapshots_async.return_value = []

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools["list_tasks"]()

        call_kwargs = mock_task_manager.list_tasks_snapshots_async.call_args
        assert call_kwargs.kwargs["status_filter"] is None
        assert call_kwargs.kwargs["task_type_filter"] is None
        assert call_kwargs.kwargs["include_result"] is False

    @pytest.mark.asyncio
    async def test_list_tasks_empty(self, bg_task_tools, mock_task_manager):
        """When no tasks exist, return an empty list."""
        mock_task_manager.list_tasks_snapshots_async.return_value = []

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["list_tasks"]()

        data = parse_json_result(result)
        assert data == []

    @pytest.mark.asyncio
    async def test_list_tasks_invalid_status(self, bg_task_tools):
        """Invalid status value returns an error."""
        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=MagicMock(),
        ):
            result = await bg_task_tools["list_tasks"](status="invalid_status")

        data = assert_error(result, expected_message="Invalid filter value")

    @pytest.mark.asyncio
    async def test_list_tasks_invalid_type(self, bg_task_tools):
        """Invalid task_type value returns an error."""
        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=MagicMock(),
        ):
            result = await bg_task_tools["list_tasks"](task_type="invalid_type")

        data = assert_error(result, expected_message="Invalid filter value")

    @pytest.mark.asyncio
    async def test_list_tasks_uses_connection_user_id(
        self, bg_task_tools, mock_task_manager
    ):
        """Task listing uses user_id from active connection."""
        mock_task_manager.list_tasks_snapshots_async.return_value = []

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools["list_tasks"]()

        call_kwargs = mock_task_manager.list_tasks_snapshots_async.call_args
        assert call_kwargs.kwargs["user_id"] == "test_user"

    @pytest.mark.asyncio
    async def test_list_tasks_default_user_when_disconnected(
        self, bg_task_tools_disconnected, mock_task_manager
    ):
        """When no active connection, uses 'default_user' for listing."""
        mock_task_manager.list_tasks_snapshots_async.return_value = []

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools_disconnected["list_tasks"]()

        call_kwargs = mock_task_manager.list_tasks_snapshots_async.call_args
        assert call_kwargs.kwargs["user_id"] == "default_user"

    @pytest.mark.asyncio
    async def test_list_tasks_exception(self, bg_task_tools):
        """When an unexpected exception occurs, return status=error."""
        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            side_effect=RuntimeError("Unexpected failure"),
        ):
            result = await bg_task_tools["list_tasks"]()

        data = assert_error(result, expected_message="Unexpected failure")


# =============================================================================
# cancel_task
# =============================================================================


class TestCancelTask:
    """Tests for the cancel_task tool."""

    @pytest.mark.asyncio
    async def test_cancel_success(self, bg_task_tools, mock_task_manager):
        """Cancelling a running task returns status=cancelled."""
        mock_task_manager.cancel_task.return_value = True

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["cancel_task"](task_id="task-to-cancel")

        data = parse_json_result(result)
        assert data["status"] == "cancelled"
        assert data["task_id"] == "task-to-cancel"
        assert "cancellation requested" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, bg_task_tools, mock_task_manager):
        """Cancelling a nonexistent task returns an error."""
        mock_task_manager.cancel_task.return_value = False

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["cancel_task"](task_id="no-such-task")

        data = assert_error(result, expected_message="not found")

    @pytest.mark.asyncio
    async def test_cancel_already_completed(self, bg_task_tools, mock_task_manager):
        """Cancelling a completed task returns an error."""
        mock_task_manager.cancel_task.return_value = False

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["cancel_task"](task_id="completed-task")

        data = assert_error(result, expected_message="already completed")

    @pytest.mark.asyncio
    async def test_cancel_uses_connection_user_id(
        self, bg_task_tools, mock_task_manager
    ):
        """Cancel passes user_id from active connection."""
        mock_task_manager.cancel_task.return_value = True

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools["cancel_task"](task_id="t1")

        mock_task_manager.cancel_task.assert_called_once_with("t1", "test_user")

    @pytest.mark.asyncio
    async def test_cancel_default_user_when_disconnected(
        self, bg_task_tools_disconnected, mock_task_manager
    ):
        """When no active connection, cancel uses 'default_user'."""
        mock_task_manager.cancel_task.return_value = True

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            await bg_task_tools_disconnected["cancel_task"](task_id="t1")

        mock_task_manager.cancel_task.assert_called_once_with("t1", "default_user")

    @pytest.mark.asyncio
    async def test_cancel_exception(self, bg_task_tools):
        """When an exception occurs during cancel, return status=error."""
        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            side_effect=RuntimeError("Manager unavailable"),
        ):
            result = await bg_task_tools["cancel_task"](task_id="t1")

        data = assert_error(result, expected_message="Manager unavailable")

    @pytest.mark.asyncio
    async def test_cancel_task_manager_cancel_exception(
        self, bg_task_tools, mock_task_manager
    ):
        """When cancel_task itself raises, return status=error."""
        mock_task_manager.cancel_task.side_effect = RuntimeError("Lock contention")

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_task_manager",
            return_value=mock_task_manager,
        ):
            result = await bg_task_tools["cancel_task"](task_id="t1")

        data = assert_error(result, expected_message="Lock contention")


# =============================================================================
# Helper functions
# =============================================================================


class TestResolveHostname:
    """Tests for the _resolve_hostname helper function."""

    def test_localhost_with_env_var(self):
        """When DERIVA_MCP_LOCALHOST_HOSTNAME is set, localhost is remapped."""
        from deriva_ml_mcp.tools.background_tasks import _resolve_hostname

        with patch.dict("os.environ", {"DERIVA_MCP_LOCALHOST_HOSTNAME": "deriva"}):
            result = _resolve_hostname("localhost")

        assert result == "deriva"

    def test_localhost_without_env_var(self):
        """When DERIVA_MCP_LOCALHOST_HOSTNAME is not set, localhost is unchanged."""
        from deriva_ml_mcp.tools.background_tasks import _resolve_hostname

        with patch.dict("os.environ", {}, clear=True):
            # Make sure the env var is not set
            import os
            os.environ.pop("DERIVA_MCP_LOCALHOST_HOSTNAME", None)
            result = _resolve_hostname("localhost")

        assert result == "localhost"

    def test_non_localhost_hostname(self):
        """Non-localhost hostnames are returned unchanged."""
        from deriva_ml_mcp.tools.background_tasks import _resolve_hostname

        result = _resolve_hostname("www.example.org")
        assert result == "www.example.org"

    def test_none_hostname(self):
        """None hostname is returned as None."""
        from deriva_ml_mcp.tools.background_tasks import _resolve_hostname

        result = _resolve_hostname(None)
        assert result is None


class TestGetUserIdFromCredential:
    """Tests for the _get_user_id_from_credential helper function."""

    def test_with_valid_hostname(self):
        """With a valid hostname, derives user_id from credentials."""
        from deriva_ml_mcp.tools.background_tasks import _get_user_id_from_credential

        with (
            patch(
                "deriva_ml_mcp.tools.background_tasks.get_credential",
                return_value={"cookie": "abc"},
            ),
            patch(
                "deriva_ml_mcp.tools.background_tasks.derive_user_id",
                return_value="user_hash_123",
            ),
        ):
            result = _get_user_id_from_credential("www.example.org")

        assert result == "user_hash_123"

    def test_with_none_hostname(self):
        """With no hostname, returns default_user."""
        from deriva_ml_mcp.tools.background_tasks import _get_user_id_from_credential

        result = _get_user_id_from_credential(None)
        assert result == "default_user"

    def test_credential_exception(self):
        """When get_credential raises, falls back to default_user."""
        from deriva_ml_mcp.tools.background_tasks import _get_user_id_from_credential

        with patch(
            "deriva_ml_mcp.tools.background_tasks.get_credential",
            side_effect=Exception("No credential file"),
        ):
            result = _get_user_id_from_credential("www.example.org")

        assert result == "default_user"

    def test_derive_user_id_exception(self):
        """When derive_user_id raises, falls back to default_user."""
        from deriva_ml_mcp.tools.background_tasks import _get_user_id_from_credential

        with (
            patch(
                "deriva_ml_mcp.tools.background_tasks.get_credential",
                return_value={"cookie": "abc"},
            ),
            patch(
                "deriva_ml_mcp.tools.background_tasks.derive_user_id",
                side_effect=ValueError("Invalid credential format"),
            ),
        ):
            result = _get_user_id_from_credential("www.example.org")

        assert result == "default_user"
