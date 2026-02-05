"""Background task management for long-running operations.

This module provides a multi-user safe task management system for operations
that may take longer than MCP connection timeouts (e.g., catalog cloning).

Design considerations for multi-user environments:
- Tasks are isolated by user_id (derived from credentials or session)
- Each user can only see and manage their own tasks
- Task state is stored in memory with periodic file backup for recovery
- Thread-safe operations using locks with minimal contention

Performance considerations:
- Lock contention minimized using copy-on-read pattern for status queries
- Async-safe methods provided for use in asyncio contexts
- Task state snapshots avoid holding locks during I/O operations

Persistence:
- Tasks are backed up to a JSON file periodically (default: every 5 seconds)
- On startup, tasks are recovered from the backup file
- Running tasks from crashed servers are marked as failed
- Completed tasks are cleaned up after retention period (default: 7 days)
"""

from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from concurrent.futures import Future

logger = logging.getLogger("deriva-mcp")


class TaskStatus(Enum):
    """Status of a background task."""

    PENDING = "pending"  # Task created but not yet started
    RUNNING = "running"  # Task is currently executing
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"  # Task finished with an error
    CANCELLED = "cancelled"  # Task was cancelled by user


class TaskType(Enum):
    """Types of background tasks."""

    CLONE_CATALOG = "clone_catalog"
    # Future task types can be added here


@dataclass
class TaskProgress:
    """Progress information for a running task."""

    current_step: str = ""
    total_steps: int = 0
    current_step_number: int = 0
    percent_complete: float = 0.0
    message: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_step_number": self.current_step_number,
            "percent_complete": self.percent_complete,
            "message": self.message,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class BackgroundTask:
    """Represents a background task."""

    task_id: str
    user_id: str
    task_type: TaskType
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    _future: Future | None = field(default=None, repr=False)

    def to_dict(self, include_result: bool = True) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Args:
            include_result: If True, include the full result (can be large).
        """
        data = {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "parameters": self.parameters,
            "progress": self.progress.to_dict(),
            "error": self.error,
        }
        if include_result and self.result is not None:
            data["result"] = self.result
        return data

    def to_persistence_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence (always includes result and user_id)."""
        data = self.to_dict(include_result=True)
        data["user_id"] = self.user_id
        return data

    @classmethod
    def from_persistence_dict(cls, data: dict[str, Any]) -> "BackgroundTask":
        """Create a BackgroundTask from a persistence dictionary."""
        # Parse progress
        progress_data = data.get("progress", {})
        progress = TaskProgress(
            current_step=progress_data.get("current_step", ""),
            total_steps=progress_data.get("total_steps", 0),
            current_step_number=progress_data.get("current_step_number", 0),
            percent_complete=progress_data.get("percent_complete", 0.0),
            message=progress_data.get("message", ""),
            updated_at=datetime.fromisoformat(progress_data["updated_at"])
            if progress_data.get("updated_at")
            else datetime.now(timezone.utc),
        )

        return cls(
            task_id=data["task_id"],
            user_id=data["user_id"],
            task_type=TaskType(data["task_type"]),
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            parameters=data.get("parameters", {}),
            result=data.get("result"),
            error=data.get("error"),
            progress=progress,
            _future=None,  # Cannot restore futures
        )


class TaskPersistence:
    """Handles periodic backup of task state to disk for recovery.

    This class provides crash recovery for the BackgroundTaskManager by:
    - Periodically saving task state to a JSON file
    - Loading tasks on startup
    - Marking running tasks from crashed servers as failed
    - Cleaning up old completed tasks

    The file format is human-readable JSON for easy debugging.
    File locking (flock) is used to prevent corruption from concurrent writes.
    """

    DEFAULT_PATH = Path.home() / ".deriva-ml" / "task_state.json"
    DEFAULT_SYNC_INTERVAL = 5  # seconds
    DEFAULT_RETENTION_HOURS = 24 * 7  # 7 days

    def __init__(
        self,
        path: Path | str | None = None,
        sync_interval: int = DEFAULT_SYNC_INTERVAL,
        retention_hours: int = DEFAULT_RETENTION_HOURS,
    ):
        """Initialize task persistence.

        Args:
            path: Path to the state file. Defaults to ~/.deriva-ml/task_state.json
            sync_interval: Seconds between automatic saves. Defaults to 5.
            retention_hours: Hours to retain completed tasks. Defaults to 168 (7 days).
        """
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.sync_interval = sync_interval
        self.retention_hours = retention_hours
        self.server_instance_id = str(uuid.uuid4())[:12]

        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._file_lock = threading.Lock()
        self._sync_thread: threading.Thread | None = None
        self._stop_sync = threading.Event()

        logger.info(f"Task persistence initialized: {self.path}")

    def start_sync_thread(self, get_tasks_fn: Callable[[], dict[str, BackgroundTask]]) -> None:
        """Start the background sync thread.

        Args:
            get_tasks_fn: Function that returns current tasks dict (called under lock by caller).
        """
        if self._sync_thread is not None:
            return

        self._get_tasks_fn = get_tasks_fn
        self._stop_sync.clear()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.debug(f"Task sync thread started (interval: {self.sync_interval}s)")

    def stop_sync_thread(self) -> None:
        """Stop the background sync thread."""
        if self._sync_thread is None:
            return

        self._stop_sync.set()
        self._sync_thread.join(timeout=self.sync_interval + 1)
        self._sync_thread = None
        logger.debug("Task sync thread stopped")

    def _sync_loop(self) -> None:
        """Background loop that periodically saves task state."""
        while not self._stop_sync.wait(timeout=self.sync_interval):
            try:
                tasks = self._get_tasks_fn()
                self.save(tasks)
            except Exception as e:
                logger.warning(f"Task sync failed: {e}")

    def save(self, tasks: dict[str, BackgroundTask]) -> None:
        """Save tasks to the state file.

        Args:
            tasks: Dictionary of task_id -> BackgroundTask
        """
        state = {
            "server_instance_id": self.server_instance_id,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "tasks": {task_id: task.to_persistence_dict() for task_id, task in tasks.items()},
        }

        with self._file_lock:
            try:
                # Write to temp file first, then rename for atomicity
                temp_path = self.path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    # Use flock for advisory locking
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(state, f, indent=2)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                # Atomic rename
                temp_path.rename(self.path)
                logger.debug(f"Saved {len(tasks)} tasks to {self.path}")

            except Exception as e:
                logger.error(f"Failed to save task state: {e}")
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()

    def load(self) -> tuple[dict[str, BackgroundTask], dict[str, list[str]]]:
        """Load tasks from the state file.

        Returns:
            Tuple of (tasks dict, user_tasks dict)
        """
        tasks: dict[str, BackgroundTask] = {}
        user_tasks: dict[str, list[str]] = {}

        if not self.path.exists():
            logger.debug(f"No task state file found at {self.path}")
            return tasks, user_tasks

        try:
            with self._file_lock:
                with open(self.path) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        state = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            old_instance_id = state.get("server_instance_id", "")
            tasks_data = state.get("tasks", {})

            for task_id, task_data in tasks_data.items():
                try:
                    task = BackgroundTask.from_persistence_dict(task_data)

                    # Mark running tasks from previous instances as failed
                    if task.status == TaskStatus.RUNNING:
                        logger.warning(f"Recovering crashed task {task_id} (was running)")
                        task.status = TaskStatus.FAILED
                        task.error = "Server crashed or restarted while task was running"
                        task.completed_at = datetime.now(timezone.utc)
                        task.progress.message = "Failed: Server crash"

                    tasks[task_id] = task

                    # Build user_tasks index
                    if task.user_id not in user_tasks:
                        user_tasks[task.user_id] = []
                    user_tasks[task.user_id].append(task_id)

                except Exception as e:
                    logger.warning(f"Failed to load task {task_id}: {e}")

            logger.info(
                f"Loaded {len(tasks)} tasks from {self.path} "
                f"(previous instance: {old_instance_id[:8]}...)"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted task state file: {e}")
        except Exception as e:
            logger.error(f"Failed to load task state: {e}")

        return tasks, user_tasks

    def cleanup_old_tasks(
        self, tasks: dict[str, BackgroundTask], user_tasks: dict[str, list[str]]
    ) -> int:
        """Remove completed tasks older than retention period.

        Args:
            tasks: Dictionary of task_id -> BackgroundTask (modified in place)
            user_tasks: Dictionary of user_id -> [task_ids] (modified in place)

        Returns:
            Number of tasks removed
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        removed = 0

        # Find tasks to remove
        to_remove = []
        for task_id, task in tasks.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                task_time = task.completed_at or task.created_at
                if task_time < cutoff:
                    to_remove.append(task_id)

        # Remove them
        for task_id in to_remove:
            task = tasks.pop(task_id)
            if task.user_id in user_tasks:
                try:
                    user_tasks[task.user_id].remove(task_id)
                except ValueError:
                    pass
            removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} old tasks (older than {self.retention_hours}h)")

        return removed


class BackgroundTaskManager:
    """Manages background tasks for long-running operations.

    This class is designed for multi-user environments:
    - Tasks are isolated by user_id
    - Thread-safe operations
    - Configurable max workers per user and globally
    - Persistent storage with periodic file backup for crash recovery
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_tasks_per_user: int = 10,
        persistence_path: Path | str | None = None,
        sync_interval: int = TaskPersistence.DEFAULT_SYNC_INTERVAL,
        retention_hours: int = TaskPersistence.DEFAULT_RETENTION_HOURS,
    ) -> None:
        """Initialize the task manager.

        Args:
            max_workers: Maximum concurrent tasks across all users.
            max_tasks_per_user: Maximum tasks (including history) per user.
            persistence_path: Path to task state file for persistence. None to disable.
            sync_interval: Seconds between automatic saves to disk.
            retention_hours: Hours to retain completed tasks.
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, BackgroundTask] = {}  # task_id -> task
        self._user_tasks: dict[str, list[str]] = {}  # user_id -> [task_ids]
        self._lock = threading.RLock()
        self._max_tasks_per_user = max_tasks_per_user

        # Initialize persistence
        self._persistence: TaskPersistence | None = None
        if persistence_path is not False:  # False explicitly disables, None uses default
            self._persistence = TaskPersistence(
                path=persistence_path,
                sync_interval=sync_interval,
                retention_hours=retention_hours,
            )
            # Load existing tasks from disk
            self._tasks, self._user_tasks = self._persistence.load()
            # Clean up old tasks
            self._persistence.cleanup_old_tasks(self._tasks, self._user_tasks)
            # Start background sync thread
            self._persistence.start_sync_thread(self._get_tasks_for_persistence)

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return str(uuid.uuid4())[:12]

    def _get_tasks_for_persistence(self) -> dict[str, BackgroundTask]:
        """Get a copy of tasks dict for persistence (thread-safe)."""
        with self._lock:
            return dict(self._tasks)

    def _cleanup_old_tasks(self, user_id: str) -> None:
        """Remove oldest completed/failed/cancelled tasks if user exceeds limit.

        Must be called with lock held.
        """
        if user_id not in self._user_tasks:
            return

        user_task_ids = self._user_tasks[user_id]
        if len(user_task_ids) <= self._max_tasks_per_user:
            return

        # Find completed tasks to remove (oldest first)
        removable = []
        for task_id in user_task_ids:
            task = self._tasks.get(task_id)
            if task and task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ):
                removable.append((task_id, task.completed_at or task.created_at))

        # Sort by completion time and remove oldest
        removable.sort(key=lambda x: x[1])
        to_remove = len(user_task_ids) - self._max_tasks_per_user

        for task_id, _ in removable[:to_remove]:
            del self._tasks[task_id]
            user_task_ids.remove(task_id)
            logger.debug(f"Cleaned up old task {task_id} for user {user_id}")

    def create_task(
        self,
        user_id: str,
        task_type: TaskType,
        task_fn: Callable[..., Any],
        parameters: dict[str, Any],
        progress_callback: Callable[[TaskProgress], None] | None = None,
    ) -> BackgroundTask:
        """Create and start a new background task.

        Args:
            user_id: Identifier for the user (for isolation).
            task_type: Type of task being created.
            task_fn: Function to execute in background.
            parameters: Parameters to pass to task_fn and store for reference.
            progress_callback: Optional callback for progress updates.

        Returns:
            The created BackgroundTask.
        """
        with self._lock:
            task_id = self._generate_task_id()

            task = BackgroundTask(
                task_id=task_id,
                user_id=user_id,
                task_type=task_type,
                status=TaskStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                parameters=parameters,
            )

            # Register task
            self._tasks[task_id] = task
            if user_id not in self._user_tasks:
                self._user_tasks[user_id] = []
            self._user_tasks[user_id].append(task_id)

            # Cleanup old tasks if needed
            self._cleanup_old_tasks(user_id)

            # Create wrapper that handles status updates
            def task_wrapper() -> Any:
                try:
                    with self._lock:
                        task.status = TaskStatus.RUNNING
                        task.started_at = datetime.now(timezone.utc)
                        task.progress.message = "Starting..."

                    # Create progress updater with minimal lock duration
                    # We use a simple assignment which is atomic for object references
                    def update_progress(progress: TaskProgress) -> None:
                        # Create a copy of the progress to avoid reference issues
                        progress_copy = TaskProgress(
                            current_step=progress.current_step,
                            total_steps=progress.total_steps,
                            current_step_number=progress.current_step_number,
                            percent_complete=progress.percent_complete,
                            message=progress.message,
                            updated_at=progress.updated_at,
                        )
                        # Atomic assignment - no lock needed for single reference update
                        # This is safe because we're replacing the entire object
                        task.progress = progress_copy
                        if progress_callback:
                            progress_callback(progress)

                    # Execute the task
                    result = task_fn(progress_updater=update_progress, **parameters)

                    with self._lock:
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now(timezone.utc)
                        task.result = result
                        task.progress.message = "Completed"
                        task.progress.percent_complete = 100.0

                    logger.info(f"Task {task_id} completed successfully")
                    return result

                except Exception as e:
                    with self._lock:
                        task.status = TaskStatus.FAILED
                        task.completed_at = datetime.now(timezone.utc)
                        task.error = str(e)
                        task.progress.message = f"Failed: {e}"

                    logger.error(f"Task {task_id} failed: {e}")
                    raise

            # Submit to executor
            future = self._executor.submit(task_wrapper)
            task._future = future

            logger.info(f"Created task {task_id} for user {user_id}: {task_type.value}")
            return task

    def get_task(self, task_id: str, user_id: str) -> BackgroundTask | None:
        """Get a task by ID, validating user ownership.

        Args:
            task_id: The task ID to retrieve.
            user_id: The user requesting the task (for access control).

        Returns:
            The task if found and owned by user, None otherwise.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.user_id == user_id:
                return task
            return None

    def get_task_snapshot(self, task_id: str, user_id: str) -> dict[str, Any] | None:
        """Get a snapshot of task state without holding the lock.

        This is the preferred method for async contexts as it minimizes
        lock contention by returning a copy of the task state.

        Args:
            task_id: The task ID to retrieve.
            user_id: The user requesting the task (for access control).

        Returns:
            Dictionary snapshot of task state, or None if not found/not owned.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.user_id == user_id:
                # Return a deep copy of the task dict to avoid holding lock
                return copy.deepcopy(task.to_dict(include_result=True))
            return None

    async def get_task_snapshot_async(self, task_id: str, user_id: str) -> dict[str, Any] | None:
        """Async version of get_task_snapshot.

        Runs the snapshot operation in a thread pool to avoid blocking
        the asyncio event loop.

        Args:
            task_id: The task ID to retrieve.
            user_id: The user requesting the task (for access control).

        Returns:
            Dictionary snapshot of task state, or None if not found/not owned.
        """
        return await asyncio.to_thread(self.get_task_snapshot, task_id, user_id)

    def list_tasks(
        self,
        user_id: str,
        status_filter: TaskStatus | None = None,
        task_type_filter: TaskType | None = None,
    ) -> list[BackgroundTask]:
        """List all tasks for a user.

        Args:
            user_id: The user whose tasks to list.
            status_filter: Optional filter by status.
            task_type_filter: Optional filter by task type.

        Returns:
            List of tasks matching criteria.
        """
        with self._lock:
            task_ids = self._user_tasks.get(user_id, [])
            tasks = []
            for task_id in task_ids:
                task = self._tasks.get(task_id)
                if task:
                    if status_filter and task.status != status_filter:
                        continue
                    if task_type_filter and task.task_type != task_type_filter:
                        continue
                    tasks.append(task)
            return tasks

    def list_tasks_snapshots(
        self,
        user_id: str,
        status_filter: TaskStatus | None = None,
        task_type_filter: TaskType | None = None,
        include_result: bool = False,
    ) -> list[dict[str, Any]]:
        """List task snapshots without holding locks during serialization.

        This is the preferred method for async contexts.

        Args:
            user_id: The user whose tasks to list.
            status_filter: Optional filter by status.
            task_type_filter: Optional filter by task type.
            include_result: If True, include full results in snapshots.

        Returns:
            List of task state dictionaries.
        """
        with self._lock:
            task_ids = self._user_tasks.get(user_id, [])
            snapshots = []
            for task_id in task_ids:
                task = self._tasks.get(task_id)
                if task:
                    if status_filter and task.status != status_filter:
                        continue
                    if task_type_filter and task.task_type != task_type_filter:
                        continue
                    # Copy within lock to get consistent state
                    snapshots.append(copy.deepcopy(task.to_dict(include_result=include_result)))
            return snapshots

    async def list_tasks_snapshots_async(
        self,
        user_id: str,
        status_filter: TaskStatus | None = None,
        task_type_filter: TaskType | None = None,
        include_result: bool = False,
    ) -> list[dict[str, Any]]:
        """Async version of list_tasks_snapshots.

        Args:
            user_id: The user whose tasks to list.
            status_filter: Optional filter by status.
            task_type_filter: Optional filter by task type.
            include_result: If True, include full results in snapshots.

        Returns:
            List of task state dictionaries.
        """
        return await asyncio.to_thread(
            self.list_tasks_snapshots,
            user_id,
            status_filter,
            task_type_filter,
            include_result,
        )

    def cancel_task(self, task_id: str, user_id: str) -> bool:
        """Cancel a pending or running task.

        Args:
            task_id: The task ID to cancel.
            user_id: The user requesting cancellation (for access control).

        Returns:
            True if task was cancelled, False if not found or already completed.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.user_id != user_id:
                return False

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False

            # Try to cancel the future
            if task._future and not task._future.done():
                task._future.cancel()

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)
            task.progress.message = "Cancelled by user"

            logger.info(f"Task {task_id} cancelled by user {user_id}")
            return True

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the task manager.

        Args:
            wait: If True, wait for running tasks to complete.
        """
        logger.info("Shutting down task manager")

        # Stop persistence sync thread and save final state
        if self._persistence:
            self._persistence.stop_sync_thread()
            # Final save with current state
            with self._lock:
                self._persistence.save(self._tasks)
            logger.info("Task state saved to disk")

        self._executor.shutdown(wait=wait)


# Global task manager instance
_task_manager: BackgroundTaskManager | None = None

# Global configuration for task manager (set via init_task_manager)
_task_manager_config: dict[str, Any] = {}


def init_task_manager(
    persistence_path: Path | str | None = None,
    sync_interval: int = TaskPersistence.DEFAULT_SYNC_INTERVAL,
    retention_hours: int = TaskPersistence.DEFAULT_RETENTION_HOURS,
    max_workers: int = 4,
    max_tasks_per_user: int = 10,
) -> BackgroundTaskManager:
    """Initialize the global task manager with configuration.

    This should be called once at server startup to configure persistence
    settings. If called multiple times, subsequent calls are ignored.

    Args:
        persistence_path: Path to task state file. None for default path.
        sync_interval: Seconds between automatic saves to disk.
        retention_hours: Hours to retain completed tasks.
        max_workers: Maximum concurrent tasks across all users.
        max_tasks_per_user: Maximum tasks (including history) per user.

    Returns:
        The initialized BackgroundTaskManager instance.
    """
    global _task_manager, _task_manager_config

    if _task_manager is not None:
        logger.debug("Task manager already initialized, ignoring reinit")
        return _task_manager

    _task_manager_config = {
        "persistence_path": persistence_path,
        "sync_interval": sync_interval,
        "retention_hours": retention_hours,
        "max_workers": max_workers,
        "max_tasks_per_user": max_tasks_per_user,
    }

    _task_manager = BackgroundTaskManager(**_task_manager_config)
    logger.info(f"Task manager initialized (persistence: {persistence_path or 'default'})")
    return _task_manager


def get_task_manager() -> BackgroundTaskManager:
    """Get the global task manager instance, creating it if needed.

    If init_task_manager() was not called, creates with default settings.
    """
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager
