"""Background task management for long-running operations.

This module provides a multi-user safe task management system for operations
that may take longer than MCP connection timeouts (e.g., catalog cloning).

Design considerations for multi-user environments:
- Tasks are isolated by user_id (derived from credentials or session)
- Each user can only see and manage their own tasks
- Task state is stored in memory (lost on server restart)
- Thread-safe operations using locks with minimal contention

Performance considerations:
- Lock contention minimized using copy-on-read pattern for status queries
- Async-safe methods provided for use in asyncio contexts
- Task state snapshots avoid holding locks during I/O operations
"""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
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


class BackgroundTaskManager:
    """Manages background tasks for long-running operations.

    This class is designed for multi-user environments:
    - Tasks are isolated by user_id
    - Thread-safe operations
    - Configurable max workers per user and globally
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_tasks_per_user: int = 10,
    ) -> None:
        """Initialize the task manager.

        Args:
            max_workers: Maximum concurrent tasks across all users.
            max_tasks_per_user: Maximum tasks (including history) per user.
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, BackgroundTask] = {}  # task_id -> task
        self._user_tasks: dict[str, list[str]] = {}  # user_id -> [task_ids]
        self._lock = threading.RLock()
        self._max_tasks_per_user = max_tasks_per_user

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return str(uuid.uuid4())[:12]

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
        self._executor.shutdown(wait=wait)


# Global task manager instance
_task_manager: BackgroundTaskManager | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Get the global task manager instance, creating it if needed."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager
