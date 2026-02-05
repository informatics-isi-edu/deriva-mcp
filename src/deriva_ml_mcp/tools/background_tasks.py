"""Background task management tools for long-running operations.

This module provides MCP tools for managing background tasks such as
catalog cloning. These tools allow users to:
- Start long-running operations asynchronously
- Check task status and progress
- List their active and completed tasks
- Cancel pending/running tasks

The task system is multi-user safe - each user can only see and manage
their own tasks.

Performance notes:
- All MCP tools are truly async, using asyncio.to_thread() for blocking operations
- Task status queries use snapshot methods to minimize lock contention
- User ID is consistently determined to avoid task lookup mismatches
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from deriva.core import get_credential

from deriva_ml_mcp.tasks import (
    TaskProgress,
    TaskStatus,
    TaskType,
    get_task_manager,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-mcp")

# Cache for user ID to ensure consistency within a session
_cached_user_id: str | None = None


def _get_user_id(hostname: str | None = None) -> str:
    """Get a user identifier from credentials.

    For multi-user isolation, we use the credential identity as user_id.
    If no credentials available, fall back to a default (single-user mode).

    IMPORTANT: This function now caches the user ID to ensure consistency
    between task creation and task lookup. The first call with a hostname
    sets the cached value which is then used for all subsequent calls.

    Args:
        hostname: Optional hostname to get credentials for.

    Returns:
        A string identifying the user.
    """
    global _cached_user_id

    # If we have a cached user ID, always use it for consistency
    if _cached_user_id is not None:
        return _cached_user_id

    try:
        if hostname:
            cred = get_credential(hostname)
            if cred and "cookie" in cred:
                # Extract webauthn from cookie for user identity
                cookie = cred.get("cookie", "")
                if "webauthn=" in cookie:
                    # Use a hash of the webauthn value for privacy
                    import hashlib

                    webauthn = cookie.split("webauthn=")[1].split(";")[0]
                    user_id = hashlib.sha256(webauthn.encode()).hexdigest()[:16]
                    _cached_user_id = user_id
                    logger.debug(f"Cached user ID from credentials: {user_id[:8]}...")
                    return user_id
        # Fall back to checking any available credential
        # In single-user mode, this is fine
        _cached_user_id = "default_user"
        logger.debug("Using default_user for single-user mode")
        return "default_user"
    except Exception:
        _cached_user_id = "default_user"
        return "default_user"


async def _get_user_id_async(hostname: str | None = None) -> str:
    """Async version of _get_user_id.

    Runs credential lookup in a thread to avoid blocking the event loop.
    """
    return await asyncio.to_thread(_get_user_id, hostname)


def _clone_catalog_task(
    progress_updater: Any,
    source_hostname: str,
    source_catalog_id: str,
    root_rid: str | None = None,
    dest_hostname: str | None = None,
    alias: str | None = None,
    add_ml_schema: bool = False,
    schema_only: bool = False,
    asset_mode: str = "refs",
    copy_annotations: bool = True,
    copy_policy: bool = True,
    exclude_schemas: list[str] | None = None,
    exclude_objects: list[str] | None = None,
    reinitialize_dataset_versions: bool = True,
    orphan_strategy: str = "fail",
    prune_hidden_fkeys: bool = False,
    truncate_oversized: bool = False,
    include_tables: list[str] | None = None,
    include_associations: bool = True,
    include_vocabularies: bool = True,
    use_export_annotation: bool = False,
) -> dict[str, Any]:
    """Execute catalog clone operation with progress tracking.

    This function is called by the task manager in a background thread.
    """
    from deriva_ml.catalog import AssetCopyMode, OrphanStrategy

    # Update progress
    progress = TaskProgress(
        current_step="Initializing clone operation",
        total_steps=4,
        current_step_number=1,
        percent_complete=5.0,
        message="Preparing to clone catalog...",
    )
    progress_updater(progress)

    # Convert string parameters to enums
    asset_mode_enum = AssetCopyMode(asset_mode)
    orphan_strategy_enum = OrphanStrategy(orphan_strategy)

    # Update progress
    progress.current_step = "Connecting to source catalog"
    progress.current_step_number = 2
    progress.percent_complete = 10.0
    progress.message = f"Connecting to {source_hostname}..."
    progress_updater(progress)

    # Determine if this is a partial or full clone
    if root_rid:
        from deriva_ml.catalog import clone_subset_catalog as do_clone

        progress.message = f"Starting partial clone from RID {root_rid}..."
        progress_updater(progress)

        result = do_clone(
            source_hostname=source_hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            include_tables=include_tables,
            exclude_objects=exclude_objects,
            exclude_schemas=exclude_schemas,
            include_associations=include_associations,
            include_vocabularies=include_vocabularies,
            use_export_annotation=use_export_annotation,
            dest_hostname=dest_hostname,
            alias=alias,
            add_ml_schema=add_ml_schema,
            asset_mode=asset_mode_enum,
            copy_annotations=copy_annotations,
            copy_policy=copy_policy,
            orphan_strategy=orphan_strategy_enum,
            prune_hidden_fkeys=prune_hidden_fkeys,
            truncate_oversized=truncate_oversized,
            reinitialize_dataset_versions=reinitialize_dataset_versions,
        )
        clone_mode = "partial"
    else:
        from deriva_ml.catalog import clone_catalog as do_clone

        progress.message = "Starting full catalog clone..."
        progress_updater(progress)

        result = do_clone(
            source_hostname=source_hostname,
            source_catalog_id=source_catalog_id,
            dest_hostname=dest_hostname,
            alias=alias,
            add_ml_schema=add_ml_schema,
            schema_only=schema_only,
            asset_mode=asset_mode_enum,
            copy_annotations=copy_annotations,
            copy_policy=copy_policy,
            exclude_schemas=exclude_schemas,
            exclude_objects=exclude_objects,
            reinitialize_dataset_versions=reinitialize_dataset_versions,
            orphan_strategy=orphan_strategy_enum,
            prune_hidden_fkeys=prune_hidden_fkeys,
            truncate_oversized=truncate_oversized,
        )
        clone_mode = "full"

    # Update progress - finalizing
    progress.current_step = "Finalizing"
    progress.current_step_number = 4
    progress.percent_complete = 95.0
    progress.message = "Building result..."
    progress_updater(progress)

    # Build response from CloneCatalogResult
    response: dict[str, Any] = {
        "status": "cloned",
        "clone_mode": clone_mode,
        "source_hostname": source_hostname,
        "source_catalog_id": source_catalog_id,
        "dest_hostname": result.hostname,
        "dest_catalog_id": result.catalog_id,
        "schema_only": schema_only,
        "asset_mode": asset_mode,
    }

    if root_rid:
        response["root_rid"] = root_rid
    if result.source_snapshot:
        response["source_snapshot"] = result.source_snapshot
    if alias:
        response["alias"] = alias
    if result.datasets_reinitialized:
        response["datasets_reinitialized"] = result.datasets_reinitialized
    if result.ml_schema_added:
        response["ml_schema_added"] = result.ml_schema_added

    # Include stats from report
    if result.report:
        response["orphan_rows_removed"] = result.report.summary.orphan_rows_removed
        response["orphan_rows_nullified"] = result.report.summary.orphan_rows_nullified
        response["fkeys_pruned"] = result.report.summary.fkeys_pruned
        response["rows_skipped"] = (
            result.report.summary.rows_skipped if hasattr(result.report.summary, "rows_skipped") else 0
        )
        if result.truncated_values:
            response["truncated_values_count"] = len(result.truncated_values)
        # Include detailed report
        response["report"] = {
            "summary": {
                "total_issues": result.report.summary.total_issues,
                "errors": result.report.summary.errors,
                "warnings": result.report.summary.warnings,
                "tables_restored": result.report.summary.tables_restored,
                "tables_failed": result.report.summary.tables_failed,
                "tables_skipped": result.report.summary.tables_skipped,
                "total_rows_restored": result.report.summary.total_rows_restored,
                "orphan_rows_removed": result.report.summary.orphan_rows_removed,
                "orphan_rows_nullified": result.report.summary.orphan_rows_nullified,
                "fkeys_applied": result.report.summary.fkeys_applied,
                "fkeys_failed": result.report.summary.fkeys_failed,
                "fkeys_pruned": result.report.summary.fkeys_pruned,
            },
            "issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category.value,
                    "message": issue.message,
                    "table": issue.table,
                    "details": issue.details,
                    "action": issue.action,
                    "row_count": issue.row_count,
                }
                for issue in result.report.issues
            ],
            "tables_restored": result.report.tables_restored,
            "tables_failed": result.report.tables_failed,
            "tables_skipped": result.report.tables_skipped,
            "orphan_details": result.report.orphan_details,
        }
        response["clone_type"] = "cross_server" if dest_hostname and dest_hostname != source_hostname else "same_server"
        response["message"] = (
            f"Catalog {'subset ' if root_rid else ''}migrated from "
            f"{source_hostname}:{source_catalog_id} to {result.hostname}:{result.catalog_id}"
        )
        response["report_summary"] = result.report.to_text()

    return response


def register_background_task_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register background task management tools with the MCP server."""

    @mcp.tool()
    async def clone_catalog_async(
        source_hostname: str,
        source_catalog_id: str,
        root_rid: str | None = None,
        dest_hostname: str | None = None,
        alias: str | None = None,
        add_ml_schema: bool = False,
        schema_only: bool = False,
        asset_mode: str = "refs",
        copy_annotations: bool = True,
        copy_policy: bool = True,
        exclude_schemas: list[str] | None = None,
        exclude_objects: list[str] | None = None,
        reinitialize_dataset_versions: bool = True,
        orphan_strategy: str = "fail",
        prune_hidden_fkeys: bool = False,
        truncate_oversized: bool = False,
        include_tables: list[str] | None = None,
        include_associations: bool = True,
        include_vocabularies: bool = True,
        use_export_annotation: bool = False,
    ) -> str:
        """Start a catalog clone operation in the background.

        This is the async version of clone_catalog - it starts the clone operation
        and immediately returns a task_id that you can use to check progress.

        Use this for large catalogs or cross-server clones that may take several
        minutes to complete. Check progress with `get_task_status(task_id)`.

        **Full clone** (root_rid=None):
        Creates a complete clone of the source catalog.

        **Partial clone** (root_rid provided):
        Creates a subset clone containing only data reachable from the root RID.

        Args:
            source_hostname: Source server hostname (e.g., "www.facebase.org").
            source_catalog_id: ID of the catalog to clone.
            root_rid: Optional RID for partial clone (e.g., "3-HXMC").
            dest_hostname: Destination hostname. If None, uses source hostname.
            alias: Optional alias name for the new catalog.
            add_ml_schema: If True, add the DerivaML schema to the clone.
            schema_only: If True, copy only schema structure without data.
            asset_mode: How to handle assets: "none", "refs" (default), or "full".
            copy_annotations: If True (default), copy all annotations.
            copy_policy: If True (default), copy ACL policies.
            exclude_schemas: Schemas to exclude from cloning.
            exclude_objects: Tables ("schema:table") to exclude.
            reinitialize_dataset_versions: If True, increment dataset versions.
            orphan_strategy: How to handle orphans: "fail", "delete", "nullify".
            prune_hidden_fkeys: Skip FKs with hidden reference data.
            truncate_oversized: Truncate values exceeding index limits.
            include_tables: (Partial) Additional starting tables.
            include_associations: (Partial) Include association tables.
            include_vocabularies: (Partial) Include vocabulary tables.
            use_export_annotation: (Partial) Use export annotation.

        Returns:
            JSON with task_id and status. Use get_task_status(task_id) to check progress.

        Example:
            # Start async clone
            clone_catalog_async("www.facebase.org", "1",
                               root_rid="3-HXMC",
                               dest_hostname="localhost",
                               alias="facebase-test")
            -> {"task_id": "abc123", "status": "pending", ...}

            # Check progress
            get_task_status("abc123")
            -> {"status": "running", "progress": {"percent_complete": 45.0, ...}}

            # When done
            get_task_status("abc123")
            -> {"status": "completed", "result": {...}}
        """
        try:
            task_manager = get_task_manager()
            # Use async credential lookup to avoid blocking
            user_id = await _get_user_id_async(source_hostname)

            # Store parameters for the task
            parameters = {
                "source_hostname": source_hostname,
                "source_catalog_id": source_catalog_id,
                "root_rid": root_rid,
                "dest_hostname": dest_hostname,
                "alias": alias,
                "add_ml_schema": add_ml_schema,
                "schema_only": schema_only,
                "asset_mode": asset_mode,
                "copy_annotations": copy_annotations,
                "copy_policy": copy_policy,
                "exclude_schemas": exclude_schemas,
                "exclude_objects": exclude_objects,
                "reinitialize_dataset_versions": reinitialize_dataset_versions,
                "orphan_strategy": orphan_strategy,
                "prune_hidden_fkeys": prune_hidden_fkeys,
                "truncate_oversized": truncate_oversized,
                "include_tables": include_tables,
                "include_associations": include_associations,
                "include_vocabularies": include_vocabularies,
                "use_export_annotation": use_export_annotation,
            }

            # Create the background task
            task = task_manager.create_task(
                user_id=user_id,
                task_type=TaskType.CLONE_CATALOG,
                task_fn=_clone_catalog_task,
                parameters=parameters,
            )

            return json.dumps(
                {
                    "status": "started",
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "message": (f"Clone operation started. Use get_task_status('{task.task_id}') to check progress."),
                    "parameters": {
                        "source": f"{source_hostname}:{source_catalog_id}",
                        "dest": dest_hostname or source_hostname,
                        "root_rid": root_rid,
                        "alias": alias,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Failed to start clone task: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def get_task_status(
        task_id: str,
        include_result: bool = True,
    ) -> str:
        """Get the status and progress of a background task.

        Args:
            task_id: The task ID returned by an async operation.
            include_result: If True, include the full result when completed.

        Returns:
            JSON with task status, progress, and optionally the result.

        Example:
            get_task_status("abc123")
            -> {
                "task_id": "abc123",
                "status": "running",
                "progress": {
                    "current_step": "Copying data",
                    "percent_complete": 45.0,
                    "message": "Copying table Subject..."
                }
            }
        """
        try:
            task_manager = get_task_manager()
            # Use cached user_id for consistency with task creation
            user_id = await _get_user_id_async()

            # Use async snapshot method to avoid blocking event loop and minimize lock contention
            task_snapshot = await task_manager.get_task_snapshot_async(task_id, user_id)
            if not task_snapshot:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Task {task_id} not found or access denied",
                    }
                )

            # If include_result is False, remove result from snapshot
            if not include_result and "result" in task_snapshot:
                del task_snapshot["result"]

            return json.dumps(task_snapshot)

        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def list_tasks(
        status: str | None = None,
        task_type: str | None = None,
    ) -> str:
        """List all background tasks for the current user.

        Args:
            status: Filter by status: "pending", "running", "completed", "failed", "cancelled".
            task_type: Filter by type: "clone_catalog".

        Returns:
            JSON list of tasks with their status and basic info.

        Example:
            list_tasks(status="running")
            -> [{"task_id": "abc123", "status": "running", ...}]
        """
        try:
            task_manager = get_task_manager()
            # Use cached user_id for consistency
            user_id = await _get_user_id_async()

            # Parse filters
            status_filter = TaskStatus(status) if status else None
            type_filter = TaskType(task_type) if task_type else None

            # Use async snapshot method to avoid blocking and minimize lock contention
            task_snapshots = await task_manager.list_tasks_snapshots_async(
                user_id=user_id,
                status_filter=status_filter,
                task_type_filter=type_filter,
                include_result=False,
            )

            return json.dumps(task_snapshots)

        except ValueError as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Invalid filter value: {e}",
                }
            )
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def cancel_task(task_id: str) -> str:
        """Cancel a pending or running background task.

        Args:
            task_id: The task ID to cancel.

        Returns:
            JSON with cancellation status.

        Note: Cancellation is best-effort. Long-running operations may not
        stop immediately.
        """
        try:
            task_manager = get_task_manager()
            # Use cached user_id for consistency
            user_id = await _get_user_id_async()

            # Run cancel in thread to avoid blocking
            cancelled = await asyncio.to_thread(task_manager.cancel_task, task_id, user_id)

            if cancelled:
                return json.dumps(
                    {
                        "status": "cancelled",
                        "task_id": task_id,
                        "message": "Task cancellation requested",
                    }
                )
            else:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Task {task_id} not found, access denied, or already completed",
                    }
                )

        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )
