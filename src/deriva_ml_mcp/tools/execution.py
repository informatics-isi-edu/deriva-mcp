"""Execution management tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-ml-mcp")

# Store active executions by connection key
_active_executions: dict[str, Any] = {}


def register_execution_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register execution management tools with the MCP server."""

    def _get_execution_key() -> str:
        """Get a key for the current active connection."""
        ml = conn_manager.get_active_or_raise()
        return f"{ml.host_name}:{ml.catalog_id}"

    @mcp.tool()
    async def create_execution(
        workflow_name: str,
        workflow_type: str,
        description: str = "",
        dataset_rids: list[str] | None = None,
        asset_rids: list[str] | None = None,
    ) -> str:
        """Create a new execution for running ML workflows.

        Creates an execution record that tracks inputs, outputs, and provenance
        for a computational or manual workflow.

        IMPORTANT: After completing your work, you MUST call upload_execution_outputs()
        to upload any registered assets to the catalog. The execution workflow is:
        1. create_execution() - Create the execution record
        2. start_execution() - Mark execution as running
        3. asset_file_path() - Register files for upload (can be called multiple times)
        4. stop_execution() - Mark execution as complete
        5. upload_execution_outputs() - Upload all registered assets to catalog

        Args:
            workflow_name: Name of the workflow to execute.
            workflow_type: Type of workflow (must exist in Workflow_Type vocabulary).
            description: Description of what this execution does.
            dataset_rids: Optional list of dataset RIDs to use as inputs.
            asset_rids: Optional list of asset RIDs to use as inputs.

        Returns:
            JSON object with execution RID and details.
        """
        try:
            from deriva_ml.execution.execution_configuration import ExecutionConfiguration
            from deriva_ml.dataset.aux_classes import DatasetSpec

            ml = conn_manager.get_active_or_raise()

            # Create workflow
            workflow = ml.create_workflow(
                name=workflow_name,
                workflow_type=workflow_type,
                description=description,
            )

            # Build dataset specs
            datasets = []
            if dataset_rids:
                for rid in dataset_rids:
                    datasets.append(DatasetSpec(rid=rid))

            # Create configuration
            config = ExecutionConfiguration(
                workflow=workflow,
                description=description,
                datasets=datasets,
                assets=asset_rids or [],
            )

            # Create execution
            execution = ml.create_execution(config)

            # Store for later use
            key = _get_execution_key()
            _active_executions[key] = execution

            return json.dumps({
                "status": "created",
                "execution_rid": execution.execution_rid,
                "workflow_rid": execution.workflow_rid,
                "description": description,
                "dataset_count": len(datasets),
                "asset_count": len(asset_rids or []),
            })
        except Exception as e:
            logger.error(f"Failed to create execution: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def start_execution() -> str:
        """Start the active execution.

        Marks the execution as running and begins tracking time.
        Must have created an execution first with create_execution.

        Returns:
            JSON object with execution status.
        """
        try:
            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution. Use create_execution first.",
                })

            execution = _active_executions[key]
            execution.execution_start()

            return json.dumps({
                "status": "started",
                "execution_rid": execution.execution_rid,
            })
        except Exception as e:
            logger.error(f"Failed to start execution: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def stop_execution() -> str:
        """Stop the active execution.

        Marks the execution as completed and records duration.

        Returns:
            JSON object with execution status and duration.
        """
        try:
            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

            execution = _active_executions[key]
            execution.execution_stop()

            return json.dumps({
                "status": "completed",
                "execution_rid": execution.execution_rid,
            })
        except Exception as e:
            logger.error(f"Failed to stop execution: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def update_execution_status(status: str, message: str) -> str:
        """Update the status of the active execution.

        Records a status update for tracking progress.

        Args:
            status: Status value (pending, running, completed, failed).
            message: Description of current state or progress.

        Returns:
            JSON object confirming the update.
        """
        try:
            from deriva_ml.core.definitions import Status

            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

            status_map = {
                "pending": Status.pending,
                "running": Status.running,
                "completed": Status.completed,
                "failed": Status.failed,
                "initializing": Status.initializing,
                "created": Status.created,
            }

            status_enum = status_map.get(status.lower(), Status.running)
            execution = _active_executions[key]
            execution.update_status(status_enum, message)

            return json.dumps({
                "status": "updated",
                "execution_rid": execution.execution_rid,
                "new_status": status,
                "message": message,
            })
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_execution_info() -> str:
        """Get information about the active execution.

        Returns details about the currently active execution including
        status, datasets, registered assets, and whether upload is pending.

        Returns:
            JSON object with execution details.
        """
        try:
            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "no_active_execution",
                    "message": "No active execution. Use create_execution first.",
                })

            execution = _active_executions[key]

            # Check if there are registered but not yet uploaded assets
            has_pending_uploads = execution.uploaded_assets is None

            return json.dumps({
                "execution_rid": execution.execution_rid,
                "workflow_rid": execution.workflow_rid,
                "status": execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
                "dataset_rids": execution.dataset_rids,
                "dataset_count": len(execution.datasets),
                "working_dir": str(execution.working_dir),
                "upload_pending": has_pending_uploads,
                "upload_reminder": "Remember to call upload_execution_outputs() when done." if has_pending_uploads else None,
            })
        except Exception as e:
            logger.error(f"Failed to get execution info: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def restore_execution(execution_rid: str) -> str:
        """Restore a previous execution.

        Reloads an existing execution by its RID, restoring the execution
        context and downloaded datasets.

        Args:
            execution_rid: RID of the execution to restore.

        Returns:
            JSON object with restored execution details.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            execution = ml.restore_execution(execution_rid)

            # Store for later use
            key = _get_execution_key()
            _active_executions[key] = execution

            return json.dumps({
                "status": "restored",
                "execution_rid": execution.execution_rid,
                "workflow_rid": execution.workflow_rid,
                "dataset_count": len(execution.datasets),
            })
        except Exception as e:
            logger.error(f"Failed to restore execution: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def asset_file_path(
        asset_name: str,
        file_name: str,
        asset_types: list[str] | None = None,
        copy_file: bool = False,
        rename_file: str | None = None,
    ) -> str:
        """Register a file for upload as an execution asset.

        Registers a file to be uploaded when upload_execution_outputs() is called.
        The file will be associated with the specified asset table and types.

        This is the primary way to add output files to an execution. Files are
        staged locally and uploaded in batch when upload_execution_outputs() is called.

        Args:
            asset_name: Name of the asset table (e.g., "Image", "Model", "Execution_Metadata").
            file_name: Path to the file to register. Can be:
                - An existing file (will be symlinked or copied)
                - A new path (returned path can be opened for writing)
            asset_types: Asset type terms from Asset_Type vocabulary.
                Defaults to the asset_name if not specified.
            copy_file: If True, copy the file instead of creating a symlink.
            rename_file: If provided, rename the file to this name.

        Returns:
            JSON object with the registered file path and details.
        """
        try:
            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution. Use create_execution first.",
                })

            execution = _active_executions[key]
            asset_path = execution.asset_file_path(
                asset_name=asset_name,
                file_name=file_name,
                asset_types=asset_types,
                copy_file=copy_file,
                rename_file=rename_file,
            )

            return json.dumps({
                "status": "registered",
                "execution_rid": execution.execution_rid,
                "asset_name": asset_name,
                "file_path": str(asset_path),
                "file_name": asset_path.file_name,
                "asset_types": asset_path.asset_types,
                "note": "Call upload_execution_outputs() to upload this file to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to register asset file: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def upload_execution_outputs(clean_folder: bool = True) -> str:
        """Upload all registered assets from the active execution to the catalog.

        IMPORTANT: This must be called after you have finished registering assets
        with asset_file_path(). This uploads all staged files to the catalog
        and records them in the execution's provenance.

        The typical workflow is:
        1. create_execution() - Create execution record
        2. start_execution() - Begin tracking
        3. asset_file_path() - Register output files (repeat as needed)
        4. stop_execution() - Mark complete
        5. upload_execution_outputs() - Upload all files to catalog

        Args:
            clean_folder: Whether to clean up output folders after upload.

        Returns:
            JSON object with upload summary including count of assets by type.
        """
        try:
            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

            execution = _active_executions[key]
            results = execution.upload_execution_outputs(clean_folder=clean_folder)

            # Summarize results
            summary = {}
            for asset_type, paths in results.items():
                summary[asset_type] = len(paths)

            return json.dumps({
                "status": "uploaded",
                "execution_rid": execution.execution_rid,
                "assets_uploaded": summary,
            })
        except Exception as e:
            logger.error(f"Failed to upload outputs: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_executions(limit: int = 50) -> str:
        """List recent executions in the catalog.

        Returns information about recent executions including their
        status, workflow, and timing.

        Args:
            limit: Maximum number of executions to return.

        Returns:
            JSON array of execution records.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            pb = ml.pathBuilder()
            execution_path = pb.schemas[ml.ml_schema].Execution

            executions = list(execution_path.entities().fetch(limit=limit))
            result = []
            for exe in executions:
                result.append({
                    "rid": exe.get("RID"),
                    "workflow": exe.get("Workflow"),
                    "status": exe.get("Status"),
                    "status_detail": exe.get("Status_Detail"),
                    "description": exe.get("Description"),
                    "duration": exe.get("Duration"),
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list executions: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def create_execution_dataset(
        description: str = "",
        dataset_types: list[str] | None = None,
    ) -> str:
        """Create a new dataset within the active execution.

        Creates a dataset that is associated with the current execution
        for provenance tracking.

        Args:
            description: Description of the dataset.
            dataset_types: Dataset type terms from vocabulary.

        Returns:
            JSON object with created dataset details.
        """
        try:
            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

            execution = _active_executions[key]
            dataset = execution.create_dataset(
                description=description,
                dataset_types=dataset_types or [],
            )

            return json.dumps({
                "status": "created",
                "dataset_rid": dataset.dataset_rid,
                "execution_rid": execution.execution_rid,
                "description": description,
                "dataset_types": dataset_types or [],
            })
        except Exception as e:
            logger.error(f"Failed to create execution dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def download_execution_dataset(
        dataset_rid: str,
        version: str | None = None,
        materialize: bool = True,
    ) -> str:
        """Download a dataset for use in the active execution.

        Downloads and materializes a dataset, making it available
        for processing in the execution.

        Args:
            dataset_rid: RID of the dataset to download.
            version: Specific version to download (optional).
            materialize: Whether to fetch all referenced files.

        Returns:
            JSON object with download details and local path.
        """
        try:
            from deriva_ml.dataset.aux_classes import DatasetSpec

            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

            execution = _active_executions[key]
            spec = DatasetSpec(rid=dataset_rid, version=version, materialize=materialize)
            bag = execution.download_dataset_bag(spec)

            return json.dumps({
                "status": "downloaded",
                "dataset_rid": bag.dataset_rid,
                "version": bag.version,
                "path": str(bag.path),
            })
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_execution_working_dir() -> str:
        """Get the working directory for the active execution.

        Returns the path where execution files should be placed
        for upload.

        Returns:
            JSON object with working directory path.
        """
        try:
            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

            execution = _active_executions[key]
            return json.dumps({
                "working_dir": str(execution.working_dir),
                "execution_rid": execution.execution_rid,
            })
        except Exception as e:
            logger.error(f"Failed to get working dir: {e}")
            return json.dumps({"status": "error", "message": str(e)})
