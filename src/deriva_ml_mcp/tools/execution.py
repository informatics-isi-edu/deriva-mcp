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
        """Create a new execution to track an ML workflow run.

        WORKFLOW: After creating, follow this sequence:
        1. start_execution() - Begin timing
        2. asset_file_path() - Register output files (repeat as needed)
        3. stop_execution() - End timing
        4. upload_execution_outputs() - REQUIRED: Upload files to catalog

        Args:
            workflow_name: Name for this workflow (e.g., "ResNet Training Run 1").
            workflow_type: Type from Workflow_Type vocabulary (e.g., "Training").
            description: What this execution does.
            dataset_rids: Input dataset RIDs to track as provenance.
            asset_rids: Input asset RIDs to track as provenance.

        Returns:
            JSON with execution_rid, workflow_rid, dataset_count, asset_count.

        Example:
            create_execution("CIFAR Training", "Training", "Train on CIFAR-10", ["1-ABC"])
        """
        try:
            from deriva_ml.execution.execution_configuration import ExecutionConfiguration
            from deriva_ml.dataset.aux_classes import DatasetSpec

            ml = conn_manager.get_active_or_raise()

            workflow = ml.create_workflow(
                name=workflow_name,
                workflow_type=workflow_type,
                description=description,
            )

            datasets = []
            if dataset_rids:
                for rid in dataset_rids:
                    datasets.append(DatasetSpec(rid=rid))

            config = ExecutionConfiguration(
                workflow=workflow,
                description=description,
                datasets=datasets,
                assets=asset_rids or [],
            )

            execution = ml.create_execution(config)
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
        """Start timing the active execution. Call after create_execution()."""
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
        """Stop timing and mark execution complete. Call before upload_execution_outputs()."""
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
        """Update the execution status with a progress message.

        Args:
            status: One of "pending", "running", "completed", "failed".
            message: Progress message or error description.

        Returns:
            JSON with execution_rid, new_status, message.
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
        """Get details about the active execution including upload status.

        Returns:
            JSON with execution_rid, status, working_dir, upload_pending flag.
        """
        try:
            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "no_active_execution",
                    "message": "No active execution. Use create_execution first.",
                })

            execution = _active_executions[key]
            has_pending_uploads = execution.uploaded_assets is None

            return json.dumps({
                "execution_rid": execution.execution_rid,
                "workflow_rid": execution.workflow_rid,
                "status": execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
                "dataset_rids": execution.dataset_rids,
                "dataset_count": len(execution.datasets),
                "working_dir": str(execution.working_dir),
                "upload_pending": has_pending_uploads,
                "upload_reminder": "Call upload_execution_outputs() when done." if has_pending_uploads else None,
            })
        except Exception as e:
            logger.error(f"Failed to get execution info: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def restore_execution(execution_rid: str) -> str:
        """Restore a previous execution to continue working with it.

        Args:
            execution_rid: RID of the execution to restore (e.g., "1-ABC").

        Returns:
            JSON with execution_rid, workflow_rid, dataset_count.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            execution = ml.restore_execution(execution_rid)

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
        """Register a file for upload as an execution output.

        Files are staged locally and uploaded when upload_execution_outputs() is called.

        Args:
            asset_name: Asset table name (e.g., "Image", "Model", "Execution_Metadata").
            file_name: Path to existing file OR new filename to create.
            asset_types: Asset type terms (defaults to asset_name).
            copy_file: True to copy file, False to symlink (default).
            rename_file: New filename if renaming.

        Returns:
            JSON with file_path to use and reminder to call upload_execution_outputs().

        Example:
            asset_file_path("Model", "/tmp/model.pt") -> registers model for upload
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
                "note": "Call upload_execution_outputs() to upload.",
            })
        except Exception as e:
            logger.error(f"Failed to register asset file: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def upload_execution_outputs(clean_folder: bool = True) -> str:
        """Upload all registered assets to the catalog. MUST be called to complete execution.

        This uploads files registered with asset_file_path() and records provenance.

        Args:
            clean_folder: Remove local staging folders after upload (default: True).

        Returns:
            JSON with assets_uploaded counts by type.

        Example:
            upload_execution_outputs() -> {"assets_uploaded": {"Model": 1, "Image": 10}}
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

        Args:
            limit: Max number to return (default: 50).

        Returns:
            JSON array of {rid, workflow, status, description, duration}.
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
        """Create a dataset that is output from the active execution.

        The dataset will be linked to this execution for provenance.

        Args:
            description: What this dataset contains.
            dataset_types: Type labels (e.g., ["Training", "Augmented"]).

        Returns:
            JSON with dataset_rid, execution_rid.
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
        """Download a dataset as input for the active execution.

        Args:
            dataset_rid: RID of the dataset to download.
            version: Specific version (default: current).
            materialize: Fetch all referenced files (default: True).

        Returns:
            JSON with dataset_rid, version, path (local directory).
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
        """Get the local working directory path for the active execution.

        Returns:
            JSON with working_dir path.
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
