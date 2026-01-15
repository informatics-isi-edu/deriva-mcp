"""Execution management tools for DerivaML MCP server.

Executions track ML workflow runs with full provenance. Key concepts:

**Execution Lifecycle (MCP Tools)**:
1. **create_execution()**: Create execution with workflow type and input datasets/assets
2. **start_execution()**: Begin timing the workflow run
3. **[Do ML work]**: Run your training, inference, or processing pipeline
4. **asset_file_path()**: Register output files for upload (repeat as needed)
5. **stop_execution()**: End timing and mark complete
6. **upload_execution_outputs()**: Upload all registered files to catalog (REQUIRED)

**Python Context Manager** (for direct Python usage):
```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)
config = ExecutionConfiguration(
    workflow=ml.create_workflow("Training", "Training"),
    datasets=[DatasetSpec(rid="1-ABC")],
)

# Context manager handles start/stop timing automatically
with ml.create_execution(config) as exe:
    # Download input data
    bag = exe.download_dataset_bag(DatasetSpec(rid="1-ABC"))

    # Do ML work...
    model = train_model(bag)

    # Register outputs for upload
    model_path = exe.asset_file_path("Model", "model.pt")
    torch.save(model, model_path)

# After context exits, call upload separately
exe.upload_execution_outputs()
```

**Provenance Tracking**:
Executions automatically track:
- **Input datasets**: Which dataset versions were used (reproducibility)
- **Input assets**: Individual files consumed by the workflow
- **Output assets**: Files produced, linked to this execution
- **Output datasets**: New datasets created by this workflow
- **Timing**: Start/stop times and duration
- **Status**: Progress updates and error messages

**Workflows**:
Each execution references a Workflow (from Workflow_Type vocabulary) that
categorizes what type of work was done: Training, Inference, Preprocessing,
Evaluation, etc.

**Asset File Paths**:
Use asset_file_path() to register files for upload. This:
- Stages files in a local working directory
- Associates files with asset tables (e.g., "Model", "Image")
- Applies asset type labels from vocabulary
- Tracks the execution that produced them

Files are NOT uploaded until upload_execution_outputs() is called.

**Working Directory**:
Each execution has a temporary working directory for staging files.
Use get_execution_working_dir() to get the path for writing outputs.
"""

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
        """Create a new execution to track an ML workflow run with provenance.

        This is the first step in the execution lifecycle. Specify input datasets
        and assets to establish provenance - these will be recorded as inputs to
        this workflow run.

        LIFECYCLE (follow in order):
        1. create_execution() - You are here
        2. start_execution() - Begin timing
        3. [Run your ML workflow]
        4. asset_file_path() - Register each output file
        5. stop_execution() - End timing
        6. upload_execution_outputs() - Upload files (REQUIRED)

        Args:
            workflow_name: Descriptive name (e.g., "ResNet50 Training Run 3").
            workflow_type: Type from Workflow_Type vocabulary (e.g., "Training", "Inference").
            description: What this execution does and why.
            dataset_rids: Input dataset RIDs for provenance tracking.
            asset_rids: Input asset RIDs for provenance tracking.

        Returns:
            JSON with execution_rid, workflow_rid, dataset_count, asset_count.

        Example:
            create_execution("CIFAR Training", "Training", "Train ResNet on CIFAR-10", ["1-ABC"])
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
        """Start timing the active execution. Call after create_execution().

        Records the start timestamp for duration tracking. The execution
        status changes to "running".
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
        """Stop timing and mark execution complete. Call before upload_execution_outputs().

        Records the stop timestamp and calculates duration. Call this after
        your ML workflow completes but BEFORE uploading outputs.
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
        """Register a file for upload as an execution output asset.

        This is the key method for uploading files produced by your workflow.
        Files are staged in the execution's working directory and uploaded
        when upload_execution_outputs() is called.

        The file will be:
        - Associated with the specified asset table (e.g., "Model", "Image")
        - Tagged with asset types from the Asset_Type vocabulary
        - Linked to this execution for provenance tracking

        Args:
            asset_name: Target asset table (e.g., "Image", "Model", "Execution_Metadata").
            file_name: Path to existing file to stage, OR filename for new file to create.
            asset_types: Asset_Type vocabulary terms (defaults to [asset_name]).
            copy_file: True to copy file, False to symlink (default, saves disk space).
            rename_file: Optionally rename the file during staging.

        Returns:
            JSON with file_path (use this path for writing), file_name, asset_types.

        Examples:
            # Register existing model file
            asset_file_path("Model", "/tmp/trained_model.pt")

            # Get path for new file to write
            asset_file_path("Execution_Metadata", "metrics.json")
            # Then write to the returned file_path
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
        """Upload all registered assets to the catalog. REQUIRED to complete execution.

        This is the final step in the execution lifecycle. Uploads all files
        registered with asset_file_path() to the catalog's object store and
        creates asset records with proper provenance linking.

        IMPORTANT: When using the Python context manager, this must be called AFTER
        exiting the `with` block (see example below). The context manager handles
        start/stop timing, while upload is a separate step.

        IMPORTANT: Outputs are NOT persisted until this is called. Always call
        this method, even if the workflow failed (to record partial results).

        Args:
            clean_folder: Remove local staging directory after upload (default: True).

        Returns:
            JSON with assets_uploaded counts by asset table type.

        Example (MCP tool sequence):
            1. create_execution("Training", "Training", ...)
            2. start_execution()
            3. [Do ML work, call asset_file_path() to register outputs]
            4. stop_execution()
            5. upload_execution_outputs()  <- Call this LAST

        Example (Python context manager):
            ```python
            with ml.create_execution(config) as exe:
                # Do work inside context manager
                path = exe.asset_file_path("Model", "model.pt")
                # Write to path...

            # Upload AFTER exiting context manager
            exe.upload_execution_outputs()
            ```

        Note:
            For progress monitoring during uploads, use the Python API directly with
            a progress_callback parameter. See UploadProgress and UploadCallback in
            deriva_ml for details.
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
        """Create a new dataset as output from this execution.

        Creates a dataset that is linked to this execution for provenance.
        Use this when your workflow produces a new curated collection of data
        (e.g., augmented training data, filtered results).

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
        version: str,
        materialize: bool = True,
    ) -> str:
        """Download a dataset bag for use in this execution.

        Downloads a dataset as a BDBag to the execution's working directory.
        The download is recorded as an input for provenance tracking.

        Args:
            dataset_rid: RID of the dataset to download.
            version: Semantic version to download (e.g., "1.0.0"). Required.
                Use get_dataset() to find the current_version if needed.
            materialize: Fetch all referenced asset files (default: True).

        Returns:
            JSON with dataset_rid, version, path (local bag directory).
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
