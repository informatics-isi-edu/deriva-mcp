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
            from deriva_ml.dataset.aux_classes import DatasetSpec
            from deriva_ml.execution.execution_configuration import ExecutionConfiguration

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
    async def download_asset(
        asset_rid: str,
        dest_dir: str | None = None,
    ) -> str:
        """Download an asset file from the catalog to the local filesystem.

        Downloads the file associated with an asset record to the execution's
        working directory (or a specified directory). Records the download as
        an input for provenance tracking.

        Args:
            asset_rid: RID of the asset to download (e.g., "1-ABC").
            dest_dir: Optional destination directory. If not provided, uses
                the execution's working directory.

        Returns:
            JSON with:
            - file_path: Local path to the downloaded file
            - filename: Name of the file
            - asset_table: The asset table name (e.g., "Model", "Image")
            - asset_types: List of asset type labels

        Example:
            download_asset("1-ABC")  # Download to execution working dir
            download_asset("1-ABC", "/tmp/models")  # Download to specific dir
        """
        try:
            from pathlib import Path

            key = _get_execution_key()
            if key not in _active_executions:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution. Use create_execution first.",
                })

            execution = _active_executions[key]

            # Use provided dest_dir or default to execution working dir
            if dest_dir:
                destination = Path(dest_dir)
            else:
                destination = execution.working_dir

            asset_path = execution.download_asset(
                asset_rid=asset_rid,
                dest_dir=destination,
            )

            return json.dumps({
                "status": "downloaded",
                "execution_rid": execution.execution_rid,
                "asset_rid": asset_rid,
                "file_path": str(asset_path.asset_path),
                "filename": asset_path.file_name,
                "asset_table": asset_path.asset_table,
                "asset_types": asset_path.asset_types,
            })
        except Exception as e:
            logger.error(f"Failed to download asset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def find_executions(
        workflow_rid: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> str:
        """List all executions in the catalog with optional filtering.

        Returns all executions, not just experiments with Hydra config. Use this
        for finding any workflow run. Use find_experiments() if you specifically
        want ML experiments with Hydra configuration.

        Args:
            workflow_rid: Optional workflow RID to filter by.
            status: Optional status filter ("pending", "running", "completed", "failed").
            limit: Max number to return (default: 50).

        Returns:
            JSON array of {execution_rid, workflow_rid, status, description}.

        Example:
            find_executions()  # All executions
            find_executions(status="completed")  # Completed only
            find_executions(workflow_rid="1-ABC")  # For specific workflow
        """
        try:
            from deriva_ml.core.definitions import Status

            ml = conn_manager.get_active_or_raise()

            # Map status string to enum if provided
            status_enum = None
            if status:
                status_map = {
                    "pending": Status.pending,
                    "running": Status.running,
                    "completed": Status.completed,
                    "failed": Status.failed,
                    "initializing": Status.initializing,
                    "created": Status.created,
                }
                status_enum = status_map.get(status.lower())

            executions = list(ml.find_executions(
                workflow_rid=workflow_rid,
                status=status_enum,
            ))

            # Limit results
            executions = executions[:limit]

            result = []
            for exe in executions:
                result.append({
                    "execution_rid": exe.execution_rid,
                    "workflow_rid": exe.workflow_rid,
                    "status": exe.status.value if hasattr(exe.status, 'value') else str(exe.status),
                    "description": exe.description,
                })

            return json.dumps({
                "status": "success",
                "count": len(result),
                "executions": result,
            })
        except Exception as e:
            logger.error(f"Failed to find executions: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def find_experiments(
        workflow_rid: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> str:
        """List experiments (executions with Hydra configuration) in the catalog.

        Only returns executions that have Hydra configuration metadata (i.e., a
        config.yaml file in Execution_Metadata assets). This distinguishes ML
        experiments from other types of executions.

        Args:
            workflow_rid: Optional workflow RID to filter by.
            status: Optional status filter ("pending", "running", "completed", "failed").
            limit: Max number to return (default: 50).

        Returns:
            JSON array of experiment summaries, each with:
            - name: Experiment name from config_choices.model_config
            - execution_rid: The execution RID
            - description: Execution description
            - status: Execution status
            - config_choices: Dictionary of Hydra config names used
            - model_config: Dictionary of model hyperparameters
            - input_datasets: List of input dataset info
            - url: Chaise URL to view execution

        Example:
            find_experiments()  # All experiments
            find_experiments(status="completed")  # Completed experiments only
        """
        try:
            from deriva_ml.core.definitions import Status

            ml = conn_manager.get_active_or_raise()

            # Map status string to enum if provided
            status_enum = None
            if status:
                status_map = {
                    "pending": Status.pending,
                    "running": Status.running,
                    "completed": Status.completed,
                    "failed": Status.failed,
                    "initializing": Status.initializing,
                    "created": Status.created,
                }
                status_enum = status_map.get(status.lower())

            # Use the new find_experiments method
            experiments = list(ml.find_experiments(
                workflow_rid=workflow_rid,
                status=status_enum,
            ))

            # Limit results
            experiments = experiments[:limit]

            # Build summaries
            result = []
            for exp in experiments:
                result.append(exp.summary())

            return json.dumps({
                "status": "success",
                "count": len(result),
                "experiments": result,
            })
        except Exception as e:
            logger.error(f"Failed to find experiments: {e}")
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
        """Download a dataset version as a BDBag for use in this execution.

        Creates a self-contained BDBag archive of the specified dataset version
        and records it as an input for provenance tracking. The bag includes
        all dataset members, nested datasets (recursively), feature values,
        vocabulary terms, and asset files.

        The bag captures the exact catalog state at the version's snapshot time,
        ensuring reproducibility regardless of later catalog changes.

        Args:
            dataset_rid: RID of the dataset to download.
            version: Semantic version to download (e.g., "1.0.0"). Required.
                Use lookup_dataset() to find the current_version if needed.
            materialize: If True (default), fetch all referenced asset files
                (images, model weights, etc.) from Hatrac storage. If False,
                bag contains only metadata and remote file references.

        Returns:
            JSON with bag attributes: dataset_rid, version, description,
            dataset_types, execution_rid, and bag_path (local filesystem path).
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
                "version": str(bag.current_version) if bag.current_version else None,
                "description": bag.description,
                "dataset_types": bag.dataset_types,
                "execution_rid": bag.execution_rid,
                "bag_path": str(bag.model.bag_path),
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

    # =========================================================================
    # Execution Nesting Tools
    # =========================================================================

    @mcp.tool()
    async def add_nested_execution(
        parent_execution_rid: str,
        child_execution_rid: str,
        sequence: int | None = None,
    ) -> str:
        """Add a child execution to a parent execution.

        Creates a parent-child relationship between executions. Use this to
        group related executions, such as:
        - Parameter sweeps (parent = sweep, children = individual runs)
        - Pipelines (parent = pipeline, children = stages)
        - Cross-validation (parent = CV experiment, children = folds)

        Args:
            parent_execution_rid: RID of the parent execution.
            child_execution_rid: RID of the child execution to nest.
            sequence: Optional ordering index (0, 1, 2...). Use None for parallel executions.

        Returns:
            JSON with parent_rid, child_rid, sequence.

        Example:
            # Create a sweep parent, then add child executions
            add_nested_execution("1-PARENT", "1-CHILD1", sequence=0)
            add_nested_execution("1-PARENT", "1-CHILD2", sequence=1)
        """
        try:
            ml = conn_manager.get_active_or_raise()
            parent_exec = ml.lookup_execution(parent_execution_rid)
            parent_exec.add_nested_execution(child_execution_rid, sequence=sequence)

            return json.dumps({
                "status": "added",
                "parent_rid": parent_execution_rid,
                "child_rid": child_execution_rid,
                "sequence": sequence,
            })
        except Exception as e:
            logger.error(f"Failed to add nested execution: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_nested_executions(
        execution_rid: str,
        recurse: bool = False,
    ) -> str:
        """List all child (nested) executions of an execution.

        Args:
            execution_rid: RID of the parent execution.
            recurse: If True, return all descendants (children, grandchildren, etc.).

        Returns:
            JSON array of {execution_rid, workflow_rid, status, description} for each child.

        Example:
            list_nested_executions("1-PARENT")  # Direct children only
            list_nested_executions("1-PARENT", recurse=True)  # All descendants
        """
        try:
            ml = conn_manager.get_active_or_raise()
            execution = ml.lookup_execution(execution_rid)
            children = execution.list_nested_executions(recurse=recurse)

            result = []
            for child in children:
                result.append({
                    "execution_rid": child.execution_rid,
                    "workflow_rid": child.workflow_rid,
                    "status": child.status.value if hasattr(child.status, 'value') else str(child.status),
                    "description": child.description,
                })

            return json.dumps({
                "parent_rid": execution_rid,
                "recurse": recurse,
                "count": len(result),
                "children": result,
            })
        except Exception as e:
            logger.error(f"Failed to list nested executions: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_parent_executions(
        execution_rid: str,
        recurse: bool = False,
    ) -> str:
        """List all parent executions that contain this execution as a child.

        Args:
            execution_rid: RID of the child execution.
            recurse: If True, return all ancestors (parents, grandparents, etc.).

        Returns:
            JSON array of {execution_rid, workflow_rid, status, description} for each parent.

        Example:
            list_parent_executions("1-CHILD")  # Direct parents only
            list_parent_executions("1-CHILD", recurse=True)  # All ancestors
        """
        try:
            ml = conn_manager.get_active_or_raise()
            execution = ml.lookup_execution(execution_rid)
            parents = execution.list_parent_executions(recurse=recurse)

            result = []
            for parent in parents:
                result.append({
                    "execution_rid": parent.execution_rid,
                    "workflow_rid": parent.workflow_rid,
                    "status": parent.status.value if hasattr(parent.status, 'value') else str(parent.status),
                    "description": parent.description,
                })

            return json.dumps({
                "child_rid": execution_rid,
                "recurse": recurse,
                "count": len(result),
                "parents": result,
            })
        except Exception as e:
            logger.error(f"Failed to list parent executions: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def lookup_experiment(execution_rid: str) -> str:
        """Look up an experiment (execution with Hydra configuration) by RID.

        Returns detailed information about an experiment including its Hydra
        configuration, model parameters, input datasets/assets, and output assets.

        Use this to get full details about a specific ML experiment. For listing
        multiple experiments, use find_experiments().

        Args:
            execution_rid: RID of the experiment/execution to look up.

        Returns:
            JSON with experiment details:
            - name: Experiment name from config
            - execution_rid: The execution RID
            - description: Execution description
            - status: Execution status
            - config_choices: Dictionary of Hydra config names used
            - model_config: Dictionary of model hyperparameters
            - input_datasets: List of input dataset summaries (dataset_rid, description, version, dataset_types)
            - input_assets: List of input asset summaries (asset_rid, asset_table, filename, description, asset_types, url)
            - output_assets: List of output asset summaries (asset_rid, asset_table, filename, description, asset_types, url)
            - metadata_assets: List of execution metadata assets (config files, hydra.yaml, etc.)
            - url: Chaise URL to view execution

        Example:
            lookup_experiment("1-ABC") -> full experiment details with config
        """
        try:
            ml = conn_manager.get_active_or_raise()
            exp = ml.lookup_experiment(execution_rid)

            return json.dumps(exp.summary())
        except Exception as e:
            logger.error(f"Failed to lookup experiment: {e}")
            return json.dumps({"status": "error", "message": str(e)})


# =============================================================================
# Storage Management Tools
# =============================================================================


def register_storage_tools(mcp_server: FastMCP, conn_manager: ConnectionManager):
    """Register storage management tools with the MCP server."""

    @mcp_server.tool()
    def clear_cache(older_than_days: int | None = None) -> str:
        """Clear the dataset cache directory.

        Removes cached dataset bags to free disk space. Can optionally
        filter by age to only remove old entries.

        Args:
            older_than_days: If provided, only remove cache entries older
                than this many days. If None, removes all cache entries.

        Returns:
            JSON object with:
            - files_removed: Number of files removed
            - dirs_removed: Number of directories removed
            - bytes_freed: Total bytes freed
            - errors: Number of removal errors

        Example:
            clear_cache()  # Clear all cache
            clear_cache(older_than_days=7)  # Only clear old entries
        """
        try:
            ml = conn_manager.get_active_or_raise()
            result = ml.clear_cache(older_than_days=older_than_days)
            return json.dumps({
                "status": "success",
                "older_than_days": older_than_days,
                **result,
            })
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp_server.tool()
    def clean_execution_dirs(
        older_than_days: int | None = None,
        exclude_rids: list[str] | None = None,
    ) -> str:
        """Clean up execution working directories.

        Removes execution output directories from the local working directory
        to free disk space. Use this to clean up completed or orphaned executions.

        Args:
            older_than_days: If provided, only remove directories older than
                this many days. If None, removes all (except excluded).
            exclude_rids: List of execution RIDs to preserve (never remove).

        Returns:
            JSON object with:
            - dirs_removed: Number of directories removed
            - bytes_freed: Total bytes freed
            - errors: Number of removal errors

        Example:
            clean_execution_dirs()  # Clean all
            clean_execution_dirs(older_than_days=30)  # Clean old only
            clean_execution_dirs(exclude_rids=["1-ABC", "1-DEF"])  # Preserve specific
        """
        try:
            ml = conn_manager.get_active_or_raise()
            result = ml.clean_execution_dirs(
                older_than_days=older_than_days,
                exclude_rids=exclude_rids,
            )
            return json.dumps({
                "status": "success",
                "older_than_days": older_than_days,
                "exclude_rids": exclude_rids,
                **result,
            })
        except Exception as e:
            logger.error(f"Failed to clean execution dirs: {e}")
            return json.dumps({"status": "error", "message": str(e)})
