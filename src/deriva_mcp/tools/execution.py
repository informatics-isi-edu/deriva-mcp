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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-mcp")


def register_execution_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register execution management tools with the MCP server."""

    def _get_active_tool_execution() -> Any | None:
        """Get the active tool-created execution for the current connection."""
        conn_info = conn_manager.get_active_connection_info_or_raise()
        return conn_info.active_tool_execution

    def _set_active_tool_execution(execution: Any) -> None:
        """Set the active tool-created execution for the current connection."""
        conn_info = conn_manager.get_active_connection_info_or_raise()
        conn_info.active_tool_execution = execution

    @mcp.tool()
    async def create_execution(
        workflow_name: str,
        workflow_type: str,
        description: str = "",
        dataset_rids: list[str] | None = None,
        asset_rids: list[str] | None = None,
        dry_run: bool = False,
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
            dry_run: If True, download input datasets/assets but skip creating
                execution records in the catalog and skip uploading results.
                Useful for testing data loading, configuration, and model
                initialization without writing to the catalog.

        Returns:
            JSON with execution_rid, workflow_rid, dataset_count, asset_count, dry_run.

        Example:
            create_execution("CIFAR Training", "Training", "Train ResNet on CIFAR-10", ["1-ABC"])
            create_execution("Test Run", "Training", "Debug data loading", dry_run=True)
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

            execution = ml.create_execution(config, dry_run=dry_run)
            _set_active_tool_execution(execution)

            return json.dumps({
                "status": "created",
                "execution_rid": execution.execution_rid,
                "workflow_rid": execution.workflow_rid,
                "description": description,
                "dataset_count": len(datasets),
                "asset_count": len(asset_rids or []),
                "dry_run": dry_run,
            })
        except Exception as e:
            logger.error(f"Failed to create execution: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def start_execution() -> str:
        """Start timing the active execution. Call after create_execution().

        Records the start timestamp for duration tracking. The execution
        status changes to "running".
        """
        try:
            execution = _get_active_tool_execution()
            if execution is None:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution. Use create_execution first.",
                })

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
            execution = _get_active_tool_execution()
            if execution is None:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

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

            execution = _get_active_tool_execution()
            if execution is None:
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
    async def set_execution_description(execution_rid: str, description: str) -> str:
        """Set or update the description for an execution.

        Updates the execution's description in the catalog. Good descriptions help
        users understand what the execution accomplished and any notable results.

        Args:
            execution_rid: RID of the execution to update.
            description: New description text.

        Returns:
            JSON with status, execution_rid, description.

        Example:
            set_execution_description("2-XYZ", "Training run with lr=0.001, achieved 95% accuracy")
        """
        try:
            ml = conn_manager.get_active_or_raise()
            record = ml.lookup_execution(execution_rid)
            record.description = description
            return json.dumps({
                "status": "updated",
                "execution_rid": execution_rid,
                "description": description,
            })
        except Exception as e:
            logger.error(f"Failed to set execution description: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_execution_info() -> str:
        """Get details about the active execution including upload status.

        Returns:
            JSON with execution_rid, status, working_dir, upload_pending flag.
        """
        try:
            execution = _get_active_tool_execution()
            if execution is None:
                return json.dumps({
                    "status": "no_active_execution",
                    "message": "No active execution. Use create_execution first.",
                })

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

            _set_active_tool_execution(execution)

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
            execution = _get_active_tool_execution()
            if execution is None:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution. Use create_execution first.",
                })

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
            execution = _get_active_tool_execution()
            if execution is None:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

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

            execution = _get_active_tool_execution()
            if execution is None:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution. Use create_execution first.",
                })

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
                "file_path": str(asset_path),
                "filename": asset_path.file_name,
                "asset_table": asset_path.asset_table,
                "asset_types": asset_path.asset_types,
            })
        except Exception as e:
            logger.error(f"Failed to download asset: {e}")
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
            execution = _get_active_tool_execution()
            if execution is None:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

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
        exclude_tables: list[str] | None = None,
        timeout: list[int] | None = None,
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
            exclude_tables: Optional list of table names to exclude from FK path
                traversal during bag export. Use when downloads are slow or
                timing out due to expensive joins through large tables.
            timeout: Optional [connect_timeout, read_timeout] in seconds for
                network requests. Default is [10, 610]. Increase read_timeout
                for large datasets with deep FK joins that need more time.

        Returns:
            JSON with bag attributes: dataset_rid, version, description,
            dataset_types, execution_rid, and bag_path (local filesystem path).
        """
        try:
            from deriva_ml.dataset.aux_classes import DatasetSpec

            execution = _get_active_tool_execution()
            if execution is None:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

            spec = DatasetSpec(
                rid=dataset_rid,
                version=version,
                materialize=materialize,
                exclude_tables=set(exclude_tables) if exclude_tables else None,
                timeout=tuple(timeout) if timeout else None,
            )
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
            execution = _get_active_tool_execution()
            if execution is None:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution.",
                })

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

# =============================================================================
# Storage Management Tools
# =============================================================================


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.1f} {units[i]}"


def _discover_cache_dirs(connected_cache_dir: str | None = None, extra_dirs: list[str] | None = None) -> list[Path]:
    """Discover all potential DerivaML cache directories.

    Scans the default ~/.deriva-ml tree for any cache/ directories,
    adds the connected catalog's cache_dir, and any explicitly provided paths.

    Returns:
        Deduplicated list of existing cache directory Paths.
    """
    from pathlib import Path

    candidates: set[Path] = set()

    # Scan default location: ~/.deriva-ml/*/*/cache/
    default_root = Path.home() / ".deriva-ml"
    if default_root.exists():
        for cache_dir in default_root.rglob("cache"):
            if cache_dir.is_dir():
                candidates.add(cache_dir.resolve())

    # Add connected catalog's cache dir
    if connected_cache_dir:
        p = Path(connected_cache_dir)
        if p.exists():
            candidates.add(p.resolve())

    # Add any explicitly provided directories
    for d in (extra_dirs or []):
        p = Path(d)
        if p.exists():
            candidates.add(p.resolve())

    return sorted(candidates)


def _parse_cache_entry(entry_path: Path) -> dict[str, Any] | None:
    """Parse a single cache directory entry into structured info.

    Cache entries have the naming convention: {dataset_rid}_{checksum}
    Inside is Dataset_{dataset_rid}/ containing the extracted bag.

    Returns:
        Dict with entry metadata, or None if not a valid cache entry.
    """
    import csv
    from datetime import datetime

    name = entry_path.name
    # Parse {rid}_{checksum} format — RID is everything before the last underscore+hex
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return None

    dataset_rid = parts[0]
    checksum = parts[1]

    # Calculate size
    try:
        size_bytes = sum(f.stat().st_size for f in entry_path.rglob("*") if f.is_file())
        mtime = datetime.fromtimestamp(entry_path.stat().st_mtime)
    except OSError:
        return None

    # Check materialization status
    materialized = (entry_path / "validated_check.txt").exists()

    # Try to extract description and version from Dataset.csv inside the bag
    description = ""
    bag_dir = entry_path / f"Dataset_{dataset_rid}"
    dataset_csv = bag_dir / "data" / "Dataset.csv"
    if dataset_csv.exists():
        try:
            with dataset_csv.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("RID") == dataset_rid:
                        description = row.get("Description", "")
                        break
        except Exception:
            pass

    # Count asset files from fetch.txt
    asset_count = 0
    asset_total_bytes = 0
    fetch_txt = bag_dir / "fetch.txt"
    if fetch_txt.exists():
        try:
            with fetch_txt.open(encoding="utf-8") as f:
                for line in f:
                    parts_line = line.strip().split("\t")
                    if len(parts_line) >= 2:
                        asset_count += 1
                        try:
                            asset_total_bytes += int(parts_line[1])
                        except ValueError:
                            pass
        except Exception:
            pass

    return {
        "dataset_rid": dataset_rid,
        "checksum": checksum[:12] + "...",
        "checksum_full": checksum,
        "size_bytes": size_bytes,
        "size": _human_readable_size(size_bytes),
        "asset_count": asset_count,
        "asset_total_bytes": asset_total_bytes,
        "asset_size": _human_readable_size(asset_total_bytes),
        "materialized": materialized,
        "description": description,
        "modified": mtime.isoformat(),
        "path": str(entry_path),
    }


def register_storage_tools(mcp_server: FastMCP, conn_manager: ConnectionManager):
    """Register storage management tools with the MCP server."""

    @mcp_server.tool()
    async def list_cache_contents(
        cache_dirs: list[str] | None = None,
    ) -> str:
        """List all cached dataset bags across all DerivaML cache directories.

        Discovers cache directories automatically by scanning ~/.deriva-ml/
        and the connected catalog's cache. Shows each cached bag with its
        dataset RID, description, size, asset count, and materialization status.

        Use this to understand what's consuming disk space before deciding
        what to delete.

        Args:
            cache_dirs: Optional additional cache directory paths to scan.
                These are added to the automatically discovered directories.

        Returns:
            JSON with:
                - cache_directories: list of discovered cache paths with entry counts
                - entries: list of cached bags, each with dataset_rid, checksum,
                  size, asset_count, materialized, description, modified, path
                - total_size: human-readable total size
                - total_size_bytes: total size in bytes
        """
        try:
            # Get connected catalog's cache dir if available
            connected_cache = None
            try:
                ml = conn_manager.get_active_or_raise()
                connected_cache = str(ml.cache_dir)
            except Exception:
                pass  # Not connected, just scan defaults

            dirs = _discover_cache_dirs(connected_cache, cache_dirs)

            all_entries: list[dict[str, Any]] = []
            dir_summaries: list[dict[str, Any]] = []

            for cache_dir in dirs:
                entries_in_dir = []
                for entry in sorted(cache_dir.iterdir()):
                    if entry.is_dir() and "_" in entry.name:
                        parsed = _parse_cache_entry(entry)
                        if parsed:
                            parsed["cache_dir"] = str(cache_dir)
                            entries_in_dir.append(parsed)

                # Derive hostname/catalog from the cache dir path
                # Pattern: ~/.deriva-ml/{hostname}/{catalog_id}/cache
                cache_parts = cache_dir.parts
                label = str(cache_dir)
                try:
                    cache_idx = cache_parts.index("cache")
                    if cache_idx >= 2:
                        label = f"{cache_parts[cache_idx - 2]}/{cache_parts[cache_idx - 1]}"
                except (ValueError, IndexError):
                    pass

                dir_summaries.append({
                    "path": str(cache_dir),
                    "label": label,
                    "entry_count": len(entries_in_dir),
                    "total_bytes": sum(e["size_bytes"] for e in entries_in_dir),
                    "total_size": _human_readable_size(
                        sum(e["size_bytes"] for e in entries_in_dir)
                    ),
                })
                all_entries.extend(entries_in_dir)

            total_bytes = sum(e["size_bytes"] for e in all_entries)

            return json.dumps({
                "status": "success",
                "cache_directories": dir_summaries,
                "entries": all_entries,
                "total_entries": len(all_entries),
                "total_size_bytes": total_bytes,
                "total_size": _human_readable_size(total_bytes),
            })
        except Exception as e:
            logger.error(f"Failed to list cache contents: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp_server.tool()
    async def delete_cache_entry(
        dataset_rid: str,
        cache_dir: str | None = None,
        confirm: bool = False,
    ) -> str:
        """Delete cached dataset bags for a specific dataset RID.

        Finds and removes all cached bags matching the given dataset RID.
        A single dataset may have multiple cached versions (different checksums).

        IMPORTANT: This permanently deletes cached data. The bags can be
        re-downloaded from the catalog but this may take significant time
        for large datasets. Always call list_cache_contents first to review
        what will be deleted.

        Args:
            dataset_rid: The dataset RID to delete cached bags for.
                All cached versions of this dataset will be removed.
            cache_dir: Optional specific cache directory to delete from.
                If not provided, searches all discovered cache directories.
            confirm: Must be set to true to actually delete. If false,
                returns what would be deleted without removing anything
                (dry run).

        Returns:
            JSON with deleted entries and bytes freed, or dry run preview.
        """
        try:
            import shutil

            connected_cache = None
            try:
                ml = conn_manager.get_active_or_raise()
                connected_cache = str(ml.cache_dir)
            except Exception:
                pass

            search_dirs = (
                [Path(cache_dir)] if cache_dir else
                _discover_cache_dirs(connected_cache)
            )

            matches: list[dict[str, Any]] = []
            for d in search_dirs:
                if not d.exists():
                    continue
                for entry in d.iterdir():
                    if entry.is_dir() and entry.name.startswith(f"{dataset_rid}_"):
                        parsed = _parse_cache_entry(entry)
                        if parsed:
                            matches.append(parsed)

            if not matches:
                return json.dumps({
                    "status": "success",
                    "message": f"No cached bags found for dataset {dataset_rid}",
                    "entries_found": 0,
                })

            if not confirm:
                return json.dumps({
                    "status": "dry_run",
                    "message": f"Found {len(matches)} cached bag(s) for dataset {dataset_rid}. "
                               f"Set confirm=true to delete.",
                    "entries": matches,
                    "total_bytes": sum(e["size_bytes"] for e in matches),
                    "total_size": _human_readable_size(
                        sum(e["size_bytes"] for e in matches)
                    ),
                })

            # Actually delete
            deleted = []
            errors = []
            bytes_freed = 0
            for entry in matches:
                try:
                    shutil.rmtree(entry["path"])
                    deleted.append(entry)
                    bytes_freed += entry["size_bytes"]
                except Exception as e:
                    errors.append({"path": entry["path"], "error": str(e)})

            return json.dumps({
                "status": "success",
                "deleted": deleted,
                "entries_deleted": len(deleted),
                "bytes_freed": bytes_freed,
                "size_freed": _human_readable_size(bytes_freed),
                "errors": errors,
            })
        except Exception as e:
            logger.error(f"Failed to delete cache entry: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp_server.tool()
    async def clear_cache(
        older_than_days: int | None = None,
        cache_dir: str | None = None,
        confirm: bool = False,
    ) -> str:
        """Clear the dataset cache directory.

        Removes cached dataset bags to free disk space. Can optionally
        filter by age to only remove old entries.

        IMPORTANT: This permanently deletes cached data. Always call
        list_cache_contents first to review what will be removed, and
        set confirm=true to proceed.

        Args:
            older_than_days: If provided, only remove cache entries older
                than this many days. If None, removes all cache entries.
            cache_dir: Optional specific cache directory to clear. If not
                provided, clears the connected catalog's cache directory.
            confirm: Must be set to true to actually delete. If false,
                returns a preview of what would be deleted (dry run).

        Returns:
            JSON with deletion results or dry run preview.
        """
        try:
            if cache_dir:
                target = Path(cache_dir)
            else:
                ml = conn_manager.get_active_or_raise()
                target = ml.cache_dir

            if not target.exists():
                return json.dumps({
                    "status": "success",
                    "message": "Cache directory does not exist or is empty.",
                    "entries_found": 0,
                })

            # Gather entries that would be affected
            import time
            cutoff_time = None
            if older_than_days is not None:
                cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

            entries_to_delete: list[dict[str, Any]] = []
            for entry in sorted(target.iterdir()):
                if not entry.is_dir():
                    continue
                if cutoff_time is not None:
                    if entry.stat().st_mtime > cutoff_time:
                        continue
                parsed = _parse_cache_entry(entry)
                if parsed:
                    entries_to_delete.append(parsed)

            if not entries_to_delete:
                return json.dumps({
                    "status": "success",
                    "message": "No cache entries match the criteria.",
                    "entries_found": 0,
                })

            total_bytes = sum(e["size_bytes"] for e in entries_to_delete)

            if not confirm:
                return json.dumps({
                    "status": "dry_run",
                    "message": f"Found {len(entries_to_delete)} cache entry/entries "
                               f"({_human_readable_size(total_bytes)}). "
                               f"Set confirm=true to delete.",
                    "older_than_days": older_than_days,
                    "entries": entries_to_delete,
                    "total_bytes": total_bytes,
                    "total_size": _human_readable_size(total_bytes),
                })

            # Actually delete
            import shutil
            deleted = []
            errors = []
            bytes_freed = 0
            for entry_info in entries_to_delete:
                try:
                    shutil.rmtree(entry_info["path"])
                    deleted.append(entry_info)
                    bytes_freed += entry_info["size_bytes"]
                except Exception as e:
                    errors.append({"path": entry_info["path"], "error": str(e)})

            return json.dumps({
                "status": "success",
                "older_than_days": older_than_days,
                "entries_deleted": len(deleted),
                "bytes_freed": bytes_freed,
                "size_freed": _human_readable_size(bytes_freed),
                "deleted": deleted,
                "errors": errors,
            })
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp_server.tool()
    async def clean_execution_dirs(
        older_than_days: int | None = None,
        exclude_rids: list[str] | None = None,
        confirm: bool = False,
    ) -> str:
        """Clean up execution working directories.

        Removes execution output directories from the local working directory
        to free disk space. Use this to clean up completed or orphaned executions.

        IMPORTANT: This permanently deletes execution outputs that have not
        been uploaded. Set confirm=true to proceed.

        Args:
            older_than_days: If provided, only remove directories older than
                this many days. If None, removes all (except excluded).
            exclude_rids: List of execution RIDs to preserve (never remove).
            confirm: Must be set to true to actually delete. If false,
                returns a preview of what would be deleted (dry run).

        Returns:
            JSON with deletion results or dry run preview.
        """
        try:
            ml = conn_manager.get_active_or_raise()

            if not confirm:
                # Dry run: list what would be deleted
                all_dirs = ml.list_execution_dirs()
                exclude_set = set(exclude_rids or [])

                import time
                cutoff_time = None
                if older_than_days is not None:
                    cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

                would_delete = []
                for d in all_dirs:
                    if d["execution_rid"] in exclude_set:
                        continue
                    if cutoff_time is not None:
                        from datetime import datetime
                        d_mtime = d["modified"].timestamp() if isinstance(d["modified"], datetime) else 0
                        if d_mtime > cutoff_time:
                            continue
                    would_delete.append({
                        "execution_rid": d["execution_rid"],
                        "size": _human_readable_size(d["size_bytes"]),
                        "size_bytes": d["size_bytes"],
                        "modified": d["modified"].isoformat() if hasattr(d["modified"], "isoformat") else str(d["modified"]),
                        "file_count": d["file_count"],
                    })

                total_bytes = sum(d["size_bytes"] for d in would_delete)
                return json.dumps({
                    "status": "dry_run",
                    "message": f"Found {len(would_delete)} execution dir(s) "
                               f"({_human_readable_size(total_bytes)}). "
                               f"Set confirm=true to delete.",
                    "entries": would_delete,
                    "total_bytes": total_bytes,
                    "total_size": _human_readable_size(total_bytes),
                })

            result = ml.clean_execution_dirs(
                older_than_days=older_than_days,
                exclude_rids=exclude_rids,
            )
            return json.dumps({
                "status": "success",
                "older_than_days": older_than_days,
                "exclude_rids": exclude_rids,
                **result,
                "size_freed": _human_readable_size(result.get("bytes_freed", 0)),
            })
        except Exception as e:
            logger.error(f"Failed to clean execution dirs: {e}")
            return json.dumps({"status": "error", "message": str(e)})
