"""Execution management tools for DerivaML MCP server.

Executions track ML workflow runs with full provenance. Key concepts:

**Execution Lifecycle (MCP Tools)**:
1. **create_execution()**: Create execution with workflow type and input datasets/assets
2. **start_execution()**: Begin timing the workflow run
3. **[Do ML work]**: Run your training, inference, or processing pipeline
4. **stop_execution()**: End timing and mark complete

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
"""

from __future__ import annotations

import json
import logging
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
        4. stop_execution() - End timing

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
        """Stop timing and mark execution complete.

        Records the stop timestamp and calculates duration. Call this after
        your ML workflow completes.
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

