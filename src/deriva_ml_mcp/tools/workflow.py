"""Workflow management tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-ml-mcp")


def register_workflow_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register workflow management tools with the MCP server."""

    @mcp.tool()
    async def lookup_workflow_by_url(url: str) -> str:
        """Find a workflow by its source URL.

        Search for a workflow that was registered with the given source URL.
        Use this to check if a workflow for a specific script or notebook
        already exists before creating a new one.

        Args:
            url: The source URL to search for (e.g., GitHub URL to script).

        Returns:
            JSON with:
            - found: True if workflow exists, False otherwise
            - workflow: Full workflow details if found

        Example:
            lookup_workflow_by_url("https://github.com/org/repo/blob/main/train.py")
        """
        try:
            ml = conn_manager.get_active_or_raise()
            workflow = ml.lookup_workflow_by_url(url)

            if workflow:
                return json.dumps({
                    "found": True,
                    "workflow": {
                        "rid": workflow.rid,
                        "name": workflow.name,
                        "workflow_type": workflow.workflow_type,
                        "description": workflow.description,
                        "url": workflow.url,
                        "checksum": workflow.checksum,
                        "version": workflow.version,
                    },
                })
            return json.dumps({"found": False, "workflow": None})
        except Exception as e:
            logger.error(f"Failed to lookup workflow by URL: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def create_workflow(
        name: str,
        workflow_type: str,
        description: str = "",
    ) -> str:
        """Create and register a new workflow definition.

        Args:
            name: Display name for the workflow.
            workflow_type: Type from Workflow_Type vocabulary (e.g., "Training", "Inference").
            description: What this workflow does.

        Returns:
            JSON with status, rid, name, workflow_type, description.

        Example:
            create_workflow("ResNet Training", "Training", "Trains ResNet50 on image data")
        """
        try:
            ml = conn_manager.get_active_or_raise()
            workflow = ml.create_workflow(
                name=name,
                workflow_type=workflow_type,
                description=description,
            )
            rid = ml.add_workflow(workflow)
            return json.dumps({
                "status": "created",
                "workflow_rid": rid,
                "name": workflow.name,
                "workflow_type": workflow.workflow_type,
                "description": workflow.description,
            })
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_workflow_description(workflow_rid: str, description: str) -> str:
        """Set or update the description for a workflow.

        Updates the workflow's description in the catalog. Good descriptions help
        users understand what the workflow does and how to use it.

        Args:
            workflow_rid: RID of the workflow to update.
            description: New description text.

        Returns:
            JSON with status, workflow_rid, description.

        Example:
            set_workflow_description("3-WKF", "Trains CNN on image data with augmentation")
        """
        try:
            ml = conn_manager.get_active_or_raise()
            workflow = ml.lookup_workflow(workflow_rid)
            workflow.description = description
            return json.dumps({
                "status": "updated",
                "workflow_rid": workflow_rid,
                "description": description,
            })
        except Exception as e:
            logger.error(f"Failed to set workflow description: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_workflow_type(
        type_name: str,
        description: str,
    ) -> str:
        """Add a new workflow type to the Workflow_Type vocabulary.

        Args:
            type_name: Name for the new workflow type.
            description: What this type of workflow does.

        Returns:
            JSON with status, name, description, rid.

        Example:
            add_workflow_type("Data Augmentation", "Workflows that augment training data")
        """
        try:
            from deriva_ml import MLVocab

            ml = conn_manager.get_active_or_raise()
            term = ml.add_term(
                table=MLVocab.workflow_type,
                term_name=type_name,
                description=description,
                exists_ok=True,
            )
            return json.dumps({
                "status": "created",
                "name": term.name,
                "description": term.description,
                "rid": term.rid,
            })
        except Exception as e:
            logger.error(f"Failed to add workflow type: {e}")
            return json.dumps({"status": "error", "message": str(e)})
