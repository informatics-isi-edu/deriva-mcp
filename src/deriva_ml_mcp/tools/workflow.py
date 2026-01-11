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
    async def list_workflows() -> str:
        """List all workflows in the catalog.

        Returns all registered workflow definitions, including their
        names, types, versions, and descriptions.

        Returns:
            JSON array of workflow objects.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            workflows = ml.list_workflows()
            result = []
            for w in workflows:
                result.append({
                    "rid": w.rid,
                    "name": w.name,
                    "workflow_type": w.workflow_type,
                    "version": w.version,
                    "description": w.description,
                    "url": w.url,
                    "checksum": w.checksum,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def lookup_workflow(url_or_checksum: str) -> str:
        """Look up a workflow by URL or checksum.

        Finds a workflow using either its source URL or code checksum.

        Args:
            url_or_checksum: The URL or checksum to search for.

        Returns:
            JSON object with workflow RID or null if not found.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            rid = ml.lookup_workflow(url_or_checksum)
            if rid:
                return json.dumps({"found": True, "rid": rid})
            return json.dumps({"found": False, "rid": None})
        except Exception as e:
            logger.error(f"Failed to lookup workflow: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def create_workflow(
        name: str,
        workflow_type: str,
        description: str = "",
    ) -> str:
        """Create and register a new workflow.

        Creates a workflow definition and registers it in the catalog.
        The workflow_type must be a term from the Workflow_Type vocabulary.

        Args:
            name: Name of the workflow.
            workflow_type: Type of workflow (must exist in Workflow_Type vocabulary).
            description: Description of what the workflow does.

        Returns:
            JSON object with the created workflow's RID and details.
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
                "rid": rid,
                "name": workflow.name,
                "workflow_type": workflow.workflow_type,
                "description": workflow.description,
            })
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_workflow_types() -> str:
        """List available workflow types.

        Returns all terms from the Workflow_Type vocabulary that can
        be used when creating workflows.

        Returns:
            JSON array of workflow type terms.
        """
        try:
            from deriva_ml import MLVocab

            ml = conn_manager.get_active_or_raise()
            terms = ml.list_vocabulary_terms(MLVocab.workflow_type)
            result = []
            for term in terms:
                result.append({
                    "name": term.name,
                    "description": term.description,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list workflow types: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_workflow_type(
        type_name: str,
        description: str,
    ) -> str:
        """Add a new workflow type to the vocabulary.

        Creates a new term in the Workflow_Type vocabulary.

        Args:
            type_name: Name for the new workflow type.
            description: Description of the workflow type.

        Returns:
            JSON object with the created term details.
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
