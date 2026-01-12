"""DerivaML MCP Server.

This module implements the Model Context Protocol server for DerivaML,
exposing DerivaML operations as MCP tools that can be used by LLM applications.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.tools import (
    register_catalog_tools,
    register_dataset_tools,
    register_vocabulary_tools,
    register_workflow_tools,
    register_feature_tools,
    register_schema_tools,
    register_execution_tools,
)

# Configure logging - NEVER use print() in STDIO MCP servers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/tmp/deriva-ml-mcp.log")],
)
logger = logging.getLogger("deriva-ml-mcp")

# Initialize FastMCP server
mcp = FastMCP(
    "deriva-ml",
    instructions="MCP server for DerivaML - manage ML workflows, datasets, and features in Deriva catalogs",
)

# Global connection manager
connection_manager = ConnectionManager()


def register_all_tools(mcp_server: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register all DerivaML tools with the MCP server."""
    register_catalog_tools(mcp_server, conn_manager)
    register_dataset_tools(mcp_server, conn_manager)
    register_vocabulary_tools(mcp_server, conn_manager)
    register_workflow_tools(mcp_server, conn_manager)
    register_feature_tools(mcp_server, conn_manager)
    register_schema_tools(mcp_server, conn_manager)
    register_execution_tools(mcp_server, conn_manager)


# Register all tools
register_all_tools(mcp, connection_manager)


def main() -> None:
    """Run the DerivaML MCP server."""
    logger.info("Starting DerivaML MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
