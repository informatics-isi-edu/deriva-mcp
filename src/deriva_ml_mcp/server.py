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
    register_annotation_tools,
    register_catalog_tools,
    register_dataset_tools,
    register_vocabulary_tools,
    register_workflow_tools,
    register_feature_tools,
    register_schema_tools,
    register_execution_tools,
    register_storage_tools,
    register_data_tools,
    register_devtools,
)
from deriva_ml_mcp.resources import register_resources
from deriva_ml_mcp.prompts import register_prompts

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
    instructions="""MCP server for DerivaML - manage ML workflows, datasets, and features in Deriva catalogs

## Connection

Always call `connect_catalog` before using other tools. This establishes the connection and auto-detects the domain schema.

## Tool Selection Guidelines

**Prefer high-level tools over low-level operations:**

- Use domain-specific tools like `create_dataset`, `add_dataset_members`, `create_feature`, `add_feature_value`, `create_execution`, etc. instead of raw `insert_records`
- High-level tools properly initialize default values, enforce data integrity constraints, and handle relationships
- Only use `insert_records` as a last resort when no high-level API exists for your use case - it bypasses business logic and may leave required fields uninitialized

**Provenance tracking:**

- Use `create_execution` to track the provenance of datasets, features, and other artifacts
- When creating datasets or features that should be tracked, create them within an execution context
- Executions link inputs, outputs, and configuration for reproducibility

## Common Workflows

**Exploring a catalog:**
1. `connect_catalog` - Connect to the catalog
2. `list_tables` or `get_schema_description` - Understand the schema
3. `list_datasets` - See available datasets
4. `list_vocabularies` / `list_vocabulary_terms` - Explore controlled vocabularies
5. `list_features` / `list_feature_values` - Examine feature definitions

**Working with datasets:**
1. `create_dataset` - Create a new dataset with types
2. `add_dataset_members` - Add assets or nested datasets as members
3. `list_dataset_members` - View dataset contents
4. `download_dataset` - Download dataset assets locally

**Adding features:**
1. `create_feature` - Define a new feature linking a target table to vocabulary terms
2. `add_feature_value` or `add_feature_value_record` - Assign feature values to records

**Running workflows:**
1. `create_execution` - Start a tracked execution
2. `start_execution` / `stop_execution` - Manage execution lifecycle
3. `upload_execution_outputs` - Upload results to the catalog

## Schema Conventions

- `deriva-ml` schema: Core ML tables (Dataset, Execution, Feature, etc.)
- Domain schema: Project-specific tables (assets, vocabularies, features)
- Vocabularies: Controlled term lists for consistent labeling
- Assets: Tables with file attachments (images, models, etc.)

## Before Calling Tools

**Always verify required parameters before calling any tool.** Check the tool's description and parameter schema to understand which parameters are required vs optional. Never assume a parameter is optional - verify first.
""",
)

# Global connection manager
connection_manager = ConnectionManager()


def register_all_tools(mcp_server: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register all DerivaML tools, resources, and prompts with the MCP server."""
    # Register tools
    register_annotation_tools(mcp_server, conn_manager)
    register_catalog_tools(mcp_server, conn_manager)
    register_dataset_tools(mcp_server, conn_manager)
    register_vocabulary_tools(mcp_server, conn_manager)
    register_workflow_tools(mcp_server, conn_manager)
    register_feature_tools(mcp_server, conn_manager)
    register_schema_tools(mcp_server, conn_manager)
    register_execution_tools(mcp_server, conn_manager)
    register_storage_tools(mcp_server, conn_manager)
    register_data_tools(mcp_server, conn_manager)
    register_devtools(mcp_server, conn_manager)

    # Register resources
    register_resources(mcp_server, conn_manager)

    # Register prompts
    register_prompts(mcp_server, conn_manager)


# Register all tools
register_all_tools(mcp, connection_manager)


def main() -> None:
    """Run the DerivaML MCP server."""
    logger.info("Starting DerivaML MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
