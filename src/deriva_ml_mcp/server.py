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

**Discovering available catalogs:**
1. `list_catalog_registry` - Query a server to see all available catalogs and aliases
2. `connect_catalog` - Connect using catalog ID or alias name

**Exploring a catalog:**
1. `connect_catalog` - Connect to the catalog
2. `list_tables` or `get_schema_description` - Understand the schema
3. `list_datasets` - See available datasets
4. `list_vocabularies` / `list_vocabulary_terms` - Explore controlled vocabularies
5. `list_features` / `list_feature_values` - Examine feature definitions

**Creating a new catalog:**
1. `create_catalog` - Create a new DerivaML catalog (optionally with an alias)
2. `clone_catalog` - Clone an existing catalog to create a copy

**Managing catalog aliases:**
- `create_catalog_alias` - Create an alias for a catalog (access by name instead of ID)
- `get_catalog_alias` - Get alias metadata (target catalog, owner)
- `update_catalog_alias` - Change alias target or owner
- `delete_catalog_alias` - Remove an alias (catalog is not deleted)

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

**Provenance queries:**
- `list_dataset_executions` - Find all executions that used a dataset
- `list_asset_executions` - Find executions that created/used an asset

## Schema Conventions

- `deriva-ml` schema: Core ML tables (Dataset, Execution, Feature, etc.)
- Domain schema: Project-specific tables (assets, vocabularies, features)
- Vocabularies: Controlled term lists for consistent labeling
- Assets: Tables with file attachments (images, models, etc.)

## Notebook Configuration Pattern

Notebooks use hydra-zen configuration as the primary source of parameters:

1. Define a config module in `src/configs/` that inherits from `BaseConfig`
2. Load configuration in the notebook using `get_notebook_configuration()`
3. Extract connection settings and parameters from the resolved config object

Example notebook setup:
```python
from configs import load_all_configs
from configs.my_notebook import MyNotebookConfigBuilds
from deriva_ml.execution import get_notebook_configuration

load_all_configs()
config = get_notebook_configuration(
    MyNotebookConfigBuilds,
    config_name="my_notebook",
    overrides=["assets=different_assets"],  # Optional
)
host = config.deriva_ml.hostname
catalog = config.deriva_ml.catalog_id
assets = config.assets
```

## Dataset Versioning and Configuration

**DatasetSpecConfig requires a version parameter.** When creating `DatasetSpecConfig` entries for hydra-zen configuration files, the `version` parameter is required:

```python
DatasetSpecConfig(rid="28EA", version="0.4.0")  # Correct
DatasetSpecConfig(rid="28EA")  # ERROR: missing required 'version'
```

**Finding the correct version:**
- Use `lookup_dataset(rid)` to get dataset info including `current_version`
- If no specific version is needed, use the `current_version` from the lookup result

**Important: Dataset versions capture catalog state at creation time.**
- A dataset version represents a snapshot of the data at the time the version was created
- If changes have been made to the catalog since the version was created (e.g., adding new features, modifying records), those changes are NOT included in existing versions
- To include recent changes, call `increment_dataset_version` first, then use the new version number
- This ensures reproducibility: the same version always returns the same data

## Running Models with deriva-ml-run

The `deriva-ml-run` CLI executes ML models with full provenance tracking. It uses hydra-zen configuration for all parameters.

**Single experiment:**
```bash
uv run deriva-ml-run +experiment=cifar10_quick
```

**Multirun experiments (using multirun_config):**
```bash
# Run a predefined multirun configuration
uv run deriva-ml-run +multirun=quick_vs_extended

# Override parameters from the multirun config
uv run deriva-ml-run +multirun=lr_sweep model_config.epochs=5

# List available configs
uv run deriva-ml-run --info
```

**Defining multirun configurations:**

Multirun configs bundle Hydra overrides with rich markdown descriptions. Define them in `configs/multiruns.py`:

```python
from deriva_ml.execution import multirun_config

multirun_config(
    "quick_vs_extended",
    overrides=[
        "+experiment=cifar10_quick,cifar10_extended",
    ],
    description='''## Model Comparison

    Comparing quick vs extended training...

    | Config | Epochs | Architecture |
    |--------|--------|--------------|
    | quick | 3 | 32->64 channels |
    | extended | 50 | 64->128 channels |
    ''',
)

multirun_config(
    "lr_sweep",
    overrides=[
        "+experiment=cifar10_lr_sweep",
        "model_config.learning_rate=0.0001,0.001,0.01,0.1",
    ],
    description="Learning rate hyperparameter sweep",
)
```

**Benefits of multirun_config:**
- No need to remember `--multirun` flag - automatically enabled
- Rich markdown descriptions for parent executions (supports tables, headers, etc.)
- Reproducible sweeps documented in code
- Same Hydra override syntax as command line

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
