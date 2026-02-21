"""DerivaML MCP Server.

This module implements the Model Context Protocol server for DerivaML,
exposing DerivaML operations as MCP tools that can be used by LLM applications.

Supports two transport modes:
- stdio (default): Standard input/output for local MCP clients
- streamable-http: HTTP transport for persistent, long-running connections

Features:
- Persistent task storage with crash recovery
- SSE keepalive for proxy-friendly long connections

Usage:
    # STDIO mode (default)
    deriva-mcp

    # HTTP mode
    deriva-mcp --transport streamable-http --host 0.0.0.0 --port 8000

    # HTTP mode with custom keepalive interval
    deriva-mcp --transport streamable-http --sse-keepalive-interval 30
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.prompts import register_prompts
from deriva_ml_mcp.resources import register_resources
from deriva_ml_mcp.tools import (
    register_annotation_tools,
    register_background_task_tools,
    register_catalog_tools,
    register_data_tools,
    register_dataset_tools,
    register_devtools,
    register_execution_tools,
    register_feature_tools,
    register_schema_tools,
    register_storage_tools,
    register_vocabulary_tools,
    register_workflow_tools,
)

# Configure logging - NEVER use print() in STDIO MCP servers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/tmp/deriva-mcp.log")],
)
logger = logging.getLogger("deriva-mcp")

# Server instructions (extracted for reuse)
SERVER_INSTRUCTIONS = """MCP server for DerivaML - manage ML workflows, datasets, and features in Deriva catalogs

## Connection

Always call `connect_catalog` before using other tools. This establishes the connection and auto-detects the domain schema.

## Tool Selection Guidelines

**Prefer high-level tools over low-level operations:**

- Use domain-specific tools like `create_dataset`, `add_dataset_members`, `create_feature`, `add_feature_value`, `create_execution`, etc. instead of raw `insert_records`
- High-level tools properly initialize default values, enforce data integrity constraints, and handle relationships
- Only use `insert_records` as a last resort when no high-level API exists for your use case - it bypasses business logic and may leave required fields uninitialized

**Prefer DerivaML API and datapath over raw ERMRest URLs:**

- Always use MCP tools or the DerivaML Python API (`ml.pathBuilder()`) for data operations
- Never construct raw ERMRest REST URLs (e.g., `https://host/ermrest/catalog/N/entity/...`) for inserts, updates, or deletes
- Raw ERMRest URLs bypass business logic, FK validation, provenance tracking, and version management
- For bulk operations not covered by MCP tools, use the DerivaML Python API with `ml.pathBuilder()` which provides type-safe datapath operations
- If you must read data and no MCP tool exists, use `query_table` rather than constructing ERMRest URLs

**Provenance tracking:**

- Use `create_execution` to track the provenance of datasets, features, and other artifacts
- When creating datasets or features that should be tracked, create them within an execution context
- Executions link inputs, outputs, and configuration for reproducibility

## Common Workflows

**Discovering available catalogs:**
1. Read `deriva-ml://registry/{hostname}` resource - See all available catalogs and aliases
2. `connect_catalog` - Connect using catalog ID or alias name

**Exploring a catalog:**
1. `connect_catalog` - Connect to the catalog
2. Read `deriva-ml://catalog/schema` resource - Understand the full schema (tables, columns, FKs, features)
3. Read `deriva-ml://catalog/datasets` resource - See available datasets
4. Read `deriva-ml://catalog/vocabularies` resource - Explore controlled vocabularies
5. Read `deriva-ml://catalog/features` resource - Examine feature definitions

**Creating a new catalog:**
1. `create_catalog` - Create a new DerivaML catalog (optionally with an alias)
2. `clone_catalog` - Clone an existing catalog to create a copy

**Managing catalog aliases:**
- `create_catalog_alias` - Create an alias for a catalog (access by name instead of ID)
- Read `deriva-ml://alias/{hostname}/{alias_name}` resource - Get alias metadata (target catalog, owner)
- `update_catalog_alias` - Change alias target or owner
- `delete_catalog_alias` - Remove an alias (catalog is not deleted)

**Working with datasets:**
1. `create_dataset` - Create a new dataset with types
2. `add_dataset_members` - Add assets or nested datasets as members
3. `list_dataset_members` - View dataset contents
4. `split_dataset` - Create train/test splits with optional stratification
5. `download_dataset` - Download dataset assets locally
6. `restructure_assets` - Organize downloaded assets into ML-ready directories

**Adding features:**
1. `create_feature` - Define a new feature linking a target table to vocabulary terms
2. `add_feature_value` or `add_feature_value_record` - Assign feature values to records

**Running workflows:**
1. `create_execution` - Start a tracked execution
2. `start_execution` / `stop_execution` - Manage execution lifecycle
3. `upload_execution_outputs` - Upload results to the catalog

**Managing workflows:**

Executions require a workflow, and workflows require a workflow type. The hierarchy is:
- **Workflow_Type** → vocabulary term (e.g., "Training", "Inference")
- **Workflow** → reusable workflow definition
- **Execution** → instance of a workflow run

Before creating an execution:
1. Read `deriva-ml://catalog/workflow-types` resource - See available workflow types
2. Read `deriva-ml://catalog/workflows` resource - Search for existing workflows
3. `lookup_workflow_by_url()` - Check if workflow exists by URL
4. `create_workflow()` - Create new workflow if needed (or let `create_execution` create it)
5. `add_workflow_type()` - Add new workflow type if needed

**Provenance queries:**
- `list_dataset_executions` - Find all executions that used a dataset
- `list_asset_executions` - Find executions that created/used an asset

## Schema Conventions

- `deriva-ml` schema: Core ML tables (Dataset, Execution, Feature, etc.)
- Domain schema: Project-specific tables (assets, vocabularies, features)
- Vocabularies: Controlled term lists for consistent labeling
- Assets: Tables with file attachments (images, models, etc.)

**IMPORTANT: Always use RID as the surrogate key for foreign key relationships.**

- Every table in Deriva has a system-generated `RID` column (unique, immutable identifier)
- When creating foreign keys between tables, always reference the `RID` column, never natural keys like `id`, `name`, or other domain-specific columns
- RID is the standard surrogate key across all Deriva catalogs and ensures consistent behavior with DerivaML tools (datasets, features, associations)
- Using non-RID keys (e.g., integer `id` columns) for FKs can cause type mismatches and break dataset member operations

## Notebook Configuration Pattern

Notebooks use hydra-zen configuration as the primary source of parameters:

1. Define a config module in `src/configs/` that inherits from `BaseConfig`
2. Load configuration in the notebook using `run_notebook()`
3. Access resolved config, DerivaML instance, and execution context

Example notebook setup:
```python
from deriva_ml.execution import run_notebook

# Single call handles all setup
ml, execution, config = run_notebook("my_analysis")

# Ready to use:
# - ml: Connected DerivaML instance
# - execution: Execution context with downloaded inputs
# - config: Resolved configuration (config.assets, config.threshold, etc.)

# Access asset descriptions
if hasattr(config.assets, 'description'):
    print(f"Analysis: {config.assets.description}")
```

**Running notebooks from CLI:**

```bash
# Run with default configuration
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb

# Use a specific named configuration
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb --config roc_lr_sweep

# Override specific parameters
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb assets=different_assets

# Show available configurations
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --info
```

## Dataset Versioning and Configuration

**DatasetSpecConfig requires a version parameter.** When creating `DatasetSpecConfig` entries for hydra-zen configuration files, the `version` parameter is required:

```python
DatasetSpecConfig(rid="28EA", version="0.4.0")  # Correct
DatasetSpecConfig(rid="28EA")  # ERROR: missing required 'version'
```

**Finding the correct version:**
- Read `deriva-ml://dataset/{dataset_rid}` resource to get dataset info including `current_version`
- If no specific version is needed, use the `current_version` from the resource result

**Important: Dataset versions capture catalog state at creation time.**
- A dataset version represents a snapshot of the data at the time the version was created
- If changes have been made to the catalog since the version was created (e.g., adding new features, modifying records), those changes are NOT included in existing versions
- To include recent changes, call `increment_dataset_version` first, then use the new version number
- This ensures reproducibility: the same version always returns the same data

## Dataset Bags (BDBags)

A **BDBag** (Big Data Bag) is a self-describing, portable archive of a specific dataset version.
Use `download_dataset(dataset_rid, version)` to export a dataset as a BDBag for local processing.

**A BDBag for a specific version contains:**

1. **All dataset members** - Records from domain tables (e.g., Image, Subject) that belong to the dataset
2. **Nested datasets** - Child datasets are included recursively with all their members
3. **Asset files** - Binary files (images, model weights, etc.) referenced by members, fetched when `materialize=True`
4. **Feature values** - All feature annotations for dataset members (e.g., Image_Classification labels)
5. **Vocabulary terms** - Controlled vocabulary terms used by features
6. **Catalog snapshot metadata** - The exact catalog state at the version's creation time

**Key characteristics:**

- **Version-specific**: A bag captures the dataset at a specific version's snapshot time
- **Self-contained**: Contains everything needed to reproduce the dataset offline
- **Checksummed**: All files have cryptographic checksums for integrity verification
- **Portable**: Can be shared, archived, or transferred to other systems

**FK path traversal in bag exports:**

When exporting a bag, DerivaML follows **all** foreign key paths from each member element type
to include related data. Paths traverse freely across element-type boundaries — if `Subject`
members have FK paths to `Image`, those `Image` records are included even if `Image` has no
explicit members in the dataset.

**What gets included:**
- All tables reachable via FK paths from member element types
- Feature tables for any reachable element types (e.g., `Image_Classification` for `Image`)
- Vocabulary terms are exported separately (not via FK paths)

**What can go wrong:**
- Deep multi-table FK joins may exceed server query time limits
- If a query fails, `download_dataset_bag()` raises a `DerivaMLException` with guidance
- **Fix**: Add the desired records as direct dataset members to use simpler association paths
  (register the table as an element type with `add_dataset_element_type()` first)

**Example:** A dataset with `Subject` members will automatically include FK-reachable `Image`
records via the `Subject → Image` path. If the join is too large, add `Image` RIDs as explicit
members instead.

**Materialization:**

When `materialize=True` (the default), the bag fetches all referenced asset files from Hatrac storage.
This creates a fully self-contained archive. With `materialize=False`, the bag contains only
metadata and remote file references (smaller but requires network access to use).

**Caching:**

Bags are cached locally by checksum in the DerivaML cache directory. When you download the same
dataset version again, the cached bag is reused without re-downloading. This makes repeated
access to the same dataset version very fast. The cache key is `{dataset_rid}_{checksum}`.

The cache location can be configured via the `cache_dir` argument when creating a DerivaML instance.
If not specified, bags are cached in a default location within the user's home directory.

**Asset Caching:**

Individual execution assets (e.g., model weights) can be cached using `AssetSpec(cache=True)`:

```python
from deriva_ml.asset.aux_classes import AssetSpec

config = ExecutionConfiguration(
    assets=[
        AssetSpec(rid="6-EPNR", cache=True),   # cached by MD5
        "6-EP56",                                # not cached
    ]
)
```

Cached assets are stored in `cache_dir/assets/{rid}_{md5}/` and symlinked into execution
directories. The MD5 checksum is compared against the catalog record to detect stale caches.
Use `cache=True` for large, immutable assets like pre-trained model weights.

For hydra-zen configs, use `AssetSpecConfig(rid=..., cache=True)`:

```python
from deriva_ml.asset.aux_classes import AssetSpecConfig

asset_store(
    with_description(
        [AssetSpecConfig(rid="6-EPNR", cache=True)],
        "MAE weights for OCT. ~3.7GB.",
    ),
    name="retfound_mae_oct_weights",
)
```

**MINID support:**

Bags can be registered with a MINID (Minimal Viable Identifier) for permanent, citable references.
This requires S3 bucket configuration on the catalog.

**Example workflow:**
```python
# Download dataset for local ML training
download_dataset("ABC123", version="1.2.0", materialize=True)
# Returns: {"bag_path": "/path/to/Dataset_ABC123", ...}

# The bag contains all members, nested datasets, features, and asset files
# at the exact catalog state when version 1.2.0 was created
```

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

**Multirun execution hierarchy:**

Multiruns create a parent-child execution structure:
- **Parent execution**: Created first, contains the multirun description and links to all child runs
- **Child executions**: One per parameter combination, each with full provenance

Example: `+multirun=lr_sweep` with 4 learning rates creates 5 executions (1 parent + 4 children).

Use `list_parent_executions` and `list_nested_executions` to navigate this hierarchy.

## Discovering Execution Outputs

After running experiments, use MCP resources and tools to discover generated assets:

```
# Browse all assets in a specific table
Read resource: deriva-ml://table/{asset_table}/assets

# Get details about a specific asset including provenance
Read resource: deriva-ml://asset/{asset_rid}

# Find executions that created/used an asset
list_asset_executions(asset_rid="<ASSET_RID>", asset_role="Output")

# Get experiment details including inputs and outputs
Read resource: deriva-ml://experiment/{execution_rid}
```

This is useful for:
- Gathering asset RIDs to create analysis configurations
- Verifying expected outputs were generated
- Building asset groups for comparative analysis (e.g., ROC curves)

## Documentation Best Practices

**Always provide descriptions for tables, columns, datasets, executions, and other catalog entities.**

Good documentation makes catalogs self-explanatory and helps users understand data provenance:

- **Tables**: Use `set_table_description` to explain what the table stores and its role in the workflow
- **Columns**: Use `set_column_description` to explain what each column contains, units, valid ranges, etc.
- **Datasets**: Always provide a `description` when creating datasets explaining their purpose and contents
- **Executions**: Include a `description` explaining what the execution does and its expected outputs
- **Vocabulary terms**: Provide clear `description` values explaining what each term means
- **Features**: Document what the feature represents and how values should be interpreted

**Asset configuration descriptions:**

When defining asset configurations in hydra-zen, use `with_description()` to document what assets are included and why:

```python
from deriva_ml.execution import with_description

asset_store(
    with_description(
        ["RID1", "RID2"],
        '''Prediction probability files from learning rate sweep.

Compares four learning rates on the small labeled split:
- lr=0.0001: Conservative, slow convergence
- lr=0.001: Standard baseline
- lr=0.01: Aggressive, may show instability
- lr=0.1: Very aggressive

Use with roc_analysis notebook to compare AUC scores.''',
    ),
    name="roc_lr_sweep",
)
```

The description should explain:
- What the assets contain (e.g., model weights, prediction probabilities)
- Which experiments/executions produced them
- Key parameters that differ between assets
- How to use them (e.g., which notebook or analysis)

**Display configuration improves usability:**

- Use `set_table_display_name` and `set_column_display_name` for human-readable names in the UI
- Use `set_row_name_pattern` to control how rows appear in dropdowns and references
- Use `set_visible_columns` to show the most relevant columns first

## Code Provenance Best Practices

**Commit code before running models.** DerivaML tracks code provenance by recording:
- Git commit hash
- Repository URL
- Working directory state

If there are uncommitted changes, the execution record won't have a valid code reference. Always:
1. Stage and commit changes before running `deriva-ml-run`
2. Use `dry_run=true` during debugging to test without creating execution records
3. Tag versions with semantic versioning before significant model runs

**Script-based workflow for catalog operations:**

For operations that modify catalog data (dataset creation, splitting, ETL, feature creation,
data loading), prefer generating a **committed script** over using interactive MCP tools:

1. Generate a Python script in `scripts/` using the DerivaML Python API
2. Test with `--dry-run` to verify correctness
3. Commit the script to the repository
4. Run for real — the execution record captures the git commit hash

This ensures every catalog mutation has a permanent, versioned code reference.
See the `catalog-operations-workflow` prompt for templates and detailed guidance.

MCP tools are still appropriate for read-only exploration, one-time admin tasks, and
queries — anything that doesn't need reproducibility tracking.

**Dry run mode:**
- `dry_run=true` downloads input datasets/assets but skips execution record creation
- Useful for testing data loading, configuration, and model initialization
- No catalog writes occur in dry run mode

## Dataset Splits for Evaluation

**Use the `split_dataset` tool to create train/test splits.** It follows scikit-learn conventions
and handles provenance tracking automatically.

**Simple random split:**
```python
split_dataset("1-ABC", test_size=0.2, seed=42)
```

**Stratified split (maintains class distribution):**
```python
split_dataset("1-ABC", test_size=0.2,
              stratify_by_column="Image_Classification_Image_Class",
              include_tables=["Image", "Image_Classification"])
```

The `stratify_by_column` uses the denormalized column name format `{TableName}_{ColumnName}`.
Use `denormalize_dataset` first to discover available column names.

**When creating train/test splits, consider whether ground truth labels are needed:**

- **Unlabeled test splits**: Test partition has no ground truth labels. Use for training pipelines where test evaluation isn't needed during training.
- **Labeled test splits**: Both train and test partitions have ground truth labels. Required for:
  - Computing accuracy metrics on test set
  - Generating ROC curves or other evaluation metrics
  - Comparing model predictions to ground truth

**Pattern:** Use `training_types=["Labeled"]` and `testing_types=["Labeled"]` when both splits need ground truth labels:
```python
split_dataset("1-ABC", test_size=0.2, seed=42,
              training_types=["Labeled"], testing_types=["Labeled"])
```

**Dry run to preview split plan:**
```python
split_dataset("1-ABC", test_size=0.2, dry_run=True)
# Returns counts and strategy without modifying catalog
```

**Code provenance for splits:**

For full provenance tracking, prefer the script-based workflow: generate a split script,
test with `--dry-run`, commit, then run. The interactive `split_dataset` MCP tool creates
execution records but cannot capture code provenance since there is no committed script.
See the `catalog-operations-workflow` prompt for the recommended workflow.

## Restructuring Assets for ML Frameworks

After downloading a dataset, use `restructure_assets` to organize files into the directory
structure expected by ML frameworks (e.g., PyTorch ImageFolder):

```python
restructure_assets(dataset_rid="1-ABC", asset_table="Image",
                   output_dir="./ml_data", group_by=["Diagnosis"])
```

This creates:
```
./ml_data/
  Training/
    Normal/
      image1.jpg
      image2.jpg
    Abnormal/
      image3.jpg
  Testing/
    Normal/
      image4.jpg
    Abnormal/
      image5.jpg
```

By default, symlinks are used to save disk space. Set `use_symlinks=False` to copy files instead.

## Notebook Display Utilities

**Dataset and Experiment classes provide markdown output methods:**

Both `Dataset` and `Experiment` classes have `to_markdown()` and `display_markdown()` methods for generating formatted output in Jupyter notebooks:

```python
# Get markdown string for custom use
md_str = dataset.to_markdown(show_children=True)
md_str = experiment.to_markdown(show_datasets=True, show_assets=True)

# Display directly in Jupyter
dataset.display_markdown(show_children=True)
experiment.display_markdown()
```

**Experiment.to_markdown()** returns:
- Header with name and link to execution
- Description (if available)
- Configuration choices (Hydra config names used)
- Model configuration (hyperparameters)
- Input datasets (with nested children)
- Input assets

**Dataset.to_markdown()** returns:
- Link to dataset record
- Version, types, and description
- Optionally nested child datasets

**Loading experiments from assets:**

When analyzing assets (e.g., prediction files), use `lookup_experiment()` to get the source experiment:

```python
# From asset path in execution context
asset = ml.lookup_asset(asset_path.asset_rid)
executions = asset.list_executions(asset_role='Output')

if executions:
    # executions returns ExecutionRecord objects with execution_rid attribute
    exp = ml.lookup_experiment(executions[0].execution_rid)
    exp.display_markdown()  # Show full experiment details
```

## Asset Management

**Use `lookup_asset()` to get detailed asset information:**

```python
# Look up an asset by RID
asset = ml.lookup_asset("3JSE")
print(f"File: {asset.filename}, Table: {asset.asset_table}")
print(f"Types: {asset.asset_types}")
print(f"Created by: {asset.execution_rid}")

# Find executions that used this asset
executions = asset.list_executions()
for exe in executions:
    print(f"Execution {exe.execution_rid}: {exe.description}")
```

**Asset methods return typed objects:**
- `ml.list_assets(table)` returns `list[Asset]`
- `ml.find_assets()` returns iterable of `Asset` objects
- `asset.list_executions()` returns `list[ExecutionRecord]`
- `ml.list_asset_executions(rid)` returns `list[ExecutionRecord]`

## ExecutionRecord vs Execution

DerivaML has two execution classes for different use cases:

**ExecutionRecord** - Lightweight catalog state representation:
- Returned by `ml.lookup_execution()`, `asset.list_executions()`, etc.
- Has `execution_rid`, `workflow_rid`, `status`, `description` properties
- Use for querying and viewing execution metadata
- Supports `update_status()` for updating catalog state

**Execution** - Full lifecycle management:
- Returned by `ml.create_execution()` and `ml.restore_execution()`
- Includes working directory, asset upload, dataset download
- Use for running ML workflows with provenance tracking

When you need to query execution history, the methods return ExecutionRecord objects.
When you need to run a workflow, use create_execution() or restore_execution().

## Citation URLs

**Use `ml.cite()` to generate permanent URLs to catalog entities:**

```python
# Get permanent citation URL (with snapshot timestamp)
url = ml.cite(rid)  # Returns: https://host/id/catalog/RID@snaptime

# Get URL to current catalog state (no snapshot)
url = ml.cite(rid, current=True)  # Returns: https://host/id/catalog/RID
```

The `cite()` method:
- Accepts either a RID string or a dictionary with a 'RID' key
- By default returns a permanent URL with snapshot timestamp for reproducibility
- With `current=True`, returns a URL to the current catalog state (useful for linking to live data)
- Validates that the RID exists in the catalog

## Before Calling Tools

**Always verify required parameters before calling any tool.** Check the tool's description and parameter schema to understand which parameters are required vs optional. Never assume a parameter is optional - verify first.

## Background Tasks for Long-Running Operations

Some operations like catalog cloning can take many minutes. Use the async versions
of these tools to avoid timeout issues:

**Starting a long-running operation:**
```python
# Instead of clone_catalog (which may timeout), use:
clone_catalog_async("www.facebase.org", "1",
                    root_rid="3-HXMC",
                    dest_hostname="localhost",
                    alias="my-clone")
# Returns immediately with: {"task_id": "abc123", "status": "started", ...}
```

**Checking progress:**
```python
get_task_status("abc123")
# Returns: {"status": "running", "progress": {"percent_complete": 45.0, ...}}
```

**When complete:**
```python
get_task_status("abc123")
# Returns: {"status": "completed", "result": {...full clone result...}}
```

**Managing tasks:**
- `list_tasks()` - See all your tasks
- `list_tasks(status="running")` - Filter by status
- `cancel_task("abc123")` - Cancel a running task

Tasks are isolated per user - you can only see and manage your own tasks.
"""

# Global connection manager (shared across server instances)
connection_manager = ConnectionManager()

# SSE keepalive interval (seconds) - patched into EventSourceResponse
_sse_keepalive_interval: int = 30


def patch_sse_keepalive(ping_interval: int) -> None:
    """Patch sse_starlette.EventSourceResponse to include keepalive pings.

    The MCP SDK uses EventSourceResponse but doesn't expose the ping parameter.
    This monkey-patch wraps the class to inject our configured ping interval,
    preventing proxy timeouts for long-running SSE connections.

    Args:
        ping_interval: Seconds between keepalive pings (0 to disable)
    """
    global _sse_keepalive_interval
    _sse_keepalive_interval = ping_interval

    if ping_interval <= 0:
        logger.info("SSE keepalive disabled")
        return

    try:
        import sse_starlette

        _original_init = sse_starlette.EventSourceResponse.__init__

        def patched_init(self, *args, **kwargs):
            # Inject ping if not already set
            if "ping" not in kwargs:
                kwargs["ping"] = _sse_keepalive_interval
            _original_init(self, *args, **kwargs)

        sse_starlette.EventSourceResponse.__init__ = patched_init
        logger.info(f"SSE keepalive enabled (interval: {ping_interval}s)")

    except Exception as e:
        logger.warning(f"Failed to patch SSE keepalive: {e}")


def register_all_tools(mcp_server: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register all DerivaML tools, resources, and prompts with the MCP server."""
    # Register tools
    register_annotation_tools(mcp_server, conn_manager)
    register_background_task_tools(mcp_server, conn_manager)
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


def create_server(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    """Create and configure the MCP server.

    Args:
        host: Host to bind for HTTP transport (default: 127.0.0.1)
        port: Port to bind for HTTP transport (default: 8000)

    Returns:
        Configured FastMCP server instance
    """
    mcp = FastMCP(
        "deriva-ml",
        host=host,
        port=port,
        instructions=SERVER_INSTRUCTIONS,
    )

    # Register all tools, resources, and prompts
    register_all_tools(mcp, connection_manager)

    # Health check endpoint - does NOT create MCP sessions
    # This is critical for Docker health checks which poll every 30 seconds.
    # Without this, each health check would create an orphan session that
    # never gets cleaned up, leading to resource exhaustion.
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> Response:
        """Health check endpoint for container orchestration.

        Returns basic server status without creating an MCP session.
        Use this endpoint for Docker/Kubernetes health probes instead of /mcp.
        """
        from deriva_ml_mcp import __version__

        return JSONResponse({
            "status": "ok",
            "service": "deriva-mcp",
            "version": __version__,
        })

    return mcp


# For backward compatibility: create default server at module load time
# This allows existing code that imports `mcp` directly to continue working
mcp = create_server()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DerivaML MCP Server - Model Context Protocol server for Deriva catalogs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with STDIO transport (default, for local MCP clients)
  deriva-mcp

  # Run with HTTP transport (for persistent connections)
  deriva-mcp --transport streamable-http --host 0.0.0.0 --port 8000

  # Run HTTP server with custom keepalive and task settings
  deriva-mcp --transport streamable-http \\
    --sse-keepalive-interval 30 \\
    --task-state-path /data/tasks.json \\
    --task-retention-hours 168

Environment variables (override defaults):
  DERIVA_MCP_TASK_STATE_PATH      - Path to task state file
  DERIVA_MCP_TASK_RETENTION_HOURS - Hours to retain completed tasks
  DERIVA_MCP_TASK_SYNC_INTERVAL   - Seconds between task state saves
  DERIVA_MCP_SSE_KEEPALIVE        - SSE keepalive interval in seconds
""",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP transport (default: 8000)",
    )

    # Task persistence options
    parser.add_argument(
        "--task-state-path",
        type=Path,
        default=None,
        help="Path to task state file (default: ~/.deriva-ml/task_state.json)",
    )
    parser.add_argument(
        "--task-retention-hours",
        type=int,
        default=None,
        help="Hours to retain completed tasks (default: 168 = 7 days)",
    )
    parser.add_argument(
        "--task-sync-interval",
        type=int,
        default=None,
        help="Seconds between task state saves (default: 5)",
    )

    # SSE keepalive options
    parser.add_argument(
        "--sse-keepalive-interval",
        type=int,
        default=None,
        help="SSE keepalive interval in seconds (default: 30, 0 to disable)",
    )

    return parser.parse_args()


def main() -> None:
    """Run the DerivaML MCP server."""
    import atexit

    from deriva_ml_mcp.tasks import TaskPersistence, init_task_manager

    args = parse_args()

    # Resolve configuration from args and environment variables
    # Environment variables take precedence over defaults, CLI args override both

    # Task persistence configuration
    task_state_path = args.task_state_path
    if task_state_path is None:
        env_path = os.environ.get("DERIVA_MCP_TASK_STATE_PATH")
        task_state_path = Path(env_path) if env_path else None

    task_retention_hours = args.task_retention_hours
    if task_retention_hours is None:
        env_retention = os.environ.get("DERIVA_MCP_TASK_RETENTION_HOURS")
        task_retention_hours = int(env_retention) if env_retention else TaskPersistence.DEFAULT_RETENTION_HOURS

    task_sync_interval = args.task_sync_interval
    if task_sync_interval is None:
        env_sync = os.environ.get("DERIVA_MCP_TASK_SYNC_INTERVAL")
        task_sync_interval = int(env_sync) if env_sync else TaskPersistence.DEFAULT_SYNC_INTERVAL

    # SSE keepalive configuration
    sse_keepalive_interval = args.sse_keepalive_interval
    if sse_keepalive_interval is None:
        env_keepalive = os.environ.get("DERIVA_MCP_SSE_KEEPALIVE")
        sse_keepalive_interval = int(env_keepalive) if env_keepalive else 30

    # Initialize task manager with persistence
    init_task_manager(
        persistence_path=task_state_path,
        sync_interval=task_sync_interval,
        retention_hours=task_retention_hours,
    )

    # Create server with appropriate settings
    server = create_server(host=args.host, port=args.port)

    # Register shutdown handler for background task manager
    def shutdown_task_manager() -> None:
        logger.info("Shutting down background task manager")
        try:
            from deriva_ml_mcp.tasks import get_task_manager

            task_manager = get_task_manager()
            task_manager.shutdown(wait=False)  # Don't block on pending tasks
        except Exception as e:
            logger.warning(f"Error shutting down task manager: {e}")

    atexit.register(shutdown_task_manager)

    transport: Literal["stdio", "streamable-http"] = args.transport
    if transport == "streamable-http":
        # Enable SSE keepalive for HTTP transport
        patch_sse_keepalive(sse_keepalive_interval)

        logger.info(
            f"Starting DerivaML MCP server (transport={transport}, "
            f"host={args.host}, port={args.port}, "
            f"sse_keepalive={sse_keepalive_interval}s)"
        )
    else:
        logger.info(f"Starting DerivaML MCP server (transport={transport})")

    server.run(transport=transport)


if __name__ == "__main__":
    main()
