# Running an ML Execution with Provenance

An execution is the fundamental unit of provenance in DerivaML. Every data transformation, model training run, or analysis should be wrapped in an execution to track what was done, with what inputs, and what outputs were produced.

## Table of Contents

- [Concepts](#concepts)
- [Python API: Context Manager Pattern](#python-api-context-manager-pattern)
- [MCP Tools Workflow](#mcp-tools-workflow)
- [ExecutionConfiguration Details](#executionconfiguration-details)
- [Downloading Execution Datasets](#downloading-execution-datasets)
- [Registering Output Files](#registering-output-files)
- [Useful Inspection and Management Tools](#useful-inspection-and-management-tools)
- [Managing Asset Types](#managing-asset-types)
- [Creating New Asset Tables](#creating-new-asset-tables)
- [Tips](#tips)

---

## Concepts

- **Execution**: A recorded unit of work with inputs (datasets, assets), outputs (files, new data), and metadata (workflow, description, status).
- **Workflow**: A reusable definition of a type of work (e.g., "Image Classification Training"). Executions reference a workflow.
- **ExecutionConfiguration**: Specifies the workflow, input datasets, and assets for an execution.

## Python API: Context Manager Pattern

The recommended approach uses a `with` block that auto-starts and auto-stops the execution:

```python
from deriva_ml import DerivaML, ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)

# 1. Find or create a workflow
workflow = ml.create_workflow(
    name="Image Classification Training",
    url="https://github.com/org/repo",
    workflow_type="Training",
    description="Train CNN on labeled image dataset"
)

# 2. Configure the execution
config = ExecutionConfiguration(
    workflow=workflow,
    datasets=["2-ABC1"],          # Dataset RIDs to use as input
    assets=["2-DEF2", "2-GHI3"]  # Individual asset RIDs
)

# 3. Run within context manager
with ml.create_execution(config) as exe:
    # Execution is automatically started

    # Download input datasets
    exe.download_execution_dataset()

    # Do your work...
    results = train_model(exe.working_dir)

    # Write output files using asset_file_path()
    output_path = exe.asset_file_path("Execution_Asset", "model_weights.pt")
    save_model(results, output_path)

    metrics_path = exe.asset_file_path("Execution_Asset", "metrics.json")
    save_metrics(results, metrics_path)

# 4. Upload AFTER exiting the context manager
exe.upload_execution_outputs()
```

**Key points about the context manager:**
- `with` block automatically calls `start_execution()` on entry and `stop_execution()` on exit.
- If an exception occurs inside the block, the execution status is set to "Failed".
- You MUST call `upload_execution_outputs()` AFTER exiting the `with` block, not inside it.
- Use `asset_file_path(asset_table, filename)` to register output files -- this both creates the file path and registers it as an output asset.

## MCP Tools Workflow

For interactive use or when working through the MCP interface:

```
Step 1: Create the execution (also finds/creates the workflow)
  -> create_execution(
       workflow_name="Image Classification Training",
       workflow_type="Training",
       description="Training run on labeled images",
       dataset_rids=["2-ABC1"],
       asset_rids=["2-DEF2"]
     )
  Returns: execution RID

Step 2: Start the execution
  -> start_execution()

Step 3: Download input data
  -> download_execution_dataset(dataset_rid="2-ABC1", version="1.0.0")

Step 4: Do your work
  (run notebooks, scripts, generate outputs)

Step 5: Register output files
  -> asset_file_path(asset_name="Execution_Asset", file_name="results.csv")

Step 6: Stop the execution
  -> stop_execution()

Step 7: Upload outputs
  -> upload_execution_outputs()
```

**Important:** MCP execution lifecycle tools (`start_execution`, `stop_execution`, `get_execution_working_dir`, `upload_execution_outputs`) operate on the **active execution** -- they take no `execution_rid` parameter.

## ExecutionConfiguration Details

```python
from deriva_ml import ExecutionConfiguration

config = ExecutionConfiguration(
    workflow=workflow_object,              # Required: Workflow object from create_workflow/lookup
    datasets=["2-ABC1", "2-ABC2"],        # Optional: input dataset RIDs
    assets=["2-DEF1"],                    # Optional: input asset RIDs
    description="Run description",         # Optional: execution description
)
```

## Downloading Execution Datasets

Once an execution is started, download all configured input datasets:

```python
# Python API
with ml.create_execution(config) as exe:
    dataset_paths = exe.download_execution_dataset()
    # Returns dict mapping dataset RID -> local directory path
```

```
# MCP tools
download_execution_dataset(dataset_rid="2-ABC1", version="1.0.0")
```

The downloaded data lands in the execution's working directory under a structured layout.

## Registering Output Files

Use `asset_file_path()` to both get the correct output path and register the file as an execution output:

```python
# Python API — two required args: asset_table and filename
output_path = exe.asset_file_path("Execution_Asset", "predictions.csv")
# Write your data to output_path
df.to_csv(output_path, index=False)
```

```
# MCP tools
asset_file_path(asset_name="Execution_Asset", file_name="predictions.csv")
```

## Useful Inspection and Management Tools

### Get execution info
```
# MCP tool
get_execution_info(execution_rid="2-YYYY")
# Returns: workflow, status, datasets, assets, nested executions, timestamps
```

### Update execution status
```
# MCP tool
update_execution_status(status="Running", message="Processing batch 3")
# Valid statuses: Pending, Running, Complete, Failed
```

### Restore a previous execution
```
# MCP tool
restore_execution(execution_rid="2-YYYY")
# Re-downloads execution assets and datasets to local working directory
# Useful for debugging or continuing work from a previous execution
```

### Get execution working directory
```
# MCP tool (operates on active execution, no params)
get_execution_working_dir()
```

### Nested executions
For multi-step pipelines, create nested executions within a parent:

```
# MCP tools
# First, create both parent and child executions, then link them:
add_nested_execution(parent_execution_rid="2-PARENT", child_execution_rid="2-CHILD")
```

### List related executions
```
list_nested_executions(execution_rid="2-YYYY")   # Child executions
list_parent_executions(execution_rid="2-YYYY")    # Parent executions
list_dataset_executions(dataset_rid="2-ABC1")     # Executions that used this dataset
list_asset_executions(asset_rid="2-DEF2")         # Executions that used this asset
```

## Managing Asset Types

Asset Types are vocabulary terms that categorize assets (e.g., "Raw Image", "Trained Model", "Preprocessed CSV").

```
# Create a new asset type
add_asset_type(type_name="Normalized Image", description="Image after intensity normalization")

# Tag an asset with a type
add_asset_type_to_asset(asset_rid="2-IMG1", asset_type="Normalized Image")

# Remove a type tag
remove_asset_type_from_asset(asset_rid="2-IMG1", asset_type="Normalized Image")
```

## Creating New Asset Tables

When you need a new category of files:

```
create_asset_table(
    asset_name="Processed_Image",
    columns=[{"name": "Resolution", "type": "text", "nullok": true, "comment": "Image resolution"}],
    referenced_tables=["Subject"],
    comment="Processed microscopy images"
)
```

This creates the table with standard asset columns (URL, Filename, Length, MD5, Description) plus any custom columns.

## Tips

- Always wrap work in an execution for provenance tracking.
- Upload outputs AFTER the `with` block exits, never inside it.
- Use `asset_file_path()` for every output file -- do not manually place files in the working directory.
- Set meaningful descriptions on workflows, executions, and output assets.
- For long-running work, use `update_execution_status()` to track progress.
- Use `restore_execution()` to resume or inspect a completed execution's local state.
- Nested executions are ideal for multi-phase pipelines (preprocessing, training, evaluation).
