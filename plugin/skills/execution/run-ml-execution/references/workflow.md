# Running an ML Execution with Provenance

An execution is the fundamental unit of provenance in DerivaML. Every data transformation, model training run, or analysis should be wrapped in an execution to track what was done, with what inputs, and what outputs were produced.

## Concepts

- **Execution**: A recorded unit of work with inputs (datasets, assets), outputs (files, new data), and metadata (workflow, description, status).
- **Workflow**: A reusable definition of a type of work (e.g., "Image Classification Training"). Executions reference a workflow.
- **ExecutionConfiguration**: Specifies the workflow, input datasets, and assets for an execution.

## Python API: Context Manager Pattern

The recommended approach uses a `with` block that auto-starts and auto-stops the execution:

```python
from deriva.ml import DerivaML, ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)

# 1. Find or create a workflow
workflows = ml.list_workflows()
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
    output_path = exe.asset_file_path("model_weights.pt", description="Trained model weights")
    save_model(results, output_path)

    metrics_path = exe.asset_file_path("metrics.json", description="Training metrics")
    save_metrics(results, metrics_path)

# 4. Upload AFTER exiting the context manager
exe.upload_execution_outputs()
```

**Key points about the context manager:**
- `with` block automatically calls `start_execution()` on entry and `stop_execution()` on exit.
- If an exception occurs inside the block, the execution status is set to "Failed".
- You MUST call `upload_execution_outputs()` AFTER exiting the `with` block, not inside it.
- Use `asset_file_path()` to register output files -- this both creates the file path and registers it as an output asset.

## MCP Tools Workflow

For interactive use or when working through the MCP interface:

```
Step 1: Find or create a workflow
  -> query_table(table="Workflow") or create_workflow(...)

Step 2: Create the execution
  -> create_execution(
       workflow_rid="2-XXXX",
       description="Training run on labeled images",
       dataset_rids=["2-ABC1"],
       asset_rids=["2-DEF2"]
     )
  Returns: execution RID

Step 3: Start the execution
  -> start_execution(execution_rid="2-YYYY")

Step 4: Download input data
  -> download_execution_dataset(execution_rid="2-YYYY")

Step 5: Do your work
  (run notebooks, scripts, generate outputs)

Step 6: Register output files
  -> asset_file_path(execution_rid="2-YYYY", filename="results.csv", description="Model predictions")

Step 7: Stop the execution
  -> stop_execution(execution_rid="2-YYYY")

Step 8: Upload outputs
  -> upload_execution_outputs(execution_rid="2-YYYY")
```

## ExecutionConfiguration Details

```python
from deriva.ml import ExecutionConfiguration

config = ExecutionConfiguration(
    workflow=workflow_rid_or_object,   # Required: which workflow
    datasets=["2-ABC1", "2-ABC2"],    # Optional: input dataset RIDs
    assets=["2-DEF1"],                # Optional: input asset RIDs
    description="Run description",     # Optional: execution description
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
download_execution_dataset(execution_rid="2-YYYY")
```

The downloaded data lands in the execution's working directory under a structured layout.

## Registering Output Files

Use `asset_file_path()` to both get the correct output path and register the file as an execution output:

```python
# Python API
output_path = exe.asset_file_path("predictions.csv", description="Model predictions on test set")
# Write your data to output_path
df.to_csv(output_path, index=False)

# For subdirectories
nested_path = exe.asset_file_path("plots/confusion_matrix.png", description="Confusion matrix plot")
```

```
# MCP tools
asset_file_path(execution_rid="2-YYYY", filename="predictions.csv", description="Model predictions")
```

## Useful Inspection and Management Tools

### Get execution info
```python
info = ml.get_execution_info(execution_rid="2-YYYY")
# Returns: workflow, status, datasets, assets, nested executions, timestamps
```

### Update execution status
```python
ml.update_execution_status(execution_rid="2-YYYY", status="Running")
# Valid statuses: Pending, Running, Complete, Failed
```

### Restore a previous execution
```python
ml.restore_execution(execution_rid="2-YYYY")
# Re-downloads execution assets and datasets to local working directory
# Useful for debugging or continuing work from a previous execution
```

### Get execution working directory
```python
working_dir = ml.get_execution_working_dir(execution_rid="2-YYYY")
```

### Nested executions
For multi-step pipelines, create nested executions within a parent:

```python
with ml.create_execution(parent_config) as parent_exe:
    # First step
    with ml.add_nested_execution(parent_exe, step_config) as step1:
        # ... do step 1 work ...

    # Second step
    with ml.add_nested_execution(parent_exe, step2_config) as step2:
        # ... do step 2 work ...
```

```
# MCP tools
add_nested_execution(parent_rid="2-PARENT", workflow_rid="2-STEP1_WF", description="Preprocessing step")
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
add_asset_type(name="Normalized Image", description="Image after intensity normalization")

# Tag an asset with a type
add_asset_type_to_asset(asset_rid="2-IMG1", asset_type="Normalized Image")

# Remove a type tag
remove_asset_type_from_asset(asset_rid="2-IMG1", asset_type="Normalized Image")
```

## Creating New Asset Tables

When you need a new category of files:

```
create_asset_table(
    table_name="Processed_Image",
    columns=[{"name": "Resolution", "type": "text", "nullok": true, "comment": "Image resolution"}],
    referenced_tables=["Subject"]
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
