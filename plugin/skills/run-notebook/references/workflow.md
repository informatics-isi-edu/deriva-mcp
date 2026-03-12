# Develop and Run a DerivaML Notebook with Execution Tracking

This skill covers how to structure, develop, and run DerivaML Jupyter notebooks with full execution tracking and provenance.

## Step 0: Verify Environment

Before starting, confirm your environment is set up. See the `setup-notebook-environment` skill for full details. Quick check:

```bash
uv sync --group=jupyter
uv run jupyter kernelspec list  # Should show your project kernel
uv run deriva-globus-auth-utils login --host ml.derivacloud.org --no-browser  # Should confirm auth
```

## Create a Notebook from Template

Start from the project's notebook template if one exists, or create a new notebook with the required structure described below.

## Required Notebook Structure

A DerivaML notebook that supports execution tracking must have these components:

### 1. Imports Cell

The first code cell should contain all imports:

```python
from pathlib import Path
from deriva_ml import DerivaML
```

### 2. Parameters Cell (Tagged "parameters")

The second code cell must be **tagged with `"parameters"`** for papermill injection. This cell contains all configurable values:

```python
# Parameters
host = "ml.derivacloud.org"
catalog_id = "1"
schema = "my_schema"
dataset_rid = "2-B4C8"
dataset_version = 3
learning_rate = 1e-3
batch_size = 32
epochs = 100
dry_run = False
```

**How to tag a cell in JupyterLab**: Click the cell, then go to the sidebar (View > Cell Inspector or click the gear icon) and add `"parameters"` to the cell tags.

When the notebook is run via papermill (by `deriva-ml-run` or the `run_notebook` MCP tool), the values in this cell are replaced by injected parameters. A new cell is inserted immediately after with the injected values.

### 3. Config Loading

Connect to the catalog:

```python
ml = DerivaML(hostname=host, catalog_id=catalog_id)
```

### 4. Execution Context

Wrap the main computation in an execution context manager. This creates an execution record in the catalog, tracks provenance, and handles cleanup:

```python
from deriva_ml import ExecutionConfiguration

workflow = ml.lookup_workflow_by_url("https://github.com/my-org/my-repo")

config = ExecutionConfiguration(
    workflow=workflow,
    datasets=[dataset_rid],
    description=f"Training run: lr={learning_rate}, bs={batch_size}, epochs={epochs}",
)

with ml.create_execution(config) as execution:
    # Datasets specified in config are auto-downloaded
    for dataset in execution.datasets:
        dataset.restructure_assets(...)  # DatasetBag objects

    # Your training logic here
    model = train(execution.working_dir, learning_rate, batch_size, epochs)

    # Save outputs using asset_file_path
    output_path = execution.asset_file_path("Execution_Asset", "model_weights.pt")
    save_model(model.state_dict(), output_path)

    metrics_path = execution.asset_file_path("Execution_Asset", "metrics.json")
    save_metrics({"accuracy": 0.95, "loss": 0.12}, metrics_path)

    # Outputs auto-uploaded on context exit
```

### 5. Save Execution RID

After the execution context closes, save the execution RID for downstream use. The convention is to use the `DERIVA_ML_SAVE_EXECUTION_RID` variable:

```python
DERIVA_ML_SAVE_EXECUTION_RID = execution.rid
print(f"Execution RID: {DERIVA_ML_SAVE_EXECUTION_RID}")
```

This variable is recognized by the DerivaML notebook runner. When the notebook is run programmatically, the runner extracts this value to link the notebook execution to the catalog record.

## Developing the Notebook

### Single Task

Keep each notebook focused on one task: training a model, evaluating results, exploring data, or generating visualizations. Do not combine unrelated tasks.

### Parameterize Everything

Every value that might change between runs should be in the parameters cell:

- Host and catalog connection details.
- Dataset RIDs and versions.
- Model hyperparameters.
- File paths and output locations.
- The `dry_run` flag.

### Clear Outputs Before Committing

Always clear all outputs before committing. If `nbstripout` is installed (see `setup-notebook-environment`), this happens automatically. Otherwise:

- JupyterLab: Kernel > Restart Kernel and Clear All Outputs.
- CLI: `uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb`

### Run End-to-End

Before committing, restart the kernel and run all cells from top to bottom. A notebook that only works with out-of-order cell execution is broken:

- JupyterLab: Kernel > Restart Kernel and Run All Cells.

### Use the Execution Context

Always use `ml.create_execution()` for tracked work. The execution context:

- Creates a timestamped execution record.
- Links the execution to datasets, workflow, and code version.
- Provides `asset_file_path()` to register output files.
- Automatically updates status on success or failure.
- Records the git commit hash and branch.

## Commit and Version Before Running

Before running a notebook for production (non-dry-run):

1. Clear outputs and save the notebook.
2. Commit all changes:
   ```bash
   git add notebook.ipynb
   git commit -m "Update training notebook"
   ```
3. Bump the version if needed:
   ```bash
   uv run bump-version patch
   ```

## Running the Notebook with Tracking

### Using Hydra Config Defaults (Recommended)

If your project has hydra-zen configs set up, the notebook can be run as part of an experiment:

```bash
uv run deriva-ml-run +experiment=baseline
```

The experiment config can specify the notebook as the task function, with parameters injected from the config.

### With Explicit Host and Catalog

```bash
uv run papermill notebook.ipynb output.ipynb \
  -p host ml.derivacloud.org \
  -p catalog_id 1 \
  -p dataset_rid 2-B4C8 \
  -p dry_run False
```

### With Parameter Overrides

```bash
uv run papermill notebook.ipynb output.ipynb \
  -p learning_rate 0.01 \
  -p epochs 200
```

### From a Parameter File

Create a `params.yaml`:
```yaml
host: ml.derivacloud.org
catalog_id: "1"
dataset_rid: "2-B4C8"
learning_rate: 0.001
epochs: 100
dry_run: false
```

Run:
```bash
uv run papermill notebook.ipynb output.ipynb -f params.yaml
```

### Inspect Before Running

Use the MCP tool to inspect the notebook structure without running it:

- `inspect_notebook` shows the parameters cell, tags, and overall structure.

### Run via MCP Tool

Use the `run_notebook` MCP tool for programmatic execution:

- `run_notebook` runs the notebook with specified parameters and returns the execution RID.

## What Happens During Execution

When a DerivaML notebook runs with execution tracking, these steps occur:

1. **Execution record created**: A new row is inserted in the Execution table with status "Running", linked to the workflow, datasets, and code version.

2. **Data downloaded**: Datasets and assets specified in the config are downloaded to the execution's working directory. Cached assets are symlinked.

3. **Notebook executed**: Papermill runs the notebook cell-by-cell, injecting parameters. Output is captured.

4. **Results uploaded**: Files saved via `execution.asset_file_path()` are uploaded to the catalog by `execution.upload_execution_outputs()`.

5. **Status updated**: The execution status is set to "Complete" on success or "Failed" on error. The output notebook (with all outputs) is attached to the execution record.

## Complete Workflow Checklist

- [ ] Environment verified (dependencies, kernel, auth)
- [ ] Notebook has imports cell
- [ ] Parameters cell is tagged `"parameters"`
- [ ] All configurable values are in the parameters cell
- [ ] Main logic is inside `ml.create_execution()` context
- [ ] `DERIVA_ML_SAVE_EXECUTION_RID` is set after execution
- [ ] Notebook runs end-to-end with Restart & Run All
- [ ] Outputs cleared before commit
- [ ] Code committed to Git
- [ ] Version bumped if needed
- [ ] Tested with `dry_run=True`
- [ ] Production run completed
- [ ] Execution verified in catalog

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `NameError` on parameter variables | Parameters cell not tagged or not first | Ensure the cell is tagged `"parameters"` and appears early |
| `No kernel named X` | Kernel not installed | Run `uv run deriva-ml-install-kernel` |
| Execution status stuck at "Running" | Notebook crashed without clean exit | Use `update_execution_status` MCP tool to set to "Failed" |
| Outputs still in committed notebook | `nbstripout` not installed | Run `uv run nbstripout --install` |
| `PapermillExecutionError` | A cell raised an exception | Check the output notebook for the traceback |
| `AuthenticationError` during execution | Credentials expired mid-run | Re-authenticate and re-run |
| `DERIVA_ML_SAVE_EXECUTION_RID` not found | Variable not set in notebook | Add the assignment after the execution context block |
| Files not appearing in catalog | `upload_execution_outputs()` not called after `with` block | Call it after exiting the context manager |
