---
name: run-notebook
description: "ALWAYS use this skill when creating, developing, or running DerivaML Jupyter notebooks with execution tracking. Triggers on: 'create notebook', 'new notebook', 'add notebook', 'scaffold notebook', 'run notebook', 'jupyter', 'notebook structure', 'deriva-ml-run-notebook', 'notebook with provenance', 'notebook_config', 'run_notebook'."
disable-model-invocation: true
---

# Create and Run a DerivaML Notebook

DerivaML notebooks support full execution tracking and provenance. The `run_notebook()` API handles connection, execution context, and config loading automatically.

## Creating a New Notebook

### Step 1: Define a config module

Create `src/configs/<notebook_name>.py`:

**Simple notebook** (standard fields only — assets, datasets, workflow):
```python
from deriva_ml.execution import notebook_config

notebook_config(
    "<notebook_name>",
    defaults={"assets": "my_assets", "datasets": "my_dataset"},
)
```

**Notebook with custom parameters:**
```python
from dataclasses import dataclass
from deriva_ml.execution import BaseConfig, notebook_config

@dataclass
class MyAnalysisConfig(BaseConfig):
    threshold: float = 0.5
    num_iterations: int = 100

notebook_config(
    "<notebook_name>",
    config_class=MyAnalysisConfig,
    defaults={"assets": "my_assets"},
)
```

Multiple named configs can share one file:
```python
notebook_config(
    "<notebook_name>",
    defaults={"assets": "my_assets"},
)

notebook_config(
    "<notebook_name>_variant",
    defaults={"assets": "other_assets", "datasets": "other_dataset"},
)
```

See the `write-hydra-config` skill for the full config API reference and rules.

### Step 2: Create the notebook

Create `notebooks/<notebook_name>.ipynb` with an initialization cell:

```python
from deriva_ml.execution import run_notebook

ml, execution, config = run_notebook("<notebook_name>")

# Ready to use:
# - ml: Connected DerivaML instance
# - execution: Execution context with downloaded inputs
# - config: Resolved configuration (config.assets, config.threshold, etc.)
```

`run_notebook()` handles catalog connection, execution creation, dataset downloading, and config resolution. There is no need to manually set up parameters cells, papermill tags, or connection boilerplate.

Add a final cell to upload outputs:
```python
execution.upload_execution_outputs()
```

Use `execution.asset_file_path("Execution_Asset", filename)` for all output files.

### Step 3: Run

```bash
# Show available configs
uv run deriva-ml-run-notebook notebooks/<notebook_name>.ipynb --info

# Run with defaults
uv run deriva-ml-run-notebook notebooks/<notebook_name>.ipynb

# Override assets or datasets (positional Hydra overrides, NOT --config)
uv run deriva-ml-run-notebook notebooks/<notebook_name>.ipynb assets=different_assets

# Override host/catalog
uv run deriva-ml-run-notebook notebooks/<notebook_name>.ipynb \
    --host www.example.org --catalog 2
```

`--config` does NOT override the `run_notebook()` config name in the notebook cell. Use positional Hydra overrides instead.

## Critical Rules

1. **Clear outputs before committing** — Use `nbstripout` or manual clear
2. **Commit before production runs** — Git hash is recorded in the execution record
3. **Test with dry run** — `uv run deriva-ml-run-notebook notebooks/<name>.ipynb dry_run=true`
4. **Use `asset_file_path("Execution_Asset", ...)`** for all output files

## MCP Tools

- `inspect_notebook` — View notebook structure, parameters, and tags without running
- `run_notebook` — Execute notebook with parameters and return execution RID

## Pre-Production Checklist

- [ ] Config module created in `src/configs/`
- [ ] `run_notebook("<name>")` call in first cell matches config name
- [ ] `upload_execution_outputs()` in final cell
- [ ] Runs end-to-end with Restart & Run All
- [ ] Outputs cleared, code committed, version bumped

For the full guide with environment setup, the manual papermill approach, and troubleshooting, read `references/workflow.md`.
