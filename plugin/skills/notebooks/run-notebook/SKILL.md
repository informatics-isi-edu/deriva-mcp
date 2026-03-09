---
name: run-notebook
description: "ALWAYS use this skill when developing or running DerivaML Jupyter notebooks with execution tracking. Triggers on: 'run notebook', 'jupyter', 'papermill', 'parameters cell', 'notebook structure', 'deriva-ml-run-notebook', 'notebook with provenance'."
---

# Develop and Run a DerivaML Notebook

DerivaML notebooks support full execution tracking and provenance when structured correctly.

## Required Notebook Structure

1. **Imports cell** — All imports in the first code cell
2. **Parameters cell** — Tagged `"parameters"` for papermill injection. Contains all configurable values (host, catalog, dataset RIDs, hyperparameters, `dry_run`)
3. **Config loading** — `ml = DerivaML(host=host, catalog_id=catalog_id, schema=schema)`
4. **Execution context** — Main logic inside `with ml.create_execution(...) as execution:` block
5. **Save execution RID** — Set `DERIVA_ML_SAVE_EXECUTION_RID = execution.rid` after the context block

## Critical Rules

1. **Tag the parameters cell** — Must have `"parameters"` tag for papermill to inject values
2. **Use `create_execution()` context manager** — Provides provenance tracking, auto-status updates
3. **Clear outputs before committing** — Use `nbstripout` or manual clear
4. **Commit before production runs** — Git hash is recorded in the execution record
5. **Test with `dry_run=True`** — Validates pipeline without catalog writes

## Running Notebooks

```bash
# Via CLI (recommended)
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb

# With config overrides
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb assets=my_assets

# Override host/catalog
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --host ml.derivacloud.org --catalog 2

# Show available configs
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --info
```

## MCP Tools

- `inspect_notebook` — View notebook structure, parameters, and tags without running
- `run_notebook` — Execute notebook with parameters and return execution RID

## Pre-Production Checklist

- [ ] Parameters cell tagged `"parameters"`
- [ ] All configurable values in parameters cell
- [ ] Main logic inside `create_execution()` context
- [ ] `DERIVA_ML_SAVE_EXECUTION_RID` set after context
- [ ] Runs end-to-end with Restart & Run All
- [ ] Outputs cleared, code committed, version bumped

For the full guide with environment setup, papermill details, and troubleshooting table, read `references/workflow.md`.
