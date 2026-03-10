---
name: run-ml-execution
description: "ALWAYS use this skill when running ML executions with provenance tracking in DerivaML — the execution lifecycle, context managers, output registration, and nested executions. Triggers on: 'create execution', 'run with provenance', 'upload outputs', 'asset_file_path', 'execution lifecycle', 'track my work'."
disable-model-invocation: true
---

# Running an ML Execution with Provenance

Every data transformation, model training run, or analysis in DerivaML should be wrapped in an execution to track inputs, outputs, and provenance.

## Execution Lifecycle

```
create_execution → start → work → stop → upload_execution_outputs
```

## Critical Rules

1. **Every execution needs a workflow** — Create or find one with `create_workflow` first.
2. **Use the context manager in Python** — `with ml.create_execution(config) as exe:` auto-starts and auto-stops.
3. **Upload AFTER the with block** — `exe.upload_execution_outputs()` must be called after exiting the context manager, never inside it.
4. **Use `asset_file_path()` for all outputs** — This both creates the path and registers the file as an output asset. Never manually place files in the working directory.
5. **Failed executions are tracked** — If an exception occurs in the with block, status is set to "Failed" automatically.

## Python API (Recommended)

```python
config = ExecutionConfiguration(
    workflow=workflow,
    datasets=["2-ABC1"],
    assets=["2-DEF2"],
)
with ml.create_execution(config) as exe:
    exe.download_execution_dataset()
    # ... do work ...
    path = exe.asset_file_path("results.csv", description="Model predictions")
    # ... write to path ...
exe.upload_execution_outputs()
```

## MCP Tools

```
create_execution(workflow_rid=..., description=..., dataset_rids=[...])
start_execution(execution_rid=...)
download_execution_dataset(execution_rid=...)
# ... do work ...
asset_file_path(execution_rid=..., filename=..., description=...)
stop_execution(execution_rid=...)
upload_execution_outputs(execution_rid=...)
```

## Key Tools

- `restore_execution` — Re-download a previous execution's assets for debugging
- `add_nested_execution` — Multi-step pipelines with parent-child structure
- `list_nested_executions` / `list_parent_executions` — Navigate execution hierarchy
- `add_asset_type` / `create_asset_table` — Manage asset categories and tables

For the full guide with ExecutionConfiguration details, nested executions, asset management, and inspection tools, read `references/workflow.md`.
