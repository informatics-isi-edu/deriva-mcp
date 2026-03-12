---
name: run-ml-execution
description: "ALWAYS use this skill when running ML executions with provenance tracking in DerivaML — the execution lifecycle, context managers, registering outputs, downloading inputs, nested executions, workflows, and execution status. Triggers on: 'create execution', 'run with provenance', 'upload outputs', 'asset_file_path', 'execution lifecycle', 'track my work', 'create workflow', 'nested execution', 'restore execution'."
disable-model-invocation: true
---

# Running ML Executions with Provenance

An execution is the fundamental unit of provenance in DerivaML. It records what work was done, with what inputs (datasets, assets), what outputs were produced (files, data), and what code and configuration were used. Every execution references a workflow, which itself has a workflow type.

For background on the execution hierarchy, statuses, nested executions, dry run mode, and the working directory layout, see `references/concepts.md`.

## Critical Rules

1. **Every execution needs a workflow** — find one with `lookup_workflow_by_url` or create one with `create_workflow` before creating the execution. When using MCP tools, `create_execution` can find or create the workflow for you.
2. **Use the context manager in Python** — `with ml.create_execution(config) as exe:` auto-starts and auto-stops, and sets status to "Failed" on exception.
3. **Upload AFTER the with block** — `exe.upload_execution_outputs()` must be called after exiting the context manager, not inside it. When using `deriva-ml-run`, upload is handled automatically.
4. **Use `asset_file_path` for all outputs** — this both stages the file and registers it as an output asset. Never manually place files in the working directory.
5. **Commit code before running** — DerivaML records the git commit hash for code provenance. Uncommitted changes mean no valid code reference.

## Lifecycle Summary

### MCP tools

1. `create_execution` — create the execution (finds/creates workflow automatically)
2. `start_execution` — set status to Running, record start time
3. `download_execution_dataset` / `download_asset` — download inputs
4. Do your work
5. `asset_file_path` — register each output file for upload
6. `stop_execution` — set status to Completed, record stop time
7. `upload_execution_outputs` — upload all registered files to catalog

Steps 2-7 operate on the **active execution** — no `execution_rid` parameter needed.

### Python API

```python
config = ExecutionConfiguration(
    workflow=workflow,
    datasets=["2-ABC1"],
    assets=["2-DEF2"],
    description="Train CNN on labeled images"
)
with ml.create_execution(config) as exe:
    # Datasets specified in config are auto-downloaded; access via exe.datasets
    for dataset in exe.datasets:
        dataset.restructure_assets(...)  # DatasetBag objects
    # ... do work ...
    path = exe.asset_file_path("Execution_Asset", "results.csv")
    # ... write to path ...
    # Outputs auto-uploaded on context exit
```

## Key Tools Beyond the Lifecycle

- `get_execution_info` — details of the active execution (no parameters)
- `update_execution_status` — track progress with status and message
- `set_execution_description` — update an execution's description
- `restore_execution` — re-download a previous execution's inputs for debugging or re-analysis
- `add_nested_execution` — link parent and child executions for multi-step pipelines
- `list_nested_executions` / `list_parent_executions` — navigate execution hierarchies
- `create_execution_dataset` — create an output dataset linked to the active execution
- `deriva://storage/execution-dirs` resource — find local execution working directories

For the full step-by-step guide with all parameters, see `references/workflow.md`.

## Reference Resources

- `references/concepts.md` — What executions are, the hierarchy, statuses, nested executions, dry run, working directory
- `references/workflow.md` — Step-by-step MCP and Python API workflows with complete examples
- `deriva://docs/execution-configuration` — Full guide to execution lifecycle and configuration
- `deriva://execution/{execution_rid}` — Execution details, status, and timing
- `deriva://experiment/{execution_rid}` — Rich view with Hydra config, model params, inputs/outputs
- `deriva://execution/{execution_rid}/inputs` — Input datasets and assets
- `deriva://catalog/workflows` — Available workflows
- `deriva://catalog/workflow-types` — Workflow type vocabulary terms
- `deriva://workflow/{workflow_rid}` — Workflow details

## Related Skills

- **`work-with-assets`** — Discovering, downloading, creating asset tables, and managing asset types
- **`run-experiment`** — Running experiments via the `deriva-ml-run` CLI with pre-flight checks
- **`configure-experiment`** — Writing Hydra-zen experiment configurations
- **`prepare-training-data`** — Downloading and restructuring data for ML frameworks
