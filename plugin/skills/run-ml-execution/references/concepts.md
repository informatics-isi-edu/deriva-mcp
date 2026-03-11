# Execution Concepts

Background on executions, workflows, and provenance in DerivaML. For the step-by-step guide, see `workflow.md`.

## Table of Contents

- [What is an Execution?](#what-is-an-execution)
- [The Execution Hierarchy](#the-execution-hierarchy)
- [Execution Statuses](#execution-statuses)
- [ExecutionConfiguration](#executionconfiguration)
- [Nested Executions](#nested-executions)
- [Dry Run Mode](#dry-run-mode)
- [Execution Working Directory](#execution-working-directory)
- [Restoring Executions](#restoring-executions)
- [Execution vs ExecutionRecord](#execution-vs-executionrecord)

---

## What is an Execution?

An execution is the fundamental unit of provenance in DerivaML. It records what work was done, with what inputs, and what outputs were produced. Every data transformation, model training run, or analysis should be wrapped in an execution.

An execution tracks:
- **Inputs** — which datasets and assets were consumed
- **Outputs** — which files were produced (model weights, predictions, plots)
- **Workflow** — what kind of work was performed
- **Timing** — when the work started and stopped
- **Status** — whether it succeeded, failed, or is still running
- **Code provenance** — git commit hash and repository URL
- **Configuration** — Hydra config choices and parameters

This means you can always answer: "Where did this model come from? What data was it trained on? What code produced it?"

## The Execution Hierarchy

Executions exist within a three-level hierarchy:

```
Workflow_Type → Workflow → Execution
```

**Workflow_Type** is a controlled vocabulary term that categorizes workflows broadly — for example, "Training", "Inference", "Analysis", "ETL", "Annotation". These are terms in the `Workflow_Type` vocabulary.

**Workflow** is a reusable definition of a specific kind of work. It has a name, description, URL (typically a GitHub repository), and one or more workflow types. For example, a workflow named "CIFAR-10 CNN Training" of type "Training" at URL `https://github.com/org/repo`. Workflows are created once and reused across many executions.

**Execution** is a single instance of running a workflow — one training run, one analysis pass, one notebook evaluation. Each execution references exactly one workflow and records its specific inputs, outputs, configuration, and timing.

Before creating an execution, you need a workflow. Before creating a workflow, the workflow type must exist in the vocabulary.

## Execution Statuses

| Status | Meaning |
|--------|---------|
| `Initializing` | Initial setup in progress |
| `Created` | Record created in catalog |
| `Pending` | Queued for execution |
| `Running` | Work in progress |
| `Completed` | Finished successfully |
| `Failed` | Encountered an error |
| `Aborted` | Manually stopped |

The context manager automatically transitions through `Initializing` → `Running` → `Completed` (or `Failed` on exception). You can also update status manually with `update_execution_status` for finer-grained progress tracking during long-running work.

## ExecutionConfiguration

In the Python API, `ExecutionConfiguration` specifies everything needed to create an execution:

```python
from deriva_ml import ExecutionConfiguration

config = ExecutionConfiguration(
    workflow=workflow,                   # Required: Workflow object
    datasets=["2-ABC1"],                # Optional: input dataset RIDs
    assets=["2-DEF2"],                  # Optional: input asset RIDs or AssetSpec objects
    description="Train CNN on batch 1", # Optional: execution description (supports Markdown)
)
```

- **workflow**: A `Workflow` object from `create_workflow` or `lookup_workflow_by_url`. Required.
- **datasets**: List of dataset RID strings. These become the execution's input datasets.
- **assets**: List of asset RID strings or `AssetSpec` objects. Use `AssetSpec(rid="...", cache=True)` for large assets that should be cached locally across executions.
- **description**: Human-readable description. Supports Markdown for rich formatting in the Chaise UI.
- **config_choices**: Dict of Hydra config group selections (auto-populated by `deriva-ml-run`).

When using MCP tools instead of the Python API, `create_execution` accepts `workflow_name`, `workflow_type`, and `description` directly — it finds or creates the workflow automatically.

## Nested Executions

For multi-step pipelines, executions can be organized into parent-child relationships:

```
Parent execution (e.g., "Hyperparameter Sweep")
├── Child 1 (e.g., "lr=0.001")
├── Child 2 (e.g., "lr=0.01")
└── Child 3 (e.g., "lr=0.1")
```

Common use cases:
- **Parameter sweeps** — parent represents the sweep, children are individual runs
- **Pipelines** — parent represents the pipeline, children are stages (preprocessing, training, evaluation)
- **Cross-validation** — parent represents the CV experiment, children are individual folds

Each child execution is a full execution with its own inputs, outputs, and provenance. The parent-child link is tracked via an association table with an optional `sequence` number for ordering.

Navigation:
- From parent → children: `list_nested_executions`
- From child → parent: `list_parent_executions`
- Both support `recurse` for deep hierarchies

## Dry Run Mode

Dry run mode lets you test the full pipeline without writing to the catalog:

- No execution record is created (uses a placeholder RID of `"0"`)
- No catalog writes occur — no provenance, no status updates
- Datasets and assets **are** still downloaded — you can verify data loading works
- Configuration is still resolved — you can verify parameters are correct
- Output files can still be written locally — you can verify the model runs

In MCP tools, pass `dry_run`: `true` to `create_execution`. In Python, pass `dry_run=True` to the runner or set it in the Hydra config.

Use dry runs to:
- Test data loading and model initialization before committing to a full run
- Debug configuration issues without cluttering the catalog with failed executions
- Verify the pipeline end-to-end on a new machine or environment

## Execution Working Directory

Each execution gets a local working directory at `<ml_working_dir>/Execution/<execution_rid>/` with this layout:

```
Execution/<execution_rid>/
├── asset/                    # Output assets staged for upload
│   ├── <schema>/
│   │   └── <AssetTable>/     # Files organized by asset table
│   └── ml/
│       └── Execution_Asset/  # Default output table
├── asset-type/               # Asset type metadata (JSONL)
├── feature/                  # Feature values organized by table/feature
└── downloaded-assets/        # Downloaded input assets
```

Access via `get_execution_working_dir` (MCP) or `execution.working_dir` (Python).

## Restoring Executions

`restore_execution` re-downloads a previous execution's datasets and assets to a local working directory. This is useful for:

- **Debugging** — inspect what data a failed execution was working with
- **Continuing work** — resume from where a previous execution left off
- **Analysis** — run new analysis on the same inputs without re-configuring

The restored execution becomes the active execution, so subsequent MCP tool calls (`get_execution_working_dir`, `asset_file_path`, etc.) operate on it.

## Execution vs ExecutionRecord

DerivaML has two Python classes for executions:

**Execution** — Full lifecycle management:
- Created via `ml.create_execution(config)` or `ml.restore_execution(rid)`
- Provides context manager (`with ... as exe:`)
- Manages working directory, downloads, uploads, status transitions
- Use for **running** workflows

**ExecutionRecord** — Lightweight catalog state:
- Returned by `ml.lookup_execution(rid)`, `asset.list_executions()`, etc.
- Has `execution_rid`, `workflow_rid`, `status`, `description` properties
- Supports `update_status()` and `set_description()` for updating catalog state
- Use for **querying** and inspecting execution history

When you need to run work, use `Execution`. When you need to look at past results, you'll typically get `ExecutionRecord` objects.
