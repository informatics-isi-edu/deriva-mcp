# Execution Concepts

Background on executions, workflows, and provenance in DerivaML. For the step-by-step guide, see `workflow.md`.

## Table of Contents

- [Executions in the Catalog](#executions-in-the-catalog)
- [Execution RIDs](#execution-rids)
- [Execution Structure](#execution-structure)
- [Execution Statuses](#execution-statuses)
- [Workflows and Workflow Types](#workflows-and-workflow-types)
- [Nested Executions](#nested-executions)
- [Execution Data Flow](#execution-data-flow)
- [Creating and Managing Executions](#creating-and-managing-executions)
- [ExecutionConfiguration](#executionconfiguration)
- [The Execution Context Manager](#the-execution-context-manager)
- [Execution Working Directory](#execution-working-directory)
- [Dry Run Mode](#dry-run-mode)
- [Restoring Executions](#restoring-executions)

---

## Executions in the Catalog

An execution is a catalog record that captures a unit of work — a model training run, a data analysis, a notebook evaluation, a feature annotation pass. Executions are the fundamental unit of provenance in DerivaML. They are persistent, queryable entities stored in the catalog alongside your data.

Every execution record answers: "What work was done? With what inputs? What was produced? What code and configuration were used?"

Executions are represented in the `Execution` table in the `deriva-ml` schema. Like all catalog records, each execution has a unique **RID** that permanently identifies it.

## Execution RIDs

Every execution has a unique RID (Resource IDentifier) — a short, immutable string like `2-YYYY` or `3-AB4C`. This RID is the primary way to reference an execution:

- **In MCP tools**: Pass `execution_rid` to `restore_execution`, `list_nested_executions`, `set_execution_description`, etc.
- **In provenance queries**: Asset and dataset provenance records reference execution RIDs
- **In the web UI**: Each execution has a Chaise page at its RID-based URL
- **In citation**: `ml.cite(rid)` generates a permanent URL for an execution
- **In nested relationships**: Parent-child links between executions use RIDs

RIDs are assigned by the catalog when the execution record is created and never change.

## Execution Structure

An execution record in the catalog has these relationships:

```
Execution
├── Workflow (FK)           — what kind of work was performed
├── Status (FK)             — current state (Running, Completed, Failed, ...)
├── Description             — human-readable purpose (supports Markdown)
├── Start/Stop timestamps   — when the work ran
├── Input Datasets          — which datasets were consumed (association table)
├── Input Assets            — which assets were consumed (association table)
├── Output Assets           — which files were produced (association table)
├── Code provenance         — git commit hash and repository URL
├── Configuration           — Hydra config choices and parameters
└── Nested Executions       — parent-child relationships (association table)
```

The input and output links are tracked through association tables with role information ("Input" or "Output"), so you can trace provenance in both directions — from an execution to its artifacts, or from an artifact back to the execution that created it.

**Querying executions:**
- Read `deriva://execution/{execution_rid}` for full details
- Read `deriva://experiment/{execution_rid}` for a richer view with Hydra config and model parameters
- Read `deriva://execution/{execution_rid}/inputs` for just the input datasets and assets
- Call `list_dataset_executions` to find all executions that used a dataset
- Call `list_asset_executions` to find executions that created or used an asset

**The ExecutionRecord class** in the Python API is the lightweight read-only representation of an execution record. It's returned by lookup and query methods:

```python
record = ml.lookup_execution("2-YYYY")
print(record.execution_rid)   # "2-YYYY"
print(record.status)          # "Completed"
print(record.description)     # "Train CNN on batch 1"
print(record.workflow_rid)    # "1-WXYZ"
```

`ExecutionRecord` is also what you get back from provenance queries like `asset.list_executions()` and `ml.list_asset_executions()`.

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

The execution context manager automatically transitions through `Initializing` → `Running` → `Completed` (or `Failed` on exception). You can also update status manually with `update_execution_status` for finer-grained progress tracking during long-running work.

## Workflows and Workflow Types

Every execution references a **workflow** — a reusable definition of a kind of work.

A workflow can represent many things:
- **A program** — a Python script, a trained model pipeline, a CLI tool
- **A person performing a process** — a pathologist annotating slides, a curator reviewing data quality
- **A workflow manager** — an Airflow DAG, a Nextflow pipeline, a Snakemake workflow
- **A notebook** — a Jupyter notebook performing analysis or visualization

What matters is that it identifies *what kind of work* was done, so that executions are traceable and reproducible.

**Workflow_Type** is a controlled vocabulary term that categorizes workflows broadly — for example, "Training", "Inference", "Analysis", "ETL", "Annotation". These are terms in the `Workflow_Type` vocabulary.

**Workflow** is the specific workflow definition. It has:
- A **name** (e.g., "CIFAR-10 CNN Training")
- A **URL** (typically a GitHub repository, but could be a documentation page or any identifier)
- One or more **workflow types**
- A **description** of what it does

Workflows are created once and reused across many executions. For example, the same "CIFAR-10 CNN Training" workflow might be used for hundreds of training runs with different hyperparameters — each run is a separate execution.

### Finding and creating workflows

Before creating an execution, you need a workflow. Check for existing workflows first:
- Read `deriva://catalog/workflows` to list all workflows
- Call `lookup_workflow_by_url` with the repository URL to find a workflow by its source

If no suitable workflow exists, create one:
- Call `create_workflow` with `name`, `workflow_type`, and optionally `description`
- If the workflow type doesn't exist yet, call `add_workflow_type` with `type_name` and `description` first

When using MCP tools, `create_execution` can find or create the workflow for you — pass `workflow_name` and `workflow_type` and it handles the lookup/creation automatically.

## Nested Executions

Executions can be organized into parent-child relationships for multi-step work:

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

Each child is a full execution with its own RID, inputs, outputs, and provenance. The parent-child link is tracked via an association table with an optional `sequence` number for ordering.

Navigation:
- From parent → children: `list_nested_executions`
- From child → parent: `list_parent_executions`
- Both support `recurse` for deep hierarchies

## Execution Data Flow

An execution consumes inputs, does work in a local working directory, and produces outputs that get uploaded back to the catalog. Understanding this flow is key to working with executions.

### Consuming inputs

An execution's inputs are **datasets** and **assets** specified when the execution is created. During execution, you download these to a local working directory:

- **Datasets** are downloaded as BDBags — self-contained, versioned archives that include all member records, asset files, feature values, and vocabulary terms at the exact catalog state when the version was created. Call `download_execution_dataset` with a dataset RID and version. See the `create-dataset` skill for how datasets and versions work, and its `references/bags.md` for details on the BDBag format.
- **Individual assets** (e.g., pretrained model weights) are downloaded directly. Call `download_asset` with an asset RID. See the `work-with-assets` skill for asset concepts including caching.

Both operations automatically record provenance — the downloaded dataset or asset is linked to the execution with role "Input".

### The working directory

Each execution gets a local working directory where all downloaded inputs and staged outputs live. This directory is created automatically and persists until cleaned up. Access it via `get_execution_working_dir` (MCP) or `execution.working_dir` (Python). See [Execution Working Directory](#execution-working-directory) for the layout.

### Producing outputs

Output files (model weights, predictions, plots, etc.) must be **registered** before they can be uploaded to the catalog. Registration is done via `asset_file_path`, which:

1. Takes an asset table name (e.g., `"Execution_Asset"`) and filename
2. Stages the file in the execution's working directory
3. Returns a file path — write your output to this path, or pass an existing file to be staged
4. Records the file's metadata (asset types, table) for upload

Registered files are **not yet in the catalog** — they exist only in the local staging area.

### Uploading outputs

After the execution's work is complete, call `upload_execution_outputs` to upload all registered files to the catalog in one batch. This:

1. Uploads each staged file to Hatrac (Deriva's file storage)
2. Creates asset records in the appropriate asset tables
3. Links each asset to the execution with role "Output"
4. Optionally cleans up the local staging directory

Until `upload_execution_outputs` is called, output files exist only locally. This is a deliberate design — it allows the execution to complete (or fail) without partial uploads.

### Recording feature values

An execution can also produce **feature values** — structured annotations on catalog records (e.g., per-image classification labels, confidence scores). Like output files, feature values are **staged locally** and uploaded when `upload_execution_outputs` is called:

- In MCP tools, call `add_feature_value` or `add_feature_value_record` during the execution.
- In Python, call `execution.add_features(records)`. This writes JSONL files to disk in the execution's `feature/` directory — the catalog is not updated until `upload_execution_outputs()` runs.

Both output files and feature values are linked to the execution for provenance. For creating features and populating values, see the `create-feature` skill.

### The complete flow

```
Create execution → Start → Download inputs → Do work → Register outputs → Stop → Upload
                            ↓                               ↓                       ↓
                     Working directory              Staging area             Catalog updated
                     (downloaded data)        (files + feature JSONL)    (assets + features)
```

## Creating and Managing Executions

Execution records are created and managed by the **Execution** class in the Python API, or by the MCP execution lifecycle tools. Unlike `ExecutionRecord` (read-only lookup), the `Execution` class is the active object that drives the data flow described above:

- Creates the execution record in the catalog
- Manages the local working directory
- Downloads input datasets and assets
- Stages output files for upload
- Transitions status (start, stop, fail)
- Uploads outputs to the catalog

In Python, `Execution` is used through a context manager:

```python
with ml.create_execution(config) as exe:
    # exe is an Execution object — manages the full lifecycle
    ...
```

In MCP tools, the lifecycle is managed through explicit tool calls (`create_execution`, `start_execution`, `stop_execution`, etc.) that operate on the active execution.

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

When using MCP tools, `create_execution` accepts `workflow_name`, `workflow_type`, and `description` directly — it finds or creates the workflow automatically.

## The Execution Context Manager

The recommended Python pattern uses a `with` block:

```python
with ml.create_execution(config) as exe:
    # On enter: creates execution record, sets status to Initializing → Running
    # Datasets specified in config are auto-downloaded
    for dataset in exe.datasets:
        dataset.restructure_assets(...)  # DatasetBag objects
    # ... do work ...
    path = exe.asset_file_path("Execution_Asset", "results.csv")
    # ... write to path ...
    # On exit: sets status to Completed (or Failed), outputs auto-uploaded
```

**Key points:**
- The `with` block automatically calls `start_execution()` on entry and `stop_execution()` on exit
- If an exception occurs, status is set to "Failed" automatically
- Call `upload_execution_outputs()` **after** exiting the `with` block, not inside it
- When using `deriva-ml-run`, upload is handled automatically by the runner

## Execution Working Directory

Each execution gets a local working directory at `<ml_working_dir>/Execution/<execution_rid>/`:

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

## Restoring Executions

`restore_execution` re-downloads a previous execution's datasets and assets to a local working directory. This is useful for:

- **Debugging** — inspect what data a failed execution was working with
- **Continuing work** — resume from where a previous execution left off
- **Analysis** — run new analysis on the same inputs without re-configuring

The restored execution becomes the active execution, so subsequent MCP tool calls (`get_execution_working_dir`, `asset_file_path`, etc.) operate on it.

### Finding execution RIDs to restore

You need the execution's RID to restore it. Several ways to find it:

- **From the catalog**: Read `deriva://execution/{execution_rid}` if you know the RID, or query the `Execution` table via `query_table` to search by workflow, status, or description.
- **From local storage**: Read the `deriva://storage/execution-dirs` resource to see execution working directories that still exist locally. Each entry includes the execution RID, a label, size, and modification time.
- **From provenance**: Call `list_asset_executions` with an asset RID to find which execution produced it, or `list_dataset_executions` with a dataset RID to find executions that used it.
- **From the web UI**: Browse executions in Chaise and copy the RID from the record page.
