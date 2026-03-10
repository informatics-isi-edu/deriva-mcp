---
name: new-model
description: "ALWAYS use this skill when creating a new DerivaML model function or adding a model to a project. Triggers on: 'create model', 'add model', 'new model', 'scaffold model', 'model function', 'write training code', 'add a training pipeline'. This skill covers authoring the model function itself and wiring it to configs, workflows, and experiments — the end-to-end process for adding a new model to a DerivaML project."
argument-hint: "[model-name]"
---

# Create a New DerivaML Model

This skill covers the end-to-end process of adding a new model to a DerivaML project: writing the model function, creating its config, defining a workflow, and wiring it into experiments.

For the config file syntax details, see the `write-hydra-config` skill.
For project structure and config group architecture, see the `configure-experiment` skill.

## Model Function Signature

DerivaML models are plain Python functions. The framework injects `ml_instance` and `execution` at runtime — everything else becomes a configurable hyperparameter via hydra-zen.

```python
def my_model(
    # Your hyperparameters (become configurable)
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    # Framework-injected (always present, always last)
    ml_instance: DerivaML = None,
    execution: Execution | None = None,
) -> None:
    ...
```

The `ml_instance` and `execution` parameters must have default `None` — `builds(..., zen_partial=True)` in the config defers their injection to runtime.

## Working with Execution Assets

Inside the model function, use the execution object for all I/O:

```python
def my_model(
    learning_rate: float = 1e-3,
    ml_instance: DerivaML = None,
    execution: Execution | None = None,
) -> None:
    # Access downloaded dataset files
    for table, assets in execution.asset_paths.items():
        for asset_path in assets:
            # asset_path is a Path with .asset_rid attribute
            data = load_data(asset_path)

    # Create output files — ALWAYS use asset_file_path
    output_path = execution.asset_file_path("Execution_Asset", "predictions.csv")
    save_predictions(output_path)

    metrics_path = execution.asset_file_path("Execution_Asset", "metrics.json")
    save_metrics(metrics_path)
```

`asset_file_path("Execution_Asset", filename)` both creates the file path and registers the file as an execution output. Use `"Execution_Asset"` for all model-produced files (predictions, metrics, checkpoints, plots). `"Execution_Metadata"` is reserved for framework-generated files (configs, environment snapshots) — never use it for model outputs.

## Steps

### 1. Create the model file

Create `src/models/<model_name>.py`:

```python
from deriva_ml import DerivaML
from deriva_ml.execution import Execution


def my_model(
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_size: int = 128,
    ml_instance: DerivaML = None,
    execution: Execution | None = None,
) -> None:
    """Train my model on the provided datasets.

    Args:
        learning_rate: Optimizer learning rate.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        hidden_size: Hidden layer dimension.
        ml_instance: Injected DerivaML instance.
        execution: Injected execution context.
    """
    # Load data from execution datasets
    for table, assets in execution.asset_paths.items():
        for asset_path in assets:
            print(f"Processing {asset_path} (RID: {asset_path.asset_rid})")

    # ... training logic ...

    # Save outputs
    output_path = execution.asset_file_path("Execution_Asset", "results.csv")
    # ... write results to output_path ...
```

### 2. Create the model config

Create `src/configs/<model_name>.py`. Use `builds()` with `zen_partial=True`:

```python
from hydra_zen import builds, store
from models.my_model import my_model

MyModelConfig = builds(
    my_model,
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    hidden_size=128,
    populate_full_signature=True,
    zen_partial=True,
)

model_store = store(group="model_config")

model_store(
    MyModelConfig,
    name="default_model",
    zen_meta={"description": "Default config: 10 epochs, lr=1e-3, batch 64."},
)

# Variants override specific parameters
model_store(
    MyModelConfig,
    name="my_model_quick",
    epochs=3,
    batch_size=128,
    zen_meta={"description": "Quick run: 3 epochs for pipeline validation."},
)
```

See the `write-hydra-config` skill for the full config API reference and rules.

### 3. Create a workflow

Add to `src/configs/workflow.py`:

```python
from hydra_zen import store, builds
from deriva_ml.execution import Workflow

MyWorkflow = builds(
    Workflow,
    name="My Model Training",
    workflow_type=["Training"],
    description="Train my model on the dataset. Outputs predictions and metrics.",
    populate_full_signature=True,
)

workflow_store = store(group="workflow")
workflow_store(MyWorkflow, name="my_workflow")
```

If this is the project's first or primary model, also set it as `default_workflow`.

### 4. Add experiments

Add to `src/configs/experiments.py`:

```python
from hydra_zen import make_config, store
from configs.base import DerivaModelConfig

experiment_store = store(group="experiment", package="_global_")

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "my_model_quick"},
            {"override /datasets": "my_dataset"},
            {"override /workflow": "my_workflow"},
        ],
        description="Quick test of my model on the small dataset",
        bases=(DerivaModelConfig,),
    ),
    name="my_model_quick",
)
```

### 5. Test

```bash
# Check config resolves
uv run deriva-ml-run +experiment=my_model_quick --info

# Dry run (downloads data, runs model, no catalog writes)
uv run deriva-ml-run +experiment=my_model_quick dry_run=true
```

## Related Skills

- **`write-hydra-config`** — Config file syntax for every config group
- **`configure-experiment`** — Project structure and config group architecture
- **`run-experiment`** — Pre-flight checklist and CLI commands
- **`run-ml-execution`** — Execution lifecycle and provenance details
