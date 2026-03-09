---
name: write-hydra-config
description: "Authoritative reference for writing and validating hydra-zen config files for DerivaML — datasets, assets, workflows, model configs, experiments, multiruns, and notebook configs. Use whenever adding, editing, or updating any config in a DerivaML project's configs/ directory, after creating catalog entities that should be reflected in configs, or when validating that config RIDs and versions match the catalog."
user-invocable: true
---

# Writing Hydra-Zen Config Files for DerivaML

This skill is the authoritative reference for the Python API used in DerivaML hydra-zen configuration files. Every config group has a specific pattern — follow the examples here exactly.

## When to Use This Skill

- Writing a new config file (datasets.py, assets.py, model.py, etc.)
- Adding a new entry to an existing config file
- After creating a catalog entity (dataset, asset, workflow) that should be added to configs
- Fixing or updating existing config entries
- Validating that config RIDs and versions exist in the catalog

For **creating a new project from scratch**, read `references/config-templates.md` — it has complete starter templates for every config file.

After any catalog-modifying action (create_dataset, split_dataset, create_workflow, etc.), proactively offer to update the relevant config file using these patterns.

## Config Groups Overview

| Group | File | Key Import | Registration |
|---|---|---|---|
| `deriva_ml` | `configs/deriva.py` | `from deriva_ml import DerivaMLConfig` | `store(group="deriva_ml")` |
| `datasets` | `configs/datasets.py` | `from deriva_ml.dataset import DatasetSpecConfig` | `store(group="datasets")` |
| `assets` | `configs/assets.py` | `from deriva_ml.execution import with_description` | `store(group="assets")` |
| `workflow` | `configs/workflow.py` | `from deriva_ml.execution import Workflow` | `store(group="workflow")` |
| `model_config` | `configs/<model>.py` | `from hydra_zen import builds` | `store(group="model_config")` |
| `experiment` | `configs/experiments.py` | `from hydra_zen import make_config` | `store(group="experiment", package="_global_")` |
| multiruns | `configs/multiruns.py` | `from deriva_ml.execution import multirun_config` | `multirun_config("name", ...)` |
| notebooks | `configs/<notebook>.py` | `from deriva_ml.execution import notebook_config` | `notebook_config("name", ...)` |

## Datasets (`configs/datasets.py`)

```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

datasets_store = store(group="datasets")

# With description (recommended)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="28DM", version="0.9.0")],
        "Complete CIFAR-10 dataset with all 10,000 images (5,000 training + 5,000 testing). "
        "Use for full-scale experiments.",
    ),
    name="cifar10_complete",
)

# Multiple datasets in one config
datasets_store(
    with_description(
        [
            DatasetSpecConfig(rid="28FC", version="0.4.0"),
            DatasetSpecConfig(rid="28FP", version="0.4.0"),
        ],
        "Small training (500) and testing (500) sets for rapid prototyping.",
    ),
    name="cifar10_small_both",
)

# Empty dataset list (for notebooks that don't need datasets)
datasets_store([], name="no_datasets")

# REQUIRED: default_dataset — plain list, no with_description()
# (with_description creates DictConfig which can't merge with BaseConfig's ListConfig)
datasets_store(
    [DatasetSpecConfig(rid="28DY", version="0.9.0")],
    name="default_dataset",
)
```

**Key rules:**
- `version` is **required** — always a semver string like `"0.9.0"`, not an integer
- Use `with_description()` for non-default configs
- Default configs use plain lists (no `with_description`) for merge compatibility
- Find the current version via the `deriva-ml://dataset/{rid}` MCP resource
- If data has changed since the version was created, call `increment_dataset_version` first

## Assets (`configs/assets.py`)

```python
from hydra_zen import store
from deriva_ml.execution import with_description

asset_store = store(group="assets")

# Plain RID strings (most common)
asset_store(
    with_description(
        ["3WS6", "3X20"],
        "Prediction probabilities from quick (3 epochs) vs extended (50 epochs) training. "
        "Use with ROC analysis notebook.",
    ),
    name="roc_quick_vs_extended",
)

# AssetSpecConfig with caching (for large immutable files like model weights)
from deriva_ml.asset.aux_classes import AssetSpecConfig

asset_store(
    with_description(
        [AssetSpecConfig(rid="3WS2", cache=True)],
        "Pre-trained weights from cifar10_quick (execution 3WR0, 3 epochs). "
        "Cached locally (~50MB) to avoid re-downloading.",
    ),
    name="quick_weights",
)

# REQUIRED: default_asset — empty list, plain (no with_description)
asset_store([], name="default_asset")

# Alias for clarity
asset_store([], name="no_assets")
```

**Key rules:**
- Plain RID strings for simple references: `["3WS6", "3X20"]`
- `AssetSpecConfig(rid=..., cache=True)` for large files that shouldn't re-download
- Default/empty configs use plain lists for merge compatibility
- Assets are typically execution outputs — note the source execution RID in the description

## Workflow (`configs/workflow.py`)

```python
from hydra_zen import store, builds
from deriva_ml.execution import Workflow

# Build the workflow config class
Cifar10CNNWorkflow = builds(
    Workflow,
    name="CIFAR-10 2-Layer CNN",
    workflow_type=["Training", "Image Classification"],  # string or list of strings
    description="""
Train a 2-layer convolutional neural network on CIFAR-10 image data.

## Architecture
- **Conv Layer 1**: 3 -> 32 channels, 3x3 kernel, ReLU, MaxPool 2x2
- **Conv Layer 2**: 32 -> 64 channels, 3x3 kernel, ReLU, MaxPool 2x2
- **FC Layer**: 64x8x8 -> 128 hidden units -> 10 classes
""".strip(),
    populate_full_signature=True,
)

workflow_store = store(group="workflow")

# REQUIRED: default_workflow
workflow_store(Cifar10CNNWorkflow, name="default_workflow")

# Named variants
workflow_store(Cifar10CNNWorkflow, name="cifar10_cnn")
```

**Key rules:**
- Use `builds(Workflow, ...)` with `populate_full_signature=True`
- `workflow_type` can be a single string or a list of strings
- `description` supports markdown — use it for architecture details
- Git URL and commit hash are captured automatically at runtime

## Model Config (`configs/<model>.py`)

```python
from hydra_zen import builds, store
from models.cifar10_cnn import cifar10_cnn

# Build the base config — zen_partial=True is critical
# (execution context is injected at runtime)
Cifar10CNNConfig = builds(
    cifar10_cnn,
    conv1_channels=32,
    conv2_channels=64,
    hidden_size=128,
    dropout_rate=0.0,
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    weight_decay=0.0,
    populate_full_signature=True,
    zen_partial=True,
)

model_store = store(group="model_config")

# REQUIRED: default_model
model_store(
    Cifar10CNNConfig,
    name="default_model",
    zen_meta={
        "description": (
            "Default CIFAR-10 CNN: 32->64 channels, 128 hidden units, 10 epochs, "
            "batch size 64, lr=1e-3. Balanced config for standard training runs."
        )
    },
)

# Variants override specific parameters
model_store(
    Cifar10CNNConfig,
    name="cifar10_quick",
    epochs=3,
    batch_size=128,
    zen_meta={
        "description": (
            "Quick training: 3 epochs, batch 128. Use for rapid iteration, "
            "debugging, and verifying the training pipeline works correctly."
        )
    },
)

model_store(
    Cifar10CNNConfig,
    name="cifar10_extended",
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=256,
    dropout_rate=0.25,
    weight_decay=1e-4,
    learning_rate=1e-3,
    epochs=50,
    zen_meta={
        "description": (
            "Extended training for best accuracy: Large model (64->128 ch, 256 hidden), "
            "regularization (dropout 0.25, weight decay 1e-4), 50 epochs."
        )
    },
)
```

**Key rules:**
- `zen_partial=True` is required — the execution context is injected later
- `populate_full_signature=True` exposes all constructor params to Hydra
- `zen_meta={"description": "..."}` documents the config variant
- Override individual params when registering variants (no need to rebuild)

## Experiments (`configs/experiments.py`)

```python
from hydra_zen import make_config, store
from configs.base import DerivaModelConfig

# package="_global_" is set on the store, not on make_config
experiment_store = store(group="experiment", package="_global_")

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Quick CIFAR-10 training: 3 epochs, 32->64 channels, batch size 128",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_quick",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_extended"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Extended CIFAR-10 training: 50 epochs, 64->128 channels, full regularization",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_extended",
)
```

**Key rules:**
- `package="_global_"` goes on the `store()` call
- `bases=(DerivaModelConfig,)` inherits from the base config
- `hydra_defaults` uses `{"override /group": "name"}` syntax
- `"_self_"` must be first in the defaults list
- `description` is a plain string on `make_config()` (not zen_meta)

## Multiruns (`configs/multiruns.py`)

```python
from deriva_ml.execution import multirun_config

multirun_config(
    "quick_vs_extended",
    overrides=[
        "+experiment=cifar10_quick,cifar10_extended",
    ],
    description="""## Quick vs Extended Training Comparison

| Config | Epochs | Architecture | Regularization |
|--------|--------|--------------|----------------|
| quick | 3 | 32->64 channels | None |
| extended | 50 | 64->128 channels | Dropout 0.25, WD 1e-4 |

**Objective:** Compare training duration vs accuracy tradeoff.
""",
)

# Hyperparameter sweep
multirun_config(
    "lr_sweep",
    overrides=[
        "+experiment=cifar10_quick",
        "model_config.epochs=10",
        "model_config.learning_rate=0.0001,0.001,0.01,0.1",
    ],
    description="Learning rate sweep: 4 values from 1e-4 to 1e-1 on quick config.",
)

# Grid search (N x M runs)
multirun_config(
    "lr_batch_grid",
    overrides=[
        "+experiment=cifar10_quick",
        "model_config.epochs=10",
        "model_config.learning_rate=0.001,0.01",
        "model_config.batch_size=64,128",
    ],
    description="LR x batch size grid: 2x2 = 4 total runs.",
)
```

**Key rules:**
- First arg is the multirun name (string), not a keyword
- `overrides` is a list of Hydra override strings (comma-separated values for sweeps)
- `description` supports rich markdown (tables, headers) — shown on the parent execution
- No `--multirun` flag needed when using `multirun_config` — it's automatic
- CLI usage: `uv run deriva-ml-run +multirun=lr_sweep`

## Notebook Configs (`configs/<notebook>.py`)

```python
from dataclasses import dataclass
from deriva_ml.execution import BaseConfig, notebook_config

@dataclass
class ROCAnalysisConfig(BaseConfig):
    """Custom parameters for this notebook."""
    show_per_class: bool = True
    confidence_threshold: float = 0.0

notebook_config(
    "roc_analysis",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_quick_vs_extended", "datasets": "no_datasets"},
    description="ROC curve analysis (default: quick vs extended training)",
)

# Simple notebook with no custom parameters
notebook_config(
    "my_analysis",
    defaults={"assets": "my_assets"},
)
```

In the notebook:
```python
from deriva_ml.execution import run_notebook

ml, execution, config = run_notebook("roc_analysis")
# config.assets, config.show_per_class, config.confidence_threshold are available
```

## Base Config (`configs/base.py`)

```python
from hydra_zen import store
from deriva_ml import DerivaML
from deriva_ml.execution import BaseConfig, DerivaBaseConfig, base_defaults, create_model_config

DerivaModelConfig = create_model_config(
    DerivaML,
    description="Simple model run",
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "default_deriva"},
        {"datasets": "default_dataset"},
        {"assets": "default_asset"},
        {"workflow": "default_workflow"},
        {"model_config": "default_model"},
    ],
)

store(DerivaModelConfig, name="deriva_model")
```

**Key rule:** Each default name must match a `name=` in the corresponding config group's store.

## Deriva Connection (`configs/deriva.py`)

```python
from hydra_zen import store
from deriva_ml import DerivaMLConfig

deriva_store = store(group="deriva_ml")

# REQUIRED: default_deriva
deriva_store(
    DerivaMLConfig,
    name="default_deriva",
    hostname="localhost",
    catalog_id=6,
    use_minid=False,
    zen_meta={
        "description": (
            "Local development catalog (localhost:6) with CIFAR-10 data. "
            "Schema: cifar10_10k."
        )
    },
)
```

## Config `__init__.py`

The `__init__.py` must re-export `load_configs` so all config modules are discovered:

```python
from deriva_ml.execution import load_configs

load_all_configs = lambda: load_configs("configs")
```

All config modules in the package are imported automatically by `load_configs()`.

## Description Mechanisms

Two mechanisms exist — use the right one for the context:

| Config Type | Mechanism | Example |
|---|---|---|
| Lists (datasets, assets) | `with_description(items, "...")` | `with_description([DatasetSpecConfig(...)], "Training images v3")` |
| `builds()` configs (models, connections) | `zen_meta={"description": "..."}` | `store(Config, name="x", zen_meta={"description": "..."})` |
| Experiments | `description=` param on `make_config()` | `make_config(..., description="Quick training run")` |
| Multiruns | `description=` param on `multirun_config()` | `multirun_config("name", ..., description="...")` |
| Notebooks | `description=` param on `notebook_config()` | `notebook_config("name", ..., description="...")` |

Descriptions are recorded in execution metadata and make experiments self-documenting. Before writing descriptions, look up catalog details via `deriva-ml://dataset/{rid}` or `deriva-ml://asset/{rid}`.

### Good Descriptions

- **Specific**: "ResNet-50 with 3-class output head, trained with cosine annealing LR schedule"
- **Quantified**: "4,500 histopathology tiles at 224x224, balanced across 3 subtypes"
- **Purposeful**: "Validation set held out by patient ID to prevent data leakage"
- **Version-aware**: "Frozen at version 3, which excludes 12 QC-failed slides"

## Validating Configs Against the Catalog

Before running experiments, validate that all RIDs and versions in config files actually exist in the connected catalog. This catches common errors like typos in RIDs, stale versions, or configs pointing at the wrong catalog.

### Validation Checklist

For each config file, check:

| Config Type | What to Validate | MCP Tool / Resource |
|---|---|---|
| `DatasetSpecConfig(rid=..., version=...)` | RID exists, version exists | `deriva-ml://dataset/{rid}` |
| Asset RID strings `["3WS6"]` | RID exists in an asset table | `validate_rids(rids=[...])` |
| `AssetSpecConfig(rid=...)` | RID exists | `validate_rids(rids=[...])` |
| `workflow_type="Training"` | Workflow type term exists | `deriva-ml://catalog/workflow-types` |

### Validation Workflow

1. **Connect to the catalog** using the same `deriva_ml` config the experiment will use
2. **Read the config files** and extract all RIDs and versions
3. **Validate RIDs** — use `validate_rids` to batch-check that all RIDs exist
4. **Check dataset versions** — for each `DatasetSpecConfig`, read `deriva-ml://dataset/{rid}` and verify the version exists. If the version is older than `current_version`, the config may be using stale data
5. **Report mismatches** — list any RIDs that don't exist, versions that are missing, or versions that are behind current

### Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `Dataset not found: RID=...` | RID doesn't exist in target catalog | Verify RID against correct catalog (dev vs prod) |
| `Version X not found` | Version never created | Use `get_current_version` to find latest, or `increment_dataset_version` |
| Stale version | Data changed since version was created | Call `increment_dataset_version`, then update config |
| Wrong catalog | Config RIDs are from a different catalog | Check `deriva_ml` config group — are you pointing at the right host/catalog? |

### Proactive Validation

After any catalog-modifying action (create_dataset, split_dataset, increment_dataset_version, etc.), proactively:

1. Note the new RID, version, and description
2. Check if existing config files reference the affected entity
3. Offer to update configs if versions are stale or new entities should be added
4. Present changes for approval before modifying files
5. Remind the user to commit config changes before running experiments
