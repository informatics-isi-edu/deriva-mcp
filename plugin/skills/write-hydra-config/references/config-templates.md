# Config File Templates

Complete starter templates for each config file in a DerivaML project. Copy and adapt these when creating a new project from scratch.

## Table of Contents

1. [configs/__init__.py](#initpy)
2. [configs/base.py](#basepy)
3. [configs/deriva.py](#derivapy)
4. [configs/datasets.py](#datasetspy)
5. [configs/assets.py](#assetspy)
6. [configs/workflow.py](#workflowpy)
7. [configs/model.py](#modelpy)
8. [configs/experiments.py](#experimentspy)
9. [configs/multiruns.py](#multirunspy)
10. [configs/notebook_example.py](#notebookpy)

---

## `__init__.py`

```python
"""Configuration Package.

All config modules are discovered automatically by load_configs().
"""
from deriva_ml.execution import load_configs

load_all_configs = lambda: load_configs("configs")
```

## `base.py`

```python
"""Base configuration for the model runner.

Experiments inherit from DerivaModelConfig.
"""
from hydra_zen import store
from deriva_ml import DerivaML
from deriva_ml.execution import BaseConfig, DerivaBaseConfig, base_defaults, create_model_config

DerivaModelConfig = create_model_config(
    DerivaML,
    description="Model training run",
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

__all__ = ["BaseConfig", "DerivaBaseConfig", "DerivaModelConfig", "base_defaults"]
```

## `deriva.py`

```python
"""DerivaML Connection Configuration.

REQUIRED: A configuration named "default_deriva" must be defined.
"""
from hydra_zen import store
from deriva_ml import DerivaMLConfig

deriva_store = store(group="deriva_ml")

# REQUIRED: default_deriva
deriva_store(
    DerivaMLConfig,
    name="default_deriva",
    hostname="YOUR_HOST_HERE",      # e.g., "ml.derivacloud.org" or "localhost"
    catalog_id=YOUR_CATALOG_ID,     # e.g., 6
    use_minid=False,
    zen_meta={
        "description": "Development catalog. Replace with your catalog details."
    },
)
```

## `datasets.py`

```python
"""Dataset Configuration.

REQUIRED: A configuration named "default_dataset" must be defined.

Usage:
    uv run deriva-ml-run datasets=my_dataset_name
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

datasets_store = store(group="datasets")

# Empty dataset list (for notebooks that don't need datasets)
datasets_store([], name="no_datasets")

# Example: add your datasets here
# datasets_store(
#     with_description(
#         [DatasetSpecConfig(rid="XXXX", version="1.0.0")],
#         "Description of what this dataset contains and its purpose.",
#     ),
#     name="my_dataset",
# )

# REQUIRED: default_dataset — plain list, no with_description()
datasets_store(
    [DatasetSpecConfig(rid="XXXX", version="1.0.0")],
    name="default_dataset",
)
```

## `assets.py`

```python
"""Asset Configuration.

REQUIRED: A configuration named "default_asset" must be defined.

Usage:
    uv run deriva-ml-run assets=my_assets
"""
from hydra_zen import store
from deriva_ml.execution import with_description

asset_store = store(group="assets")

# REQUIRED: default_asset — empty list
asset_store([], name="default_asset")

# Alias for clarity
asset_store([], name="no_assets")

# Example: add your assets here
# asset_store(
#     with_description(
#         ["RID1", "RID2"],
#         "Description of what these assets are and where they came from.",
#     ),
#     name="my_assets",
# )

# Example: cached asset (for large files like model weights)
# from deriva_ml.asset.aux_classes import AssetSpecConfig
# asset_store(
#     with_description(
#         [AssetSpecConfig(rid="XXXX", cache=True)],
#         "Pre-trained weights (~500MB). Cached locally.",
#     ),
#     name="pretrained_weights",
# )
```

## `workflow.py`

```python
"""Workflow Configuration.

REQUIRED: A configuration named "default_workflow" must be defined.

Usage:
    uv run deriva-ml-run workflow=my_workflow
"""
from hydra_zen import store, builds
from deriva_ml.execution import Workflow

MyWorkflow = builds(
    Workflow,
    name="My ML Workflow",
    workflow_type="Training",  # or ["Training", "Image Classification"]
    description="""
Describe what this workflow does.

## Architecture
- Describe the model or pipeline

## Outputs
- What files/artifacts are produced
""".strip(),
    populate_full_signature=True,
)

workflow_store = store(group="workflow")

# REQUIRED: default_workflow
workflow_store(MyWorkflow, name="default_workflow")
```

## `model.py`

```python
"""Model Configuration.

REQUIRED: A configuration named "default_model" must be defined.

Usage:
    uv run deriva-ml-run model_config=my_variant
    uv run deriva-ml-run model_config.learning_rate=0.01
"""
from hydra_zen import builds, store
from my_project.models import my_model_function  # Your model's entry point

# Build the base config
# zen_partial=True is critical — execution context is injected at runtime
MyModelConfig = builds(
    my_model_function,
    # Add your model's parameters here:
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    populate_full_signature=True,
    zen_partial=True,
)

model_store = store(group="model_config")

# REQUIRED: default_model
model_store(
    MyModelConfig,
    name="default_model",
    zen_meta={
        "description": "Default configuration. Describe hyperparameters and intended use."
    },
)

# Add variants by overriding specific parameters
# model_store(
#     MyModelConfig,
#     name="quick",
#     epochs=3,
#     zen_meta={"description": "Quick test: 3 epochs for pipeline validation."},
# )
```

## `experiments.py`

```python
"""Experiment definitions.

Usage:
    uv run deriva-ml-run +experiment=my_experiment
"""
from hydra_zen import make_config, store
from configs.base import DerivaModelConfig

experiment_store = store(group="experiment", package="_global_")

# Example experiment
# experiment_store(
#     make_config(
#         hydra_defaults=[
#             "_self_",
#             {"override /model_config": "quick"},
#             {"override /datasets": "my_dataset"},
#         ],
#         description="Quick test run with small dataset",
#         bases=(DerivaModelConfig,),
#     ),
#     name="quick_test",
# )
```

## `multiruns.py`

```python
"""Multirun configurations for experiment sweeps.

Usage:
    uv run deriva-ml-run +multirun=my_sweep
"""
from deriva_ml.execution import multirun_config

# Example: compare two experiments
# multirun_config(
#     "compare_models",
#     overrides=[
#         "+experiment=quick_test,extended_test",
#     ],
#     description="Compare quick vs extended training configurations.",
# )

# Example: hyperparameter sweep
# multirun_config(
#     "lr_sweep",
#     overrides=[
#         "+experiment=quick_test",
#         "model_config.learning_rate=0.0001,0.001,0.01,0.1",
#     ],
#     description="Learning rate sweep: 4 values from 1e-4 to 1e-1.",
# )
```

## `notebook_example.py`

```python
"""Configuration for a Jupyter notebook.

Usage in notebook:
    from deriva_ml.execution import run_notebook
    ml, execution, config = run_notebook("my_analysis")

From CLI:
    uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --config my_analysis
"""
from dataclasses import dataclass
from deriva_ml.execution import BaseConfig, notebook_config


@dataclass
class MyAnalysisConfig(BaseConfig):
    """Custom parameters for this notebook."""
    threshold: float = 0.5
    show_plots: bool = True


# Simple notebook (no custom params)
# notebook_config(
#     "simple_analysis",
#     defaults={"assets": "my_assets"},
# )

# Notebook with custom parameters
# notebook_config(
#     "my_analysis",
#     config_class=MyAnalysisConfig,
#     defaults={"assets": "my_assets", "datasets": "no_datasets"},
#     description="Analysis notebook with configurable threshold",
# )
```
