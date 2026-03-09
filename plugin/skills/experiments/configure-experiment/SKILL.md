---
name: configure-experiment
description: "ALWAYS use this skill when setting up a DerivaML experiment project, adding config groups, or understanding how experiments compose. Triggers on: 'set up experiment', 'config groups', 'project structure', 'hydra defaults', 'DerivaModelConfig', 'experiment preset', 'new project from template'."
---

# Configure ML Experiments with hydra-zen and DerivaML

This covers the structure of a DerivaML experiment project: config groups, how they compose, and project setup. For exact Python API patterns for each config type, see the `write-hydra-config` skill.

## Config Groups

| Group | Purpose | File |
|---|---|---|
| `deriva_ml` | Catalog connection (host, catalog ID) | `configs/deriva.py` |
| `datasets` | Dataset RIDs and versions | `configs/datasets.py` |
| `assets` | Pre-trained weights, reference files | `configs/assets.py` |
| `workflow` | What the code does | `configs/workflow.py` |
| `model_config` | Hyperparameters and architecture | `configs/<model>.py` |
| `experiment` | Named combinations of the above | `configs/experiments.py` |
| `multiruns` | Sweeps over experiments/parameters | `configs/multiruns.py` |

## How Experiments Compose

```
Base config (defaults for every group)
  + Experiment overrides (swap specific groups)
    + CLI overrides (fine-tune individual parameters)
```

Example: `uv run deriva-ml-run +experiment=cifar10_quick` loads base defaults, then overrides `model_config` and `datasets` from the experiment preset.

## Critical Rules

1. **Every group needs a default** — `default_deriva`, `default_dataset`, `default_asset`, `default_workflow`, `default_model`
2. **Pin dataset versions** — Use `DatasetSpecConfig(rid="...", version="...")` for reproducibility
3. **Use meaningful names** — `resnet50_extended` not `config2`
4. **Test with `--info`** — `uv run deriva-ml-run +experiment=X --info` to inspect resolved config

## Setup Steps

1. Clone the model template or create `configs/` directory
2. Configure each group in order: `deriva.py` → `datasets.py` → `assets.py` → `workflow.py` → `<model>.py` → `base.py` → `experiments.py`
3. Verify: `uv run deriva-ml-run --info`

For the full project structure, `base.py` template, and setup walkthrough, read `references/workflow.md`.

## Related Skills

- **`write-hydra-config`** — Exact Python API patterns for each config type
- **`run-experiment`** — Pre-flight checklist and CLI commands for running
