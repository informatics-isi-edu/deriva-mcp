---
name: route-run-workflows
description: "Use this skill for running and configuring DerivaML experiments, notebooks, and executions. Covers running experiments with deriva-ml-run, executing notebooks, writing Hydra-Zen configs, creating new model functions, managing execution lifecycle and provenance, and troubleshooting execution failures. Choose this when users want to run, configure, create, or debug any ML workflow."
---

# ML Workflows — Experiments, Notebooks, Executions, and Configuration

You are a router skill. Based on the user's request, load the appropriate specialized skill.

## Routing Rules

Analyze the user's intent and read the matching skill:

### Running experiments
- **Running experiments with deriva-ml-run, pre-flight checks, dry runs, CLI overrides, named multiruns, verifying results** → Read and follow `../run-experiment/SKILL.md`

### Running notebooks
- **Creating, developing, or running DerivaML Jupyter notebooks, notebook structure, run_notebook(), deriva-ml-run-notebook, papermill parameters** → Read and follow `../run-notebook/SKILL.md`

### Execution lifecycle (Python)
- **Creating executions in Python code, context managers, ml.create_execution(), registering outputs with asset_file_path(), nested executions, restoring executions** → Read and follow `../run-ml-execution/SKILL.md`

### Configuration
- **Writing or editing Hydra-Zen config files — DatasetSpecConfig, AssetSpecConfig, builds(), experiment_config, multirun_config, with_description, notebook_config** → Read and follow `../write-hydra-config/SKILL.md`
- **Setting up experiment project structure, understanding config composition, hydra defaults, DerivaModelConfig** → Read and follow `../configure-experiment/SKILL.md`

### Creating new models
- **Creating a new model function, scaffolding model code, wiring a model into configs/workflows/experiments** → Read and follow `../new-model/SKILL.md`

### Troubleshooting
- **Any execution failure, error, stuck status, authentication issue, upload timeout, missing files, version mismatch, permission denied** → Read and follow `../troubleshoot-execution/SKILL.md`

## Important

After identifying the correct skill, read its SKILL.md file completely and follow its instructions. Do not attempt to handle the request from this routing skill alone.
