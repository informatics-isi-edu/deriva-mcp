---
name: run-experiment
description: "ALWAYS use this skill when running experiments with deriva-ml-run — pre-flight checks, dry runs, CLI commands, and result verification. Triggers on: 'run experiment', 'deriva-ml-run', 'dry run', 'multirun', 'sweep', 'pre-flight', 'check git status before running'."
disable-model-invocation: true
---

# Run an Experiment with deriva-ml-run

This covers the pre-flight checks and CLI commands for running experiments. Assumes configs are already set up (see `configure-experiment`).

## Pre-Flight Checklist

Complete these before every production run. **Stop and fix any issues.**

1. **Git clean** — `git status` must show no uncommitted changes (commit hash is recorded)
2. **Version current** — Bump with `uv run bump-version patch|minor` if meaningful changes since last run
3. **Lock file valid** — `uv lock --check` must pass
4. **User confirmation** — Present commit, version, branch, experiment name; get explicit approval

## Key Rule: Dry Run First

Always test with `dry_run=True` before a production run:

```bash
uv run deriva-ml-run +experiment=baseline dry_run=True
```

This downloads data and runs the model but does not upload results to the catalog.

## CLI Quick Reference

```bash
# Inspect config without running
uv run deriva-ml-run --info
uv run deriva-ml-run +experiment=baseline --info

# Single experiment
uv run deriva-ml-run +experiment=baseline

# Override host/catalog
uv run deriva-ml-run --host ml-dev.derivacloud.org --catalog 99 +experiment=baseline

# Override parameters
uv run deriva-ml-run +experiment=baseline model_config.learning_rate=0.001

# Named multirun (no --multirun flag needed)
uv run deriva-ml-run +multirun=lr_sweep

# Ad-hoc multirun
uv run deriva-ml-run +experiment=baseline model_config.learning_rate=1e-2,1e-3,1e-4 --multirun
```

## Verify Results

After a run, check the execution was recorded:
- Read `deriva://execution/{rid}` for details
- Read `deriva://chaise-url/{rid}` for the web UI link
- Verify: status is "Completed", correct datasets linked, output assets attached, git hash matches

## Reference Resources

- `deriva://execution/{rid}` — Execution details and status after a run
- `deriva://experiment/{rid}` — Full experiment info with inputs and outputs
- `deriva://chaise-url/{rid}` — Web UI link for viewing results in Chaise

For the full guide with troubleshooting table, Hydra override syntax, and multirun details, read `references/workflow.md`.
