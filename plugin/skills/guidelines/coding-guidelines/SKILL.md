---
name: coding-guidelines
description: "Coding standards and project setup for DerivaML projects — uv/pyproject.toml configuration, Git workflow, Google docstrings, ruff linting, type hints. Use when setting up a new project or establishing development practices."
---

# Coding Guidelines for DerivaML Projects

This skill covers the recommended workflow, coding standards, and best practices for developing DerivaML-based machine learning projects.

## Repository Setup

### Start with Your Own Repository

Every DerivaML project should live in its own Git repository. Do not develop inside the DerivaML library itself.

```bash
mkdir my-ml-project
cd my-ml-project
git init
uv init
```

### Use `uv` as the Package Manager

DerivaML projects use `uv` for dependency management. Always:

- Define dependencies in `pyproject.toml`.
- Use `uv add` to add dependencies (not `pip install`).
- **Commit `uv.lock`** to version control. This ensures reproducible environments across machines and over time.

```bash
uv add deriva-ml
uv add torch torchvision  # ML framework deps
```

### Typical `pyproject.toml` Structure

```toml
[project]
name = "my-ml-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "deriva-ml>=0.5.0",
]

[project.optional-dependencies]
jupyter = ["jupyterlab", "papermill"]
dev = ["pytest", "ruff"]

[dependency-groups]
jupyter = ["jupyterlab", "papermill"]

[project.scripts]
deriva-ml-run = "my_project.configs:main"
```

## Environment Management

- Use `uv sync` to install the project in development mode.
- Use `uv sync --group=jupyter` when you need Jupyter support.
- Use `uv run` to execute commands within the project environment.
- Never install packages globally for project work.

## Git Workflow

### Branch Strategy

- Use feature branches for all work: `git checkout -b feature/add-segmentation-model`.
- Keep `main` clean and passing.
- Use pull requests for code review.

### Commit Before Running

**Always commit your code before running an experiment.** DerivaML records the git state (commit hash, branch, dirty status) in the execution metadata. If you run with uncommitted changes:

- The execution is marked as having a dirty working tree.
- Reproducing the exact run later becomes difficult or impossible.

### Version Bumping

Use `bump-version` (via the MCP tool or CLI) before production runs:

```bash
uv run bump-version patch  # 0.1.0 -> 0.1.1
```

Versioning conventions:
- **patch**: Bug fixes, small parameter tweaks.
- **minor**: New experiment configurations, new model architectures.
- **major**: Breaking changes to the training pipeline or data format.

Commit the version bump before running.

## Coding Standards

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def train_model(config: ModelConfig, dataset_path: Path) -> dict[str, float]:
    """Train the classification model on the provided dataset.

    Args:
        config: Model hyperparameters and architecture configuration.
        dataset_path: Path to the downloaded and extracted dataset.

    Returns:
        Dictionary of metric names to final values, e.g.
        {"accuracy": 0.95, "loss": 0.12}.

    Raises:
        ValueError: If the dataset contains no samples.
    """
```

### Type Hints

Use type hints on all function signatures. Use modern Python typing (Python 3.11+):

```python
from pathlib import Path

def load_images(directory: Path, extensions: list[str] | None = None) -> list[Path]:
    ...

def compute_metrics(predictions: dict[str, list[float]]) -> dict[str, float]:
    ...
```

### Code Formatting and Linting

- Use `ruff` for linting and formatting.
- Configure in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

## Semantic Versioning

DerivaML projects follow semantic versioning. The version is recorded in every execution, creating a direct link between code and results.

| Change Type | Version Bump | Example |
|---|---|---|
| Fix a bug in data loading | patch | 0.1.0 -> 0.1.1 |
| Add a new model architecture | minor | 0.1.1 -> 0.2.0 |
| Restructure the config system | major | 0.2.0 -> 1.0.0 |

The version lives in `pyproject.toml` and is managed by the `bump-version` tool.

## Notebook Guidelines

Never commit notebook outputs to Git -- install `nbstripout` to strip them automatically. Keep each notebook focused on one task, and ensure it runs end-to-end with Restart & Run All.

See `setup-notebook-environment` for full environment setup and `run-notebook` for execution tracking.

## Experiments and Data

- Define experiment configs in hydra-zen (see `configure-experiment`)
- Always test with `dry_run=True` before production runs (see `run-experiment`)
- Never commit data files to Git -- store in Deriva catalogs and pin dataset versions (see `dataset-versioning`)
- Wrap all data operations in executions for provenance (see `run-ml-execution`)

## Extensibility

Prefer inheritance and composition over modifying DerivaML library code:

```python
from deriva.ml import DerivaML

class MyProjectML(DerivaML):
    """Extended DerivaML with project-specific helpers."""

    def load_training_data(self, dataset_rid: str) -> pd.DataFrame:
        ...
```

## Summary Checklist

- [ ] Own repository with `uv` and committed `uv.lock`
- [ ] Feature branches and pull requests
- [ ] Google docstrings and type hints on all public APIs
- [ ] `nbstripout` installed for notebooks
- [ ] No data files in Git -- store in Deriva catalogs
- [ ] Version bumped and committed before production runs
