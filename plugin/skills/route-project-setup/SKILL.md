---
name: route-project-setup
description: "Use this skill when setting up or managing a DerivaML development environment. Covers checking and updating package versions, installing Jupyter kernels, configuring nbstripout, authenticating with Deriva/Globus, setting up pyproject.toml, managing uv dependencies, and establishing coding standards and Git workflow."
---

# Project Setup — Environment, Versions, and Standards

You are a router skill. Based on the user's request, load the appropriate specialized skill.

## Routing Rules

Analyze the user's intent and read the matching skill:

### Environment setup for notebooks
- **Setting up Jupyter environment, installing kernels, uv sync --group=jupyter, configuring nbstripout, Deriva/Globus authentication, PyTorch dependencies** → Read and follow `../setup-notebook-environment/SKILL.md`

### Version checking and updates
- **Checking if DerivaML packages are up to date, updating packages, version queries** → Read and follow `../check-versions/SKILL.md`

### Coding standards and project setup
- **Project setup from scratch, pyproject.toml structure, uv configuration, Git workflow, Google docstrings, ruff linting, type hints, version bumping** → Read and follow `../coding-guidelines/SKILL.md`

## Important

After identifying the correct skill, read its SKILL.md file completely and follow its instructions. Do not attempt to handle the request from this routing skill alone.
