---
name: dataset-versioning
description: "Dataset version management rules for DerivaML — always use explicit versions in DatasetSpecConfig, increment after catalog changes, check versions before experiments. Use when pinning versions, debugging version mismatches, or understanding the versioning lifecycle."
user-invocable: false
disable-model-invocation: true
---

# Dataset Versioning Rules

Dataset versioning is essential for reproducible ML experiments. Follow these rules strictly.

## Rule 1: Always Use Explicit Versions for Real Experiments

NEVER use "current" or "latest" for production or real experiment runs.

**Correct:**
```python
DatasetSpecConfig(rid="1-ABC4", version="1.2.0")
```

**Wrong:**
```python
DatasetSpecConfig(rid="1-ABC4", version="current")
DatasetSpecConfig(rid="1-ABC4")  # implies current
```

The ONLY acceptable use of "current" is for debugging and dry runs where reproducibility is not required.

## Rule 2: Increment Version After Catalog Changes

Dataset versions are snapshots of the catalog state at a point in time. If you modify the catalog in any way that affects a dataset's contents, those changes are NOT visible in existing versions.

Changes that require a version increment:
- Adding new features or feature values
- Fixing or correcting labels
- Adding new images or assets
- Modifying asset metadata
- Adding or removing dataset members
- Changing vocabulary terms used by features

After any such change:
1. Call `increment_dataset_version()` with a description of what changed
2. Update configuration files to reference the new version
3. Commit the config changes before running experiments

## Rule 3: Always Provide Version Descriptions

Version descriptions are strongly recommended and should explain:
- **What** changed in this version
- **Why** the change was made
- **Impact** on experiments or downstream usage

**Good descriptions** — state what changed, why, and impact on downstream usage:
- "Added severity grading feature (mild/moderate/severe) to all 12,450 images. Required for new stratified training pipeline"
- "Fixed 47 mislabeled pneumonia images identified in audit review. Retraining recommended for any model trained on v1.1.0"
- "Added 2,000 new COVID-19 images from March 2026 collection. Increases COVID class from 3,200 to 5,200 images"

**Bad descriptions:**
- "Updated"
- "New version"
- "Changes"
- "" (empty)

## Semantic Versioning

Follow semantic versioning for dataset versions:

| Version Component | When to Increment | Examples |
|-------------------|-------------------|----------|
| **Major** (X.0.0) | Breaking changes, schema modifications, incompatible structure changes | New column requirements, removed features, restructured tables |
| **Minor** (0.X.0) | New data, new features, non-breaking additions | Added images, new feature annotations, expanded vocabulary |
| **Patch** (0.0.X) | Bug fixes, label corrections, metadata fixes | Fixed mislabeled images, corrected metadata, typo fixes |

## Workflow

### After Creating a Dataset

1. Create the dataset with `create_dataset`
2. Add it to config with explicit version:
   ```python
   training_v1 = builds(DatasetSpec, rid="1-ABC4", version="1.0.0")
   ```
3. Commit the config

### After Modifying the Catalog

1. Make catalog changes (add features, fix labels, etc.)
2. Increment version with description:
   ```
   increment_dataset_version(dataset_rid="1-ABC4", description="Added severity grading feature")
   ```
3. Update config to new version:
   ```python
   training_v1 = builds(DatasetSpec, rid="1-ABC4", version="1.1.0")
   ```
4. Commit config changes
5. Run experiments with the new version

## Reference Resources

- `deriva://docs/datasets` — Full guide to dataset versioning, snapshots, and version increment workflows. Read this for detailed examples and edge cases beyond what this skill covers.
- `deriva://dataset/{rid}/versions` — View version history for a dataset
- `deriva://dataset/{rid}` — Current dataset info including version

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Running without explicit version | Results not reproducible | Always specify version in config |
| Expecting catalog changes in old versions | Old versions are frozen snapshots | Increment version to capture changes |
| Empty or vague version descriptions | Cannot understand version history | Write specific, informative descriptions |
| Not updating config after increment | Experiments still use old version | Update config immediately after incrementing |
| Not committing config before running | Git hash doesn't match config state | Always commit, then run |

## Pre-Experiment Checklist

Before running any experiment:

- [ ] Dataset version is explicitly specified (not "current")
- [ ] Config file is updated with correct version
- [ ] Config changes are committed to git

After any catalog modification:

- [ ] Version has been incremented with a descriptive message
- [ ] All affected config files are updated to the new version
- [ ] Config changes are committed to git
