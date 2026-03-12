---
name: create-dataset
description: "ALWAYS use this skill when creating, populating, splitting, or managing datasets in DerivaML — including adding members, registering element types, train/test splits, versioning, nested datasets, and provenance. Triggers on: 'create a dataset', 'split dataset', 'add members', 'training/testing split', 'dataset types'."
disable-model-invocation: true
---

# Creating and Managing Datasets in DerivaML

Datasets are the primary unit for organizing data in DerivaML. A dataset is a versioned collection of records (members) drawn from one or more catalog tables, with full provenance tracking through executions.

For background on what datasets are, how versioning and element types work, nested datasets, and FK traversal in bag exports, see `references/concepts.md`.

## Description Guidance

Every dataset should have a description that explains its composition, purpose, and key characteristics. The description is visible in the Chaise UI and in execution records.

**Good dataset descriptions:**
- "500 CIFAR-10 images (50 per class), balanced across all 10 categories, for rapid iteration during development"
- "80/20 patient-level stratified split of the full imaging cohort. Split at patient level to prevent data leakage from multiple images per subject"
- "Complete labeled dataset of 12,450 chest X-rays with ground truth diagnosis annotations. Source data for all training and evaluation experiments"

**Bad dataset descriptions:**
- "Training data" or "My dataset" or "Images"
- Leaving the description empty

For split datasets, the description should note the split strategy and rationale (why this ratio, why this stratification column).

## Workflow Summary

The standard sequence for creating a dataset:

1. `create_execution` — start an execution for provenance
2. `create_dataset` — create the dataset within the active execution
3. `add_dataset_type` — label the dataset (Training, Complete, Labeled, etc.)
4. `add_dataset_element_type` — register source tables (catalog-level, before adding members)
5. `validate_rids` — check RIDs exist before adding
6. `add_dataset_members` — add records by RID (auto-increments version)
7. `split_dataset` (optional) — create train/test/val children (auto-increments version). Use `dry_run=true` to preview, `seed` for reproducibility, `*_types=["Labeled"]` when splits need ground truth.
8. `stop_execution` — finalize (no `upload_execution_outputs` needed — dataset operations don't produce output files)

For the full step-by-step guide with code examples (both MCP tools and Python API), see `references/workflow.md`.

## Reference Resources

- `references/concepts.md` — What datasets are, versioning, element types, nested datasets
- `references/bags.md` — BDBag exports: what they contain, FK traversal, timeouts, caching
- `references/workflow.md` — Step-by-step how-to with MCP and Python examples
- `deriva://catalog/datasets` — Browse existing datasets before creating new ones
- `deriva://dataset/{rid}` — Dataset details including current version
- `deriva://catalog/dataset-element-types` — Check which element types are registered

## Related Skills

- **`prepare-training-data`** — Downloading, extracting, and preparing dataset data for ML training pipelines.
- **`debug-bag-contents`** — Diagnosing missing data, FK traversal issues, and export problems in dataset bags.
- **`dataset-versioning`** — Full versioning rules, semantic versioning conventions, and pre-experiment checklist.
- **`run-ml-execution`** — Wrap dataset creation in an execution for provenance tracking.
