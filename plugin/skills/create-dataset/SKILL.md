---
name: create-dataset
description: "ALWAYS use this skill when creating, populating, splitting, or managing datasets in DerivaML — including adding members, registering element types, train/test splits, versioning, nested datasets, and provenance. Triggers on: 'create a dataset', 'split dataset', 'add members', 'training/testing split', 'dataset types'."
disable-model-invocation: true
---

# Creating and Managing Datasets in DerivaML

Datasets are the primary unit for organizing data in DerivaML. A dataset is a named, versioned collection of records (members) drawn from one or more catalog tables, with full provenance tracking through executions.

## Key Concepts

- **Element types** — Tables registered as allowed member sources for a dataset. Must be registered before adding members from that table.
- **Members** — Individual records (by RID) included in the dataset. Can be added in bulk by RID list or grouped by table.
- **Types** — Labels describing purpose: Training, Testing, Validation, Complete, Labeled, Unlabeled. Custom types can also be created.
- **Versioning** — Monotonically increasing version number. Increment after any modification (add/remove members, change element types).
- **Nested datasets** — Parent-child relationships. Used for train/test/validation splits. `split_dataset` creates these automatically.
- **Provenance** — Datasets are created within executions, linking them to workflows and tracking lineage.

## Splitting Datasets

Three split types are available via `split_dataset`:

| Split Type | Use Case | Key Parameter |
|-----------|----------|---------------|
| `random` | General train/test/val splits | `ratios`, `seed` |
| `stratified` | Preserve label distribution across splits | `ratios`, `stratify_feature`, `seed` |
| `labeled` | Separate records with/without feature values | `feature_name` |

All splits create nested child datasets automatically. Use `dry_run=true` to preview before committing.

## Critical Rules and Gotchas

1. **Always create datasets within an execution** — Use `create_execution_dataset`, not bare `create_dataset`, to maintain provenance.
2. **Register element types before adding members** — `add_dataset_element_type` must be called for each source table before `add_dataset_members`.
3. **FK traversal in bag exports** — Downloaded bags include all FK-reachable records. Deep join chains (Image -> Sample -> Subject -> Study) can cause timeouts. Workaround: add intermediate table records as direct members.
4. **Version after every modification** — Call `increment_dataset_version` after adding/removing members or changing element types.
5. **Validate RIDs first** — Use `validate_rids` before `add_dataset_members` to catch invalid RIDs early.
6. **Set seeds on splits** — Always pass `seed` to `split_dataset` for reproducibility.
7. **Deleting a dataset removes only the container** — Member records (images, subjects, etc.) are not deleted, only the dataset and its member associations.

## Workflow Summary

The standard sequence for creating a dataset:

1. Create execution (with a workflow for provenance)
2. `create_execution_dataset` — create the dataset within the execution
3. `add_dataset_type` — label the dataset (Training, Complete, etc.)
4. `add_dataset_element_type` — register source tables
5. `add_dataset_members` — add records by RID
6. `split_dataset` (optional) — create train/test/val child datasets
7. `increment_dataset_version` — version the dataset
8. `stop_execution` + `upload_execution_outputs` — finalize

For the full step-by-step guide with code examples (both Python API and MCP tools), see `references/workflow.md`.

## Related Skills

- **`prepare-training-data`** — Downloading, extracting, and preparing dataset data for ML training pipelines.
- **`debug-bag-contents`** — Diagnosing missing data, FK traversal issues, and export problems in dataset bags.
