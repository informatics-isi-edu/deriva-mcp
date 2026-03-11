# Dataset Concepts

Background on datasets in DerivaML. For the step-by-step guide to creating and managing datasets, see `workflow.md`.

## Table of Contents

- [What is a Dataset?](#what-is-a-dataset)
- [Dataset Types](#dataset-types)
- [Dataset Element Types](#dataset-element-types)
- [Dataset Versioning](#dataset-versioning)
- [Identifying a Dataset: RID + Version](#identifying-a-dataset-rid--version)
- [Nested Datasets](#nested-datasets)
- [Downloading Datasets as Bags](#downloading-datasets-as-bags)
- [Operations Summary](#operations-summary)

---

## What is a Dataset?

A dataset is a versioned collection of records (members) from one or more catalog tables. Datasets organize data for ML workflows — training sets, evaluation sets, curated subsets — with full provenance tracking.

Each dataset has:
- **An RID** — unique identifier, like any other catalog record
- **Members** — specific records included in the dataset, referenced by RID
- **Element types** — which tables can contribute members
- **Types** — labels describing the dataset's purpose (Training, Testing, etc.)
- **A version** — monotonically increasing, tied to a catalog snapshot for reproducibility
- **Provenance** — which execution created it, which executions have used it

Datasets can be heterogeneous: a single dataset can contain records from multiple tables (e.g., both Image and Subject records). DerivaML manages the relationships between these records and makes them accessible from all FK paths.

## Dataset Types

Dataset types are labels from the `Dataset_Type` controlled vocabulary. They describe the dataset's purpose and are used for organizing, filtering, and discovery. A dataset can have multiple types simultaneously.

**Standard types available by default:**

| Type | Purpose |
|------|---------|
| Training | Data for model training |
| Testing | Data for model evaluation |
| Validation | Data for hyperparameter tuning |
| Complete | Full dataset before splitting |
| Split | Parent container created by `split_dataset` |
| Labeled | Data with ground truth feature annotations |
| Unlabeled | Data without feature annotations |

Custom types can be created with `create_dataset_type_term`. Types are additive — a dataset labeled both "Training" and "Labeled" is a training set that includes ground truth annotations.

## Dataset Element Types

Before adding records from a table to any dataset, that table must be registered as a **dataset element type**. This is a catalog-level operation — once registered, records from that table can be added to any dataset in the catalog.

Registration creates the association table (`Dataset_{TableName}`) that links datasets to records in that table. Without this association table, the catalog has no way to track which records belong to which datasets.

**Key points:**
- Register with `add_dataset_element_type(table_name="Image")` (MCP) or `ml.add_dataset_element_type("Image")` (Python)
- Registration is idempotent — calling it again for an already-registered table is harmless
- Check what's registered via the `deriva://catalog/dataset-element-types` resource
- Element types also determine the starting points for FK traversal during bag export (see below)

## Dataset Versioning

Every dataset has a version number using semantic versioning (`major.minor.patch`):

| Component | Meaning | Example |
|-----------|---------|---------|
| Major | Schema change to objects in the dataset | Table columns added/removed |
| Minor | New elements added to the dataset | Members added, split created |
| Patch | Minor alterations | Description updated, data cleaned |

DerivaML assigns version `0.1.0` when a dataset is created. The tools `add_dataset_members` and `split_dataset` auto-increment the minor version. For other changes, call `increment_dataset_version` manually.

**Versions are snapshots.** Each version is tied to a catalog snapshot timestamp. When you download a specific version, you get the exact data that existed when that version was created — not the current state. This is the foundation of reproducibility: the same dataset RID + version always produces the same data.

**If you've modified data since the last version** (added features, updated records), those changes are not included in existing versions. Call `increment_dataset_version` to create a new version that captures the current state.

### Dataset History

Every version increment is recorded in the dataset's history — a chronological log of all versions with their snapshot timestamps, descriptions, and the execution that created them.

```python
# Python API
history = dataset.dataset_history()
for entry in history:
    print(f"Version {entry.dataset_version}: {entry.description} (snapshot: {entry.snapshot})")
```

Each `DatasetHistory` entry contains:
- `dataset_version` — the version number (e.g., `0.3.0`)
- `snapshot` — catalog snapshot timestamp (ties this version to an exact catalog state)
- `description` — why this version was created
- `execution_rid` — which execution created it (provenance)
- `minid` — permanent identifier URL, if registered

See the `dataset-versioning` skill for full versioning rules and the pre-experiment checklist.

## Identifying a Dataset: RID + Version

A dataset is uniquely identified by its **RID** (Resource Identifier), like any catalog record. But because datasets evolve over time, the combination of **RID + version** is what identifies a specific, reproducible snapshot of the data.

This pair is captured in a **DatasetSpec** — the standard way to reference a dataset in code:

```python
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetSpecConfig

# Python API
DatasetSpec(rid="28EA", version="0.4.0")

# Hydra-zen configuration (version is required)
DatasetSpecConfig(rid="28EA", version="0.4.0")
```

Use the `get_dataset_spec` MCP tool to generate the correct `DatasetSpecConfig` string for a dataset, including its current version. The `deriva://dataset/{rid}` resource also shows the current version.

## Nested Datasets

Datasets can contain other datasets as children, forming hierarchies. The most common use is train/test/validation splits:

```
Complete Dataset (parent)
└── Split (created by split_dataset)
    ├── Training (70%)
    ├── Validation (10%)
    └── Testing (20%)
```

Child datasets automatically inherit their parent's element types. They are independent datasets with their own RIDs, versions, and types — the parent-child relationship is purely organizational.

`split_dataset` creates nested datasets automatically. You can also create them manually with `add_dataset_child`.

Navigate hierarchies with `list_dataset_children` (supports `recurse=true` for all descendants) and `list_dataset_parents`.

## Downloading Datasets as Bags

Datasets can be downloaded as **BDBag** archives — self-describing, checksummed packages containing all member records, related data, asset files, feature values, and vocabulary terms. The same dataset RID + version always produces the same bag.

For full details on what bags contain, how FK traversal determines bag contents, materialization, caching, timeout handling, and Hydra-Zen configuration, see `bags.md`.

For diagnosing missing data in bag exports, see the `debug-bag-contents` skill.

## Operations Summary

| Operation | MCP Tool | What it does |
|-----------|----------|--------------|
| Create dataset | `create_dataset` | Create within active execution (provenance) |
| Add types | `add_dataset_type` | Label a dataset (Training, Labeled, etc.) |
| Remove types | `remove_dataset_type` | Remove a type label |
| Create custom type | `create_dataset_type_term` | Add to the Dataset_Type vocabulary |
| Register element type | `add_dataset_element_type` | Allow a table's records as members (catalog-level) |
| Add members | `add_dataset_members` | Add records by RID (auto-increments version) |
| Remove members | `delete_dataset_members` | Remove records from dataset |
| List members | `list_dataset_members` | View dataset contents |
| Split | `split_dataset` | Create train/test/val children (auto-increments version) |
| Increment version | `increment_dataset_version` | Manual version bump |
| Add child | `add_dataset_child` | Manually nest datasets |
| List children | `list_dataset_children` | Navigate hierarchy |
| List parents | `list_dataset_parents` | Navigate hierarchy |
| Download | `download_dataset` | Export as BDBag |
| Estimate size | `estimate_bag_size` | Preview bag contents |
| Validate RIDs | `validate_rids` | Check RIDs exist before adding |
| Delete | `delete_dataset` | Remove dataset (not member records) |
