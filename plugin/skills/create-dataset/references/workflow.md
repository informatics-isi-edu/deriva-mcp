# Dataset Workflow Reference

Step-by-step MCP tool examples for creating and managing datasets. For background concepts, see `concepts.md`. For bag downloads, see `bags.md`.

## Table of Contents

1. [Creating a Dataset](#creating-a-dataset) — Check resources, create execution, add members
2. [Managing Types](#managing-types) — Add, remove, create custom types
3. [Managing Members](#managing-members) — Add, remove, validate, list
4. [Splitting Datasets](#splitting-datasets) — Random, stratified, labeled, dry run, navigation
5. [Versioning](#versioning) — When and how to increment
6. [Downloading](#downloading) — Preview and download
7. [Provenance](#provenance) — Track dataset lineage
8. [Deleting](#deleting) — Remove datasets
9. [Complete Example](#complete-example) — End-to-end workflow

---

## Creating a Dataset

### Check existing resources first

Before creating a dataset, review what already exists:

- Call `query_table` with `table_name`: `"Dataset"` to list existing datasets.
- Call `query_table` with `table_name`: `"Dataset_Type"` to list available types.
- Read the `deriva://catalog/dataset-element-types` resource to see which tables are registered as element types.

### MCP Tools

Each step below is a separate MCP tool call. Use the RID returned by each tool in subsequent calls.

**Step 1: Create and start an execution for provenance**

Call `create_execution` with:
- `workflow_name`: `"Dataset Curation"`
- `workflow_type`: `"Data Management"`
- `description`: `"Create training dataset"`

Then call `start_execution`.

**Step 2: Create the dataset**

Call `create_dataset` with:
- `description`: `"Curated set of labeled tumor histology images"`
- `dataset_types`: `["Training", "Labeled"]`

Note the returned dataset RID (e.g., `"2-DS01"`) — you'll need it for subsequent steps.

**Step 3: Register element types** (catalog-level, idempotent)

Call `add_dataset_element_type` with `table_name`: `"Image"`.
Call `add_dataset_element_type` with `table_name`: `"Subject"`.

**Step 4: Add members**

Call `add_dataset_members` with:
- `dataset_rid`: the RID from step 2
- `member_rids`: `["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5"]`
- `description`: `"Initial population of labeled tumor images"`

This auto-increments the dataset version; the description is recorded in version history.

Alternatively, add members grouped by table (faster for large datasets):
- `dataset_rid`: the RID from step 2
- `members_by_table`: `{"Image": ["2-IMG1", "2-IMG2"], "Subject": ["2-SUB1"]}`
- `description`: `"Added remaining images and subjects"`

**Step 5: Finalize**

Call `stop_execution`. (No need to call `upload_execution_outputs` — dataset operations don't produce output files.)

### Python API

For creating datasets in Python scripts with full provenance, see the `run-ml-execution` skill which covers `ExecutionConfiguration` and context manager patterns. A brief example:

```python
from deriva_ml import DerivaML, ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)
workflow = ml.create_workflow(
    name="Dataset Curation",
    workflow_type="Data Management",
    description="Curate and organize training datasets"
)

with ml.create_execution(ExecutionConfiguration(workflow=workflow)) as exe:
    dataset = exe.create_dataset(
        description="Labeled tumor images",
        dataset_types=["Training", "Labeled"]
    )
    ml.add_dataset_element_type("Image")
    dataset.add_dataset_members(
        members=["2-IMG1", "2-IMG2", "2-IMG3"],
        description="Initial labeled images"
    )
```

## Managing Types

To **add a type** to a dataset, call `add_dataset_type` with `dataset_rid` and `dataset_type` (e.g., `"Training"`).

To **remove a type**, call `remove_dataset_type` with the same parameters (e.g., `dataset_type`: `"Complete"`).

To **create a new custom type**, call `create_dataset_type_term` with `type_name` (e.g., `"Preprocessed"`) and `description`.

To **delete a custom type**, call `delete_dataset_type_term` with `type_name`.

## Managing Members

To **list current members**, call `list_dataset_members` with `dataset_rid`.

To **validate RIDs** before adding (catches invalid RIDs early), call `validate_rids` with `dataset_rids` (for dataset RIDs) or `asset_rids` (for asset RIDs).

To **add more members**, call `add_dataset_members` with:
- `dataset_rid`: the dataset's RID
- `member_rids`: list of RIDs to add
- `description`: why these members are being added (recorded in version history)

This auto-increments the dataset version.

To **remove members**, call `delete_dataset_members` with `dataset_rid` and `member_rids`.

## Splitting Datasets

`split_dataset` creates nested child datasets from a parent. It follows the same conventions as scikit-learn's [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — parameters like `test_size`, `train_size`, `shuffle`, `seed`, and stratification work the same way. It auto-increments the dataset version. Always use `dry_run=true` first to preview the split plan.

### Basic usage

To **preview a split** without modifying the catalog, call `split_dataset` with `source_dataset_rid`, `test_size`, `seed`, and `dry_run`: `true`.

To **create a two-way split** (e.g., 80% Training / 20% Testing), call `split_dataset` with:
- `source_dataset_rid`: the dataset's RID
- `test_size`: `0.2`
- `seed`: `42`

To **create a three-way split**, also include `val_size` (e.g., `0.1` for 10% Validation).

### Stratified and labeled splits

To maintain class distribution, add `stratify_by_column` with the denormalized column name. Derive this from the table schema — do **not** call `denormalize_dataset` just to discover column names.

**Finding the stratify column name:**

1. Read `deriva://catalog/schema` or `deriva://catalog/features` to find the feature table name and its columns
2. Construct the denormalized column name as `{FeatureTableName}_{ColumnName}`

For example, if the feature table is `Execution_Image_Image_Classification` and the column is `Image_Class`, the stratify column is `Execution_Image_Image_Classification_Image_Class`.

`include_tables` is required when using stratification — use the feature table name from the schema.

**Handling missing values in the stratify column:** Not all members may have a value for the stratify column (e.g., unlabeled images in a labeled feature table). Use `stratify_missing` to control this:

| Policy | Behavior |
|--------|----------|
| `"error"` (default) | Raise an error reporting the count and percentage of nulls |
| `"drop"` | Exclude rows with missing values — only labeled rows are split |
| `"include"` | Treat nulls as a distinct class — missing-value rows are distributed proportionally |

To label partitions with ground truth metadata (needed for evaluation, ROC curves, etc.), add `training_types`, `testing_types`, and/or `validation_types` (e.g., `["Labeled"]`).

**Example:** A stratified, labeled three-way split would use:
- `source_dataset_rid`: the dataset's RID
- `test_size`: `0.2`, `val_size`: `0.1`, `seed`: `42`
- `stratify_by_column`: `"Image_Classification_Image_Class"`
- `include_tables`: `["Image", "Image_Classification"]`
- `stratify_missing`: `"drop"` (if some images lack labels)
- `training_types`: `["Labeled"]`, `testing_types`: `["Labeled"]`, `validation_types`: `["Labeled"]`

### Parameter reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_dataset_rid` | `str` | *(required)* | RID of the dataset to split |
| `test_size` | `float` | `0.2` | Fraction for testing (0-1) |
| `train_size` | `float \| None` | `None` | Fraction for training. Default: complement of test + val |
| `val_size` | `float \| None` | `None` | Fraction for validation. When set, creates 3-way split |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `shuffle` | `bool` | `True` | Shuffle before splitting |
| `stratify_by_column` | `str \| None` | `None` | Denormalized column name for stratified split |
| `stratify_missing` | `str` | `"error"` | Policy for nulls in stratify column: `"error"`, `"drop"`, `"include"` |
| `element_table` | `str \| None` | `None` | Table to split. Auto-detected if dataset has one element type |
| `include_tables` | `list[str] \| None` | `None` | Tables for denormalization. Required with `stratify_by_column` |
| `training_types` | `list[str] \| None` | `None` | Additional types for training set (e.g., `["Labeled"]`) |
| `testing_types` | `list[str] \| None` | `None` | Additional types for testing set |
| `validation_types` | `list[str] \| None` | `None` | Additional types for validation set |
| `split_description` | `str` | `""` | Description for the parent Split dataset |
| `dry_run` | `bool` | `False` | Preview without modifying catalog |

### Navigating split results

`split_dataset` creates a parent "Split" dataset with child datasets for each partition.

To **list children** of a dataset, call `list_dataset_children` with `dataset_rid`. Add `recurse`: `true` to include all descendants, or `version` to list children at a specific version.

To **list parents** of a child dataset, call `list_dataset_parents` with `dataset_rid`.

To **list members across nested datasets**, call `list_dataset_members` with `dataset_rid`, `recurse`: `true`, and optionally `limit`.

To create parent-child relationships manually (without `split_dataset`), call `add_dataset_child` with `parent_rid` and `child_rid`. See `concepts.md` for background on nested dataset hierarchies.

## Versioning

`add_dataset_members` and `split_dataset` auto-increment the minor version. Manual incrementation is only needed for other changes (removing members, changing element types, data cleanup).

To **manually increment**, call `increment_dataset_version` with `dataset_rid`. Optionally specify `component` (`"major"`, `"minor"`, or `"patch"`) and `description` (e.g., `"Corrected mislabeled records"`).

See the `dataset-versioning` skill for full rules and the pre-experiment checklist.

## Downloading

To **preview** what a bag will contain, call `estimate_bag_size` with `dataset_rid` and `version`.

For downloading, preparing, and restructuring dataset data for ML training, see the `prepare-training-data` skill. For details on bag contents, FK traversal, and timeout handling, see `bags.md`. For diagnosing missing data, see the `debug-bag-contents` skill.

## Provenance

To find **which executions used a dataset**, call `list_dataset_executions` with `dataset_rid`.

To find **which executions used an asset**, call `list_asset_executions` with `asset_rid`.

## Deleting

To **delete a dataset** (removes the container and member associations, not the member records themselves), call `delete_dataset` with `dataset_rid`.

To **delete a dataset and all its children**, add `recurse`: `true`.

## Complete Example

End-to-end workflow: create a dataset, add members, and execute a stratified labeled split.

```python
from deriva_ml import DerivaML, ExecutionConfiguration
from deriva_ml.dataset.split import split_dataset

ml = DerivaML(hostname, catalog_id)

workflow = ml.create_workflow(
    name="Image Dataset Curation",
    workflow_type="Data Management",
    description="Curate and split image datasets for training"
)

config = ExecutionConfiguration(
    workflow=workflow,
    description="Create and split tumor image dataset"
)

with ml.create_execution(config) as exe:
    # 1. Register element types (catalog-level, idempotent)
    ml.add_dataset_element_type("Image")

    # 2. Create the master dataset
    dataset = exe.create_dataset(
        description="All labeled tumor histology images as of 2025-06",
        dataset_types=["Complete", "Labeled"]
    )

    # 3. Add all labeled images (description records why this version was created)
    dataset.add_dataset_members(
        members=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5",
                 "2-IMG6", "2-IMG7", "2-IMG8", "2-IMG9", "2-IMG10"],
        description="Initial population of labeled tumor images"
    )

    # 4. Preview the split
    result = split_dataset(
        ml, dataset.dataset_rid,
        test_size=0.15, val_size=0.15,
        stratify_by_column="Image_Classification_Image_Class",
        stratify_missing="drop",  # exclude unlabeled images
        include_tables=["Image", "Image_Classification"],
        seed=42, dry_run=True
    )
    print(f"Plan: {result.training.count} train, "
          f"{result.validation.count} val, {result.testing.count} test")

    # 5. Execute the split
    result = split_dataset(
        ml, dataset.dataset_rid,
        test_size=0.15, val_size=0.15,
        stratify_by_column="Image_Classification_Image_Class",
        stratify_missing="drop",
        include_tables=["Image", "Image_Classification"],
        training_types=["Labeled"],
        testing_types=["Labeled"],
        validation_types=["Labeled"],
        seed=42
    )
    print(f"Training: {result.training.rid}, Testing: {result.testing.rid}")

# Execution is automatically stopped and outputs uploaded on context exit
```
