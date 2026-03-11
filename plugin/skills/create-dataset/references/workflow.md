# Creating and Managing Datasets in DerivaML

Datasets are the primary mechanism for organizing data for ML workflows in DerivaML. They group records from multiple tables into a versioned, downloadable collection with full provenance tracking.

## Table of Contents

1. [Understanding Datasets](#understanding-datasets) — What datasets are, types, element types, FK traversal
2. [Step 1: Check Existing Resources](#step-1-check-existing-resources)
3. [Step 2: Create a Dataset via Execution](#step-2-create-a-dataset-via-execution) — MCP tools and Python API
4. [Step 3: Manage Dataset Types](#step-3-manage-dataset-types)
5. [Step 4: Manage Dataset Members](#step-4-manage-dataset-members)
6. [Step 5: Split Datasets](#step-5-split-datasets) — Random, stratified, three-way, dry run
7. [Step 6: Version Management](#step-6-version-management)
8. [Nested Datasets](#nested-datasets) — Manual creation, navigation, automatic via split
9. [Downloading and Using Datasets](#downloading-and-using-datasets)
10. [Complete Example: End-to-End Dataset Workflow](#complete-example-end-to-end-dataset-workflow)
11. [Provenance Tracking](#provenance-tracking)
12. [Deleting Datasets](#deleting-datasets)
13. [Tips](#tips)

---

## Understanding Datasets

### What is a Dataset?

A dataset is a named collection of records (members) from one or more tables, with:
- **Element types**: Which tables can contribute members
- **Members**: Specific records (by RID) included in the dataset
- **Types**: Labels describing the dataset's purpose (Training, Testing, etc.)
- **Versioning**: Monotonically increasing version numbers for reproducibility
- **Nested datasets**: Parent-child relationships for train/test splits
- **Provenance**: Tracking of which executions created or used the dataset

### Dataset Types

Standard dataset types available by default:

| Type | Description |
|------|-------------|
| Training | Data for model training |
| Testing | Data for model evaluation |
| Validation | Data for hyperparameter tuning |
| Complete | Full dataset before splitting |
| Labeled | Data with feature annotations |
| Unlabeled | Data without feature annotations |

Custom types can be created:
```
create_dataset_type_term(type_name="Augmented", description="Dataset with augmented samples")
```

### Element Types

Before adding records from a table to a dataset, you must register that table as an element type. This is a **catalog-level** operation (not per-dataset) — once registered, records from that table can be added to any dataset.

### FK Traversal in Bag Exports

When a dataset is downloaded as a BDBag, the export follows foreign key relationships from registered element types:

- **Starting points are member records from registered element types.** Only tables registered via `add_dataset_element_type` that have members serve as traversal starting points.
- **Both FK directions are followed.** Outgoing FKs (this table references another) and incoming FKs (another table references this one) are both traversed.
- **Vocabulary tables are natural terminators.** Controlled vocabulary terms are collected and exported separately — they don't generate further FK traversal.
- **Feature tables are automatically included.** Feature annotation tables for reachable element types are added to the export.
- **Versions capture catalog snapshots.** A bag for a specific version reflects the exact catalog state when that version was created. Changes made after the version was created are not included.

#### When downloads are slow or timing out

Deep FK chains (e.g., Image -> Sample -> Subject -> Study -> Institution) can produce expensive server-side joins that timeout. Three solutions, in order of preference:

1. **Increase the download timeout** — The default read timeout is 610 seconds (~10 min). For large datasets, increase it:
   ```
   download_dataset(dataset_rid="2-XXXX", version="1.0.0", timeout=[10, 1800])
   ```
   This gives the server 30 minutes per query. The first value (connect timeout) rarely needs changing.

2. **Exclude tables from the FK graph** — If you don't need data from certain tables, prune them:
   ```
   download_dataset(dataset_rid="2-XXXX", version="1.0.0", exclude_tables=["Study", "Institution"])
   ```
   This prevents traversal into those tables entirely.

3. **Add intermediate records as direct members** — Register intermediate tables as element types and add their records as members. This replaces deep FK joins with simpler association lookups.

For Hydra-Zen configs, both `timeout` and `exclude_tables` are available on `DatasetSpecConfig`:
```python
DatasetSpecConfig(rid="28EA", version="0.4.0", timeout=[10, 1800])
DatasetSpecConfig(rid="28EA", version="0.4.0", exclude_tables=["Study", "Institution"])
```

## Step 1: Check Existing Resources

```
# List existing datasets
query_table(table="Dataset")

# List existing dataset types
query_table(table="Dataset_Type")

# Check what element types are registered
# (reads the deriva://catalog/dataset-element-types resource)
```

## Step 2: Create a Dataset via Execution

Datasets should be created within an execution for provenance.

### MCP Tools

```
# Step 1: Create execution
create_execution(
    workflow_rid="2-WKFL",
    description="Create training dataset"
)
start_execution(execution_rid="2-EXEC")

# Step 2: Create dataset within execution (no 'name' param — use description)
create_dataset(
    description="Curated set of labeled tumor histology images",
    dataset_types=["Training", "Labeled"]
)
# Returns dataset RID

# Step 3: Register element types (catalog-level, not per-dataset)
add_dataset_element_type(table_name="Image")
add_dataset_element_type(table_name="Subject")

# Step 4: Add members by RID list
add_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5"]
)

# Or add by table (faster for large datasets)
add_dataset_members(
    dataset_rid="2-DS01",
    members_by_table={"Image": ["2-IMG1", "2-IMG2"], "Subject": ["2-SUB1"]}
)

# Step 5: Finalize
stop_execution(execution_rid="2-EXEC")
upload_execution_outputs(execution_rid="2-EXEC")
```

### Python API

```python
from deriva_ml import DerivaML, ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)

# Get or create a workflow
workflow = ml.create_workflow(
    name="Dataset Curation",
    url="https://github.com/org/repo",
    workflow_type="Data Management",
    description="Curate and organize training datasets"
)

config = ExecutionConfiguration(workflow=workflow)

with ml.create_execution(config) as exe:
    # Create the dataset (no 'name' param — use description)
    dataset = exe.create_dataset(
        description="Curated set of labeled tumor histology images",
        dataset_types=["Training", "Labeled"]
    )

    dataset_rid = dataset.rid

    # Register element types (catalog-level operation on ml, not exe)
    ml.add_dataset_element_type("Image")
    ml.add_dataset_element_type("Subject")

    # Add members by RID list
    dataset.add_dataset_members(
        members=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5"]
    )

    # Or add members by table dict (faster)
    dataset.add_dataset_members(
        members={
            "Image": ["2-IMG1", "2-IMG2", "2-IMG3"],
            "Subject": ["2-SUB1", "2-SUB2"]
        }
    )
```

## Step 3: Manage Dataset Types

```
# Add a type to a dataset
add_dataset_type(dataset_rid="2-DS01", type_name="Training")

# Remove a type
remove_dataset_type(dataset_rid="2-DS01", type_name="Complete")

# Create a new custom type
create_dataset_type_term(type_name="Preprocessed", description="Data that has been preprocessed and normalized")

# Delete a custom type
delete_dataset_type_term(type_name="Preprocessed")
```

## Step 4: Manage Dataset Members

```
# List current members
list_dataset_members(dataset_rid="2-DS01")

# Add more members
add_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG6", "2-IMG7"]
)

# Remove members
delete_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG7"]
)

# Validate that RIDs exist
validate_rids(rids=["2-IMG1", "2-IMG2", "2-FAKE"])
# Returns which RIDs are valid and which are not
```

## Step 5: Split Datasets

The `split_dataset` tool creates nested child datasets from a parent dataset. It follows scikit-learn's `train_test_split` conventions.

### Two-way Random Split (default)

```
split_dataset(
    source_dataset_rid="2-DS01",
    test_size=0.2,
    seed=42
)
```

Creates two child datasets:
- "... - Training" (80% of members)
- "... - Testing" (20% of members)

### Three-way Split (train/val/test)

```
split_dataset(
    source_dataset_rid="2-DS01",
    test_size=0.2,
    val_size=0.1,
    seed=42
)
```

Creates three child datasets:
- "... - Training" (70% of members)
- "... - Validation" (10% of members)
- "... - Testing" (20% of members)

### Stratified Split

Maintains class distribution across splits. Requires `stratify_by_column` and `include_tables`:

```
split_dataset(
    source_dataset_rid="2-DS01",
    test_size=0.2,
    stratify_by_column="Image_Classification_Image_Class",
    include_tables=["Image", "Image_Classification"],
    seed=42
)
```

The column name uses the denormalized format `{TableName}_{ColumnName}`. Use `denormalize_dataset` first to discover available column names.

### Labeled Splits

Add type labels to splits so child datasets carry ground truth metadata:

```
split_dataset(
    source_dataset_rid="2-DS01",
    test_size=0.2,
    val_size=0.1,
    seed=42,
    training_types=["Labeled"],
    testing_types=["Labeled"],
    validation_types=["Labeled"]
)
```

### Dry Run Preview

Preview a split before executing it:

```
split_dataset(
    source_dataset_rid="2-DS01",
    test_size=0.2,
    seed=42,
    dry_run=true
)
```

Returns the planned split without creating any datasets. Use this to verify sizes before committing.

### Full Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_dataset_rid` | `str` | *(required)* | RID of the dataset to split |
| `test_size` | `float` | `0.2` | Fraction for testing (0-1) |
| `train_size` | `float \| None` | `None` | Fraction for training. Default: complement of test + val |
| `val_size` | `float \| None` | `None` | Fraction for validation. When set, creates 3-way split |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `shuffle` | `bool` | `True` | Shuffle before splitting |
| `stratify_by_column` | `str \| None` | `None` | Denormalized column name for stratified split |
| `element_table` | `str \| None` | `None` | Table to split. Auto-detected if not set |
| `include_tables` | `list[str] \| None` | `None` | Tables for denormalization. Required with `stratify_by_column` |
| `training_types` | `list[str] \| None` | `None` | Additional types for training set (e.g., `["Labeled"]`) |
| `testing_types` | `list[str] \| None` | `None` | Additional types for testing set |
| `validation_types` | `list[str] \| None` | `None` | Additional types for validation set |
| `split_description` | `str` | `""` | Description for the parent Split dataset |
| `dry_run` | `bool` | `False` | Preview without modifying catalog |

## Step 6: Version Management

After any modification (adding/removing members, changing element types), increment the dataset version:

```
increment_dataset_version(dataset_rid="2-DS01")
```

Note: `add_dataset_members` and `split_dataset` auto-increment the minor version, so manual incrementation is typically only needed for other changes.

See the `dataset-versioning` skill for full versioning rules, semantic versioning conventions, and the pre-experiment checklist.

## Nested Datasets

Nested datasets create parent-child relationships, commonly used for train/test splits.

### Create manually
```
# Create child dataset (within an execution)
create_dataset(
    description="Training portion of the main dataset"
)

# Link as child of parent
add_dataset_child(
    parent_rid="2-DS01",
    child_rid="2-DS02"
)
```

### Navigate the hierarchy
```
# List children of a dataset
list_dataset_children(dataset_rid="2-DS01")

# List parents of a dataset
list_dataset_parents(dataset_rid="2-DS02")
```

### Automatic nesting via split
When using `split_dataset`, child datasets are automatically nested under a parent "Split" dataset.

## Downloading and Using Datasets

Before downloading, use `estimate_bag_size` to preview what the bag will contain:
```
estimate_bag_size(dataset_rid="2-DS01", version="1.0.0")
```
Returns row counts and asset sizes per table, so you can decide whether to use `exclude_tables` or adjust `timeout` before committing to the full download.

For extracting, downloading, and preparing dataset data for ML training, see the `prepare-training-data` skill. For diagnosing missing data in bag exports, see the `debug-bag-contents` skill.

## Complete Example: End-to-End Dataset Workflow

```
# 1. Create workflow for dataset management
create_workflow(
    name="Image Dataset Curation",
    url="https://github.com/lab/protocols",
    workflow_type="Data Management",
    description="Curate and split image datasets for training"
)

# 2. Create execution
create_execution(
    workflow_rid="2-WKFL",
    description="Create and split tumor image dataset"
)
start_execution(execution_rid="2-EXEC")

# 3. Create the master dataset
create_dataset(
    description="All labeled tumor histology images as of 2025-06",
    dataset_types=["Complete", "Labeled"]
)

# 4. Register element types
add_dataset_element_type(table_name="Image")

# 5. Add all labeled images
add_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5",
                 "2-IMG6", "2-IMG7", "2-IMG8", "2-IMG9", "2-IMG10"]
)

# 6. Preview the split
split_dataset(
    source_dataset_rid="2-DS01",
    test_size=0.15,
    val_size=0.15,
    stratify_by_column="Image_Classification_Image_Class",
    include_tables=["Image", "Image_Classification"],
    seed=42,
    dry_run=true
)

# 7. Execute the split
split_dataset(
    source_dataset_rid="2-DS01",
    test_size=0.15,
    val_size=0.15,
    stratify_by_column="Image_Classification_Image_Class",
    include_tables=["Image", "Image_Classification"],
    training_types=["Labeled"],
    testing_types=["Labeled"],
    validation_types=["Labeled"],
    seed=42
)

# 8. Finalize
stop_execution(execution_rid="2-EXEC")
upload_execution_outputs(execution_rid="2-EXEC")
```

## Provenance Tracking

Track which executions created or used datasets:

```
# Which executions used this dataset as input?
list_dataset_executions(dataset_rid="2-DS01")

# Which executions used this asset?
list_asset_executions(asset_rid="2-IMG1")
```

This lets you trace the full lineage: which workflow created the dataset, what data it contains, which experiments used it, and what results were produced.

## Deleting Datasets

```
# Delete a dataset (removes the dataset record and member associations)
delete_dataset(dataset_rid="2-DS01")
```

**Warning:** Deleting a dataset does not delete the member records themselves (e.g., images, subjects). It only removes the dataset container and its member associations.

## Tips

- Always create datasets within an execution for provenance tracking.
- Register element types before adding members of that type.
- Use `validate_rids` before adding members to catch invalid RIDs early.
- Use `dry_run=true` on `split_dataset` to preview splits before committing.
- Set a `seed` on splits for reproducibility.
- `add_dataset_members` and `split_dataset` auto-increment the version.
- For large datasets with deep FK chains, add intermediate table records as direct members to avoid export timeouts.
- Use stratified splits when your labels are imbalanced to ensure each split has representative samples.
- Nested datasets maintain the parent's element types automatically.
- Use `list_dataset_children` and `list_dataset_parents` to navigate dataset hierarchies.
