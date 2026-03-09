# Creating and Managing Datasets in DerivaML

Datasets are the primary mechanism for organizing data for ML workflows in DerivaML. They group records from multiple tables into a versioned, downloadable collection with full provenance tracking.

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
create_dataset_type_term(name="Augmented", description="Dataset with augmented samples")
```

### Element Types

Before adding records from a table to a dataset, you must register that table as an element type. This tells the dataset system which tables are allowed as sources.

### FK Traversal in Bag Exports

When a dataset is downloaded as a BDBag, the export follows foreign key relationships:

- **All FK-reachable tables are included**: If you add Image records and Image has a FK to Subject, Subject records are automatically exported.
- **Vocabulary tables are exported separately**: Controlled vocabulary terms are included in a separate section of the bag.
- **Deep join chains can timeout**: If your schema has many levels of FK relationships (e.g., Image -> Sample -> Subject -> Study -> Institution), the export query may timeout. Solution: add records from intermediate tables as direct members to avoid deep joins.

## Step 1: Check Existing Resources

```
# List existing datasets
query_table(table="Dataset")

# List existing dataset types
query_table(table="Dataset_Type")

# Check what element types are registered for a dataset
list_dataset_members(dataset_rid="2-XXXX")
```

## Step 2: Create a Dataset via Execution

Datasets should be created within an execution for provenance.

### Python API

```python
from deriva.ml import DerivaML, ExecutionConfiguration

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
    # Create the dataset
    dataset = exe.create_execution_dataset(
        name="Tumor Image Dataset v1",
        description="Curated set of labeled tumor histology images",
        dataset_types=["Training", "Labeled"]
    )

    dataset_rid = dataset["RID"]

    # Register element types
    exe.add_dataset_element_type(dataset_rid=dataset_rid, table="Image")
    exe.add_dataset_element_type(dataset_rid=dataset_rid, table="Subject")

    # Add members by RID
    exe.add_dataset_members(
        dataset_rid=dataset_rid,
        member_rids=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5"]
    )

    # Or add members by table query
    exe.add_dataset_members(
        dataset_rid=dataset_rid,
        members_by_table={
            "Image": ["2-IMG1", "2-IMG2", "2-IMG3"],
            "Subject": ["2-SUB1", "2-SUB2"]
        }
    )

exe.upload_execution_outputs()
```

### MCP Tools

```
# Step 1: Create execution
create_execution(
    workflow_rid="2-WKFL",
    description="Create training dataset"
)
start_execution(execution_rid="2-EXEC")

# Step 2: Create dataset within execution
create_execution_dataset(
    execution_rid="2-EXEC",
    name="Tumor Image Dataset v1",
    description="Curated set of labeled tumor histology images"
)
# Returns dataset RID

# Step 3: Add dataset types
add_dataset_type(dataset_rid="2-DS01", type_name="Training")
add_dataset_type(dataset_rid="2-DS01", type_name="Labeled")

# Step 4: Register element types
add_dataset_element_type(dataset_rid="2-DS01", table="Image")
add_dataset_element_type(dataset_rid="2-DS01", table="Subject")

# Step 5: Add members
add_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5"]
)

# Or add by table
add_dataset_members(
    dataset_rid="2-DS01",
    members_by_table={"Image": ["2-IMG1", "2-IMG2"], "Subject": ["2-SUB1"]}
)

# Step 6: Finalize
stop_execution(execution_rid="2-EXEC")
upload_execution_outputs(execution_rid="2-EXEC")
```

## Step 3: Manage Dataset Types

```
# Add a type to a dataset
add_dataset_type(dataset_rid="2-DS01", type_name="Training")

# Remove a type
remove_dataset_type(dataset_rid="2-DS01", type_name="Complete")

# Create a new custom type
create_dataset_type_term(name="Preprocessed", description="Data that has been preprocessed and normalized")

# Delete a custom type
delete_dataset_type_term(name="Preprocessed")
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

The `split_dataset` tool creates nested child datasets from a parent dataset, typically for train/test/validation splits.

### Random Split

```
split_dataset(
    dataset_rid="2-DS01",
    split_type="random",
    ratios={"Training": 0.7, "Testing": 0.2, "Validation": 0.1},
    seed=42
)
```

Creates three child datasets:
- "Tumor Image Dataset v1 - Training" (70% of members)
- "Tumor Image Dataset v1 - Testing" (20% of members)
- "Tumor Image Dataset v1 - Validation" (10% of members)

### Stratified Split

Split while maintaining the distribution of a label:

```
split_dataset(
    dataset_rid="2-DS01",
    split_type="stratified",
    ratios={"Training": 0.8, "Testing": 0.2},
    stratify_feature="Tumor_Classification",
    seed=42
)
```

Each child dataset will have approximately the same proportion of each Tumor_Classification grade as the parent.

### Labeled Split

Split into labeled and unlabeled subsets based on whether records have feature values:

```
split_dataset(
    dataset_rid="2-DS01",
    split_type="labeled",
    feature_name="Tumor_Classification"
)
```

Creates two child datasets:
- "Tumor Image Dataset v1 - Labeled" (records with Tumor_Classification values)
- "Tumor Image Dataset v1 - Unlabeled" (records without Tumor_Classification values)

### Dry Run Preview

Preview a split before executing it:

```
split_dataset(
    dataset_rid="2-DS01",
    split_type="random",
    ratios={"Training": 0.7, "Testing": 0.3},
    seed=42,
    dry_run=true
)
```

Returns the planned split without creating any datasets. Use this to verify sizes and distributions before committing.

## Step 6: Version Management

After any modification (adding/removing members, changing element types), increment the dataset version:

```
increment_dataset_version(dataset_rid="2-DS01")
```

See the `dataset-versioning` skill for full versioning rules, semantic versioning conventions, and the pre-experiment checklist.

## Nested Datasets

Nested datasets create parent-child relationships, commonly used for train/test splits.

### Create manually
```
# Create child dataset
create_dataset(
    name="Training Subset",
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
When using `split_dataset`, child datasets are automatically nested under the parent.

## Downloading and Using Datasets

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
create_execution_dataset(
    execution_rid="2-EXEC",
    name="Tumor Histology Complete",
    description="All labeled tumor histology images as of 2025-06"
)

# 4. Configure types and elements
add_dataset_type(dataset_rid="2-DS01", type_name="Complete")
add_dataset_type(dataset_rid="2-DS01", type_name="Labeled")
add_dataset_element_type(dataset_rid="2-DS01", table="Image")

# 5. Add all labeled images
add_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5",
                 "2-IMG6", "2-IMG7", "2-IMG8", "2-IMG9", "2-IMG10"]
)

# 6. Preview the split
split_dataset(
    dataset_rid="2-DS01",
    split_type="stratified",
    ratios={"Training": 0.7, "Testing": 0.15, "Validation": 0.15},
    stratify_feature="Tumor_Classification",
    seed=42,
    dry_run=true
)

# 7. Execute the split
split_dataset(
    dataset_rid="2-DS01",
    split_type="stratified",
    ratios={"Training": 0.7, "Testing": 0.15, "Validation": 0.15},
    stratify_feature="Tumor_Classification",
    seed=42
)

# 8. Finalize
increment_dataset_version(dataset_rid="2-DS01")
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
- Increment the dataset version after any modification.
- For large datasets with deep FK chains, add intermediate table records as direct members to avoid export timeouts.
- Use stratified splits when your labels are imbalanced to ensure each split has representative samples.
- Nested datasets maintain the parent's element types automatically.
- Use `list_dataset_children` and `list_dataset_parents` to navigate dataset hierarchies.
