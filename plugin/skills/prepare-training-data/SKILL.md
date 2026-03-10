---
name: prepare-training-data
description: "Prepare a DerivaML dataset for ML training — denormalize to DataFrame, download BDBag, build training features and labels, extract images, restructure assets. Use when getting data out of the catalog and into a format for model training or analysis."
disable-model-invocation: true
---

# Preparing Training Data from a DerivaML Dataset

This guide walks through the process of taking a DerivaML dataset and preparing it for use in ML training, evaluation, or analysis.

## Step 1: Explore the Dataset

Start by understanding what is in the dataset using catalog resources.

```
# List available datasets
query_table(table="Dataset")

# Get details about a specific dataset
get_record(table="Dataset", rid="2-XXXX")

# See what element types the dataset contains
list_dataset_members(dataset_rid="2-XXXX")

# View the dataset specification (element types, export configuration)
get_dataset_spec(dataset_rid="2-XXXX")
```

## Step 2: Understand the Table Structure

Before extracting data, understand the schema of the tables involved.

```
# Get table schema and columns via resources
Read resource: deriva://table/Image/schema
Read resource: deriva://table/Subject/schema

# View sample data
query_table(table="Image", limit=5)
query_table(table="Subject", limit=5)
```

Key things to look for:
- Which columns contain the features you need (e.g., image URLs, measurements, labels)
- Foreign key relationships between tables (e.g., Image -> Subject -> Diagnosis)
- Vocabulary columns that contain categorical labels

## Step 3: Choose Your Data Extraction Approach

### Option A: `denormalize_dataset` -- Best for Training

Joins all dataset tables into a single flat DataFrame, ideal for feeding into ML frameworks.

```
denormalize_dataset(
    dataset_rid="2-XXXX",
    include_tables=["Image", "Subject", "Diagnosis"]  # Optional: limit which tables to join
)
```

**What it does:**
- Follows foreign key relationships to join related tables
- Produces a single flat table with columns from all joined tables
- Column names are prefixed with the table name (e.g., `Image.URL`, `Subject.Age`, `Diagnosis.Label`)
- Handles many-to-one and one-to-one relationships automatically

**When to use:** Interactive exploration, quick prototyping, building training DataFrames.

**Parameters:**
- `dataset_rid` (required): The dataset to denormalize
- `include_tables` (optional): List of table names to include. If omitted, all tables in the dataset are joined.

### Option B: `query_table` -- Best for Specific Tables

Query individual tables when you need fine-grained control or only need data from one table.

```
# Get all images in a dataset
query_table(
    table="Image",
    filters=[{"column": "Dataset", "operator": "=", "value": "2-XXXX"}]
)

# Get specific columns
query_table(
    table="Subject",
    columns=["RID", "Age", "Sex", "Species"]
)
```

**When to use:** When you need data from a single table, need to apply filters, or need precise column selection.

### Option C: `download_dataset` -- Best for Production

Downloads the full dataset as a BDBag archive with all assets (files, images).

**Preview size before downloading:**
```
estimate_bag_size(dataset_rid="2-XXXX", version="1.0.0")
```
Returns row counts and asset file sizes per table so you know what to expect.

**Download:**
```
download_dataset(dataset_rid="2-XXXX", version="1.0.0")
```

**What it does:**
- Downloads all data tables as CSV files
- Downloads all referenced assets (images, files, etc.)
- Creates a reproducible BDBag with checksums
- Captures the exact catalog state at the version's snapshot time — the same version always returns the same data

**How bag contents are determined:**
- The export starts from each registered element type that has members in the dataset
- From those starting records, it follows all foreign key paths (both directions) to include related data
- Vocabulary tables are natural terminators — they're exported separately, not traversed further
- Feature tables for reachable element types are automatically included
- Nested child datasets are included recursively with all their members

**When downloads are slow or timing out:**
- Deep FK chains (e.g., Image → Sample → Subject → Study) can cause expensive joins
- **First**, increase the read timeout (default is 610s / ~10 min):
  ```
  download_dataset(dataset_rid="2-XXXX", version="1.0.0", timeout=[10, 1800])
  ```
- If the query is still too expensive, use `exclude_tables` to prune tables from the FK graph:
  ```
  download_dataset(dataset_rid="2-XXXX", version="1.0.0", exclude_tables=["Study", "Protocol"])
  ```
- Alternatively, add records from intermediate tables as direct dataset members to flatten the traversal
- Use `materialize=False` to skip downloading actual asset files (only metadata)

**When to use:** Production training pipelines, reproducible experiments, when you need actual files (not just URLs).

## Step 4: Use the Data

### Common Column Patterns

After denormalization, columns follow the pattern `TableName.ColumnName`:

| Pattern | Example | Description |
|---------|---------|-------------|
| `Image.URL` | `https://...` | Asset download URL |
| `Image.Filename` | `img_001.png` | Original filename |
| `Subject.Age` | `42` | Numeric feature |
| `Subject.Sex` | `Male` | Categorical feature from vocabulary |
| `Diagnosis.Label` | `Malignant` | Classification label from vocabulary |
| `Measurement.Value` | `3.14` | Numeric measurement |

### Building a Training DataFrame

```python
# After denormalize_dataset returns data:
import pandas as pd

# The denormalized result gives you a flat table
# Select features and labels
features = df[["Subject.Age", "Subject.Sex", "Measurement.Value"]]
labels = df["Diagnosis.Label"]

# Handle categorical variables
features_encoded = pd.get_dummies(features, columns=["Subject.Sex"])

# Split (or use pre-split nested datasets)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_encoded, labels, test_size=0.2)
```

### Working with Image Data

```python
# For image classification tasks
image_urls = df["Image.URL"]
labels = df["Diagnosis.Label"]

# Download images using the URLs, or use download_dataset for batch download
# If using download_dataset, images are already local
```

## Step 5: Version Pinning for Reproducibility

Datasets in DerivaML support versioning. Always pin to a specific version for reproducible experiments.

```
# Check current dataset version
get_record(table="Dataset", rid="2-XXXX")
# Look for the Version field

# Increment version after changes
increment_dataset_version(dataset_rid="2-XXXX")
```

**Best practices for reproducibility:**
- Record the dataset RID and version in your experiment configuration
- Use `create_execution()` to formally track which dataset version was used
- After finalizing a dataset, increment its version before using it in training
- Use nested datasets (train/test/validation splits) with `split_dataset` for consistent splits across experiments
- Download datasets within an execution context so the provenance is automatically recorded

## Tips

- Start with `denormalize_dataset` for quick exploration, then move to `download_dataset` for production.
- Use `include_tables` in denormalize to limit the join to only the tables you need -- this avoids unnecessary data and speeds up the operation.
- If denormalization produces unexpected results, check the foreign key paths between tables using `get_table()`.
- For large datasets, use `query_table` with filters to work with subsets before processing the full dataset.
- Always wrap data preparation in an execution for provenance tracking.
