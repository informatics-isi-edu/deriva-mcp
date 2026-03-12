# Script Pattern Templates for Catalog Operations

Reusable Python script templates for common DerivaML catalog operations. Each pattern follows the Develop, Test, Commit, Run workflow described in the parent skill.

## Table of Contents

- [Base Script Template](#base-script-template)
- [Dataset Creation](#dataset-creation)
- [Dataset Splitting](#dataset-splitting)
- [Feature Creation and Population](#feature-creation-and-population)
- [ETL / Data Loading](#etl--data-loading)

---

## Base Script Template

The foundation for all catalog operation scripts. Every script should follow this structure.

```python
#!/usr/bin/env python
"""<Description of what this script does>."""

import argparse
from deriva_ml import DerivaML, ExecutionConfiguration

def main():
    parser = argparse.ArgumentParser(description="<Script description>")
    parser.add_argument("--dry-run", action="store_true", help="Test without creating records")
    # Add script-specific arguments
    args = parser.parse_args()

    ml = DerivaML(hostname="...", catalog_id=...)

    workflow = ml.create_workflow(
        name="My Operation",
        url="https://github.com/org/repo",
        workflow_type="ETL",
        description="..."
    )

    config = ExecutionConfiguration(
        workflow=workflow,
        description="...",
    )

    with ml.create_execution(config, dry_run=args.dry_run) as execution:
        # Perform operations
        ...

    execution.upload_execution_outputs()

if __name__ == "__main__":
    main()
```

Key elements:
- `argparse` for CLI arguments
- `--dry-run` flag for testing without side effects
- `ExecutionConfiguration` with a `Workflow` object (not a string)
- `execution.upload_execution_outputs()` called after the with block to record results

---

## Dataset Creation

Create a new dataset and populate it with members.

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    dataset = execution.create_dataset(
        dataset_types=["Training"],
        description="Training dataset with 10,000 balanced images.",
    )
    dataset.add_dataset_members(member_rids)
    # Outputs auto-uploaded on context exit
```

---

## Dataset Splitting

Split an existing dataset into train/val/test partitions with optional stratification.

```python
from deriva_ml.dataset.split import split_dataset

# Derive the stratify column name from the schema:
#   Feature table: Execution_Image_Image_Classification
#   Column: Image_Class
#   Denormalized name: Execution_Image_Image_Classification_Image_Class

splits = split_dataset(
    ml,
    source_dataset_rid="1-ABC4",
    test_size=0.1,
    val_size=0.1,
    stratify_by_column="Execution_Image_Image_Classification_Image_Class",
    stratify_missing="drop",  # "error" (default), "drop", or "include"
    include_tables=["Image", "Execution_Image_Image_Classification"],
    seed=42,
)
```

---

## Feature Creation and Population

Create a new feature and populate it with values.

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    ml.create_feature(
        table_name="Image",
        feature_name="Severity",
        terms=["Severity_Grade"],
        comment="Severity grading for chest X-ray findings.",
    )

    feature = ml.lookup_feature("Image", "Severity")
    RecordClass = feature.feature_record_class()

    records = []
    for image_rid, severity in annotations.items():
        records.append(RecordClass(Image=image_rid, Severity_Grade=severity))
    execution.add_features(records)

execution.upload_execution_outputs()
```

---

## ETL / Data Loading

Load data from an external source, transform it, and insert into the catalog.

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    # Load data from external source
    data = load_external_data(args.source)

    # Transform and insert using pathBuilder
    pb = ml.pathBuilder()
    table = pb.schemas[ml.domain_schema].tables["TargetTable"]
    table.insert(transform(data))

execution.upload_execution_outputs()
```
