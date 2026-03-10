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

    config = ExecutionConfiguration(
        workflow_rid="...",
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
- `ExecutionConfiguration` context manager for provenance tracking
- `execution.upload_execution_outputs()` called after the with block to record results

---

## Dataset Creation

Create a new dataset and populate it with members.

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    dataset = execution.create_dataset(
        name="training-v1",
        dataset_type="Training",
        description="Training dataset with 10,000 balanced images.",
    )
    execution.add_dataset_members(dataset.rid, member_rids)

execution.upload_execution_outputs()
```

---

## Dataset Splitting

Split an existing dataset into train/val/test partitions with optional stratification and grouping.

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    splits = execution.split_dataset(
        source_rid="1-ABC4",
        splits={"train": 0.8, "val": 0.1, "test": 0.1},
        stratify_by="Diagnosis",
        group_by="Subject",
    )

execution.upload_execution_outputs()
```

---

## Feature Creation and Population

Create a new feature column on a table and populate it with values.

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    feature = execution.create_feature(
        name="Severity",
        target_table="Image",
        vocabulary="Severity_Grade",
        description="Severity grading for chest X-ray findings.",
    )
    for image_rid, severity in annotations.items():
        execution.add_feature_value(feature.name, image_rid, severity)

execution.upload_execution_outputs()
```

---

## ETL / Data Loading

Load data from an external source, transform it, and insert into the catalog.

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    # Load data from external source
    data = load_external_data(args.source)
    # Transform and insert
    for record in transform(data):
        execution.insert_record("TargetTable", record)

execution.upload_execution_outputs()
```
