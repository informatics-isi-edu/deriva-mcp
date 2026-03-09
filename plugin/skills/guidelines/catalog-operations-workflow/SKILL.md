---
name: catalog-operations-workflow
description: "ALWAYS use: When performing Deriva catalog operations (dataset creation, splitting, ETL, feature creation), generate a committed script for full provenance tracking"
user-invocable: false
---

# Script-Based Workflow for Catalog Operations

For catalog operations that need to be reproducible, auditable, or shareable — dataset creation, splitting, ETL, feature creation, and data loading — use a committed script rather than interactive MCP tools.

## When to Use Scripts vs Interactive MCP

| Situation | Approach |
|-----------|----------|
| One-off exploration, quick queries, checking state | Interactive MCP tools |
| Setting descriptions, display names, annotations | Interactive MCP tools |
| Operations you'll need to reproduce or share | Committed script |
| Dataset creation, splitting, ETL, data loading | Committed script |
| Operations others need to audit or re-run | Committed script |

The key distinction: DerivaML records the git commit hash with every execution. A committed script gives the execution record a code reference that anyone can trace back. Interactive MCP operations have no such reference.

## Workflow: Develop, Test, Commit, Run

### 1. Generate Python Script

Create a script in the `scripts/` directory using the DerivaML Python API:

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
- `execution.upload_execution_outputs()` to record results

### 2. Test with Dry Run

```bash
python scripts/my_operation.py --dry-run
```

Verify the script works correctly without creating any records in the catalog.

### 3. Commit Script

```bash
git add scripts/my_operation.py
git commit -m "Add script for <operation description>"
```

The script MUST be committed before running for real. This ensures the git commit hash in the execution record points to the actual code that was run.

### 4. Run for Real

```bash
python scripts/my_operation.py
```

The execution record will capture:
- Git commit hash
- Repository URL
- Input datasets and their versions
- Output assets and datasets
- Execution parameters

## Common Script Patterns

### Dataset Creation

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

### Dataset Splitting

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

### Feature Creation and Population

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

### ETL / Data Loading

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    # Load data from external source
    data = load_external_data(args.source)
    # Transform and insert
    for record in transform(data):
        execution.insert_record("TargetTable", record)
    execution.upload_execution_outputs()
```

## When MCP Tools Are Still Appropriate

Not everything needs a script. Use MCP tools directly for:

- **Exploratory work**: Browsing catalog structure, querying data, checking entity states
- **One-time admin tasks**: Setting descriptions, display names, annotations
- **Read-only operations**: Listing datasets, viewing features, checking versions
- **Quick debugging**: Inspecting specific records, checking execution status

## When to Suggest Scripts

When a user asks to perform a data-modifying operation interactively, suggest:

> "For full provenance tracking, I recommend creating a script that we can commit. This ensures the operation is reproducible and the execution record will reference the exact code. Shall I generate the script?"

Then follow the Develop, Test, Commit, Run workflow.
