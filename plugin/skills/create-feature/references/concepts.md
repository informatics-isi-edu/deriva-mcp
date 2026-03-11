# Feature Concepts

Background on features in DerivaML. For the step-by-step guide, see `workflow.md`.

## Table of Contents

- [What is a Feature?](#what-is-a-feature)
- [Feature Types](#feature-types)
- [Metadata Columns](#metadata-columns)
- [Multivalued Features](#multivalued-features)
- [Feature Selection](#feature-selection)
- [Feature Value Table Naming](#feature-value-table-naming)
- [Operations Summary](#operations-summary)

---

## What is a Feature?

A feature links domain objects (e.g., Image, Subject) to vocabulary terms, assets, or computed values, creating a structured annotation system for ML with full provenance tracking.

Features are the primary way to attach meaning to records in DerivaML. Common uses include:
- **Classification labels** — human-assigned or model-predicted categories (e.g., tumor grade, cell type, diagnosis)
- **Transformed data** — processed versions of source records (e.g., normalized images, cropped regions, augmented samples)
- **Downsampled data** — reduced-resolution or summarized representations (e.g., thumbnails, compressed spectrograms)
- **Statistical values** — computed aggregates (e.g., max intensity, mean pixel value, standard deviation, cell count)
- **Quality scores** — numeric assessments (e.g., image quality, focus score, confidence)
- **Segmentation masks** — pixel-level or region annotations linked as assets
- **Review annotations** — status tracking with reviewer provenance

Each feature has:
- **A name** — identifies the annotation dimension (e.g., "Tumor_Classification", "Image_Quality")
- **A target table** — which domain table's records are being annotated (e.g., Image, Subject)
- **Value columns** — vocabulary terms, asset references, or both
- **Optional metadata columns** — additional structured data like confidence scores or reviewer references
- **Provenance** — every feature value is tied to the execution that created it

Features are inherently **multivalued**: a single record can have multiple values for the same feature (e.g., labels from different annotators or model runs), and the same term can be applied to many records. This is by design — it enables inter-annotator agreement analysis, model comparison, and audit trails. When you need a single value per record, use feature selection (see below).

## Feature Types

| Type | `create_feature` parameter | Use case |
|------|---------------------------|----------|
| Term-based | `terms=["Tumor_Grade"]` | Classification labels, categories |
| Asset-based | `assets=["Mask_Image"]` | Segmentation masks, annotation overlays |
| Mixed | `terms=[...], assets=[...]` | Labels with associated files |
| With metadata | `metadata=[...]` | Confidence scores, reviewer references, notes |

The `terms` and `assets` parameters take lists of vocabulary or asset table names. At least one of `terms` or `assets` is required.

## Metadata Columns

Features can include additional columns beyond the standard term/asset columns. The `metadata` parameter accepts a list where each item is either:

- **A string** — treated as a table name, adds a foreign key reference to that table (e.g., `"Reviewer"` adds an FK to the Reviewer table)
- **A dict** — column definition with `name` and `type` keys:
  - `type` must be `{"typename": "<type>"}` where type is one of: `text`, `int2`, `int4`, `int8`, `float4`, `float8`, `boolean`, `date`, `timestamp`, `timestamptz`, `json`, `jsonb`
  - Optional keys: `nullok` (bool), `default`, `comment`

Example: `metadata=[{"name": "confidence", "type": {"typename": "float4"}}, "Reviewer"]` adds both a float confidence column and an FK to the Reviewer table.

## Multivalued Features

Because features track provenance through executions, a single record can accumulate multiple values for the same feature over time:

- **Multiple annotators** — different pathologists label the same image in separate executions
- **Multiple model runs** — different model versions produce different predictions
- **Corrections** — a later execution overrides an earlier label

This is by design — it enables inter-annotator agreement analysis, model comparison, and audit trails. But when you need a single value per record (e.g., for training), you need feature selection.

## Feature Selection

Feature selection resolves multivalued features to one value per target record. DerivaML provides two mechanisms:

### MCP tool: `fetch_table_features`

Call `fetch_table_features` with `table_name` and optionally:
- `feature_name`: fetch only a specific feature (otherwise fetches all features on the table)
- `selector`: `"newest"` — picks the value with the most recent creation time (RCT) per record
- `workflow`: a Workflow RID or Workflow_Type name — filters to values from that workflow, then picks newest per record

`selector` and `workflow` are mutually exclusive.

### MCP resources

- `deriva://table/{table_name}/feature-values` — all feature values for a table, grouped by feature
- `deriva://table/{table_name}/feature-values/newest` — deduplicated to newest per target record

### Python API

The `fetch_table_features` method accepts a `selector` callable:

```python
from deriva_ml.feature import FeatureRecord

# Built-in: newest by creation time
features = ml.fetch_table_features("Image", selector=FeatureRecord.select_newest)

# Custom: pick the record with highest confidence
def select_best(records):
    return max(records, key=lambda r: getattr(r, "Confidence", 0))

features = ml.fetch_table_features("Image", selector=select_best)
```

The selector receives a list of FeatureRecord instances for the same target object and returns the one to keep.

## Feature Value Table Naming

When you create a feature, DerivaML creates an association table to store feature values. The table name follows the pattern `{FeatureName}_Feature_Value` — for example, creating a feature named `"Tumor_Classification"` on the `Image` table creates a `Tumor_Classification_Feature_Value` table.

This table contains columns for:
- The target record (FK to the target table, e.g., `Image`)
- Each vocabulary term column (FK to the vocabulary table, e.g., `Tumor_Grade`)
- Each asset column (FK to the asset table)
- Each metadata column
- `Execution` (FK to the Execution table — provenance)

## Operations Summary

| Operation | MCP Tool | What it does |
|-----------|----------|--------------|
| Create feature | `create_feature` | Define a new feature on a target table |
| Add values (simple) | `add_feature_value` | Assign term/asset values to records (batch) |
| Add values (multi-column) | `add_feature_value_record` | Assign values with metadata columns (batch) |
| Fetch values | `fetch_table_features` | Get feature values with optional deduplication |
| Delete feature | `delete_feature` | Remove feature and its value table |
| Browse features | `deriva://catalog/features` | List all features in the catalog |
| Feature details | `deriva://feature/{table}/{name}` | Column types and requirements |
| Feature values | `deriva://feature/{table}/{name}/values` | All values with provenance |
