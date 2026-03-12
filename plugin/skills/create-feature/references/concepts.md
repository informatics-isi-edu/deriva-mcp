# Feature Concepts

Background on features in DerivaML. For the step-by-step guide, see `workflow.md`.

## Table of Contents

- [What is a Feature?](#what-is-a-feature)
- [Feature Types](#feature-types)
- [Metadata Columns](#metadata-columns)
- [Multivalued Features](#multivalued-features)
- [Feature Selection](#feature-selection)
- [Feature Value Table Naming](#feature-value-table-naming)
- [Feature Records (Python API)](#feature-records-python-api)
- [Operations Summary](#operations-summary)

---

## What is a Feature?

A feature links domain objects (e.g., Image, Subject) to a set of values — which could be controlled vocabulary terms, computed values, or assets. It is the primary way to attach structured meaning to records in DerivaML.

Common uses include:
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
- **Value columns** — controlled vocabulary terms, asset references, or both
- **Optional metadata columns** — additional structured data like confidence scores or reviewer references

Every feature value is associated with an **execution**, which provides full provenance. This means you can differentiate between multiple values for the same record by execution RID, workflow, execution description, timestamp, or any other execution attribute. For example, you can distinguish labels from "Pathologist A's review" vs "Model v2 predictions" vs "QC pipeline run #47".

Features are inherently **multivalued**: a single record can accumulate multiple values for the same feature over time (e.g., labels from different annotators or model runs), and the same term can be applied to many records. This is by design — it enables inter-annotator agreement analysis, model comparison, and audit trails. When you need a single value per record, use feature selection (see below).

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

When you need to *use* feature values — for training a model, computing metrics, or building a DataFrame — you typically need exactly one value per record. **Feature selection** is the process of choosing which value to keep when multiple exist for the same record and feature.

The choice depends on your use case:
- **Newest** — use the most recent value regardless of source (good for "latest state")
- **By workflow** — use values from a specific workflow type, e.g., only expert annotations or only model predictions
- **By execution** — use values from a specific execution run, e.g., comparing Run A vs Run B

DerivaML provides three ways to access feature values, from simplest to most flexible:

### MCP resources (browsing and exploration)

Resources provide quick, no-parameter access for exploring what feature data exists. They return all features on a table in a single response.

| Resource | What it returns |
|----------|----------------|
| `deriva://table/{table}/feature-values` | All feature values for a table, grouped by feature name. Includes duplicates from multiple executions. |
| `deriva://table/{table}/feature-values/newest` | Same, but deduplicated to one value per target record per feature — picks the most recently created value (by RCT). |
| `deriva://feature/{table}/{feature}/values` | All values for a specific feature, with full provenance (Execution RID, RCT). |

Use resources when you want a quick look at what annotations exist or need the newest values without any filtering.

### MCP tool: `fetch_table_features` (parameterized queries)

The tool provides filtering options that resources can't express. It returns a JSON dict mapping feature names to lists of feature value records.

Call `fetch_table_features` with:
- `table_name` (required): the target table (e.g., `"Image"`)
- `feature_name` (optional): fetch only a specific feature; if omitted, fetches all features on the table
- One of the following selection options (mutually exclusive):
  - `selector="newest"` — picks the value with the most recent creation time (RCT) per record
  - `workflow` — a Workflow RID (e.g., `"2-ABC1"`) or Workflow_Type name (e.g., `"Training"`). Filters to values from executions of that workflow, then picks newest per record
  - `execution` — an Execution RID. Filters to values from that specific execution run, then picks newest per record. Use this when multiple executions of the same workflow exist and you want one specific run's results

If none of `selector`, `workflow`, or `execution` is specified, all values are returned (including duplicates).

### Python API

The `ml.fetch_table_features()` method accepts a `selector` callable with signature `(list[FeatureRecord]) -> FeatureRecord`. The selector is called once per group of records sharing the same target RID, and returns the single record to keep.

**Built-in selectors:**

```python
from deriva_ml.feature import FeatureRecord

# Newest by creation time (equivalent to selector="newest" in MCP)
features = ml.fetch_table_features("Image", selector=FeatureRecord.select_newest)

# Filter by execution RID, then pick newest
# select_by_execution is a closure factory — call it to get the selector function
features = ml.fetch_table_features(
    "Image",
    selector=FeatureRecord.select_by_execution("3WY2"),
)
```

**Workflow-based selection** uses `ml.select_by_workflow()`, a method on the DerivaML instance (it needs catalog access to resolve workflow type names to execution RIDs). Unlike the other selectors, it operates on a pre-grouped list of records rather than being passed as a `selector=` argument:

```python
from collections import defaultdict

features = ml.fetch_table_features("Image", feature_name="Classification")
records = features.get("Classification", [])

# Group by target record, then select by workflow
grouped = defaultdict(list)
for r in records:
    grouped[r.Image].append(r)

selected = [ml.select_by_workflow(group, "Training") for group in grouped.values()]
```

The MCP tool's `workflow` parameter handles this grouping automatically — it's the recommended way for interactive use.

**Custom selectors** can implement any logic:

```python
# Pick the record with highest confidence score
def select_best(records):
    return max(records, key=lambda r: getattr(r, "Confidence", 0))

features = ml.fetch_table_features("Image", selector=select_best)
```

## Feature Value Table Naming

When you create a feature, DerivaML creates an association table to store feature values. The table name follows the pattern `{FeatureName}_Feature_Value` — for example, creating a feature named `"Tumor_Classification"` on the `Image` table creates a `Tumor_Classification_Feature_Value` table.

This table contains columns for:
- The target record (FK to the target table, e.g., `Image`)
- Each vocabulary term column (FK to the vocabulary table, e.g., `Tumor_Grade`)
- Each asset column (FK to the asset table)
- Each metadata column
- `Execution` (FK to the Execution table — provenance)

## Feature Records (Python API)

In the Python API, feature values are represented as **FeatureRecord** objects. Each feature has a dynamically generated record class whose fields match the feature's columns (target table, vocabulary terms, assets, metadata).

To get the record class for a feature:

```python
feature = exe.catalog.lookup_feature("Image", "Tumor_Classification")
RecordClass = feature.feature_record_class()
```

The returned `RecordClass` is a Pydantic model. Construct instances by passing column values as keyword arguments, where each key is a column name from the feature value table:

```python
record = RecordClass(Image="2-IMG1", Tumor_Grade="Grade II")
```

- The target table column (e.g., `Image`) takes the record's RID
- Vocabulary term columns (e.g., `Tumor_Grade`) take the term name
- Asset columns take the asset RID
- Metadata columns take the appropriate typed value (e.g., `confidence=0.95`)
- The `Execution` column is set automatically when you call `exe.add_features()`

Add records in batch with `exe.add_features(records)`. The execution RID is populated automatically from the active execution context.

## Operations Summary

| Operation | MCP Tool / Resource | What it does |
|-----------|---------------------|--------------|
| Create feature | `create_feature` | Define a new feature on a target table |
| Add values (simple) | `add_feature_value` | Assign term/asset values to records (batch) |
| Add values (multi-column) | `add_feature_value_record` | Assign values with metadata columns (batch) |
| Fetch with selection | `fetch_table_features` | Get feature values with filtering by selector, workflow, or execution |
| Delete feature | `delete_feature` | Remove feature and its value table |
| Browse features | `deriva://catalog/features` | List all features in the catalog |
| Feature details | `deriva://feature/{table}/{name}` | Column types and requirements |
| Feature values (all) | `deriva://feature/{table}/{name}/values` | All values for one feature, with provenance |
| Table values (all) | `deriva://table/{table}/feature-values` | All feature values for a table, grouped by feature |
| Table values (newest) | `deriva://table/{table}/feature-values/newest` | Deduplicated to one value per record per feature |
