---
name: create-feature
description: "ALWAYS use this skill when creating features, adding labels or annotations to records, setting up classification categories, or working with feature values in DerivaML. Triggers on: 'create feature', 'add labels', 'annotate images', 'classification', 'ground truth', 'confidence score', 'vocabulary terms for labeling'."
disable-model-invocation: true
---

# Creating and Populating Features in DerivaML

Features link domain objects (e.g., Image, Subject) to a set of values — controlled vocabulary terms, computed values, or assets. Every feature value is associated with an execution, so you can differentiate between multiple values by execution RID, workflow, description, or timestamp. Features are inherently multivalued, enabling inter-annotator agreement, model comparison, and audit trails.

Common uses include classification labels, transformed data, statistical aggregates, quality scores, segmentation masks, and any structured annotation that needs provenance.

For background on feature types, metadata columns, multivalued features, and feature selection, see `references/concepts.md`.

## Description Guidance

Every feature should have a description that explains what it measures or annotates, what values it takes, and its role in the ML workflow.

**Good feature descriptions:**
- "Diagnostic classification of chest X-ray images. Values from the Diagnosis vocabulary (normal, pneumonia, COVID-19). Primary ground truth label for training classification models"
- "Model-predicted class probabilities for each CIFAR-10 category. Float values 0-1 per class. Used for ROC analysis and model comparison across experiments"
- "Image quality score assigned during manual review. Values from Quality vocabulary (acceptable, borderline, rejected). Used to filter training data"

**Bad feature descriptions:**
- "Classification" or "Labels" or "Feature for Image"
- Leaving the description empty

Since features are multivalued (multiple executions can produce different values for the same record), note whether the feature is intended for ground truth annotation, model predictions, or computed metrics.

## Critical Rules

1. **Vocabulary must exist first** — create the vocabulary table and add terms before creating a term-based feature.
2. **Feature values require an active execution** — this is a hard requirement for provenance tracking.
3. **Use the right tool for the job**:
   - `add_feature_value` — simple features with a single term or asset column
   - `add_feature_value_record` — features with multiple columns (e.g., term + confidence score)
4. **Use feature selection for multivalued features** — when a record has multiple values, use `fetch_table_features` with one of these options to resolve to one value per record:
   - `selector="newest"` — picks the most recent value by creation time
   - `workflow` — filters by workflow RID or workflow type name, then picks newest
   - `execution` — filters by a specific execution RID, then picks newest

## Workflow Summary

1. `create_vocabulary` + `add_term` — define the label set (if needed; see `manage-vocabulary` skill)
2. `create_feature` — link a target table to vocabulary terms, assets, or both
3. `create_execution` + `start_execution` — start provenance tracking
4. `add_feature_value` / `add_feature_value_record` — assign values to records in batch
5. `stop_execution` — finalize (feature values are written directly to the catalog by `add_feature_value`; `upload_execution_outputs` is only needed if you also registered file assets)

For the full step-by-step guide with code examples (both MCP tools and Python API), see `references/workflow.md`.

## Reference Resources

- `references/concepts.md` — What features are, types, metadata, multivalued features, selection
- `references/workflow.md` — Step-by-step how-to with MCP and Python examples
- `deriva://docs/features` — Full user guide to features in DerivaML
- `deriva://catalog/features` — Browse existing features
- `deriva://feature/{table_name}/{feature_name}` — Feature details and column schema
- `deriva://feature/{table_name}/{feature_name}/values` — Feature values with provenance
- `deriva://table/{table_name}/feature-values/newest` — Deduplicated to newest per record

## Related Skills

- **`manage-vocabulary`** — Create and manage the controlled vocabularies that features reference.
- **`create-dataset`** — Features annotate records that belong to datasets. Feature values are included in dataset bag exports.
