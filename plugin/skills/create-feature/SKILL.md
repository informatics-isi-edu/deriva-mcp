---
name: create-feature
description: "ALWAYS use this skill when creating features, adding labels or annotations to records, setting up classification categories, or working with feature values in DerivaML. Triggers on: 'create feature', 'add labels', 'annotate images', 'classification', 'ground truth', 'confidence score', 'vocabulary terms for labeling'."
disable-model-invocation: true
---

# Creating and Populating Features in DerivaML

Features link domain objects (e.g., Image, Subject) to vocabulary terms, assets, or computed values, creating a structured annotation system for ML with full provenance tracking. Features are inherently multivalued — the same record can be labeled multiple times by different annotators, model runs, or workflows, enabling inter-annotator agreement, model comparison, and audit trails.

Common uses include classification labels, quality scores, segmentation masks, derived data representations, and any structured annotation that needs provenance.

For background on feature types, metadata columns, multivalued features, and feature selection, see `references/concepts.md`.

## Critical Rules

1. **Vocabulary must exist first** — create the vocabulary table and add terms before creating a term-based feature.
2. **Feature values require an active execution** — this is a hard requirement for provenance tracking.
3. **Use the right tool for the job**:
   - `add_feature_value` — simple features with a single term or asset column
   - `add_feature_value_record` — features with multiple columns (e.g., term + confidence score)
4. **Use feature selection for multivalued features** — when a record has multiple values, use `fetch_table_features` with `selector="newest"` or `workflow` to resolve to one value per record.

## Workflow Summary

1. `create_vocabulary` + `add_term` — define the label set (if needed; see `manage-vocabulary` skill)
2. `create_feature` — link a target table to vocabulary terms, assets, or both
3. `create_execution` + `start_execution` — start provenance tracking
4. `add_feature_value` / `add_feature_value_record` — assign values to records in batch
5. `stop_execution` — finalize (no `upload_execution_outputs` needed — feature operations don't produce output files)

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
