---
name: create-feature
description: "ALWAYS use this skill when creating features, adding labels or annotations to records, setting up classification categories, attaching metadata like confidence scores to domain objects, or working with feature values. Covers vocabulary creation, feature definition, and populating feature values with provenance."
---

# Creating and Populating Features in DerivaML

Features link domain objects (e.g., Image, Subject) to vocabulary terms, assets, or computed values, creating a structured labeling system for ML with full provenance tracking.

## Key Concepts

- **Feature** — A named labeling dimension (e.g., "Tumor_Classification", "Image_Quality"). Created with `create_feature`.
- **Vocabulary** — A controlled set of terms used as labels. Must exist before creating a term-based feature.
- **Feature value** — One record labeled with one term/asset, within one execution. Created with `add_feature_value` or `add_feature_value_record`.
- **Metadata columns** — Additional data on feature values (e.g., confidence scores, reviewer references).

## Critical Rules

1. **Vocabulary must exist first** — Create the vocabulary table and add terms before creating a feature that references it.
2. **Feature values require an active execution** — This is a hard requirement for provenance. Use `create_execution` first.
3. **Use the right tool for the job**:
   - `add_feature_value` — Simple features with a single term or asset column
   - `add_feature_value_record` — Features with multiple columns (e.g., term + confidence score)
4. **Always provide term descriptions** — They appear in the UI and help annotators understand labels.
5. **Multiple values per record are allowed** — An image can be labeled by multiple annotators, each in a separate execution.

## Workflow Summary

1. `create_vocabulary` + `add_term` — Define the label set (if needed)
2. `create_feature` — Link a target table to vocabulary terms/assets
3. `create_execution` + `start_execution` — Start provenance tracking
4. `add_feature_value` / `add_feature_value_record` — Assign labels to records
5. `stop_execution` + `upload_execution_outputs` — Finalize

## Feature Types

| Type | Parameter | Example |
|------|-----------|---------|
| Term-based | `terms=["Tumor_Grade"]` | Classification labels |
| Asset-based | `assets=["Mask_Image"]` | Segmentation masks |
| Mixed | `terms=[...], assets=[...]` | Labels with overlay images |
| With metadata | `metadata=[{"name": "confidence", "type": {"typename": "float4"}}]` | Scores, reviewer refs |

For the full step-by-step guide with code examples (both Python API and MCP tools), read `references/workflow.md`.
