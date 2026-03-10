---
name: route-data-management
description: "Use this skill for DerivaML dataset and data preparation tasks. Covers creating datasets, adding members, splitting train/test/validation, dataset versioning, creating features and labels, adding annotations, preparing data for ML training (denormalize, download BDBag, restructure assets), working with assets and provenance, troubleshooting bag exports, and writing Python scripts for batch data operations."
---

# Data Management — Datasets, Features, Assets, and Preparation

You are a router skill. Based on the user's request, load the appropriate specialized skill.

## Routing Rules

Analyze the user's intent and read the matching skill:

### Datasets
- **Creating datasets, adding members, registering element types, nesting datasets, dataset types, or splitting into train/test/validation** → Read and follow `../create-dataset/SKILL.md`
- **Dataset version management — incrementing versions, pinning versions, version in DatasetSpecConfig, reproducibility** → Read and follow `../dataset-versioning/SKILL.md`

### Features and labels
- **Creating features, adding labels or annotations, classification categories, ground truth, confidence scores, feature values** → Read and follow `../create-feature/SKILL.md`

### Preparing data for ML
- **Denormalizing datasets into DataFrames, downloading BDBags, building training features/labels, restructuring assets for PyTorch/ImageFolder** → Read and follow `../prepare-training-data/SKILL.md`

### Assets
- **Finding asset tables, downloading assets, checking asset provenance, tracing which executions created an asset** → Read and follow `../work-with-assets/SKILL.md`

### Troubleshooting data exports
- **Missing data in downloaded dataset bags, FK traversal issues, materialization problems, bag export timeouts, validate_dataset_bag** → Read and follow `../debug-bag-contents/SKILL.md`

### Scripted catalog operations
- **Writing Python scripts for batch data loading, ETL, dataset creation pipelines, or any catalog operation needing code provenance** → Read and follow `../catalog-operations-workflow/SKILL.md`

## Important

After identifying the correct skill, read its SKILL.md file completely and follow its instructions. Do not attempt to handle the request from this routing skill alone.
