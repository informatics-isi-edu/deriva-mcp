---
name: generate-descriptions
description: "ALWAYS use when creating any Deriva catalog entity (dataset, execution, feature, table, column, vocabulary, workflow) and the user hasn't provided a description. Auto-generate a meaningful description from context."
user-invocable: false
---

# Generate Descriptions for Catalog Entities

Every catalog entity that accepts a description MUST have one. If the user doesn't provide a description, generate a meaningful one based on context from the repository, conversation, and catalog state. Descriptions support GitHub-flavored Markdown which renders in the Chaise web UI.

## Entities Requiring Descriptions

- **Datasets**: `create_dataset` -- description parameter
- **Experiments**: `experiment_store()` in `configs/experiments.py` -- description parameter on `make_config()`
- **Multiruns**: `multirun_config()` in `configs/multiruns.py` -- description parameter
- **Executions and Workflows**: `create_execution`, Workflow configuration -- description parameter
- **Features**: `create_feature` -- description parameter
- **Vocabularies**: `create_vocabulary` -- comment parameter
- **Vocabulary Terms**: `add_term` -- description parameter
- **Tables and Columns**: `create_table` (uses `comment` parameter), `set_table_description`, `set_column_description`
- **Assets**: asset metadata descriptions

For hydra-zen configuration descriptions (`with_description()` and `zen_meta`), see the `write-hydra-config` skill.

## How to Generate Descriptions

Gather context from:

1. The user's request and stated intent
2. Repository structure (README, config files, existing code)
3. Existing catalog entities and their descriptions (for consistency)
4. Configuration files (hydra-zen configs, dataset specs)
5. Conversation history and decisions made

Create a description that answers:

- **What** is this entity?
- **Why** does it exist?
- **How** is it used or created?
- **What does it contain** (for datasets, tables)?

Always confirm the generated description with the user before creating the entity.

## Templates by Entity Type

### Datasets

Dataset descriptions should cover composition (what's in it), purpose (what it's for), and any important characteristics (balance, splits, provenance). For split datasets, note the split strategy and rationale.

```
<Purpose> of <source> with <count> <items>. <Key characteristics>. <Usage guidance>.
```

Example: "Training dataset of chest X-ray images with 12,450 DICOM files. Balanced across 3 diagnostic categories (normal, pneumonia, COVID-19). Use with v2.1.0+ feature annotations."

**For split datasets**, include the split rationale:

Example: "80/20 patient-level stratified split of dataset `2-B4C8`. Split at patient level to prevent data leakage from multiple images per subject. Stratified by diagnosis to maintain class balance across partitions."

### Experiments

Experiment descriptions answer *why this experiment exists* — the goal, hypothesis, or question being tested. Technical parameters are already captured in the config; the description provides the scientific or engineering motivation.

```
<Goal/hypothesis>. <What is being compared or evaluated>. <Expected outcome or success criteria>.
```

Example: "Test whether dropout 0.25 reduces overfitting on the small labeled split compared to the unregularized baseline. Expect improved validation accuracy at the cost of slower convergence. Success: val accuracy within 2% of train accuracy by epoch 50."

**For multiruns/sweeps**, describe what question the sweep answers:

Example: "Sweep learning rates [1e-4 to 1e-1] to find the optimal convergence/stability tradeoff for the 2-layer CNN. Lower rates may underfit within the epoch budget; higher rates risk training instability."

### Executions

Execution descriptions capture what was done and the key parameters. For experiments, prefer the experiment description template above; execution descriptions are for ad-hoc or one-off runs.

```
<Action> <target> using <method>. <Key parameters>. <Expected outputs>.
```

Example: "Train ResNet-50 classifier on chest X-ray dataset 1-ABC4 v1.2.0. Learning rate 0.001, batch size 32, 100 epochs. Outputs: model weights, training metrics, confusion matrix."

Use markdown tables for complex workflows with multiple steps or parameters.

### Features

Feature descriptions should explain what the feature measures or annotates, what values it takes, and how it's used in the ML workflow. Since features are multivalued (multiple executions can produce different values), note whether it's intended for ground truth, model predictions, or computed metrics.

```
<What it labels/measures> for <target table>. Values from <vocabulary or type>. <Usage context — ground truth, prediction, metric>.
```

Example: "Diagnostic classification label for Image table. Values from Diagnosis vocabulary (normal, pneumonia, COVID-19). Primary ground truth label for training classification models. Multiple annotators may label the same image; use `selector='newest'` or filter by execution for a single value per record."

### Vocabularies

Vocabulary table descriptions explain the classification scheme and its scope:

```
<What this vocabulary classifies>. <Domain context>. <How terms relate to each other>.
```

Example: "Classification of chest X-ray diagnostic findings. Terms are mutually exclusive primary diagnoses. Used by the Image_Diagnosis feature for ground truth labeling."

### Vocabulary Terms

Term descriptions define the term's meaning in context — not just restating the name. Include when to use (and when not to), and how the term relates to other terms in the vocabulary.

```
<Definition>. <When to use>. <Relationship to other terms>.
```

Example: "Pneumonia detected in chest X-ray. Use when radiological signs of pneumonia are present regardless of etiology. Mutually exclusive with 'normal'; may co-occur with 'pleural effusion'."

### Tables

```
<What records represent>. <Key relationships>. <Primary use case>.
```

Example: "Individual chest X-ray images with associated metadata. Links to Subject (patient) and Study (imaging session) tables. Primary asset table for image classification experiments."

### Columns

```
<What value represents>. <Format/units>. <Constraints or valid values>.
```

Example: "Patient age at time of imaging in years. Integer value, range 0-120. Required for demographic stratification in training splits."

## Formatting with Markdown

Descriptions support **GitHub-flavored Markdown** which renders in the Chaise web UI. Use markdown to make descriptions more readable, especially for longer or structured content:

- **Bold** and *italic* for emphasis
- Bulleted or numbered lists for multi-part descriptions
- `code` formatting for RIDs, column names, or config values
- Markdown tables for parameter summaries or comparisons
- Headers for long execution descriptions that cover multiple phases

For example, an execution description might use:

```markdown
Train ResNet-50 on chest X-ray dataset `1-ABC4` v1.2.0.

**Parameters:**
- Learning rate: 0.001
- Batch size: 32
- Epochs: 100

**Expected outputs:** model weights, training metrics, confusion matrix.
```

Keep simple descriptions as plain text — markdown is most useful for executions, datasets, and multirun descriptions where structured detail helps.

## Quality Checklist

Before finalizing any description, verify it is:

- **Specific**: Avoids generic language like "a dataset" or "some data"
- **Informative**: Provides enough context for someone unfamiliar with the project
- **Accurate**: Correctly reflects the entity's actual contents and purpose
- **Concise**: No unnecessary words, but complete enough to be useful
- **Consistent**: Matches the tone and style of existing descriptions in the catalog
- **Actionable**: Helps users understand how to use the entity

## Workflow

1. Check if the user provided a description
2. If not, gather context from all available sources
3. Draft a description using the appropriate template
4. Present the draft to the user for confirmation
5. Create the entity with the approved description
