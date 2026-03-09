---
name: generate-descriptions
description: "ALWAYS use when creating any Deriva catalog entity (dataset, execution, feature, table, column, vocabulary, workflow) and the user hasn't provided a description. Auto-generate a meaningful description from context."
user-invocable: false
---

# Generate Descriptions for Catalog Entities

Every catalog entity that accepts a description MUST have one. If the user doesn't provide a description, generate a meaningful one based on context from the repository, conversation, and catalog state. Descriptions support GitHub-flavored Markdown which renders in the Chaise web UI.

## Entities Requiring Descriptions

- **Datasets**: `create_dataset` -- description parameter
- **Executions and Workflows**: `create_execution`, Workflow configuration -- description parameter
- **Features**: `create_feature` -- description parameter
- **Vocabulary Terms**: `add_term` -- description parameter
- **Tables and Columns**: `create_table`, `set_table_description`, `set_column_description`
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

```
<Purpose> of <source> with <count> <items>. <Key characteristics>. <Usage guidance>.
```

Example: "Training dataset of chest X-ray images with 12,450 DICOM files. Balanced across 3 diagnostic categories (normal, pneumonia, COVID-19). Use with v2.1.0+ feature annotations."

### Executions

```
<Action> <target> using <method>. <Key parameters>. <Expected outputs>.
```

Example: "Train ResNet-50 classifier on chest X-ray dataset 1-ABC4 v1.2.0. Learning rate 0.001, batch size 32, 100 epochs. Outputs: model weights, training metrics, confusion matrix."

Use markdown tables for complex workflows with multiple steps or parameters.

### Features

```
<What it labels> for <target table>. Values from <vocabulary>. <Usage context>.
```

Example: "Diagnostic classification label for Image table. Values from Diagnosis vocabulary (normal, pneumonia, COVID-19). Primary label for training classification models."

### Vocabulary Terms

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
