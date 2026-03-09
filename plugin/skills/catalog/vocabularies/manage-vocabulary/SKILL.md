---
name: manage-vocabulary
description: "Create and manage controlled vocabularies in Deriva — create vocabulary tables, add terms with descriptions, add synonyms, and browse existing vocabularies. Use whenever working with categorical data, labels, or controlled term lists independent of features."
user-invocable: true
---

# Managing Controlled Vocabularies

Controlled vocabularies are the standard way to represent categorical data in Deriva. They provide consistent labeling, faceted search in Chaise, and synonym support for discoverability. Every vocabulary is a table with standard columns: Name, Description, Synonyms, ID, and URI.

Vocabularies are used by features (see `create-feature`), dataset types, workflow types, asset types, and any categorical column in your domain schema.

## Exploring Existing Vocabularies

Before creating a new vocabulary, check what already exists.

### List all vocabularies

```
# MCP resource — lists all vocabulary tables with term counts
Read resource: deriva-ml://catalog/vocabularies
```

### Browse terms in a vocabulary

```
# MCP resource — lists all terms with descriptions and synonyms
Read resource: deriva-ml://vocabulary/Species

# Or query directly
query_table(table="Species")
```

### Search for a term across vocabularies

```
# Find a term by name or synonym
lookup_term(term_name="Mouse")
```

`lookup_term` searches both term names and synonyms, so it catches alternate spellings like "Xray" matching "X-ray".

## Creating a Vocabulary

```
create_vocabulary(
    vocab_name="Tissue_Type",
    description="Classification of biological tissue types for histology analysis"
)
```

This creates a table named `Tissue_Type` in the domain schema with the standard vocabulary columns.

**Naming conventions:**
- Use `PascalCase` with underscores between words: `Tissue_Type`, `Image_Quality`, `Stain_Protocol`
- Name should be the singular form of what the terms represent
- Keep names concise but specific

## Adding Terms

```
add_term(
    table="Tissue_Type",
    name="Epithelial",
    description="Cells lining body surfaces, cavities, and glands"
)

add_term(
    table="Tissue_Type",
    name="Connective",
    description="Supportive tissue including bone, cartilage, and blood"
)

add_term(
    table="Tissue_Type",
    name="Muscle",
    description="Contractile tissue — skeletal, cardiac, or smooth"
)

add_term(
    table="Tissue_Type",
    name="Nervous",
    description="Neurons and supporting glial cells"
)
```

**Every term should have a description.** Descriptions appear as tooltips in the Chaise UI and help collaborators understand exactly what each term means. Avoid descriptions that just restate the name.

### Good vs Bad Descriptions

| Term | Bad | Good |
|---|---|---|
| Grade I | "Grade one" | "Well-differentiated, low mitotic rate, favorable prognosis" |
| Normal | "Normal tissue" | "No pathological findings, intact cellular architecture" |
| Artifact | "An artifact" | "Non-biological element (air bubble, fold, ink mark) in image" |

## Adding Synonyms

Synonyms make terms discoverable under alternative names, abbreviations, or common misspellings.

```
add_synonym(table="Tissue_Type", term_name="Connective", synonym="CT")
add_synonym(table="Tissue_Type", term_name="Connective", synonym="Connective Tissue")
add_synonym(table="Tissue_Type", term_name="Muscle", synonym="Muscular")
```

Synonyms are searched by `lookup_term`, so a search for "CT" will find "Connective".

### When to Use Synonyms vs New Terms

| Situation | Action |
|---|---|
| Same concept, different spelling ("X-ray" vs "Xray") | Add synonym |
| Same concept, different language ("Hund" for "Dog") | Add synonym |
| Common abbreviation ("CT" for "Connective Tissue") | Add synonym |
| Related but distinct concept ("Cartilage" vs "Connective") | Add new term |
| More specific version ("Hyaline Cartilage") | Add new term |

## Removing Terms and Synonyms

```
# Remove a synonym
remove_synonym(table="Tissue_Type", term_name="Connective", synonym="CT")

# Delete a term entirely (only if not referenced by any records)
delete_term(table="Tissue_Type", term_name="Artifact")
```

Deleting a term that is referenced by feature values or other records will fail with a foreign key constraint error. Remove the references first.

## Updating Term Descriptions

```
update_term_description(
    table="Tissue_Type",
    term_name="Epithelial",
    description="Cells forming continuous sheets that line body surfaces, cavities, and glands. Includes squamous, cuboidal, and columnar subtypes."
)
```

## Common Vocabulary Patterns

### Built-in Vocabularies

DerivaML catalogs come with several built-in vocabularies:

| Vocabulary | Purpose |
|---|---|
| `Dataset_Type` | Categorize datasets (Training, Testing, Validation, Labeled, etc.) |
| `Workflow_Type` | Categorize workflows (Training, Inference, Analysis, ETL, etc.) |
| `Execution_Status_Type` | Execution states (Running, Complete, Failed) |

Browse them with:
```
Read resource: deriva-ml://catalog/vocabularies
```

### Adding Types to Built-in Vocabularies

You can extend built-in vocabularies with domain-specific terms:

```
# Add a new dataset type
add_dataset_type(name="Augmented", description="Dataset containing augmented samples")

# Add a new workflow type
add_workflow_type(name="Quality Control", description="Automated QC pipeline for image screening")
```

### Vocabulary for Domain-Specific Categories

Common patterns for scientific data:

```
# Species vocabulary
create_vocabulary(vocab_name="Species", description="Biological species for experimental subjects")
add_term(table="Species", name="Homo sapiens", description="Human")
add_term(table="Species", name="Mus musculus", description="House mouse, common lab strain")
add_synonym(table="Species", term_name="Mus musculus", synonym="Mouse")

# Diagnosis vocabulary
create_vocabulary(vocab_name="Diagnosis", description="Clinical diagnostic categories")
add_term(table="Diagnosis", name="Normal", description="No pathological findings")
add_term(table="Diagnosis", name="Benign", description="Non-cancerous abnormality")
add_term(table="Diagnosis", name="Malignant", description="Cancerous, requires staging")

# Stain type vocabulary
create_vocabulary(vocab_name="Stain_Type", description="Histological staining protocols")
add_term(table="Stain_Type", name="H&E", description="Hematoxylin and eosin, standard morphology stain")
add_term(table="Stain_Type", name="IHC", description="Immunohistochemistry for protein detection")
add_synonym(table="Stain_Type", term_name="H&E", synonym="HE")
add_synonym(table="Stain_Type", term_name="H&E", synonym="Hematoxylin and Eosin")
```

### Using Vocabularies as Column Values

To use a vocabulary as a column type in a domain table, create a foreign key:

```
create_table(
    table_name="Subject",
    columns=[
        {"name": "Name", "type": "text", "nullok": false},
        {"name": "Species", "type": "text", "nullok": false},
    ],
    foreign_keys=[
        {"columns": ["Species"], "referenced_table": "Species"}
    ]
)
```

The FK to the vocabulary table enables dropdown selection in the Chaise entry form and faceted search in the compact view.

## Workflow: Adding Terms to an Existing Vocabulary

1. **Search first** — use `lookup_term` to check if the term or a synonym already exists
2. **Check semantic-awareness** — the `semantic-awareness` skill auto-triggers to prevent duplicates
3. **Add the term** with a meaningful description
4. **Add synonyms** for common alternate names
5. **Verify** — read `deriva-ml://vocabulary/{vocab_name}` to confirm

## Tips

- Vocabulary tables support faceted search in Chaise automatically — no extra configuration needed
- Terms are ordered alphabetically by name in the UI by default
- The `ID` and `URI` columns are auto-generated — you only need to provide Name and Description
- For large vocabularies (100+ terms), consider hierarchical naming (e.g., "Carcinoma:Ductal", "Carcinoma:Lobular") or multiple smaller vocabularies
- When a vocabulary is used by a feature, the feature creates an association table. See `create-feature` for details
