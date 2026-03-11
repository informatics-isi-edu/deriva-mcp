# Vocabulary Patterns and Reference

Reference material for vocabulary conventions, built-in vocabularies, and common usage patterns. For the step-by-step workflow, see the main `SKILL.md`.

## Table of Contents

- [Built-in Vocabularies](#built-in-vocabularies)
- [Extending Built-in Vocabularies](#extending-built-in-vocabularies)
- [Domain-Specific Vocabulary Examples](#domain-specific-vocabulary-examples)
- [Using Vocabularies as Column Values](#using-vocabularies-as-column-values)
- [Writing Good Descriptions](#writing-good-descriptions)
- [Synonyms vs New Terms](#synonyms-vs-new-terms)
- [Tips](#tips)

---

## Built-in Vocabularies

DerivaML catalogs come with several built-in vocabularies:

| Vocabulary | Purpose |
|---|---|
| `Dataset_Type` | Categorize datasets (Training, Testing, Validation, Labeled, etc.) |
| `Workflow_Type` | Categorize workflows (Training, Inference, Analysis, ETL, etc.) |
| `Execution_Status_Type` | Execution states (Running, Complete, Failed) |

Browse all vocabularies by reading the `deriva://catalog/vocabularies` resource.

## Extending Built-in Vocabularies

Built-in vocabularies can be extended with domain-specific terms using dedicated tools:

- To add a **dataset type**, call `create_dataset_type_term` with `type_name` and `description`.
- To add a **workflow type**, call `add_workflow_type` with `type_name` and `description`.

For general vocabularies, use `add_term` instead (see the main skill).

## Domain-Specific Vocabulary Examples

Common patterns for scientific data:

**Species vocabulary:**
1. Call `create_vocabulary` with `vocabulary_name`: `"Species"`, `comment`: `"Biological species for experimental subjects"`
2. Call `add_term` with `vocabulary_name`: `"Species"`, `term_name`: `"Homo sapiens"`, `description`: `"Human"`
3. Call `add_term` with `vocabulary_name`: `"Species"`, `term_name`: `"Mus musculus"`, `description`: `"House mouse, common lab strain"`, `synonyms`: `["Mouse"]`

**Diagnosis vocabulary:**
1. Call `create_vocabulary` with `vocabulary_name`: `"Diagnosis"`, `comment`: `"Clinical diagnostic categories"`
2. Call `add_term` with terms like `"Normal"` (`"No pathological findings"`), `"Benign"` (`"Non-cancerous abnormality"`), `"Malignant"` (`"Cancerous, requires staging"`)

**Stain type vocabulary:**
1. Call `create_vocabulary` with `vocabulary_name`: `"Stain_Type"`, `comment`: `"Histological staining protocols"`
2. Call `add_term` with `term_name`: `"H&E"`, `description`: `"Hematoxylin and eosin, standard morphology stain"`, `synonyms`: `["HE", "Hematoxylin and Eosin"]`
3. Call `add_term` with `term_name`: `"IHC"`, `description`: `"Immunohistochemistry for protein detection"`

## Using Vocabularies as Column Values

To use a vocabulary as a column type in a domain table, create a foreign key from the column to the vocabulary table. For example, a `Subject` table with a `Species` column would have an FK to the `Species` vocabulary table.

Call `create_table` with a `foreign_keys` entry linking the column to the vocabulary:
- `column`: the column name (e.g., `"Species"`)
- `referenced_table`: the vocabulary table name (e.g., `"Species"`)

The FK to the vocabulary table enables dropdown selection in the Chaise entry form and faceted search in the compact view.

## Writing Good Descriptions

Every term should have a description. Descriptions appear as tooltips in the Chaise UI and help collaborators understand exactly what each term means. Avoid descriptions that just restate the name.

| Term | Bad | Good |
|---|---|---|
| Grade I | "Grade one" | "Well-differentiated, low mitotic rate, favorable prognosis" |
| Normal | "Normal tissue" | "No pathological findings, intact cellular architecture" |
| Artifact | "An artifact" | "Non-biological element (air bubble, fold, ink mark) in image" |

## Synonyms vs New Terms

| Situation | Action |
|---|---|
| Same concept, different spelling ("X-ray" vs "Xray") | Add synonym |
| Same concept, different language ("Hund" for "Dog") | Add synonym |
| Common abbreviation ("CT" for "Connective Tissue") | Add synonym |
| Related but distinct concept ("Cartilage" vs "Connective") | Add new term |
| More specific version ("Hyaline Cartilage") | Add new term |

## Tips

- Vocabulary tables support faceted search in Chaise automatically — no extra configuration needed
- Terms are ordered alphabetically by name in the UI by default
- The `ID` and `URI` columns are auto-generated — you only need to provide Name and Description
- For large vocabularies (100+ terms), consider hierarchical naming (e.g., "Carcinoma:Ductal", "Carcinoma:Lobular") or multiple smaller vocabularies
- When a vocabulary is used by a feature, the feature creates an association table. See the `create-feature` skill for details
