---
name: create-table
description: "ALWAYS use this skill when creating tables, asset tables, or adding columns in a Deriva catalog. Triggers on: 'create table', 'add column', 'asset table', 'foreign key', 'define schema', 'new table for images/subjects/samples', 'column types'."
disable-model-invocation: true
---

# Creating Domain Tables in Deriva

Tables are the foundation of a Deriva catalog schema. Choose the right table type, follow naming conventions, and document everything.

## Table Types

| Type | Tool | When to Use |
|------|------|-------------|
| Standard table | `create_table` | Regular data with columns and foreign keys |
| Asset table | `create_asset_table` | Files with auto URL/Filename/Length/MD5 columns |
| Vocabulary table | `create_vocabulary` | Controlled term lists for categorical data |

## Key Decisions

### Naming Conventions
- **Tables**: Singular nouns with underscores (`Subject`, `Blood_Sample`)
- **Columns**: Descriptive with underscores (`Age_At_Enrollment`, `Cell_Count`)
- **FK columns**: Match the referenced table name (`Subject` column → `Subject` table)

### Column Type Selection
- Prefer `float8` over `float4` for scientific data (precision matters)
- Prefer `timestamptz` over `timestamp` (avoid timezone ambiguity)
- Prefer `jsonb` over `json` (better query performance)
- Use `markdown` only when you need rich text rendering in the UI

### Foreign Key on_delete
- `CASCADE` — Delete children when parent is deleted (strong ownership)
- `SET NULL` — Nullify FK when parent is deleted (optional relationship)
- `NO ACTION` (default) — Prevent parent deletion if children exist

### Documentation (Required)
- Always set `comment` on tables and columns — these are the primary way users understand catalog structure
- Use `set_row_name_pattern` so records are identifiable in the UI
- Use `set_table_display_name` / `set_column_display_name` for user-friendly names

**Table descriptions** should explain what records represent, key relationships, and primary use case:
- Good: "Individual chest X-ray images with associated metadata. Links to Subject (patient) and Study (imaging session). Primary asset table for classification experiments."
- Bad: "Images" or "Table for storing image data"

**Column descriptions** should explain what the value represents, units/format, and constraints:
- Good: "Patient age at time of imaging in years. Integer 0-120. Used for demographic stratification in training splits."
- Bad: "Age" or "The age column"

## Quick Reference

```
# Standard table with FK
create_table(table_name="Sample", columns=[...], foreign_keys=[...], comment="...")

# Asset table
create_asset_table(asset_name="Slide_Image", columns=[...], comment="...")

# Add column to existing table
add_column(table_name="Subject", column_name="Weight_kg", column_type="float8", comment="...")
```

## Reference Resources

- `deriva://catalog/schema` — Full catalog schema to check existing tables
- `deriva://table/{table_name}/schema` — Table details including columns and foreign keys
- `deriva://docs/ermrest/naming` — ERMrest naming conventions

For the full guide with column types table, FK specification, common patterns, and examples, read `references/workflow.md`.
