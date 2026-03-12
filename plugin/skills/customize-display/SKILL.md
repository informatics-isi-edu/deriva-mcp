---
name: customize-display
description: "Customize the Chaise web UI display for Deriva catalog tables using MCP annotation tools. Use when setting visible columns, reordering columns, changing display names, configuring row name patterns, or adjusting how tables and records appear in the browser UI."
disable-model-invocation: true
---

# Customizing Chaise Web UI Display

Deriva catalogs are browsed through the Chaise web application. The display is controlled by annotations -- JSON metadata attached to schemas, tables, and columns. MCP tools provide a high-level interface for setting these annotations without writing raw JSON.

**This skill covers the interactive MCP tool approach.** For production Python code using type-safe builder classes (better for scripts, notebooks, and version-controlled configurations), see the `use-annotation-builders` skill instead.

## Quick Start

Apply sensible default annotations to the entire catalog:

```
apply_catalog_annotations()
```

This sets up reasonable defaults for display names, visible columns, row name patterns, and foreign key display across all tables. It is safe to run multiple times -- it will update existing annotations.

**Important:** After making any annotation changes, you must call `apply_annotations()` to persist them to the catalog.

## Step 1: Check Current Annotations

Use catalog resources to see the current state:

```
# View table annotations and column details
# Read the deriva://table/Image/annotations resource

# View sample data to understand what users see
get_table_sample_data(table_name="Image", limit=5)
```

## Step 2: Understand Display Contexts

Annotations can be set per-context, controlling how data appears in different Chaise views:

| Context | Description |
|---------|-------------|
| `compact` | Table/record list view (summary rows) |
| `compact/brief` | Inline compact display (e.g., in popups) |
| `compact/select` | Record selector dropdowns |
| `detailed` | Single record detail view |
| `entry` | Record creation form |
| `entry/edit` | Record edit form |
| `entry/create` | Record creation form (overrides entry) |
| `filter/compact` | Facet panel display |
| `row_name` | How a record is identified (used in FK links, breadcrumbs) |
| `row_name/compact` | Row name in compact context |
| `row_name/detailed` | Row name in detailed context |
| `*` | Default for all contexts |

## Step 3: Customize Display Names

### Table display name
```
set_table_display_name(table_name="Image", display_name="Images")
```

### Column display name
```
set_column_display_name(table_name="Image", column_name="URL", display_name="Image File")
set_column_display_name(table_name="Subject", column_name="Age_At_Enrollment", display_name="Age at Enrollment")
```

### Column description (tooltip)
```
set_column_description(table_name="Image", column_name="URL", description="Direct download link for the image file")
```

## Step 4: Configure Visible Columns

Control which columns appear and in what order for each context.

### View current visible columns
```
# Read the deriva://table/Image/annotations resource
# Check the visible_columns annotation in the response
```

### Add a column to a context
```
add_visible_column(table_name="Image", context="compact", column="Filename")
add_visible_column(table_name="Image", context="compact", column="Subject")
add_visible_column(table_name="Image", context="compact", column="Diagnosis")
```

### Remove a column from a context
```
remove_visible_column(table_name="Image", context="compact", column="RCT")
remove_visible_column(table_name="Image", context="compact", column="RMT")
```

### Reorder columns
```
reorder_visible_columns(
    table_name="Image",
    context="compact",
    new_order=["Filename", "Subject", "Diagnosis", "Image_Type", "URL"]
)
```

### Set all visible columns at once
```
set_visible_columns(
    table_name="Image",
    annotation={"compact": ["Filename", "Subject", "Diagnosis", "Image_Type", "URL"]}
)
```

### Different columns per context
```
set_visible_columns(
    table_name="Image",
    annotation={
        "compact": ["Filename", "Subject", "Diagnosis"],
        "detailed": ["Filename", "Subject", "Diagnosis", "Image_Type", "URL", "File_Size", "Description"],
        "entry": ["Filename", "Subject", "Diagnosis", "Image_Type", "Description"]
    }
)
```

## Step 5: Configure Row Names

Row names determine how a record is identified when referenced from other tables (e.g., in foreign key links, breadcrumbs, and search results).

### Simple row name from a column
```
set_row_name_pattern(table_name="Subject", pattern="{{{Name}}}")
```

### Composite row name with multiple columns
```
set_row_name_pattern(table_name="Subject", pattern="{{{Last_Name}}}, {{{First_Name}}}")
```

### Row name with related data using Handlebars
```
set_row_name_pattern(table_name="Image", pattern="{{{Filename}}} ({{{Diagnosis}}})")
```

### Table display with row ordering
```
set_table_display(
    table_name="Subject",
    annotation={
        "compact": {
            "row_markdown_pattern": "{{{Name}}} (Age: {{{Age}}})",
            "row_order": [{"column": "Name", "descending": false}]
        }
    }
)
```

## Step 6: Configure Visible Foreign Keys

Control which related tables are shown as sections on the detail page of a record.

### Add a related table section
```
add_visible_foreign_key(table_name="Subject", context="detailed", foreign_key=["deriva-ml", "Image_Subject_fkey"])
```

### Remove a related table section
```
remove_visible_foreign_key(table_name="Subject", context="detailed", foreign_key=["deriva-ml", "Image_Subject_fkey"])
```

### Reorder related table sections
```
reorder_visible_foreign_keys(
    table_name="Subject",
    context="detailed",
    new_order=[["deriva-ml", "Image_Subject_fkey"], ["deriva-ml", "Sample_Subject_fkey"], ["deriva-ml", "Diagnosis_Subject_fkey"]]
)
```

### Set all visible foreign keys at once
```
set_visible_foreign_keys(
    table_name="Subject",
    annotation={
        "detailed": [
            {"source": [{"inbound": ["deriva-ml", "Image_Subject_fkey"]}, "RID"]},
            {"source": [{"inbound": ["deriva-ml", "Sample_Subject_fkey"]}, "RID"]}
        ]
    }
)
```

## Step 7: Apply Annotations

**This step is required.** After making changes, persist them to the catalog:

```
apply_annotations()
```

This writes all pending annotation changes to the catalog server. If you skip this step, your changes will be lost.

For common annotation recipes and patterns, see `references/common-recipes.md`.

## Reference Resources

For detailed reference material beyond what this skill covers, read these MCP resources:

- `deriva://docs/annotation-contexts` — Complete JSON reference of all valid Chaise annotation contexts and their usage. Read this when you need to know which contexts are available or what a specific context controls.
- `deriva://docs/annotations` — Full guide to the annotation builder classes and annotation JSON structure. Read this for details on pseudo-column source syntax, facet configuration options, or advanced Handlebars patterns.
- `deriva://docs/chaise/config` — Chaise web UI configuration options. Read this when customizing Chaise behavior beyond annotations (e.g., default page sizes, navbar, login config).

To inspect current annotations on a specific table or column:
- `deriva://table/{table_name}/annotations` — Display-related annotations currently set on a table
- `deriva://table/{table_name}/column/{column_name}/annotations` — Display-related annotations on a column

## Tips

- Always call `apply_annotations()` as the final step after making changes.
- Use `apply_catalog_annotations()` first to get reasonable defaults, then customize specific tables.
- The `compact` context is the most commonly customized -- it controls the table listing view.
- Row name patterns use Handlebars syntax: `{{{column_name}}}` for column values.
- Foreign key columns are automatically rendered as links to the related record in Chaise.
- Test your changes by viewing the table in Chaise after applying annotations.
- If something looks wrong, use `get_table(table_name=...)` to inspect the current annotations.
