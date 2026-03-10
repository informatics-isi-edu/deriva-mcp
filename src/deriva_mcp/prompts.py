"""MCP Prompts for DerivaML.

Auto-generated from skill SKILL.md files by scripts/generate_prompts.py.
Do not edit manually — regenerate with:
    python scripts/generate_prompts.py

Prompts provide interactive, step-by-step guidance for common tasks:
- ML execution lifecycle (training, inference workflows)
- Dataset preparation and management
- Catalog operations (tables, features, annotations)
- Experiment configuration and execution
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from deriva_mcp.connection import ConnectionManager


def register_prompts(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register all DerivaML prompts with the MCP server."""
    @mcp.prompt(
        name="customize-display",
        description="Customize the Chaise web UI display for Deriva catalog tables using MCP annotation tools. Use when setting visible columns, reordering columns, changing display names, configuring row name patterns, or adjusting how tables and records appear in the browser UI.",
    )
    def customize_display_prompt() -> str:
        """customize-display workflow guide."""
        return """# Customizing Chaise Web UI Display

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
get_table(table="Image")

# View sample data to understand what users see
get_table_sample_data(table="Image", limit=5)
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
set_table_display_name(table="Image", display_name="Images")
# Or set it contextually
set_display_annotation(table="Image", display_name="Images")
```

### Column display name
```
set_column_display_name(table="Image", column="URL", display_name="Image File")
set_column_display_name(table="Subject", column="Age_At_Enrollment", display_name="Age at Enrollment")
```

### Column description (tooltip)
```
set_column_description(table="Image", column="URL", description="Direct download link for the image file")
```

## Step 4: Configure Visible Columns

Control which columns appear and in what order for each context.

### View current visible columns
```
get_table(table="Image")
# Check the visible_columns annotation in the response
```

### Add a column to a context
```
add_visible_column(table="Image", context="compact", column="Filename")
add_visible_column(table="Image", context="compact", column="Subject")
add_visible_column(table="Image", context="compact", column="Diagnosis")
```

### Remove a column from a context
```
remove_visible_column(table="Image", context="compact", column="RCT")
remove_visible_column(table="Image", context="compact", column="RMT")
```

### Reorder columns
```
reorder_visible_columns(
    table="Image",
    context="compact",
    columns=["Filename", "Subject", "Diagnosis", "Image_Type", "URL"]
)
```

### Set all visible columns at once
```
set_visible_columns(
    table="Image",
    context="compact",
    columns=["Filename", "Subject", "Diagnosis", "Image_Type", "URL"]
)
```

### Different columns per context
```
# Compact view: show summary
set_visible_columns(
    table="Image",
    context="compact",
    columns=["Filename", "Subject", "Diagnosis"]
)

# Detailed view: show everything
set_visible_columns(
    table="Image",
    context="detailed",
    columns=["Filename", "Subject", "Diagnosis", "Image_Type", "URL", "File_Size", "Description"]
)

# Entry form: only editable fields
set_visible_columns(
    table="Image",
    context="entry",
    columns=["Filename", "Subject", "Diagnosis", "Image_Type", "Description"]
)
```

## Step 5: Configure Row Names

Row names determine how a record is identified when referenced from other tables (e.g., in foreign key links, breadcrumbs, and search results).

### Simple row name from a column
```
set_row_name_pattern(table="Subject", pattern="{{{Name}}}")
```

### Composite row name with multiple columns
```
set_row_name_pattern(table="Subject", pattern="{{{Last_Name}}}, {{{First_Name}}}")
```

### Row name with related data using Handlebars
```
set_row_name_pattern(table="Image", pattern="{{{Filename}}} ({{{Diagnosis}}})")
```

### Table display with row ordering
```
set_table_display(
    table="Subject",
    context="compact",
    row_markdown_pattern="{{{Name}}} (Age: {{{Age}}})",
    row_order=[{"column": "Name", "descending": false}]
)
```

## Step 6: Configure Visible Foreign Keys

Control which related tables are shown as sections on the detail page of a record.

### Add a related table section
```
add_visible_foreign_key(table="Subject", context="detailed", foreign_key="Image_Subject_fkey")
```

### Remove a related table section
```
remove_visible_foreign_key(table="Subject", context="detailed", foreign_key="Image_Subject_fkey")
```

### Reorder related table sections
```
reorder_visible_foreign_keys(
    table="Subject",
    context="detailed",
    foreign_keys=["Image_Subject_fkey", "Sample_Subject_fkey", "Diagnosis_Subject_fkey"]
)
```

### Set all visible foreign keys at once
```
set_visible_foreign_keys(
    table="Subject",
    context="detailed",
    foreign_keys=["Image_Subject_fkey", "Sample_Subject_fkey"]
)
```

## Step 7: Apply Annotations

**This step is required.** After making changes, persist them to the catalog:

```
apply_annotations()
```

This writes all pending annotation changes to the catalog server. If you skip this step, your changes will be lost.

## Column Display Formatting

Control how individual column values are rendered:

```
set_column_display(
    table="Image",
    column="URL",
    context="compact",
    markdown_pattern="[Download]({{{URL}}})"
)

set_column_display(
    table="Measurement",
    column="Value",
    context="compact",
    markdown_pattern="{{{Value}}} {{{Units}}}"
)
```

## Common Recipes

### Hide system columns from compact view
```
remove_visible_column(table="Image", context="compact", column="RID")
remove_visible_column(table="Image", context="compact", column="RCT")
remove_visible_column(table="Image", context="compact", column="RMT")
remove_visible_column(table="Image", context="compact", column="RCB")
remove_visible_column(table="Image", context="compact", column="RMB")
```

### Make a table's compact view show key info only
```
set_visible_columns(
    table="Subject",
    context="compact",
    columns=["Name", "Age", "Sex", "Species", "Diagnosis"]
)
set_row_name_pattern(table="Subject", pattern="{{{Name}}}")
apply_annotations()
```

### Configure a vocabulary table display
```
set_visible_columns(
    table="Diagnosis",
    context="compact",
    columns=["Name", "Description", "Synonyms"]
)
set_row_name_pattern(table="Diagnosis", pattern="{{{Name}}}")
apply_annotations()
```

## Tips

- Always call `apply_annotations()` as the final step after making changes.
- Use `apply_catalog_annotations()` first to get reasonable defaults, then customize specific tables.
- The `compact` context is the most commonly customized -- it controls the table listing view.
- Row name patterns use Handlebars syntax: `{{{column_name}}}` for column values.
- Foreign key columns are automatically rendered as links to the related record in Chaise.
- Test your changes by viewing the table in Chaise after applying annotations.
- If something looks wrong, use `get_table()` to inspect the current annotations."""

    @mcp.prompt(
        name="use-annotation-builders",
        description="Write Python scripts using type-safe annotation builder classes (ColumnAnnotation, TableAnnotation, KeyAnnotation) for production Deriva catalog code. Use when writing Python code to configure catalog display, not when using interactive MCP tools.",
    )
    def use_annotation_builders_prompt() -> str:
        """use-annotation-builders workflow guide."""
        return """# Using Annotation Builder Classes

DerivaML provides Python builder classes for constructing Deriva annotations with full type safety and IDE autocompletion. These are ideal for production code, scripts, and notebooks where you need programmatic control over catalog annotations.

**This skill covers the Python builder class approach.** For quick interactive setup using MCP tools (better for one-off tweaks and exploration), see the `customize-display` skill instead.

## When to Use Builders vs MCP Tools

| Use Case | Approach |
|----------|----------|
| Interactive catalog setup | MCP tools (`set_visible_columns`, `apply_annotations`, etc.) |
| One-off display tweaks | MCP tools |
| Production deployment scripts | Builders |
| Reusable catalog configuration | Builders |
| Complex pseudo-columns and facets | Builders |
| Code that needs IDE autocompletion | Builders |
| Sharing configuration across catalogs | Builders |

## Available Builder Classes

### Display -- Names, Markdown, Styles

Controls how a table or column is displayed.

```python
from deriva.ml.annotation_builders import Display

# Simple display name
display = Display(name="Labeled Images")

# With markdown pattern for row naming
display = Display(
    name="Images",
    markdown_name="**Images**",
    markdown_pattern="{{{Filename}}} ({{{Diagnosis}}})"
)

# Apply to a table
table.annotations[Display.tag] = display.to_dict()
```

### VisibleColumns -- Per-Context Column Lists

Defines which columns appear in each Chaise context, with method chaining.

```python
from deriva.ml.annotation_builders import VisibleColumns, Context

vc = VisibleColumns()

# Set columns for compact view
vc.set(Context.COMPACT, [
    "Filename",
    "Subject",
    "Diagnosis",
    "Image_Type"
])

# Set columns for detailed view (more columns)
vc.set(Context.DETAILED, [
    "Filename",
    "Subject",
    "Diagnosis",
    "Image_Type",
    "URL",
    "File_Size",
    "Width",
    "Height",
    "Description"
])

# Set entry form columns
vc.set(Context.ENTRY, [
    "Filename",
    "Subject",
    "Diagnosis",
    "Image_Type",
    "Description"
])

# Apply to table
table.annotations[VisibleColumns.tag] = vc.to_dict()
```

### VisibleForeignKeys -- Related Table Sections

Controls which related tables appear on the detail page.

```python
from deriva.ml.annotation_builders import VisibleForeignKeys, Context

vfk = VisibleForeignKeys()

vfk.set(Context.DETAILED, [
    {"source": [{"outbound": ["schema", "Image_Subject_fkey"]}, "RID"]},
    {"source": [{"outbound": ["schema", "Sample_Subject_fkey"]}, "RID"]}
])

table.annotations[VisibleForeignKeys.tag] = vfk.to_dict()
```

### TableDisplay -- Row Naming and Ordering

Controls table-level display behavior including row naming and default sort order.

```python
from deriva.ml.annotation_builders import TableDisplay, Context

td = TableDisplay()

# Row name pattern (how records appear in FK links)
td.set_row_name(Context.ROW_NAME, "{{{Last_Name}}}, {{{First_Name}}}")

# Row ordering
td.set_row_order(Context.COMPACT, [
    {"column": "Name", "descending": False}
])

# Page size
td.set_page_size(Context.COMPACT, 25)

table.annotations[TableDisplay.tag] = td.to_dict()
```

### ColumnDisplay -- Value Formatting

Controls how individual column values are rendered.

```python
from deriva.ml.annotation_builders import ColumnDisplay, Context

cd = ColumnDisplay()

# Render URL as a download link
cd.set(Context.COMPACT, markdown_pattern="[Download]({{{URL}}})")

# Pre-format: transform value before display
cd.set(Context.DETAILED, pre_format={"format": "%d", "unit": "bytes"})

column.annotations[ColumnDisplay.tag] = cd.to_dict()
```

### PseudoColumns -- Computed and FK-Traversed Values

PseudoColumns let you display values from related tables or computed expressions in column lists.

```python
from deriva.ml.annotation_builders import PseudoColumn, OutboundFK, InboundFK, Aggregate

# Follow an outbound foreign key to show a related column
# Image -> Subject -> Name
subject_name = PseudoColumn(
    source=[
        OutboundFK("schema", "Image_Subject_fkey"),
        "Name"
    ],
    markdown_name="Subject Name"
)

# Follow an inbound foreign key with aggregation
# Count of images related to a subject
image_count = PseudoColumn(
    source=[
        InboundFK("schema", "Image_Subject_fkey"),
        "RID"
    ],
    aggregate=Aggregate.COUNT_DISTINCT,
    markdown_name="# Images"
)

# Array aggregation -- collect all values
all_diagnoses = PseudoColumn(
    source=[
        InboundFK("schema", "Diagnosis_Subject_fkey"),
        OutboundFK("schema", "Diagnosis_Type_fkey"),
        "Name"
    ],
    aggregate=Aggregate.ARRAY_DISTINCT,
    markdown_name="Diagnoses"
)

# Use in visible columns
vc.set(Context.COMPACT, [
    "Name",
    subject_name.to_dict(),
    image_count.to_dict(),
    all_diagnoses.to_dict()
])
```

**FK path helpers:**
- `OutboundFK(schema, constraint)` -- follow FK from this table to a related table
- `InboundFK(schema, constraint)` -- follow FK from a related table to this table
- Chain multiple hops: `[OutboundFK(...), OutboundFK(...), "Column"]`

**Aggregate functions:**
- `Aggregate.COUNT` -- count of values
- `Aggregate.COUNT_DISTINCT` -- count of unique values
- `Aggregate.ARRAY` -- array of all values
- `Aggregate.ARRAY_DISTINCT` -- array of unique values
- `Aggregate.MIN`, `Aggregate.MAX` -- min/max value

### FacetList and Facet -- Faceted Search Configuration

Configure the facet panel for filtering records.

```python
from deriva.ml.annotation_builders import FacetList, Facet, OutboundFK

facets = FacetList()

# Simple column facet
facets.add(Facet(
    source="Species",
    markdown_name="Species",
    open=True
))

# FK-based facet (filter by related table value)
facets.add(Facet(
    source=[OutboundFK("schema", "Image_Diagnosis_fkey"), "Name"],
    markdown_name="Diagnosis",
    open=True
))

# Range facet for numeric column
facets.add(Facet(
    source="Age",
    markdown_name="Age",
    ranges=[{"min": 0, "max": 120}]
))

# Choice facet with specific options
facets.add(Facet(
    source="Status",
    markdown_name="Status",
    choices=["Active", "Completed", "Failed"]
))

table.annotations[FacetList.tag] = facets.to_dict()
```

### Handlebars Templates

Row name patterns and markdown patterns use Handlebars syntax:

```python
# Simple column reference
pattern = "{{{Name}}}"

# Multiple columns
pattern = "{{{Last_Name}}}, {{{First_Name}}}"

# Conditional display
pattern = "{{#if Description}}{{{Description}}}{{else}}{{{Name}}}{{/if}}"

# Iteration over array values
pattern = "{{#each Diagnoses}}{{{this}}}{{#unless @last}}, {{/unless}}{{/each}}"

# URL encoding for links
pattern = "[{{{Name}}}](/chaise/record/#{{{$catalog.id}}}/Schema:Table/RID={{{$url_encode.RID}}})"
```

**Template validation tools:**
```
validate_template_syntax(template="{{{Name}}} ({{{Age}}})")
get_handlebars_template_variables(template="{{{Name}}} ({{{Age}}})")
preview_handlebars_template(template="{{{Name}}}", table="Subject", rid="2-XXXX")
```

### Context Constants

Use context constants instead of raw strings:

```python
from deriva.ml.annotation_builders import Context

Context.COMPACT           # "compact"
Context.COMPACT_BRIEF     # "compact/brief"
Context.COMPACT_SELECT    # "compact/select"
Context.DETAILED          # "detailed"
Context.ENTRY             # "entry"
Context.ENTRY_EDIT        # "entry/edit"
Context.ENTRY_CREATE      # "entry/create"
Context.FILTER_COMPACT    # "filter/compact"
Context.ROW_NAME          # "row_name"
Context.ROW_NAME_COMPACT  # "row_name/compact"
Context.ROW_NAME_DETAILED # "row_name/detailed"
Context.DEFAULT           # "*"
```

## Complete Example: Configuring an Image Table

```python
from deriva.ml import DerivaML
from deriva.ml.annotation_builders import (
    Display, VisibleColumns, VisibleForeignKeys, TableDisplay,
    ColumnDisplay, PseudoColumn, OutboundFK, InboundFK,
    Aggregate, FacetList, Facet, Context
)

ml = DerivaML(hostname, catalog_id)

# Table display name
display = Display(name="Images", markdown_name="**Images**")

# Visible columns per context
vc = VisibleColumns()

# Pseudo-column: subject name via FK
subject_name = PseudoColumn(
    source=[OutboundFK("deriva-ml", "Image_Subject_fkey"), "Name"],
    markdown_name="Subject"
)

# Pseudo-column: diagnosis via FK chain
diagnosis = PseudoColumn(
    source=[
        OutboundFK("deriva-ml", "Image_Subject_fkey"),
        OutboundFK("deriva-ml", "Subject_Diagnosis_fkey"),
        "Name"
    ],
    markdown_name="Diagnosis"
)

vc.set(Context.COMPACT, [
    "Filename",
    subject_name.to_dict(),
    diagnosis.to_dict(),
    "Image_Type"
])

vc.set(Context.DETAILED, [
    "Filename",
    subject_name.to_dict(),
    diagnosis.to_dict(),
    "Image_Type",
    "URL",
    "File_Size",
    "Width",
    "Height",
    "Description"
])

# Table display: row name and ordering
td = TableDisplay()
td.set_row_name(Context.ROW_NAME, "{{{Filename}}}")
td.set_row_order(Context.COMPACT, [{"column": "Filename", "descending": False}])

# Column display: render URL as download link
url_display = ColumnDisplay()
url_display.set(Context.COMPACT, markdown_pattern="[Download]({{{URL}}})")

# Facets for filtering
facets = FacetList()
facets.add(Facet(
    source=[OutboundFK("deriva-ml", "Image_ImageType_fkey"), "Name"],
    markdown_name="Image Type",
    open=True
))
facets.add(Facet(
    source=[OutboundFK("deriva-ml", "Image_Subject_fkey"), "Name"],
    markdown_name="Subject"
))

# Visible foreign keys on detail page
vfk = VisibleForeignKeys()
vfk.set(Context.DETAILED, [
    {"source": [{"inbound": ["deriva-ml", "Feature_Value_Image_fkey"]}, "RID"]}
])

# Apply all annotations to the table
table = ml.model.schemas["deriva-ml"].tables["Image"]
table.annotations[Display.tag] = display.to_dict()
table.annotations[VisibleColumns.tag] = vc.to_dict()
table.annotations[TableDisplay.tag] = td.to_dict()
table.annotations[VisibleForeignKeys.tag] = vfk.to_dict()
table.annotations[FacetList.tag] = facets.to_dict()

# Apply column-level annotation
table.columns["URL"].annotations[ColumnDisplay.tag] = url_display.to_dict()

# Push all changes to the catalog
ml.apply_annotations()
```

## Tips

- Builders produce the same JSON that MCP tools set -- they are two ways to do the same thing.
- Use builders when you need to version-control your catalog configuration in Python scripts.
- Use MCP tools for quick interactive changes.
- Always call `ml.apply_annotations()` (Python) or `apply_annotations()` (MCP) after making changes.
- PseudoColumns are powerful for showing related data without changing the data model.
- Test complex Handlebars patterns with `preview_handlebars_template` before applying them."""

    @mcp.prompt(
        name="query-catalog-data",
        description="Guide for querying, filtering, searching, or browsing data in a Deriva catalog",
    )
    def query_catalog_data_prompt() -> str:
        """query-catalog-data workflow guide."""
        return """# Querying and Exploring Data in a Deriva Catalog

This skill covers how to find, filter, and explore data in a Deriva catalog using MCP tools and resources.

## Discovery Resources

| Resource | Purpose |
|----------|---------|
| `deriva://catalog/tables` | All tables with descriptions and row counts |
| `deriva://catalog/schema` | Full schema with relationships |
| `deriva://table/{name}/schema` | Column names, types, descriptions |
| `deriva://table/{name}/sample` | Sample rows |
| `deriva://table/{name}/features` | Features on a table |
| `deriva://vocabulary/{name}` | Vocabulary terms |
| `deriva://dataset/{rid}` | Dataset details and versions |
| `deriva://chaise-url/{table}/{rid}` | Web UI link |

## Key Tools

| Tool | Purpose |
|------|---------|
| `query_table` | Query with filters, columns, limit/offset |
| `count_table` | Count matching records |
| `get_record` | Fetch a single record by RID |
| `validate_rids` | Check if RIDs exist |
| `denormalize_dataset` | Join dataset tables into flat DataFrame |
| `download_dataset` | Download full dataset as BDBag |
| `list_dataset_members` | List records in a dataset |
| `list_asset_executions` | Find executions that created/used an asset |

## Common Patterns

```
# Query with filter
query_table(table_name="Subject", filters={"Species": "Mouse"}, limit=50)

# Paginate
query_table(table_name="Image", limit=100, offset=200)

# Get specific record
get_record(table_name="Subject", rid="2-B4C8")

# ML-ready flat data
denormalize_dataset(dataset_rid="2-B4C8")
```

## Tips

- Always use `limit` for large tables to avoid timeouts
- Column names are case-sensitive — check schema first
- Use `denormalize_dataset` to resolve FK RIDs into readable values
- Pin to specific dataset versions for reproducibility

For the full guide with query patterns, feature queries, provenance tracking, and troubleshooting, read `references/workflow.md`.

---

# Detailed Guide

# Querying and Exploring Data in a Deriva Catalog

This skill covers how to query, filter, and explore data stored in a Deriva catalog using MCP tools and resources.

## Understanding the Schema

Before querying, understand what tables and columns are available.

### Catalog-Level Overview

Read these MCP resources to get oriented:

- `deriva://catalog/tables` -- Lists all tables in the current schema with descriptions and row counts.
- `deriva://catalog/schema` -- Full schema overview with table relationships.

### Table-Level Details

For a specific table:

- `deriva://table/{table_name}/schema` -- Column names, types, nullability, and descriptions.
- `deriva://table/{table_name}/sample` -- A few sample rows to understand the data shape.

Use the `get_table` MCP tool for programmatic access to table metadata.

## Simple Queries

### Query All Rows

Use the `query_table` MCP tool:

```
query_table(table_name="Subject")
```

This returns all rows. For large tables, use `limit` and `offset` for pagination.

### Specific Columns

```
query_table(table_name="Subject", columns=["RID", "Name", "Species"])
```

### Limit Results

```
query_table(table_name="Subject", limit=10)
```

### Paginate Through Results

```
query_table(table_name="Subject", limit=100, offset=0)    # First 100
query_table(table_name="Subject", limit=100, offset=100)   # Next 100
query_table(table_name="Subject", limit=100, offset=200)   # Next 100
```

## Filter Queries

### Equality Filter

```
query_table(table_name="Subject", filter={"Species": "Mouse"})
```

### Multiple AND Conditions

```
query_table(
    table_name="Subject",
    filter={"Species": "Mouse", "Status": "Active"}
)
```

This returns rows where Species is "Mouse" AND Status is "Active".

### Count Rows

Use the `count_table` MCP tool to get the number of matching rows without fetching data:

```
count_table(table_name="Subject")
count_table(table_name="Subject", filter={"Species": "Mouse"})
```

## Get Specific Records

### By RID

Use the `get_record` MCP tool to fetch a single record by its RID (Row ID):

```
get_record(table_name="Subject", rid="2-B4C8")
```

This returns the complete record with all columns.

### Resolve a RID

Use the `validate_rids` MCP tool to check if RIDs exist and determine their table:

```
validate_rids(rids=["2-B4C8", "2-D1E2"])
```

## Query Related Data

### Denormalize for ML

Use the `denormalize_dataset` MCP tool to get ML-ready joined data from a dataset:

```
denormalize_dataset(dataset_rid="2-B4C8")
```

This joins the dataset's member tables, resolving foreign keys into human-readable values. The result is a flat table suitable for loading into a DataFrame.

### Query a Single Table

For simpler needs, `query_table` on the relevant table is sufficient:

```
query_table(table_name="Image", filter={"Subject": "2-A1B2"})
```

### Download a Full Dataset

Use the `download_dataset` MCP tool to get a complete local copy of a dataset:

```
download_dataset(dataset_rid="2-B4C8", version=3)
```

This downloads all dataset members and assets to a local directory.

## Vocabulary Lookups

Deriva uses controlled vocabularies for categorical values. Look them up via MCP resources:

- `deriva://vocabulary/{vocab_name}` -- Lists all terms in a vocabulary with descriptions.
- `deriva://vocabulary/{vocab_name}/{term}` -- Details for a specific term.

Common vocabularies include dataset types, workflow types, species, and status values.

Use `query_table` to query vocabulary tables directly:

```
query_table(table_name="Species")
query_table(table_name="Dataset_Type")
```

## Feature Queries

Features in DerivaML represent measured or computed properties of entities.

### List Features for a Table

Read the MCP resource:

- `deriva://table/{table_name}/features` -- Lists all features associated with a table.

### Feature Structure

Features have:
- A **feature name** (e.g., "Cell_Count", "Mean_Intensity").
- A **feature table** that stores the values.
- An **association** to a base table (e.g., Image, Subject).

### Query Feature Values

Use `query_table` on the feature table:

```
query_table(table_name="Image_Cell_Count")
```

Or use the `get_table_sample_data` MCP tool for a quick preview:

```
get_table_sample_data(table_name="Image_Cell_Count")
```

## Common Query Patterns

### Find Images for a Subject

```
query_table(table_name="Image", filter={"Subject": "2-A1B2"})
```

### Find All Subjects in a Dataset

First get the dataset members:
```
list_dataset_members(dataset_rid="2-B4C8")
```

### Date Range Queries

Deriva supports date filtering. Use ISO 8601 format:

```
query_table(
    table_name="Execution",
    filter={"Status": "Complete"},
    sort=[{"column": "RCT", "descending": True}],
    limit=20
)
```

Note: For complex date range queries, you may need to use the ERMrest API directly or filter results client-side.

### Export Data for ML

To get data ready for ML training:

1. **Identify the dataset**: `get_record(table_name="Dataset", rid="2-B4C8")`
2. **Get the members**: `list_dataset_members(dataset_rid="2-B4C8")`
3. **Denormalize**: `denormalize_dataset(dataset_rid="2-B4C8")`
4. **Download assets**: `download_dataset(dataset_rid="2-B4C8", version=3)`

## Historical Queries with Versions

Datasets in Deriva can be versioned. Query specific versions:

```
download_dataset(dataset_rid="2-B4C8", version=3)
```

To see available versions, read:

- `deriva://dataset/{rid}` -- Includes version history.

Always pin to a specific version for reproducible experiments.

## View in Web Interface

To get the Chaise (web UI) URL for any record, use the MCP resource:

- `deriva://chaise-url/{table_name}/{rid}` -- Direct URL to view a record in the browser.
- `deriva://chaise-url/{table_name}` -- URL to the table's record set view.

These URLs are useful for sharing records with collaborators or viewing complex relationships that are easier to navigate in the web interface.

## Complete Example Workflow

Here is a typical workflow for exploring and extracting data from a catalog:

1. **Orient yourself**: Read `deriva://catalog/tables` to see what is available.

2. **Explore a table**: Read `deriva://table/Subject/schema` to understand columns, then `get_table_sample_data(table_name="Subject")` for sample rows.

3. **Count records**: `count_table(table_name="Subject")` and `count_table(table_name="Subject", filter={"Species": "Mouse"})`.

4. **Query with filters**: `query_table(table_name="Subject", filter={"Species": "Mouse"}, limit=50)`.

5. **Inspect a specific record**: `get_record(table_name="Subject", rid="2-A1B2")`.

6. **Find related data**: `query_table(table_name="Image", filter={"Subject": "2-A1B2"})`.

7. **Check features**: Read `deriva://table/Image/features`, then `query_table(table_name="Image_Cell_Count", filter={"Image": "2-C3D4"})`.

8. **Get dataset for ML**: `denormalize_dataset(dataset_rid="2-B4C8")` for a flat view, or `download_dataset(dataset_rid="2-B4C8", version=3)` for a full local copy.

9. **Share with a colleague**: Read `deriva://chaise-url/Subject/2-A1B2` to get a shareable URL.

## Asset Provenance

Every asset can be traced back to the execution(s) that produced or consumed it:

```
# Find executions that created or used an asset
list_asset_executions(asset_rid="2-IMG1")
# Returns executions with role "Output" (created it) or "Input" (consumed it)

# Look up a specific asset by RID
get_record(table_name="Slide_Image", rid="2-IMG1")

# Download a specific asset
download_asset(asset_rid="2-IMG1")
```

## Tips and Troubleshooting

- **Large tables**: Always use `limit` and `offset` for tables with more than a few hundred rows. Fetching the entire table can be slow and may time out.
- **Column names are case-sensitive**: Use the exact column names from the schema. `"Species"` is not the same as `"species"`.
- **RID format**: RIDs look like `2-B4C8` (a number, a dash, and an alphanumeric string). They are unique within a catalog.
- **Foreign keys**: Many columns contain RIDs referencing other tables. Use `denormalize_dataset` to resolve these into readable values, or `get_record` to look up individual references.
- **Empty results**: If a query returns no rows, double-check the filter values. Use `query_table` without filters first to verify the table has data, then add filters incrementally.
- **Schema mismatch**: If a table is not found, verify you are connected to the correct schema. Use `set_default_schema` if needed.
- **Stale data**: Catalog data can change. If you need a stable snapshot, use versioned datasets."""

    @mcp.prompt(
        name="create-feature",
        description="Guide for creating features, adding labels or annotations to records, setting up classification categories, or working with feature values in DerivaML",
    )
    def create_feature_prompt() -> str:
        """create-feature workflow guide."""
        return """# Creating and Populating Features in DerivaML

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

---

# Detailed Guide

# Creating and Populating a Feature

Features in DerivaML link a target table (e.g., Image, Subject) to vocabulary terms, creating a structured labeling system for ML. A feature is essentially a many-to-many relationship between domain records and vocabulary terms, with provenance tracking through executions.

## Concepts

- **Feature**: A named labeling dimension (e.g., "Tumor Classification", "Image Quality Score").
- **Target table**: The table whose records are being labeled (e.g., Image, Subject).
- **Vocabulary**: A controlled set of terms used as labels (e.g., "Benign", "Malignant", "Unknown").
- **Feature value**: An individual label assignment -- one record labeled with one term, within one execution.

## Step 1: Check Existing Features

```
# List existing features
query_table(table_name="Feature")

# Check existing vocabularies
query_table(table_name="Diagnosis")
query_table(table_name="Image_Quality")
```

## Step 2: Create a Vocabulary (if needed)

If your feature needs a new set of terms, create a vocabulary first.

```
# Create a new vocabulary table
create_vocabulary(
    vocabulary_name="Tumor_Grade",
    comment="Histological grading of tumor samples"
)
```

Then add terms to the vocabulary:

```
# Add individual terms with descriptions
add_term(
    vocabulary_name="Tumor_Grade",
    term_name="Grade I",
    description="Well-differentiated, low grade"
)

add_term(
    vocabulary_name="Tumor_Grade",
    term_name="Grade II",
    description="Moderately differentiated, intermediate grade"
)

add_term(
    vocabulary_name="Tumor_Grade",
    term_name="Grade III",
    description="Poorly differentiated, high grade"
)

add_term(
    vocabulary_name="Tumor_Grade",
    term_name="Grade IV",
    description="Undifferentiated, high grade"
)
```

**Always provide meaningful descriptions for terms.** They appear in the UI and help users understand what each label means.

You can also add synonyms for terms:

```
add_synonym(vocabulary_name="Tumor_Grade", term_name="Grade I", synonym="Low Grade")
add_synonym(vocabulary_name="Tumor_Grade", term_name="Grade III", synonym="High Grade")
```

## Step 3: Create the Feature

Create the feature, linking a target table to vocabulary terms.

### Feature with vocabulary terms

```
create_feature(
    table_name="Image",
    feature_name="Tumor_Classification",
    terms=["Tumor_Grade"],         # Vocabulary table(s) to use as labels
    comment="Classification of tumor grade from histology images"
)
```

This creates:
- A `Tumor_Classification` feature record
- A `Tumor_Classification_Feature_Value` association table linking Image records to Tumor_Grade terms

### Feature with asset terms

If the feature values are files (e.g., segmentation masks, annotation overlays):

```
create_feature(
    table_name="Image",
    feature_name="Segmentation_Mask",
    assets=["Mask_Image"],         # Asset table(s) to use as values
    comment="Pixel-level segmentation masks for images"
)
```

### Feature with both terms and assets

```
create_feature(
    table_name="Image",
    feature_name="Annotated_Region",
    terms=["Region_Label"],
    assets=["Region_Overlay"],
    comment="Labeled regions with overlay images"
)
```

### Feature with metadata columns

Features can include additional columns for structured metadata like confidence scores, reviewer references, or notes. Use the `metadata` parameter:

```
# Feature with a confidence score
create_feature(
    table_name="Image",
    feature_name="Diagnosis",
    terms=["Diagnosis_Type"],
    metadata=[{"name": "confidence", "type": {"typename": "float4"}}],
    comment="Diagnosis with confidence score"
)

# Feature referencing another table (e.g., a Reviewer table)
create_feature(
    table_name="Image",
    feature_name="Review",
    terms=["Review_Status"],
    metadata=["Reviewer"],
    comment="Review annotations with reviewer tracking"
)
```

Each metadata item can be:
- **A string**: Treated as a table name, adds a foreign key reference to that table
- **A dict**: Column definition with `name` and `type` keys. Valid type names: `text`, `int2`, `int4`, `int8`, `float4`, `float8`, `boolean`, `date`, `timestamp`, `timestamptz`, `json`, `jsonb`. Optional keys: `nullok`, `default`, `comment`.

## Step 4: Add Feature Values

Feature values require an active execution for provenance tracking. Every label assignment is tied to the execution that created it.

### MCP Tools

The MCP execution tools operate on the **active execution** -- you don't pass execution RIDs to start/stop/upload.

```
# Step 1: Create execution (sets it as active)
create_execution(
    workflow_name="Manual Tumor Grading",
    workflow_type="Annotation",
    description="Expert pathologist tumor grading session"
)

# Step 2: Start timing
start_execution()

# Step 3: Add feature values
# For simple features (single term or asset column), use add_feature_value:
add_feature_value(
    table_name="Image",
    feature_name="Tumor_Classification",
    target_rid="2-IMG1",
    value="Grade II"
)

add_feature_value(
    table_name="Image",
    feature_name="Tumor_Classification",
    target_rid="2-IMG2",
    value="Grade III"
)

# For features with multiple columns (e.g., term + confidence), use add_feature_value_record:
add_feature_value_record(
    table_name="Image",
    feature_name="Diagnosis",
    target_rid="2-IMG1",
    values={"Diagnosis_Type": "Normal", "confidence": 0.95}
)

# Step 4: Stop timing
stop_execution()

# Step 5: Upload outputs
upload_execution_outputs()
```

### Python API with Context Manager

```python
from deriva_ml import DerivaML, ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)

config = ExecutionConfiguration(
    workflow_name="Manual Tumor Grading",
    workflow_type="Annotation",
    description="Expert pathologist tumor grading"
)

with ml.create_execution(config) as exe:
    exe.add_feature_value(
        feature_name="Tumor_Classification",
        target_rid="2-IMG1",
        term_name="Grade II",
        comment="Clear moderately differentiated pattern"
    )

    exe.add_feature_value(
        feature_name="Tumor_Classification",
        target_rid="2-IMG2",
        term_name="Grade III",
        comment="Poorly differentiated with high mitotic rate"
    )

    # Bulk feature values
    for image_rid, grade in labeling_results.items():
        exe.add_feature_value(
            feature_name="Tumor_Classification",
            target_rid=image_rid,
            term_name=grade
        )

exe.upload_execution_outputs()
```

## Step 5: Query Feature Values

After populating feature values, query them for analysis or training.

```
# Get all feature values for a feature
query_table(table_name="Tumor_Classification_Feature_Value")

# Get feature values for a specific image
query_table(
    table_name="Tumor_Classification_Feature_Value",
    filters={"Image": "2-IMG1"}
)

# Get all images with a specific grade
query_table(
    table_name="Tumor_Classification_Feature_Value",
    filters={"Tumor_Grade": "Grade III"}
)
```

## Complete Example: Image Classification Labels

```
# 1. Create vocabulary
create_vocabulary(
    vocabulary_name="Cell_Type",
    comment="Cell type classifications for microscopy images"
)

add_term(vocabulary_name="Cell_Type", term_name="Epithelial", description="Epithelial cells lining surfaces and cavities")
add_term(vocabulary_name="Cell_Type", term_name="Stromal", description="Connective tissue support cells")
add_term(vocabulary_name="Cell_Type", term_name="Immune", description="Immune system cells including lymphocytes and macrophages")
add_term(vocabulary_name="Cell_Type", term_name="Necrotic", description="Dead or dying cells")
add_term(vocabulary_name="Cell_Type", term_name="Artifact", description="Non-biological artifact in image")

# 2. Create the feature
create_feature(
    table_name="Image",
    feature_name="Cell_Classification",
    terms=["Cell_Type"],
    comment="Primary cell type visible in microscopy image"
)

# 3. Add values within an execution
create_execution(
    workflow_name="Expert Cell Annotation",
    workflow_type="Annotation",
    description="Expert cell type annotation - batch 1"
)
start_execution()

add_feature_value(
    table_name="Image",
    feature_name="Cell_Classification",
    target_rid="2-IMG1",
    value="Epithelial"
)

add_feature_value(
    table_name="Image",
    feature_name="Cell_Classification",
    target_rid="2-IMG2",
    value="Immune"
)

stop_execution()
upload_execution_outputs()
```

## Managing Features

### Delete a feature
```
delete_feature(table_name="Image", feature_name="Tumor_Classification")
```
This removes the feature and its feature value table. Existing data will be lost.

### List all features
```
query_table(table_name="Feature")
```

## Tips

- **Feature values require an active execution.** This is a hard requirement for provenance tracking.
- **Always provide descriptions** for vocabulary terms -- they appear in the UI and help annotators.
- Use separate executions for different labeling sessions or annotators to track who labeled what.
- Multiple features can target the same table (e.g., an Image can have both "Tumor_Classification" and "Image_Quality" features).
- A single record can have multiple values for the same feature (e.g., an image labeled by multiple annotators).
- Use `add_synonym` to make vocabulary terms discoverable under alternative names.
- Use `add_feature_value` for simple features (single term/asset) and `add_feature_value_record` for features with multiple columns.
- Feature values are queryable like any other table, making them easy to use for training data preparation."""

    @mcp.prompt(
        name="create-table",
        description="Guide for creating tables, asset tables, or adding columns in a Deriva catalog",
    )
    def create_table_prompt() -> str:
        """create-table workflow guide."""
        return """# Creating Domain Tables in Deriva

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
- Always set `comment` on tables and columns
- Use `set_row_name_pattern` so records are identifiable in the UI
- Use `set_table_display_name` / `set_column_display_name` for user-friendly names

## Quick Reference

```
# Standard table with FK
create_table(table_name="Sample", columns=[...], foreign_keys=[...], comment="...")

# Asset table
create_asset_table(table_name="Slide_Image", columns=[...], comment="...")

# Add column to existing table
add_column(table="Subject", column_name="Weight_kg", column_type="float8", comment="...")
```

For the full guide with column types table, FK specification, common patterns, and examples, read `references/workflow.md`.

---

# Detailed Guide

# Creating Domain Tables in Deriva

This guide covers creating standard domain tables and asset tables in a Deriva catalog, including column types, foreign keys, and documentation best practices.

## Table Types

| Type | Tool | Description |
|------|------|-------------|
| Standard table | `create_table` | Regular data table with columns and foreign keys |
| Asset table | `create_asset_table` | Table with built-in file upload/download support (URL, Filename, Length, MD5, etc.) |
| Vocabulary table | `create_vocabulary` | Controlled vocabulary with Name, Description, Synonyms, ID, URI |

## Planning Your Table Structure

### Naming Conventions

- **Table names**: Singular nouns with underscores (e.g., `Subject`, `Image_Annotation`, `Blood_Sample`)
- **Column names**: Descriptive with underscores (e.g., `Age_At_Enrollment`, `Sample_Date`, `Cell_Count`)
- **Foreign key columns**: Match the referenced table name (e.g., `Subject` column references `Subject` table)

### Column Types

| Type | Description | Example Values |
|------|-------------|----------------|
| `text` | Variable-length string | "John Doe", "Sample A" |
| `markdown` | Text with Markdown rendering | "**Bold** and *italic*" |
| `int2` | 16-bit integer (-32768 to 32767) | Small counts, codes |
| `int4` | 32-bit integer | Standard integers, counts |
| `int8` | 64-bit integer | Large IDs, big counts |
| `float4` | 32-bit floating point | Approximate measurements |
| `float8` | 64-bit floating point | Precise measurements |
| `boolean` | True/false | `true`, `false` |
| `date` | Calendar date | "2025-01-15" |
| `timestamp` | Date and time (no timezone) | "2025-01-15T10:30:00" |
| `timestamptz` | Date and time with timezone | "2025-01-15T10:30:00-05:00" |
| `json` | JSON data (text storage) | `{"key": "value"}` |
| `jsonb` | JSON data (binary storage, queryable) | `{"key": "value"}` |

## Creating a Simple Table

```
create_table(
    table_name="Subject",
    columns=[
        {"name": "Name", "type": "text", "nullok": false, "comment": "Full name of the subject"},
        {"name": "Age", "type": "int4", "nullok": true, "comment": "Age in years at time of enrollment"},
        {"name": "Sex", "type": "text", "nullok": true, "comment": "Biological sex (Male, Female, Unknown)"},
        {"name": "Species", "type": "text", "nullok": false, "comment": "Species of the subject"},
        {"name": "Date_Of_Birth", "type": "date", "nullok": true, "comment": "Date of birth"},
        {"name": "Notes", "type": "markdown", "nullok": true, "comment": "Additional notes in Markdown format"}
    ],
    comment="Research subjects enrolled in the study"
)
```

**Column specification fields:**
- `name` (required): Column name
- `type` (required): One of the types from the table above
- `nullok` (optional, default `true`): Whether NULL values are allowed. Set to `false` for required fields.
- `comment` (optional but strongly recommended): Description of the column's purpose

## Creating a Table with Foreign Keys

Foreign keys link tables together, establishing relationships.

```
create_table(
    table_name="Sample",
    columns=[
        {"name": "Sample_ID", "type": "text", "nullok": false, "comment": "Unique sample identifier"},
        {"name": "Collection_Date", "type": "date", "nullok": false, "comment": "Date sample was collected"},
        {"name": "Sample_Type", "type": "text", "nullok": false, "comment": "Type of sample (Blood, Tissue, etc.)"},
        {"name": "Volume_mL", "type": "float8", "nullok": true, "comment": "Sample volume in milliliters"},
        {"name": "Notes", "type": "markdown", "nullok": true, "comment": "Collection notes"}
    ],
    foreign_keys=[
        {
            "column": "Subject",
            "referenced_table": "Subject",
            "on_delete": "CASCADE",
            "comment": "The subject this sample was collected from"
        }
    ],
    comment="Biological samples collected from subjects"
)
```

**Foreign key specification fields:**
- `column` (required): Name of the FK column to create in this table (auto-created if not in columns list)
- `referenced_table` (required): Name of the table being referenced
- `on_delete` (optional): What happens when the referenced record is deleted
  - `CASCADE`: Delete this record too (use for strong ownership)
  - `SET NULL`: Set the FK column to NULL (use for optional relationships)
  - `NO ACTION` (default): Prevent deletion of the referenced record
  - `RESTRICT`: Same as NO ACTION but checked immediately
- `comment` (optional): Description of the relationship

## Creating an Asset Table

Asset tables have built-in file management columns (URL, Filename, Length, MD5, Description).

```
create_asset_table(
    table_name="Slide_Image",
    columns=[
        {"name": "Magnification", "type": "text", "nullok": true, "comment": "Microscope magnification (e.g., 10x, 40x)"},
        {"name": "Stain", "type": "text", "nullok": true, "comment": "Staining protocol used"},
        {"name": "Width", "type": "int4", "nullok": true, "comment": "Image width in pixels"},
        {"name": "Height", "type": "int4", "nullok": true, "comment": "Image height in pixels"}
    ],
    foreign_keys=[
        {
            "column": "Sample",
            "referenced_table": "Sample",
            "on_delete": "CASCADE",
            "comment": "The sample this slide image was taken from"
        }
    ],
    comment="Microscopy slide images of biological samples"
)
```

Asset tables automatically include these columns:
- `URL` -- Hatrac object store URL for the file
- `Filename` -- Original filename
- `Length` -- File size in bytes
- `MD5` -- MD5 checksum for integrity verification
- `Description` -- Text description of the asset

## Verifying Your Table

After creation, verify the table was created correctly:

```
# View the full table schema
get_table(table="Sample")

# View sample data (will be empty for new tables)
get_table_sample_data(table="Sample", limit=5)

# Count records
count_table(table="Sample")
```

## Common Patterns

### Subject -> Sample -> Measurement Hierarchy

A typical biomedical data model:

```
# Level 1: Research subjects
create_table(
    table_name="Subject",
    columns=[
        {"name": "Name", "type": "text", "nullok": false, "comment": "Subject identifier"},
        {"name": "Species", "type": "text", "nullok": false, "comment": "Species"}
    ],
    comment="Research subjects"
)

# Level 2: Samples from subjects
create_table(
    table_name="Sample",
    columns=[
        {"name": "Sample_ID", "type": "text", "nullok": false, "comment": "Sample identifier"},
        {"name": "Collection_Date", "type": "date", "nullok": false, "comment": "Collection date"}
    ],
    foreign_keys=[
        {"column": "Subject", "referenced_table": "Subject", "on_delete": "CASCADE",
         "comment": "Subject this sample was collected from"}
    ],
    comment="Samples collected from subjects"
)

# Level 3: Measurements on samples
create_table(
    table_name="Measurement",
    columns=[
        {"name": "Value", "type": "float8", "nullok": false, "comment": "Measured value"},
        {"name": "Units", "type": "text", "nullok": false, "comment": "Unit of measurement"},
        {"name": "Measurement_Date", "type": "timestamptz", "nullok": false, "comment": "When measured"}
    ],
    foreign_keys=[
        {"column": "Sample", "referenced_table": "Sample", "on_delete": "CASCADE",
         "comment": "Sample this measurement was taken from"},
        {"column": "Measurement_Type", "referenced_table": "Measurement_Type", "on_delete": "NO ACTION",
         "comment": "Type of measurement (vocabulary)"}
    ],
    comment="Quantitative measurements on samples"
)
```

### Protocol with Versioning

```
create_table(
    table_name="Protocol",
    columns=[
        {"name": "Name", "type": "text", "nullok": false, "comment": "Protocol name"},
        {"name": "Version", "type": "text", "nullok": false, "comment": "Protocol version string"},
        {"name": "Description", "type": "markdown", "nullok": false, "comment": "Full protocol description in Markdown"},
        {"name": "Effective_Date", "type": "date", "nullok": false, "comment": "Date this version became effective"},
        {"name": "Is_Active", "type": "boolean", "nullok": false, "comment": "Whether this protocol version is currently active"}
    ],
    comment="Experimental protocols with version tracking"
)
```

## Adding Columns to Existing Tables

If you need to add a column after table creation:

```
add_column(
    table="Subject",
    column_name="Weight_kg",
    column_type="float8",
    nullok=true,
    comment="Subject weight in kilograms"
)
```

## Modifying Column Properties

```
# Make a column required or optional
set_column_nullok(table="Subject", column="Notes", nullok=true)

# Update column description
set_column_description(table="Subject", column="Age", description="Age in years at enrollment, rounded down")

# Set column display name
set_column_display_name(table="Subject", column="Age_At_Enrollment", display_name="Enrollment Age")
```

## Documentation Best Practices

1. **Always comment tables**: The `comment` parameter on `create_table` is shown in the UI and in schema documentation.
2. **Always comment columns**: Column comments appear as tooltips in Chaise and serve as documentation.
3. **Set display names**: Use `set_table_display_name` and `set_column_display_name` for user-friendly names that differ from the technical names.
4. **Set row name patterns**: After creating a table, set a row name pattern so records are identifiable:
   ```
   set_row_name_pattern(table="Subject", pattern="{{{Name}}}")
   set_row_name_pattern(table="Sample", pattern="{{{Sample_ID}}} ({{{Subject}}})")
   ```
5. **Set table descriptions**: Use `set_table_description` for longer descriptions beyond the initial comment.

## Tips

- Use `text` for most string columns. Use `markdown` only when you want rich text rendering in the UI.
- Use `float8` over `float4` unless storage is a concern -- the precision difference is significant for scientific data.
- Use `timestamptz` over `timestamp` to avoid timezone ambiguity.
- Use `jsonb` over `json` for better query performance on JSON data.
- Set `nullok=false` for columns that should always have a value. This enforces data quality at the database level.
- Use `CASCADE` on delete for parent-child relationships where children should not exist without parents.
- Use `SET NULL` for optional associations where the child record is still valid without the parent.
- Vocabulary tables (created with `create_vocabulary`) are preferred over free-text columns for categorical data -- they enable faceted search and consistent labeling."""

    @mcp.prompt(
        name="manage-vocabulary",
        description="Create and manage controlled vocabularies in Deriva — create vocabulary tables, add terms with descriptions, add synonyms, and browse existing vocabularies. Use whenever working with categorical data, labels, or controlled term lists independent of features.",
    )
    def manage_vocabulary_prompt() -> str:
        """manage-vocabulary workflow guide."""
        return """# Managing Controlled Vocabularies

Controlled vocabularies are the standard way to represent categorical data in Deriva. They provide consistent labeling, faceted search in Chaise, and synonym support for discoverability. Every vocabulary is a table with standard columns: Name, Description, Synonyms, ID, and URI.

Vocabularies are used by features (see `create-feature`), dataset types, workflow types, asset types, and any categorical column in your domain schema.

## Exploring Existing Vocabularies

Before creating a new vocabulary, check what already exists.

### List all vocabularies

```
# MCP resource — lists all vocabulary tables with term counts
Read resource: deriva://catalog/vocabularies
```

### Browse terms in a vocabulary

```
# MCP resource — lists all terms with descriptions and synonyms
Read resource: deriva://vocabulary/Species

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
Read resource: deriva://catalog/vocabularies
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
5. **Verify** — read `deriva://vocabulary/{vocab_name}` to confirm

## Tips

- Vocabulary tables support faceted search in Chaise automatically — no extra configuration needed
- Terms are ordered alphabetically by name in the UI by default
- The `ID` and `URI` columns are auto-generated — you only need to provide Name and Description
- For large vocabularies (100+ terms), consider hierarchical naming (e.g., "Carcinoma:Ductal", "Carcinoma:Lobular") or multiple smaller vocabularies
- When a vocabulary is used by a feature, the feature creates an association table. See `create-feature` for details"""

    @mcp.prompt(
        name="create-dataset",
        description="Guide for creating, populating, splitting, or managing datasets in DerivaML — including adding members, registering element types, train/test splits, versioning, nested datasets, and provenance",
    )
    def create_dataset_prompt() -> str:
        """create-dataset workflow guide."""
        return """# Creating and Managing Datasets in DerivaML

Datasets are the primary unit for organizing data in DerivaML. A dataset is a named, versioned collection of records (members) drawn from one or more catalog tables, with full provenance tracking through executions.

## Key Concepts

- **Element types** — Tables registered as allowed member sources for a dataset. Must be registered before adding members from that table.
- **Members** — Individual records (by RID) included in the dataset. Can be added in bulk by RID list or grouped by table.
- **Types** — Labels describing purpose: Training, Testing, Validation, Complete, Labeled, Unlabeled. Custom types can also be created.
- **Versioning** — Monotonically increasing version number. Increment after any modification (add/remove members, change element types).
- **Nested datasets** — Parent-child relationships. Used for train/test/validation splits. `split_dataset` creates these automatically.
- **Provenance** — Datasets are created within executions, linking them to workflows and tracking lineage.

## Splitting Datasets

Three split types are available via `split_dataset`:

| Split Type | Use Case | Key Parameter |
|-----------|----------|---------------|
| `random` | General train/test/val splits | `ratios`, `seed` |
| `stratified` | Preserve label distribution across splits | `ratios`, `stratify_feature`, `seed` |
| `labeled` | Separate records with/without feature values | `feature_name` |

All splits create nested child datasets automatically. Use `dry_run=true` to preview before committing.

## Critical Rules and Gotchas

1. **Always create datasets within an execution** — Use `create_execution_dataset`, not bare `create_dataset`, to maintain provenance.
2. **Register element types before adding members** — `add_dataset_element_type` must be called for each source table before `add_dataset_members`.
3. **FK traversal in bag exports** — Downloaded bags include all FK-reachable records. Deep join chains (Image -> Sample -> Subject -> Study) can cause timeouts. Workaround: add intermediate table records as direct members.
4. **Version after every modification** — Call `increment_dataset_version` after adding/removing members or changing element types.
5. **Validate RIDs first** — Use `validate_rids` before `add_dataset_members` to catch invalid RIDs early.
6. **Set seeds on splits** — Always pass `seed` to `split_dataset` for reproducibility.
7. **Deleting a dataset removes only the container** — Member records (images, subjects, etc.) are not deleted, only the dataset and its member associations.

## Workflow Summary

The standard sequence for creating a dataset:

1. Create execution (with a workflow for provenance)
2. `create_execution_dataset` — create the dataset within the execution
3. `add_dataset_type` — label the dataset (Training, Complete, etc.)
4. `add_dataset_element_type` — register source tables
5. `add_dataset_members` — add records by RID
6. `split_dataset` (optional) — create train/test/val child datasets
7. `increment_dataset_version` — version the dataset
8. `stop_execution` + `upload_execution_outputs` — finalize

For the full step-by-step guide with code examples (both Python API and MCP tools), see `references/workflow.md`.

## Related Skills

- **`prepare-training-data`** — Downloading, extracting, and preparing dataset data for ML training pipelines.
- **`debug-bag-contents`** — Diagnosing missing data, FK traversal issues, and export problems in dataset bags.

---

# Detailed Guide

# Creating and Managing Datasets in DerivaML

Datasets are the primary mechanism for organizing data for ML workflows in DerivaML. They group records from multiple tables into a versioned, downloadable collection with full provenance tracking.

## Understanding Datasets

### What is a Dataset?

A dataset is a named collection of records (members) from one or more tables, with:
- **Element types**: Which tables can contribute members
- **Members**: Specific records (by RID) included in the dataset
- **Types**: Labels describing the dataset's purpose (Training, Testing, etc.)
- **Versioning**: Monotonically increasing version numbers for reproducibility
- **Nested datasets**: Parent-child relationships for train/test splits
- **Provenance**: Tracking of which executions created or used the dataset

### Dataset Types

Standard dataset types available by default:

| Type | Description |
|------|-------------|
| Training | Data for model training |
| Testing | Data for model evaluation |
| Validation | Data for hyperparameter tuning |
| Complete | Full dataset before splitting |
| Labeled | Data with feature annotations |
| Unlabeled | Data without feature annotations |

Custom types can be created:
```
create_dataset_type_term(name="Augmented", description="Dataset with augmented samples")
```

### Element Types

Before adding records from a table to a dataset, you must register that table as an element type. This tells the dataset system which tables are allowed as sources.

### FK Traversal in Bag Exports

When a dataset is downloaded as a BDBag, the export follows foreign key relationships:

- **All FK-reachable tables are included**: If you add Image records and Image has a FK to Subject, Subject records are automatically exported.
- **Vocabulary tables are exported separately**: Controlled vocabulary terms are included in a separate section of the bag.
- **Deep join chains can timeout**: If your schema has many levels of FK relationships (e.g., Image -> Sample -> Subject -> Study -> Institution), the export query may timeout. Solution: add records from intermediate tables as direct members to avoid deep joins.

## Step 1: Check Existing Resources

```
# List existing datasets
query_table(table="Dataset")

# List existing dataset types
query_table(table="Dataset_Type")

# Check what element types are registered for a dataset
list_dataset_members(dataset_rid="2-XXXX")
```

## Step 2: Create a Dataset via Execution

Datasets should be created within an execution for provenance.

### Python API

```python
from deriva.ml import DerivaML, ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)

# Get or create a workflow
workflow = ml.create_workflow(
    name="Dataset Curation",
    url="https://github.com/org/repo",
    workflow_type="Data Management",
    description="Curate and organize training datasets"
)

config = ExecutionConfiguration(workflow=workflow)

with ml.create_execution(config) as exe:
    # Create the dataset
    dataset = exe.create_execution_dataset(
        name="Tumor Image Dataset v1",
        description="Curated set of labeled tumor histology images",
        dataset_types=["Training", "Labeled"]
    )

    dataset_rid = dataset["RID"]

    # Register element types
    exe.add_dataset_element_type(dataset_rid=dataset_rid, table="Image")
    exe.add_dataset_element_type(dataset_rid=dataset_rid, table="Subject")

    # Add members by RID
    exe.add_dataset_members(
        dataset_rid=dataset_rid,
        member_rids=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5"]
    )

    # Or add members by table query
    exe.add_dataset_members(
        dataset_rid=dataset_rid,
        members_by_table={
            "Image": ["2-IMG1", "2-IMG2", "2-IMG3"],
            "Subject": ["2-SUB1", "2-SUB2"]
        }
    )

exe.upload_execution_outputs()
```

### MCP Tools

```
# Step 1: Create execution
create_execution(
    workflow_rid="2-WKFL",
    description="Create training dataset"
)
start_execution(execution_rid="2-EXEC")

# Step 2: Create dataset within execution
create_execution_dataset(
    execution_rid="2-EXEC",
    name="Tumor Image Dataset v1",
    description="Curated set of labeled tumor histology images"
)
# Returns dataset RID

# Step 3: Add dataset types
add_dataset_type(dataset_rid="2-DS01", type_name="Training")
add_dataset_type(dataset_rid="2-DS01", type_name="Labeled")

# Step 4: Register element types
add_dataset_element_type(dataset_rid="2-DS01", table="Image")
add_dataset_element_type(dataset_rid="2-DS01", table="Subject")

# Step 5: Add members
add_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5"]
)

# Or add by table
add_dataset_members(
    dataset_rid="2-DS01",
    members_by_table={"Image": ["2-IMG1", "2-IMG2"], "Subject": ["2-SUB1"]}
)

# Step 6: Finalize
stop_execution(execution_rid="2-EXEC")
upload_execution_outputs(execution_rid="2-EXEC")
```

## Step 3: Manage Dataset Types

```
# Add a type to a dataset
add_dataset_type(dataset_rid="2-DS01", type_name="Training")

# Remove a type
remove_dataset_type(dataset_rid="2-DS01", type_name="Complete")

# Create a new custom type
create_dataset_type_term(name="Preprocessed", description="Data that has been preprocessed and normalized")

# Delete a custom type
delete_dataset_type_term(name="Preprocessed")
```

## Step 4: Manage Dataset Members

```
# List current members
list_dataset_members(dataset_rid="2-DS01")

# Add more members
add_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG6", "2-IMG7"]
)

# Remove members
delete_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG7"]
)

# Validate that RIDs exist
validate_rids(rids=["2-IMG1", "2-IMG2", "2-FAKE"])
# Returns which RIDs are valid and which are not
```

## Step 5: Split Datasets

The `split_dataset` tool creates nested child datasets from a parent dataset, typically for train/test/validation splits.

### Random Split

```
split_dataset(
    dataset_rid="2-DS01",
    split_type="random",
    ratios={"Training": 0.7, "Testing": 0.2, "Validation": 0.1},
    seed=42
)
```

Creates three child datasets:
- "Tumor Image Dataset v1 - Training" (70% of members)
- "Tumor Image Dataset v1 - Testing" (20% of members)
- "Tumor Image Dataset v1 - Validation" (10% of members)

### Stratified Split

Split while maintaining the distribution of a label:

```
split_dataset(
    dataset_rid="2-DS01",
    split_type="stratified",
    ratios={"Training": 0.8, "Testing": 0.2},
    stratify_feature="Tumor_Classification",
    seed=42
)
```

Each child dataset will have approximately the same proportion of each Tumor_Classification grade as the parent.

### Labeled Split

Split into labeled and unlabeled subsets based on whether records have feature values:

```
split_dataset(
    dataset_rid="2-DS01",
    split_type="labeled",
    feature_name="Tumor_Classification"
)
```

Creates two child datasets:
- "Tumor Image Dataset v1 - Labeled" (records with Tumor_Classification values)
- "Tumor Image Dataset v1 - Unlabeled" (records without Tumor_Classification values)

### Dry Run Preview

Preview a split before executing it:

```
split_dataset(
    dataset_rid="2-DS01",
    split_type="random",
    ratios={"Training": 0.7, "Testing": 0.3},
    seed=42,
    dry_run=true
)
```

Returns the planned split without creating any datasets. Use this to verify sizes and distributions before committing.

## Step 6: Version Management

After any modification (adding/removing members, changing element types), increment the dataset version:

```
increment_dataset_version(dataset_rid="2-DS01")
```

See the `dataset-versioning` skill for full versioning rules, semantic versioning conventions, and the pre-experiment checklist.

## Nested Datasets

Nested datasets create parent-child relationships, commonly used for train/test splits.

### Create manually
```
# Create child dataset
create_dataset(
    name="Training Subset",
    description="Training portion of the main dataset"
)

# Link as child of parent
add_dataset_child(
    parent_rid="2-DS01",
    child_rid="2-DS02"
)
```

### Navigate the hierarchy
```
# List children of a dataset
list_dataset_children(dataset_rid="2-DS01")

# List parents of a dataset
list_dataset_parents(dataset_rid="2-DS02")
```

### Automatic nesting via split
When using `split_dataset`, child datasets are automatically nested under the parent.

## Downloading and Using Datasets

For extracting, downloading, and preparing dataset data for ML training, see the `prepare-training-data` skill. For diagnosing missing data in bag exports, see the `debug-bag-contents` skill.

## Complete Example: End-to-End Dataset Workflow

```
# 1. Create workflow for dataset management
create_workflow(
    name="Image Dataset Curation",
    url="https://github.com/lab/protocols",
    workflow_type="Data Management",
    description="Curate and split image datasets for training"
)

# 2. Create execution
create_execution(
    workflow_rid="2-WKFL",
    description="Create and split tumor image dataset"
)
start_execution(execution_rid="2-EXEC")

# 3. Create the master dataset
create_execution_dataset(
    execution_rid="2-EXEC",
    name="Tumor Histology Complete",
    description="All labeled tumor histology images as of 2025-06"
)

# 4. Configure types and elements
add_dataset_type(dataset_rid="2-DS01", type_name="Complete")
add_dataset_type(dataset_rid="2-DS01", type_name="Labeled")
add_dataset_element_type(dataset_rid="2-DS01", table="Image")

# 5. Add all labeled images
add_dataset_members(
    dataset_rid="2-DS01",
    member_rids=["2-IMG1", "2-IMG2", "2-IMG3", "2-IMG4", "2-IMG5",
                 "2-IMG6", "2-IMG7", "2-IMG8", "2-IMG9", "2-IMG10"]
)

# 6. Preview the split
split_dataset(
    dataset_rid="2-DS01",
    split_type="stratified",
    ratios={"Training": 0.7, "Testing": 0.15, "Validation": 0.15},
    stratify_feature="Tumor_Classification",
    seed=42,
    dry_run=true
)

# 7. Execute the split
split_dataset(
    dataset_rid="2-DS01",
    split_type="stratified",
    ratios={"Training": 0.7, "Testing": 0.15, "Validation": 0.15},
    stratify_feature="Tumor_Classification",
    seed=42
)

# 8. Finalize
increment_dataset_version(dataset_rid="2-DS01")
stop_execution(execution_rid="2-EXEC")
upload_execution_outputs(execution_rid="2-EXEC")
```

## Provenance Tracking

Track which executions created or used datasets:

```
# Which executions used this dataset as input?
list_dataset_executions(dataset_rid="2-DS01")

# Which executions used this asset?
list_asset_executions(asset_rid="2-IMG1")
```

This lets you trace the full lineage: which workflow created the dataset, what data it contains, which experiments used it, and what results were produced.

## Deleting Datasets

```
# Delete a dataset (removes the dataset record and member associations)
delete_dataset(dataset_rid="2-DS01")
```

**Warning:** Deleting a dataset does not delete the member records themselves (e.g., images, subjects). It only removes the dataset container and its member associations.

## Tips

- Always create datasets within an execution for provenance tracking.
- Register element types before adding members of that type.
- Use `validate_rids` before adding members to catch invalid RIDs early.
- Use `dry_run=true` on `split_dataset` to preview splits before committing.
- Set a `seed` on splits for reproducibility.
- Increment the dataset version after any modification.
- For large datasets with deep FK chains, add intermediate table records as direct members to avoid export timeouts.
- Use stratified splits when your labels are imbalanced to ensure each split has representative samples.
- Nested datasets maintain the parent's element types automatically.
- Use `list_dataset_children` and `list_dataset_parents` to navigate dataset hierarchies."""

    @mcp.prompt(
        name="dataset-versioning",
        description="Dataset version management rules for DerivaML — always use explicit versions in DatasetSpecConfig, increment after catalog changes, check versions before experiments. Use when pinning versions, debugging version mismatches, or understanding the versioning lifecycle.",
    )
    def dataset_versioning_prompt() -> str:
        """dataset-versioning workflow guide."""
        return """# Dataset Versioning Rules

Dataset versioning is essential for reproducible ML experiments. Follow these rules strictly.

## Rule 1: Always Use Explicit Versions for Real Experiments

NEVER use "current" or "latest" for production or real experiment runs.

**Correct:**
```python
DatasetSpecConfig(rid="1-ABC4", version="1.2.0")
```

**Wrong:**
```python
DatasetSpecConfig(rid="1-ABC4", version="current")
DatasetSpecConfig(rid="1-ABC4")  # implies current
```

The ONLY acceptable use of "current" is for debugging and dry runs where reproducibility is not required.

## Rule 2: Increment Version After Catalog Changes

Dataset versions are snapshots of the catalog state at a point in time. If you modify the catalog in any way that affects a dataset's contents, those changes are NOT visible in existing versions.

Changes that require a version increment:
- Adding new features or feature values
- Fixing or correcting labels
- Adding new images or assets
- Modifying asset metadata
- Adding or removing dataset members
- Changing vocabulary terms used by features

After any such change:
1. Call `increment_dataset_version()` with a description of what changed
2. Update configuration files to reference the new version
3. Commit the config changes before running experiments

## Rule 3: Always Provide Version Descriptions

Version descriptions are required and must explain:
- **What** changed in this version
- **Why** the change was made
- **Impact** on experiments or downstream usage

**Good descriptions:**
- "Added severity grading feature (mild/moderate/severe) to all 12,450 images"
- "Fixed 47 mislabeled pneumonia images identified in audit review"
- "Added 2,000 new COVID-19 images from March 2026 collection"

**Bad descriptions:**
- "Updated"
- "New version"
- "Changes"
- "" (empty)

## Semantic Versioning

Follow semantic versioning for dataset versions:

| Version Component | When to Increment | Examples |
|-------------------|-------------------|----------|
| **Major** (X.0.0) | Breaking changes, schema modifications, incompatible structure changes | New column requirements, removed features, restructured tables |
| **Minor** (0.X.0) | New data, new features, non-breaking additions | Added images, new feature annotations, expanded vocabulary |
| **Patch** (0.0.X) | Bug fixes, label corrections, metadata fixes | Fixed mislabeled images, corrected metadata, typo fixes |

## Workflow

### After Creating a Dataset

1. Create the dataset with `create_dataset`
2. Add it to config with explicit version:
   ```python
   training_v1 = builds(DatasetSpec, rid="1-ABC4", version="1.0.0")
   ```
3. Commit the config

### After Modifying the Catalog

1. Make catalog changes (add features, fix labels, etc.)
2. Increment version with description:
   ```
   increment_dataset_version(rid="1-ABC4", description="Added severity grading feature")
   ```
3. Update config to new version:
   ```python
   training_v1 = builds(DatasetSpec, rid="1-ABC4", version="1.1.0")
   ```
4. Commit config changes
5. Run experiments with the new version

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Running without explicit version | Results not reproducible | Always specify version in config |
| Expecting catalog changes in old versions | Old versions are frozen snapshots | Increment version to capture changes |
| Empty or vague version descriptions | Cannot understand version history | Write specific, informative descriptions |
| Not updating config after increment | Experiments still use old version | Update config immediately after incrementing |
| Not committing config before running | Git hash doesn't match config state | Always commit, then run |

## Pre-Experiment Checklist

Before running any experiment:

- [ ] Dataset version is explicitly specified (not "current")
- [ ] Config file is updated with correct version
- [ ] Config changes are committed to git

After any catalog modification:

- [ ] Version has been incremented with a descriptive message
- [ ] All affected config files are updated to the new version
- [ ] Config changes are committed to git"""

    @mcp.prompt(
        name="debug-bag-contents",
        description="Diagnose missing data in DerivaML dataset bag (BDBag) exports — FK traversal issues, missing tables, materialization problems, export timeouts. Use when a downloaded dataset bag is missing expected records, images, or feature values.",
    )
    def debug_bag_contents_prompt() -> str:
        """debug-bag-contents workflow guide."""
        return """# Debugging Dataset Bag Contents

When a dataset bag export is missing expected data, follow this step-by-step diagnostic process to identify and fix the issue.

---

## Step 1: Check Dataset Members

Dataset members are the explicit records that belong to a dataset. If data is missing from a bag, the first question is whether the right members are in the dataset.

- **Resource**: Check the dataset resource to see the dataset's summary and member counts.
- **Tool**: `list_dataset_members` with the dataset RID to get the full list of members, grouped by table.
- Verify that the records you expect are listed as members. If they are missing, add them with `add_dataset_members`.

---

## Step 2: Check Element Type Registration

Every table that contributes members to a dataset must be registered as a **dataset element type**. If a table is not registered, its members will be silently excluded from the bag.

- **Resource**: Check the dataset element types resource to see which tables are registered for this dataset type.
- **Tool**: `add_dataset_element_type` to register a table as an element type if it is missing. You need to specify the dataset type and the table name.
- Common tables that should be registered: `Subject`, `Observation`, `Image` (or other asset tables), and any custom tables whose records appear as dataset members.

---

## Step 3: Preview Bag Export Paths

Before downloading a full bag, preview what the export will contain.

- **Resource**: Check the dataset bag preview resource to see the projected file paths and record counts per table.
- This preview shows which tables will be included and how many rows each will have, without actually downloading anything.
- Compare the preview counts against your expectations to spot discrepancies early.

---

## Step 4: Understand FK Path Traversal

The bag export algorithm uses foreign key (FK) path traversal to determine which related records to include. Understanding this is critical for diagnosing missing data.

### Key rules:
1. **Starting points are dataset members only from registered element types.** Records in tables that are not registered as element types will not serve as starting points for traversal, even if they are dataset members.
2. **FK traversal follows both directions.** From each starting point record, the export follows foreign keys both outward (this table references another) and inward (another table references this one).
3. **Vocabulary table endpoints are exported separately.** Vocabulary/controlled-vocabulary tables encountered during traversal are collected and exported in their own section of the bag, not inline with the data tables.
4. **Traversal depth is bounded.** The export does not follow FK chains indefinitely. It follows direct FK relationships from the member records.

### How traversal works in practice:
- If `Subject` is a registered element type and you have Subject members, the export will:
  - Include those Subject records.
  - Follow FKs from Subject to related tables (e.g., Subject_Phenotype).
  - Follow FKs pointing back to Subject from other tables (e.g., Image.Subject_RID -> Subject).
  - Export vocabulary terms referenced by any included records.

---

## Step 5: Diagnose Common Scenarios

### Scenario: Images missing from a Subject-only dataset

**Problem**: Dataset has Subject members but the exported bag does not include the associated Image records.

**Diagnosis**:
- Images are in a separate asset table with an FK to Subject.
- The FK traversal should find Images that reference the Subject members.

**Fix checklist**:
1. Verify the Image table has a direct FK to Subject (not through an intermediate table).
2. If the FK path goes through an intermediate table (e.g., `Observation`), that intermediate table may need to be registered as an element type, or intermediate records need to be added as members.
3. Alternatively, add the Image records directly as dataset members and register the Image table as an element type.

### Scenario: Observation data missing

**Problem**: Observations associated with Subjects are not in the bag.

**Diagnosis**:
- Check whether Observation has a direct FK to Subject.
- If yes, the FK traversal from Subject members should pick up Observations.
- If not, the path may be indirect and not traversed.

**Fix**:
- Add Observation records as explicit dataset members and register `Observation` as an element type.
- Or ensure there is a direct FK link between the tables.

### Scenario: Vocabulary terms missing

**Problem**: Controlled vocabulary values referenced by data records are not in the bag.

**Diagnosis**:
- Vocabulary terms are exported separately from data tables.
- Check that the vocabulary table is properly configured as a vocabulary (not a regular table).

**Fix**:
- Vocabulary terms referenced by included records should be automatically exported. If they are missing, verify the FK relationship between the data table and the vocabulary table is intact.
- **Tool**: `get_table` on the vocabulary table to confirm its structure.

---

## Step 6: Download and Validate the Bag

Use the validation tool to get a detailed comparison of expected vs. actual bag contents.

- **Tool**: `validate_dataset_bag` with the dataset RID.
  - Returns a **per-table comparison** showing:
    - Expected RIDs (based on dataset members and FK traversal).
    - Actual RIDs present in the downloaded bag.
    - **Missing RIDs**: records that should be in the bag but are not.
    - **Extra RIDs**: records in the bag that were not expected (usually not a problem but worth investigating).
  - Use the missing RIDs to identify exactly which records are being dropped and from which tables.

---

## Step 7: Check FK Paths for All Element Types

For each registered element type, examine the FK paths that the export will follow.

- **Resource**: Check the FK path resource for each element type to see the full traversal graph.
- Look for:
  - **Missing links**: Tables you expect to be reachable but are not connected by FKs.
  - **Indirect paths**: FK chains that go through intermediate tables, which may not be traversed if those intermediates are not included.
  - **Circular references**: These are handled correctly but may cause confusion when reading the path graph.

---

## Step 8: Fix Common Issues

### Deep join timeouts
**Problem**: FK traversal through many intermediate tables causes slow exports or timeouts.

**Fix**: Add records from intermediate tables as direct dataset members rather than relying on deep FK traversal. This flattens the traversal and speeds up the export.

### Missing element type registration
**Problem**: Records from a table are added as members but the table is not a registered element type, so those records are ignored during export.

**Fix**:
- **Tool**: `add_dataset_element_type` to register the table.
- Then re-export the bag.

### Stale dataset version
**Problem**: The bag reflects an older version of the dataset, missing recently added members.

**Fix**:
- **Tool**: `increment_dataset_version` to create a new version that captures current membership.
- Re-export the bag after incrementing.

### Records exist but FK not established
**Problem**: Related records exist in the catalog but are not linked via FK to the member records.

**Fix**:
- Check the FK columns on the related records. Ensure they contain the correct RID values pointing to the dataset member records.
- **Tool**: `query_table` with filters to verify FK column values.

---

## Quick Diagnostic Checklist

Use this checklist when data is missing from a bag:

1. **Are the records dataset members?**
   - `list_dataset_members` -- check if expected records appear.
   - If not: `add_dataset_members`.

2. **Is the table a registered element type?**
   - Check element types resource.
   - If not: `add_dataset_element_type`.

3. **Is there a direct FK path?**
   - Check FK paths resource for the element type.
   - If not: add intermediate records as members, or restructure FKs.

4. **Does validation show the discrepancy?**
   - `validate_dataset_bag` -- look at missing RIDs per table.

5. **Is the version current?**
   - `increment_dataset_version` if members were recently changed.

6. **Preview before full download.**
   - Check the bag preview resource to confirm expected counts before downloading.

## Related Tools

| Tool | Purpose |
|------|---------|
| `list_dataset_members` | List all members of a dataset |
| `add_dataset_members` | Add records to a dataset |
| `delete_dataset_members` | Remove records from a dataset |
| `add_dataset_element_type` | Register a table as dataset element type |
| `validate_dataset_bag` | Validate bag contents against expectations |
| `increment_dataset_version` | Bump dataset version after changes |
| `get_dataset_spec` | View dataset specification |
| `download_dataset` | Download the dataset bag |
| `denormalize_dataset` | Flatten dataset for analysis |
| `query_table` | Inspect FK column values |
| `get_table` | Check table schema and FK relationships |"""

    @mcp.prompt(
        name="prepare-training-data",
        description="Prepare a DerivaML dataset for ML training — denormalize to DataFrame, download BDBag, build training features and labels, extract images, restructure assets. Use when getting data out of the catalog and into a format for model training or analysis.",
    )
    def prepare_training_data_prompt() -> str:
        """prepare-training-data workflow guide."""
        return """# Preparing Training Data from a DerivaML Dataset

This guide walks through the process of taking a DerivaML dataset and preparing it for use in ML training, evaluation, or analysis.

## Step 1: Explore the Dataset

Start by understanding what is in the dataset using catalog resources.

```
# List available datasets
query_table(table="Dataset")

# Get details about a specific dataset
get_record(table="Dataset", rid="2-XXXX")

# See what element types the dataset contains
list_dataset_members(dataset_rid="2-XXXX")

# View the dataset specification (element types, export configuration)
get_dataset_spec(dataset_rid="2-XXXX")
```

## Step 2: Understand the Table Structure

Before extracting data, understand the schema of the tables involved.

```
# Get table schema and columns via resources
Read resource: deriva://table/Image/schema
Read resource: deriva://table/Subject/schema

# View sample data
query_table(table="Image", limit=5)
query_table(table="Subject", limit=5)
```

Key things to look for:
- Which columns contain the features you need (e.g., image URLs, measurements, labels)
- Foreign key relationships between tables (e.g., Image -> Subject -> Diagnosis)
- Vocabulary columns that contain categorical labels

## Step 3: Choose Your Data Extraction Approach

### Option A: `denormalize_dataset` -- Best for Training

Joins all dataset tables into a single flat DataFrame, ideal for feeding into ML frameworks.

```
denormalize_dataset(
    dataset_rid="2-XXXX",
    include_tables=["Image", "Subject", "Diagnosis"]  # Optional: limit which tables to join
)
```

**What it does:**
- Follows foreign key relationships to join related tables
- Produces a single flat table with columns from all joined tables
- Column names are prefixed with the table name (e.g., `Image.URL`, `Subject.Age`, `Diagnosis.Label`)
- Handles many-to-one and one-to-one relationships automatically

**When to use:** Interactive exploration, quick prototyping, building training DataFrames.

**Parameters:**
- `dataset_rid` (required): The dataset to denormalize
- `include_tables` (optional): List of table names to include. If omitted, all tables in the dataset are joined.

### Option B: `query_table` -- Best for Specific Tables

Query individual tables when you need fine-grained control or only need data from one table.

```
# Get all images in a dataset
query_table(
    table="Image",
    filters=[{"column": "Dataset", "operator": "=", "value": "2-XXXX"}]
)

# Get specific columns
query_table(
    table="Subject",
    columns=["RID", "Age", "Sex", "Species"]
)
```

**When to use:** When you need data from a single table, need to apply filters, or need precise column selection.

### Option C: `download_dataset` -- Best for Production

Downloads the full dataset as a BDBag archive with all assets (files, images).

```
download_dataset(dataset_rid="2-XXXX")
```

**What it does:**
- Downloads all data tables as CSV files
- Downloads all referenced assets (images, files, etc.)
- Creates a reproducible BDBag with checksums
- Preserves the exact dataset state at download time

**When to use:** Production training pipelines, reproducible experiments, when you need actual files (not just URLs).

## Step 4: Use the Data

### Common Column Patterns

After denormalization, columns follow the pattern `TableName.ColumnName`:

| Pattern | Example | Description |
|---------|---------|-------------|
| `Image.URL` | `https://...` | Asset download URL |
| `Image.Filename` | `img_001.png` | Original filename |
| `Subject.Age` | `42` | Numeric feature |
| `Subject.Sex` | `Male` | Categorical feature from vocabulary |
| `Diagnosis.Label` | `Malignant` | Classification label from vocabulary |
| `Measurement.Value` | `3.14` | Numeric measurement |

### Building a Training DataFrame

```python
# After denormalize_dataset returns data:
import pandas as pd

# The denormalized result gives you a flat table
# Select features and labels
features = df[["Subject.Age", "Subject.Sex", "Measurement.Value"]]
labels = df["Diagnosis.Label"]

# Handle categorical variables
features_encoded = pd.get_dummies(features, columns=["Subject.Sex"])

# Split (or use pre-split nested datasets)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_encoded, labels, test_size=0.2)
```

### Working with Image Data

```python
# For image classification tasks
image_urls = df["Image.URL"]
labels = df["Diagnosis.Label"]

# Download images using the URLs, or use download_dataset for batch download
# If using download_dataset, images are already local
```

## Step 5: Version Pinning for Reproducibility

Datasets in DerivaML support versioning. Always pin to a specific version for reproducible experiments.

```
# Check current dataset version
get_record(table="Dataset", rid="2-XXXX")
# Look for the Version field

# Increment version after changes
increment_dataset_version(dataset_rid="2-XXXX")
```

**Best practices for reproducibility:**
- Record the dataset RID and version in your experiment configuration
- Use `create_execution()` to formally track which dataset version was used
- After finalizing a dataset, increment its version before using it in training
- Use nested datasets (train/test/validation splits) with `split_dataset` for consistent splits across experiments
- Download datasets within an execution context so the provenance is automatically recorded

## Tips

- Start with `denormalize_dataset` for quick exploration, then move to `download_dataset` for production.
- Use `include_tables` in denormalize to limit the join to only the tables you need -- this avoids unnecessary data and speeds up the operation.
- If denormalization produces unexpected results, check the foreign key paths between tables using `get_table()`.
- For large datasets, use `query_table` with filters to work with subsets before processing the full dataset.
- Always wrap data preparation in an execution for provenance tracking."""

    @mcp.prompt(
        name="run-ml-execution",
        description="Guide for running ML executions with provenance tracking in DerivaML — the execution lifecycle, context managers, output registration, and nested executions",
    )
    def run_ml_execution_prompt() -> str:
        """run-ml-execution workflow guide."""
        return """# Running an ML Execution with Provenance

Every data transformation, model training run, or analysis in DerivaML should be wrapped in an execution to track inputs, outputs, and provenance.

## Execution Lifecycle

```
create_execution → start → work → stop → upload_execution_outputs
```

## Critical Rules

1. **Every execution needs a workflow** — Create or find one with `create_workflow` first.
2. **Use the context manager in Python** — `with ml.create_execution(config) as exe:` auto-starts and auto-stops.
3. **Upload AFTER the with block** — `exe.upload_execution_outputs()` must be called after exiting the context manager, never inside it.
4. **Use `asset_file_path()` for all outputs** — This both creates the path and registers the file as an output asset. Never manually place files in the working directory.
5. **Failed executions are tracked** — If an exception occurs in the with block, status is set to "Failed" automatically.

## Python API (Recommended)

```python
config = ExecutionConfiguration(
    workflow=workflow,
    datasets=["2-ABC1"],
    assets=["2-DEF2"],
)
with ml.create_execution(config) as exe:
    exe.download_execution_dataset()
    # ... do work ...
    path = exe.asset_file_path("results.csv", description="Model predictions")
    # ... write to path ...
exe.upload_execution_outputs()
```

## MCP Tools

```
create_execution(workflow_rid=..., description=..., dataset_rids=[...])
start_execution(execution_rid=...)
download_execution_dataset(execution_rid=...)
# ... do work ...
asset_file_path(execution_rid=..., filename=..., description=...)
stop_execution(execution_rid=...)
upload_execution_outputs(execution_rid=...)
```

## Key Tools

- `restore_execution` — Re-download a previous execution's assets for debugging
- `add_nested_execution` — Multi-step pipelines with parent-child structure
- `list_nested_executions` / `list_parent_executions` — Navigate execution hierarchy
- `add_asset_type` / `create_asset_table` — Manage asset categories and tables

For the full guide with ExecutionConfiguration details, nested executions, asset management, and inspection tools, read `references/workflow.md`.

---

# Detailed Guide

# Running an ML Execution with Provenance

An execution is the fundamental unit of provenance in DerivaML. Every data transformation, model training run, or analysis should be wrapped in an execution to track what was done, with what inputs, and what outputs were produced.

## Concepts

- **Execution**: A recorded unit of work with inputs (datasets, assets), outputs (files, new data), and metadata (workflow, description, status).
- **Workflow**: A reusable definition of a type of work (e.g., "Image Classification Training"). Executions reference a workflow.
- **ExecutionConfiguration**: Specifies the workflow, input datasets, and assets for an execution.

## Python API: Context Manager Pattern

The recommended approach uses a `with` block that auto-starts and auto-stops the execution:

```python
from deriva.ml import DerivaML, ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)

# 1. Find or create a workflow
workflows = ml.list_workflows()
workflow = ml.create_workflow(
    name="Image Classification Training",
    url="https://github.com/org/repo",
    workflow_type="Training",
    description="Train CNN on labeled image dataset"
)

# 2. Configure the execution
config = ExecutionConfiguration(
    workflow=workflow,
    datasets=["2-ABC1"],          # Dataset RIDs to use as input
    assets=["2-DEF2", "2-GHI3"]  # Individual asset RIDs
)

# 3. Run within context manager
with ml.create_execution(config) as exe:
    # Execution is automatically started

    # Download input datasets
    exe.download_execution_dataset()

    # Do your work...
    results = train_model(exe.working_dir)

    # Write output files using asset_file_path()
    output_path = exe.asset_file_path("model_weights.pt", description="Trained model weights")
    save_model(results, output_path)

    metrics_path = exe.asset_file_path("metrics.json", description="Training metrics")
    save_metrics(results, metrics_path)

# 4. Upload AFTER exiting the context manager
exe.upload_execution_outputs()
```

**Key points about the context manager:**
- `with` block automatically calls `start_execution()` on entry and `stop_execution()` on exit.
- If an exception occurs inside the block, the execution status is set to "Failed".
- You MUST call `upload_execution_outputs()` AFTER exiting the `with` block, not inside it.
- Use `asset_file_path()` to register output files -- this both creates the file path and registers it as an output asset.

## MCP Tools Workflow

For interactive use or when working through the MCP interface:

```
Step 1: Find or create a workflow
  -> query_table(table="Workflow") or create_workflow(...)

Step 2: Create the execution
  -> create_execution(
       workflow_rid="2-XXXX",
       description="Training run on labeled images",
       dataset_rids=["2-ABC1"],
       asset_rids=["2-DEF2"]
     )
  Returns: execution RID

Step 3: Start the execution
  -> start_execution(execution_rid="2-YYYY")

Step 4: Download input data
  -> download_execution_dataset(execution_rid="2-YYYY")

Step 5: Do your work
  (run notebooks, scripts, generate outputs)

Step 6: Register output files
  -> asset_file_path(execution_rid="2-YYYY", filename="results.csv", description="Model predictions")

Step 7: Stop the execution
  -> stop_execution(execution_rid="2-YYYY")

Step 8: Upload outputs
  -> upload_execution_outputs(execution_rid="2-YYYY")
```

## ExecutionConfiguration Details

```python
from deriva.ml import ExecutionConfiguration

config = ExecutionConfiguration(
    workflow=workflow_rid_or_object,   # Required: which workflow
    datasets=["2-ABC1", "2-ABC2"],    # Optional: input dataset RIDs
    assets=["2-DEF1"],                # Optional: input asset RIDs
    description="Run description",     # Optional: execution description
)
```

## Downloading Execution Datasets

Once an execution is started, download all configured input datasets:

```python
# Python API
with ml.create_execution(config) as exe:
    dataset_paths = exe.download_execution_dataset()
    # Returns dict mapping dataset RID -> local directory path
```

```
# MCP tools
download_execution_dataset(execution_rid="2-YYYY")
```

The downloaded data lands in the execution's working directory under a structured layout.

## Registering Output Files

Use `asset_file_path()` to both get the correct output path and register the file as an execution output:

```python
# Python API
output_path = exe.asset_file_path("predictions.csv", description="Model predictions on test set")
# Write your data to output_path
df.to_csv(output_path, index=False)

# For subdirectories
nested_path = exe.asset_file_path("plots/confusion_matrix.png", description="Confusion matrix plot")
```

```
# MCP tools
asset_file_path(execution_rid="2-YYYY", filename="predictions.csv", description="Model predictions")
```

## Useful Inspection and Management Tools

### Get execution info
```python
info = ml.get_execution_info(execution_rid="2-YYYY")
# Returns: workflow, status, datasets, assets, nested executions, timestamps
```

### Update execution status
```python
ml.update_execution_status(execution_rid="2-YYYY", status="Running")
# Valid statuses: Pending, Running, Complete, Failed
```

### Restore a previous execution
```python
ml.restore_execution(execution_rid="2-YYYY")
# Re-downloads execution assets and datasets to local working directory
# Useful for debugging or continuing work from a previous execution
```

### Get execution working directory
```python
working_dir = ml.get_execution_working_dir(execution_rid="2-YYYY")
```

### Nested executions
For multi-step pipelines, create nested executions within a parent:

```python
with ml.create_execution(parent_config) as parent_exe:
    # First step
    with ml.add_nested_execution(parent_exe, step_config) as step1:
        # ... do step 1 work ...

    # Second step
    with ml.add_nested_execution(parent_exe, step2_config) as step2:
        # ... do step 2 work ...
```

```
# MCP tools
add_nested_execution(parent_rid="2-PARENT", workflow_rid="2-STEP1_WF", description="Preprocessing step")
```

### List related executions
```
list_nested_executions(execution_rid="2-YYYY")   # Child executions
list_parent_executions(execution_rid="2-YYYY")    # Parent executions
list_dataset_executions(dataset_rid="2-ABC1")     # Executions that used this dataset
list_asset_executions(asset_rid="2-DEF2")         # Executions that used this asset
```

## Managing Asset Types

Asset Types are vocabulary terms that categorize assets (e.g., "Raw Image", "Trained Model", "Preprocessed CSV").

```
# Create a new asset type
add_asset_type(name="Normalized Image", description="Image after intensity normalization")

# Tag an asset with a type
add_asset_type_to_asset(asset_rid="2-IMG1", asset_type="Normalized Image")

# Remove a type tag
remove_asset_type_from_asset(asset_rid="2-IMG1", asset_type="Normalized Image")
```

## Creating New Asset Tables

When you need a new category of files:

```
create_asset_table(
    table_name="Processed_Image",
    columns=[{"name": "Resolution", "type": "text", "nullok": true, "comment": "Image resolution"}],
    referenced_tables=["Subject"]
)
```

This creates the table with standard asset columns (URL, Filename, Length, MD5, Description) plus any custom columns.

## Tips

- Always wrap work in an execution for provenance tracking.
- Upload outputs AFTER the `with` block exits, never inside it.
- Use `asset_file_path()` for every output file -- do not manually place files in the working directory.
- Set meaningful descriptions on workflows, executions, and output assets.
- For long-running work, use `update_execution_status()` to track progress.
- Use `restore_execution()` to resume or inspect a completed execution's local state.
- Nested executions are ideal for multi-phase pipelines (preprocessing, training, evaluation)."""

    @mcp.prompt(
        name="work-with-assets",
        description="Discover, query, and download Deriva assets (files, images, model weights, CSVs) — find asset tables, check provenance, download files, trace which executions created an asset. For uploading assets, see run-ml-execution.",
    )
    def work_with_assets_prompt() -> str:
        """work-with-assets workflow guide."""
        return """# Working with Deriva Assets

Assets in DerivaML are managed files (images, CSVs, models, etc.) stored in asset tables. Each asset table has standard columns: `URL`, `Filename`, `Length`, `MD5`, `Description`, plus any custom columns.

## Discovering and Querying Assets

Use the catalog resources and query tools to find assets:

```
# List all tables — asset tables have URL, Filename, Length, MD5 columns
query_table(table="Slide_Image", limit=5)

# Search assets with filters
query_table(table_name="Slide_Image", filter={"Subject": "2-A1B2"})

# Look up a specific asset
get_record(table_name="Slide_Image", rid="2-IMG1")
```

See the `query-catalog-data` skill for comprehensive querying patterns.

## Asset Provenance

Every asset can be traced to the execution(s) that produced or consumed it:

```
list_asset_executions(asset_rid="2-IMG1")
# Returns executions with role "Output" (created it) or "Input" (consumed it)
```

## Downloading Assets

```
# Download a specific asset by RID
download_asset(asset_rid="2-IMG1")

# Download all assets in an execution's input configuration
download_execution_dataset(execution_rid="2-EXEC")

# Find where downloaded assets are located
get_execution_working_dir(execution_rid="2-EXEC")
```

## Uploading Assets and Creating Asset Tables

For uploading assets as execution outputs, creating new asset tables, and managing asset types, see the `run-ml-execution` skill. Assets should always be uploaded within an execution context for provenance tracking."""

    @mcp.prompt(
        name="configure-experiment",
        description="Guide for setting up a DerivaML experiment project, adding config groups, or understanding how experiments compose",
    )
    def configure_experiment_prompt() -> str:
        """configure-experiment workflow guide."""
        return """# Configure ML Experiments with hydra-zen and DerivaML

This covers the structure of a DerivaML experiment project: config groups, how they compose, and project setup. For exact Python API patterns for each config type, see the `write-hydra-config` skill.

## Config Groups

| Group | Purpose | File |
|---|---|---|
| `deriva_ml` | Catalog connection (host, catalog ID) | `configs/deriva.py` |
| `datasets` | Dataset RIDs and versions | `configs/datasets.py` |
| `assets` | Pre-trained weights, reference files | `configs/assets.py` |
| `workflow` | What the code does | `configs/workflow.py` |
| `model_config` | Hyperparameters and architecture | `configs/<model>.py` |
| `experiment` | Named combinations of the above | `configs/experiments.py` |
| `multiruns` | Sweeps over experiments/parameters | `configs/multiruns.py` |

## How Experiments Compose

```
Base config (defaults for every group)
  + Experiment overrides (swap specific groups)
    + CLI overrides (fine-tune individual parameters)
```

Example: `uv run deriva-ml-run +experiment=cifar10_quick` loads base defaults, then overrides `model_config` and `datasets` from the experiment preset.

## Critical Rules

1. **Every group needs a default** — `default_deriva`, `default_dataset`, `default_asset`, `default_workflow`, `default_model`
2. **Pin dataset versions** — Use `DatasetSpecConfig(rid="...", version="...")` for reproducibility
3. **Use meaningful names** — `resnet50_extended` not `config2`
4. **Test with `--info`** — `uv run deriva-ml-run +experiment=X --info` to inspect resolved config

## Setup Steps

1. Clone the model template or create `configs/` directory
2. Configure each group in order: `deriva.py` → `datasets.py` → `assets.py` → `workflow.py` → `<model>.py` → `base.py` → `experiments.py`
3. Verify: `uv run deriva-ml-run --info`

For the full project structure, `base.py` template, and setup walkthrough, read `references/workflow.md`.

## Update Experiments.md

After adding or modifying experiment or multirun configs, regenerate `Experiments.md` in the project root. This file is a human-readable summary of all defined experiments — it should always reflect the current state of the config code.

1. **Read the config source** — `experiments.py`, `multiruns.py`, and any model config files they reference
2. **Extract each experiment's** name, config group overrides, key parameters (epochs, lr, batch size, architecture), and purpose
3. **Extract each multirun's** name, overrides, sweep ranges, and description
4. **Write `Experiments.md`** with a quick-reference table, a multiruns table, and a detail section per experiment

Include `Experiments.md` in the same commit as the config changes — it should travel with the code it describes.

### Format

```markdown
# Experiments

Human-readable registry of all defined experiments and multiruns.
Generated from `src/configs/experiments.py` and `src/configs/multiruns.py`.

## Experiments

| Experiment | Model Config | Dataset | Description |
|------------|-------------|---------|-------------|
| `name` | `model_config_name` | `dataset_name` | Brief purpose |

## Multiruns

| Multirun | Overrides | Description |
|----------|----------|-------------|
| `name` | override summary | Brief purpose |

## Experiment Details

### `experiment_name`

- **Config group overrides**: `model_config=X`, `datasets=Y`
- **Parameters**: epochs, channels, batch size, learning rate, etc.
- **Purpose**: Why this experiment exists
```

## Related Skills

- **`write-hydra-config`** — Exact Python API patterns for each config type
- **`run-experiment`** — Pre-flight checklist and CLI commands for running

---

# Detailed Guide

# Configure ML Experiments with hydra-zen and DerivaML

This skill covers the high-level structure of a DerivaML experiment project: what config groups exist, how they compose into experiments, and how to set up a new project. For the exact Python API and code patterns for each config type, see the `write-hydra-config` skill.

## Config Groups

DerivaML experiments are organized into config groups, each controlling a different aspect of the run:

| Config Group | Purpose | File |
|---|---|---|
| `deriva_ml` | Catalog connection (host, catalog ID) | `configs/deriva.py` |
| `datasets` | Which datasets and versions to use | `configs/datasets.py` |
| `assets` | Additional files (weights, predictions) | `configs/assets.py` |
| `workflow` | What the code does (name, type, description) | `configs/workflow.py` |
| `model_config` | Hyperparameters and architecture | `configs/<model>.py` |
| `experiment` | Named combinations of the above | `configs/experiments.py` |
| multiruns | Sweeps over experiments/parameters | `configs/multiruns.py` |
| notebooks | Notebook-specific configs | `configs/<notebook>.py` |

## How Experiments Compose

An experiment is a named combination of config group choices. The composition hierarchy:

```
DerivaModelConfig (base.py)
  ├── default_deriva    (deriva.py)
  ├── default_dataset   (datasets.py)
  ├── default_asset     (assets.py)
  ├── default_workflow   (workflow.py)
  └── default_model     (cifar10_cnn.py)

+experiment=cifar10_quick (experiments.py)
  overrides:
    /model_config → cifar10_quick
    /datasets → cifar10_small_labeled_split
```

The base config defines defaults for every group. Experiments override specific groups. Multiruns sweep over experiments or parameters.

### Execution Flow

```
uv run deriva-ml-run +experiment=cifar10_quick
```

1. Load base config (`DerivaModelConfig`) with its defaults
2. Apply experiment overrides (model_config, datasets, etc.)
3. Apply any CLI overrides (`model_config.epochs=5`)
4. Resolve the final config and execute

### Required Defaults

Every config group needs a `default_*` entry. These are the fallback when no override is specified:

- `default_deriva` — catalog connection
- `default_dataset` — dataset list
- `default_asset` — asset list (typically empty `[]`)
- `default_workflow` — workflow definition
- `default_model` — model hyperparameters

## Project Structure

```
my-project/
  src/
    configs/
      __init__.py           # Re-exports load_configs
      base.py               # create_model_config() — the root config
      deriva.py             # DerivaMLConfig — catalog connections
      datasets.py           # DatasetSpecConfig lists
      assets.py             # Asset RID lists / AssetSpecConfig
      workflow.py            # Workflow definitions
      cifar10_cnn.py        # Model hyperparameters (one file per model type)
      experiments.py        # Named experiment presets
      multiruns.py          # Named multirun sweeps
      multirun_descriptions.py  # Long markdown descriptions for multiruns
      roc_analysis.py       # Notebook config example
    models/
      cifar10_cnn.py        # Model code (the task_function)
  notebooks/
    roc_analysis.ipynb      # Notebook using notebook_config
  pyproject.toml
  uv.lock
```

### `__init__.py`

```python
from deriva_ml.execution import load_configs

load_all_configs = lambda: load_configs("configs")
```

`load_configs()` discovers and imports all config modules in the package automatically.

### `base.py` — The Root Config

```python
from hydra_zen import store
from deriva_ml import DerivaML
from deriva_ml.execution import BaseConfig, DerivaBaseConfig, base_defaults, create_model_config

DerivaModelConfig = create_model_config(
    DerivaML,
    description="Simple model run",
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "default_deriva"},
        {"datasets": "default_dataset"},
        {"assets": "default_asset"},
        {"workflow": "default_workflow"},
        {"model_config": "default_model"},
    ],
)

store(DerivaModelConfig, name="deriva_model")
```

Each default name must match a `name=` in its config group's store.

## Setting Up a New Project

### Step 1: Scaffold the Project

If starting from the model template:
```bash
# Clone the template
git clone https://github.com/informatics-isi-edu/deriva-ml-model-template my-project
cd my-project
uv sync
```

If adding configs to an existing project, create the `configs/` directory structure above.

### Step 2: Configure Each Group

Work through the config groups in order. For each one, see the `write-hydra-config` skill for the exact Python API:

1. **`deriva.py`** — Set your catalog hostname and ID
2. **`datasets.py`** — Add your datasets with RIDs and versions
3. **`assets.py`** — Add any pre-trained weights or reference files
4. **`workflow.py`** — Describe what your code does
5. **`<model>.py`** — Define your model's hyperparameters with `builds()`
6. **`base.py`** — Wire up the defaults
7. **`experiments.py`** — Create named presets
8. **`multiruns.py`** — Define any sweeps

### Step 3: Verify

```bash
# Check config resolves correctly
uv run deriva-ml-run --info

# Check a specific experiment
uv run deriva-ml-run +experiment=my_experiment --info

# Dry run (downloads data, runs model, doesn't persist)
uv run deriva-ml-run +experiment=my_experiment dry_run=True
```

## Running Experiments

See the `run-experiment` skill for the full pre-flight checklist, CLI commands, and troubleshooting. Quick reference:

```bash
uv run deriva-ml-run +experiment=baseline              # Single experiment
uv run deriva-ml-run +multirun=lr_sweep                # Named multirun
uv run deriva-ml-run +experiment=quick,extended --multirun  # Ad-hoc multirun
uv run deriva-ml-run --info                            # Inspect resolved config
```

## Best Practices

- **Pin dataset versions** so runs are reproducible
- **Use meaningful names** — `resnet50_extended` not `config2`
- **Add descriptions everywhere** — they're recorded in execution metadata
- **Test with `dry_run=True`** before production runs
- **Commit before running** — git state is recorded in the execution
- **Use `--info`** to inspect resolved config without running"""

    @mcp.prompt(
        name="run-experiment",
        description="Guide for running experiments with deriva-ml-run — pre-flight checks, dry runs, CLI commands, and result verification",
    )
    def run_experiment_prompt() -> str:
        """run-experiment workflow guide."""
        return """# Run an Experiment with deriva-ml-run

This covers the pre-flight checks and CLI commands for running experiments. Assumes configs are already set up (see `configure-experiment`).

## Pre-Flight Checklist

Complete these before every production run. **Stop and fix any issues.**

1. **Git clean** — `git status` must show no uncommitted changes (commit hash is recorded)
2. **Version current** — Bump with `uv run bump-version patch|minor` if meaningful changes since last run
3. **Lock file valid** — `uv lock --check` must pass
4. **User confirmation** — Present commit, version, branch, experiment name; get explicit approval

## Key Rule: Dry Run First

Always test with `dry_run=True` before a production run:

```bash
uv run deriva-ml-run +experiment=baseline dry_run=True
```

This downloads data and runs the model but does not upload results to the catalog.

## CLI Quick Reference

```bash
# Inspect config without running
uv run deriva-ml-run --info
uv run deriva-ml-run +experiment=baseline --info

# Single experiment
uv run deriva-ml-run +experiment=baseline

# Override host/catalog
uv run deriva-ml-run --host ml-dev.derivacloud.org --catalog 99 +experiment=baseline

# Override parameters
uv run deriva-ml-run +experiment=baseline model_config.learning_rate=0.001

# Named multirun (no --multirun flag needed)
uv run deriva-ml-run +multirun=lr_sweep

# Ad-hoc multirun
uv run deriva-ml-run +experiment=baseline model_config.learning_rate=1e-2,1e-3,1e-4 --multirun
```

## Verify Results

After a run, check the execution was recorded:
- Read `deriva://execution/{rid}` for details
- Read `deriva://chaise-url/Execution/{rid}` for the web UI link
- Verify: status is "Complete", correct datasets linked, output assets attached, git hash matches

For the full guide with troubleshooting table, Hydra override syntax, and multirun details, read `references/workflow.md`.

---

# Detailed Guide

# Run an Experiment with deriva-ml-run

This skill covers the pre-flight checks and CLI commands for running experiments using `deriva-ml-run`. It assumes your hydra-zen configs are already set up (see the `configure-experiment` skill).

## Pre-Flight Checklist

Before running any experiment, complete these checks. **Stop and fix any issues before proceeding.**

### 1. Check Git Status

```bash
git status
```

**Stop if there are uncommitted changes.** Every execution records the git commit hash. Uncommitted changes make runs non-reproducible.

If there are changes:
```bash
git add -A
git commit -m "Prepare for experiment run"
```

### 2. Check Version

```bash
uv run python -c "import my_project; print(my_project.__version__)"
```

If you have made meaningful changes since the last run, bump the version:

```bash
# Use the MCP tool or CLI:
uv run bump-version patch   # For small changes
uv run bump-version minor   # For new features
```

Then commit the bump:
```bash
git add pyproject.toml
git commit -m "Bump version to X.Y.Z"
```

### 3. Verify Lock File

```bash
uv lock --check
```

If this fails, regenerate and commit:
```bash
uv lock
git add uv.lock
git commit -m "Update uv.lock"
```

### 4. Get User Confirmation

Before running, present a summary to the user:

```
Ready to run experiment:
  Commit:  abc1234
  Version: 0.3.1
  Branch:  feature/new-model
  Status:  clean (no uncommitted changes)
  Experiment: +experiments=baseline

Proceed? [y/N]
```

Do not run without confirmation for production (non-dry-run) experiments.

## Verify Configuration

Before running, inspect the resolved configuration:

```bash
uv run deriva-ml-run --info
```

This prints the full resolved Hydra config without executing anything. Verify:

- The correct host and catalog are selected.
- The expected datasets and versions appear.
- Model parameters are what you intend.
- `dry_run` is set as expected.

For a specific experiment:

```bash
uv run deriva-ml-run +experiments=baseline --info
```

## Running Experiments

### Test First with Dry Run

Always run with `dry_run=True` first to validate the pipeline end-to-end without persisting results:

```bash
uv run deriva-ml-run +experiments=baseline dry_run=True
```

This will:
- Resolve and validate the full config.
- Download datasets and assets.
- Run the training function.
- **Not** upload results to the catalog.

### Production Run

```bash
uv run deriva-ml-run +experiments=baseline
```

Or explicitly:

```bash
uv run deriva-ml-run +experiments=baseline dry_run=False
```

### CLI Options

| Option | Purpose | Example |
|---|---|---|
| `--host` | Override the Deriva host | `--host ml-dev.derivacloud.org` |
| `--catalog` | Override the catalog ID | `--catalog 99` |
| `--info` | Print resolved config and exit | `--info` |
| `--multirun` | Enable multirun mode for sweeps | `--multirun` |

### Hydra Overrides

Override any config value from the command line:

```bash
# Select a named experiment
uv run deriva-ml-run +experiments=baseline

# Override dataset config group
uv run deriva-ml-run datasets=cell_images_v3

# Override model config group
uv run deriva-ml-run model_config=resnet50_long

# Override individual parameters
uv run deriva-ml-run model_config.learning_rate=0.001 model_config.epochs=50

# Set dry_run
uv run deriva-ml-run +experiments=baseline dry_run=True

# Combine overrides
uv run deriva-ml-run +experiments=baseline model_config.learning_rate=0.01 dry_run=False
```

### Running Sweeps (Multirun)

For parameter sweeps defined with `multirun_config()`:

```bash
uv run deriva-ml-run +experiments=lr_batch_sweep --multirun
```

For ad-hoc sweeps using Hydra's comma syntax:

```bash
uv run deriva-ml-run +experiments=baseline model_config.learning_rate=1e-2,1e-3,1e-4 --multirun
```

For running multiple named experiments:

```bash
uv run deriva-ml-run +experiments=baseline,long_training --multirun
```

## Verify Results

### Check Executions

After the run completes, verify the execution was recorded. Use the MCP resource:

- Read `deriva://executions` to list recent executions.
- Read `deriva://execution/{rid}` for details on a specific execution.

### View in Chaise

Open the execution in the web interface. The Chaise URL is typically:

```
https://{host}/chaise/record/#{catalog_id}/{schema}:Execution/RID={execution_rid}
```

The MCP resource `deriva://chaise-url/Execution/{rid}` provides the direct URL.

Verify:
- The execution status is "Complete".
- The correct datasets and versions are linked.
- Output assets and metrics are attached.
- The git commit hash matches your expectation.

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Config error: could not find experiments/X` | Experiment name not registered in store | Check `experiments.py` for the `name=` parameter |
| `Connection refused` | Wrong host or host is down | Verify `--host` value, check network |
| `Authentication error` | Expired or missing credentials | Run `deriva-globus-auth-utils login --host {host}` |
| `Dataset not found: RID=...` | RID does not exist in the target catalog | Verify RIDs match the target catalog (dev vs prod) |
| `Version X not found for dataset` | Requested version does not exist | Check available versions with `deriva://dataset/{rid}` |
| `Dirty git state warning` | Uncommitted changes when running | Commit changes before running |
| `Lock file out of date` | `uv.lock` does not match `pyproject.toml` | Run `uv lock` and commit |
| `ModuleNotFoundError` | Dependencies not installed | Run `uv sync` |
| `Multirun requires --multirun flag` | Using `multirun_config` without the flag | Add `--multirun` to the command |
| `dry_run output looks wrong` | Config resolution issue | Use `--info` to inspect the resolved config |"""

    @mcp.prompt(
        name="write-hydra-config",
        description="Write and validate hydra-zen config files for DerivaML — DatasetSpecConfig, asset_store, builds(), experiment_config, multirun_config, with_description. Use when adding, editing, or updating any config in configs/, or when validating that config RIDs and versions match the catalog.",
    )
    def write_hydra_config_prompt() -> str:
        """write-hydra-config workflow guide."""
        return """# Writing Hydra-Zen Config Files for DerivaML

This skill is the authoritative reference for the Python API used in DerivaML hydra-zen configuration files. Every config group has a specific pattern — follow the examples here exactly.

## When to Use This Skill

- Writing a new config file (datasets.py, assets.py, model.py, etc.)
- Adding a new entry to an existing config file
- After creating a catalog entity (dataset, asset, workflow) that should be added to configs
- Fixing or updating existing config entries
- Validating that config RIDs and versions exist in the catalog

For **creating a new project from scratch**, read `references/config-templates.md` — it has complete starter templates for every config file.

After any catalog-modifying action (create_dataset, split_dataset, create_workflow, etc.), proactively offer to update the relevant config file using these patterns.

## Config Groups Overview

| Group | File | Key Import | Registration |
|---|---|---|---|
| `deriva_ml` | `configs/deriva.py` | `from deriva_ml import DerivaMLConfig` | `store(group="deriva_ml")` |
| `datasets` | `configs/datasets.py` | `from deriva_ml.dataset import DatasetSpecConfig` | `store(group="datasets")` |
| `assets` | `configs/assets.py` | `from deriva_ml.execution import with_description` | `store(group="assets")` |
| `workflow` | `configs/workflow.py` | `from deriva_ml.execution import Workflow` | `store(group="workflow")` |
| `model_config` | `configs/<model>.py` | `from hydra_zen import builds` | `store(group="model_config")` |
| `experiment` | `configs/experiments.py` | `from hydra_zen import make_config` | `store(group="experiment", package="_global_")` |
| multiruns | `configs/multiruns.py` | `from deriva_ml.execution import multirun_config` | `multirun_config("name", ...)` |
| notebooks | `configs/<notebook>.py` | `from deriva_ml.execution import notebook_config` | `notebook_config("name", ...)` |

## Datasets (`configs/datasets.py`)

```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

datasets_store = store(group="datasets")

# With description (recommended)
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="28DM", version="0.9.0")],
        "Complete CIFAR-10 dataset with all 10,000 images (5,000 training + 5,000 testing). "
        "Use for full-scale experiments.",
    ),
    name="cifar10_complete",
)

# Multiple datasets in one config
datasets_store(
    with_description(
        [
            DatasetSpecConfig(rid="28FC", version="0.4.0"),
            DatasetSpecConfig(rid="28FP", version="0.4.0"),
        ],
        "Small training (500) and testing (500) sets for rapid prototyping.",
    ),
    name="cifar10_small_both",
)

# Empty dataset list (for notebooks that don't need datasets)
datasets_store([], name="no_datasets")

# REQUIRED: default_dataset — plain list, no with_description()
# (with_description creates DictConfig which can't merge with BaseConfig's ListConfig)
datasets_store(
    [DatasetSpecConfig(rid="28DY", version="0.9.0")],
    name="default_dataset",
)
```

**Key rules:**
- `version` is **required** — always a semver string like `"0.9.0"`, not an integer
- Use `with_description()` for non-default configs
- Default configs use plain lists (no `with_description`) for merge compatibility
- Find the current version via the `deriva://dataset/{rid}` MCP resource
- If data has changed since the version was created, call `increment_dataset_version` first

## Assets (`configs/assets.py`)

```python
from hydra_zen import store
from deriva_ml.execution import with_description

asset_store = store(group="assets")

# Plain RID strings (most common)
asset_store(
    with_description(
        ["3WS6", "3X20"],
        "Prediction probabilities from quick (3 epochs) vs extended (50 epochs) training. "
        "Use with ROC analysis notebook.",
    ),
    name="roc_quick_vs_extended",
)

# AssetSpecConfig with caching (for large immutable files like model weights)
from deriva_ml.asset.aux_classes import AssetSpecConfig

asset_store(
    with_description(
        [AssetSpecConfig(rid="3WS2", cache=True)],
        "Pre-trained weights from cifar10_quick (execution 3WR0, 3 epochs). "
        "Cached locally (~50MB) to avoid re-downloading.",
    ),
    name="quick_weights",
)

# REQUIRED: default_asset — empty list, plain (no with_description)
asset_store([], name="default_asset")

# Alias for clarity
asset_store([], name="no_assets")
```

**Key rules:**
- Plain RID strings for simple references: `["3WS6", "3X20"]`
- `AssetSpecConfig(rid=..., cache=True)` for large files that shouldn't re-download
- Default/empty configs use plain lists for merge compatibility
- Assets are typically execution outputs — note the source execution RID in the description

## Workflow (`configs/workflow.py`)

```python
from hydra_zen import store, builds
from deriva_ml.execution import Workflow

# Build the workflow config class
Cifar10CNNWorkflow = builds(
    Workflow,
    name="CIFAR-10 2-Layer CNN",
    workflow_type=["Training", "Image Classification"],  # string or list of strings
    description=\"\"\"
Train a 2-layer convolutional neural network on CIFAR-10 image data.

## Architecture
- **Conv Layer 1**: 3 -> 32 channels, 3x3 kernel, ReLU, MaxPool 2x2
- **Conv Layer 2**: 32 -> 64 channels, 3x3 kernel, ReLU, MaxPool 2x2
- **FC Layer**: 64x8x8 -> 128 hidden units -> 10 classes
\"\"\".strip(),
    populate_full_signature=True,
)

workflow_store = store(group="workflow")

# REQUIRED: default_workflow
workflow_store(Cifar10CNNWorkflow, name="default_workflow")

# Named variants
workflow_store(Cifar10CNNWorkflow, name="cifar10_cnn")
```

**Key rules:**
- Use `builds(Workflow, ...)` with `populate_full_signature=True`
- `workflow_type` can be a single string or a list of strings
- `description` supports markdown — use it for architecture details
- Git URL and commit hash are captured automatically at runtime

## Model Config (`configs/<model>.py`)

```python
from hydra_zen import builds, store
from models.cifar10_cnn import cifar10_cnn

# Build the base config — zen_partial=True is critical
# (execution context is injected at runtime)
Cifar10CNNConfig = builds(
    cifar10_cnn,
    conv1_channels=32,
    conv2_channels=64,
    hidden_size=128,
    dropout_rate=0.0,
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    weight_decay=0.0,
    populate_full_signature=True,
    zen_partial=True,
)

model_store = store(group="model_config")

# REQUIRED: default_model
model_store(
    Cifar10CNNConfig,
    name="default_model",
    zen_meta={
        "description": (
            "Default CIFAR-10 CNN: 32->64 channels, 128 hidden units, 10 epochs, "
            "batch size 64, lr=1e-3. Balanced config for standard training runs."
        )
    },
)

# Variants override specific parameters
model_store(
    Cifar10CNNConfig,
    name="cifar10_quick",
    epochs=3,
    batch_size=128,
    zen_meta={
        "description": (
            "Quick training: 3 epochs, batch 128. Use for rapid iteration, "
            "debugging, and verifying the training pipeline works correctly."
        )
    },
)

model_store(
    Cifar10CNNConfig,
    name="cifar10_extended",
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=256,
    dropout_rate=0.25,
    weight_decay=1e-4,
    learning_rate=1e-3,
    epochs=50,
    zen_meta={
        "description": (
            "Extended training for best accuracy: Large model (64->128 ch, 256 hidden), "
            "regularization (dropout 0.25, weight decay 1e-4), 50 epochs."
        )
    },
)
```

**Key rules:**
- `zen_partial=True` is required — the execution context is injected later
- `populate_full_signature=True` exposes all constructor params to Hydra
- `zen_meta={"description": "..."}` documents the config variant
- Override individual params when registering variants (no need to rebuild)

## Experiments (`configs/experiments.py`)

```python
from hydra_zen import make_config, store
from configs.base import DerivaModelConfig

# package="_global_" is set on the store, not on make_config
experiment_store = store(group="experiment", package="_global_")

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Quick CIFAR-10 training: 3 epochs, 32->64 channels, batch size 128",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_quick",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_extended"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Extended CIFAR-10 training: 50 epochs, 64->128 channels, full regularization",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_extended",
)
```

**Key rules:**
- `package="_global_"` goes on the `store()` call
- `bases=(DerivaModelConfig,)` inherits from the base config
- `hydra_defaults` uses `{"override /group": "name"}` syntax
- `"_self_"` must be first in the defaults list
- `description` is a plain string on `make_config()` (not zen_meta)

## Multiruns (`configs/multiruns.py`)

```python
from deriva_ml.execution import multirun_config

multirun_config(
    "quick_vs_extended",
    overrides=[
        "+experiment=cifar10_quick,cifar10_extended",
    ],
    description=\"\"\"## Quick vs Extended Training Comparison

| Config | Epochs | Architecture | Regularization |
|--------|--------|--------------|----------------|
| quick | 3 | 32->64 channels | None |
| extended | 50 | 64->128 channels | Dropout 0.25, WD 1e-4 |

**Objective:** Compare training duration vs accuracy tradeoff.
\"\"\",
)

# Hyperparameter sweep
multirun_config(
    "lr_sweep",
    overrides=[
        "+experiment=cifar10_quick",
        "model_config.epochs=10",
        "model_config.learning_rate=0.0001,0.001,0.01,0.1",
    ],
    description="Learning rate sweep: 4 values from 1e-4 to 1e-1 on quick config.",
)

# Grid search (N x M runs)
multirun_config(
    "lr_batch_grid",
    overrides=[
        "+experiment=cifar10_quick",
        "model_config.epochs=10",
        "model_config.learning_rate=0.001,0.01",
        "model_config.batch_size=64,128",
    ],
    description="LR x batch size grid: 2x2 = 4 total runs.",
)
```

**Key rules:**
- First arg is the multirun name (string), not a keyword
- `overrides` is a list of Hydra override strings (comma-separated values for sweeps)
- `description` supports rich markdown (tables, headers) — shown on the parent execution
- No `--multirun` flag needed when using `multirun_config` — it's automatic
- CLI usage: `uv run deriva-ml-run +multirun=lr_sweep`

## Notebook Configs (`configs/<notebook>.py`)

```python
from dataclasses import dataclass
from deriva_ml.execution import BaseConfig, notebook_config

@dataclass
class ROCAnalysisConfig(BaseConfig):
    \"\"\"Custom parameters for this notebook.\"\"\"
    show_per_class: bool = True
    confidence_threshold: float = 0.0

notebook_config(
    "roc_analysis",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_quick_vs_extended", "datasets": "no_datasets"},
    description="ROC curve analysis (default: quick vs extended training)",
)

# Simple notebook with no custom parameters
notebook_config(
    "my_analysis",
    defaults={"assets": "my_assets"},
)
```

In the notebook:
```python
from deriva_ml.execution import run_notebook

ml, execution, config = run_notebook("roc_analysis")
# config.assets, config.show_per_class, config.confidence_threshold are available
```

## Base Config (`configs/base.py`)

```python
from hydra_zen import store
from deriva_ml import DerivaML
from deriva_ml.execution import BaseConfig, DerivaBaseConfig, base_defaults, create_model_config

DerivaModelConfig = create_model_config(
    DerivaML,
    description="Simple model run",
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "default_deriva"},
        {"datasets": "default_dataset"},
        {"assets": "default_asset"},
        {"workflow": "default_workflow"},
        {"model_config": "default_model"},
    ],
)

store(DerivaModelConfig, name="deriva_model")
```

**Key rule:** Each default name must match a `name=` in the corresponding config group's store.

## Deriva Connection (`configs/deriva.py`)

```python
from hydra_zen import store
from deriva_ml import DerivaMLConfig

deriva_store = store(group="deriva_ml")

# REQUIRED: default_deriva
deriva_store(
    DerivaMLConfig,
    name="default_deriva",
    hostname="localhost",
    catalog_id=6,
    use_minid=False,
    zen_meta={
        "description": (
            "Local development catalog (localhost:6) with CIFAR-10 data. "
            "Schema: cifar10_10k."
        )
    },
)
```

## Config `__init__.py`

The `__init__.py` must re-export `load_configs` so all config modules are discovered:

```python
from deriva_ml.execution import load_configs

load_all_configs = lambda: load_configs("configs")
```

All config modules in the package are imported automatically by `load_configs()`.

## Description Mechanisms

Two mechanisms exist — use the right one for the context:

| Config Type | Mechanism | Example |
|---|---|---|
| Lists (datasets, assets) | `with_description(items, "...")` | `with_description([DatasetSpecConfig(...)], "Training images v3")` |
| `builds()` configs (models, connections) | `zen_meta={"description": "..."}` | `store(Config, name="x", zen_meta={"description": "..."})` |
| Experiments | `description=` param on `make_config()` | `make_config(..., description="Quick training run")` |
| Multiruns | `description=` param on `multirun_config()` | `multirun_config("name", ..., description="...")` |
| Notebooks | `description=` param on `notebook_config()` | `notebook_config("name", ..., description="...")` |

Descriptions are recorded in execution metadata and make experiments self-documenting. Before writing descriptions, look up catalog details via `deriva://dataset/{rid}` or `deriva://asset/{rid}`.

### Good Descriptions

- **Specific**: "ResNet-50 with 3-class output head, trained with cosine annealing LR schedule"
- **Quantified**: "4,500 histopathology tiles at 224x224, balanced across 3 subtypes"
- **Purposeful**: "Validation set held out by patient ID to prevent data leakage"
- **Version-aware**: "Frozen at version 3, which excludes 12 QC-failed slides"

## Validating Configs Against the Catalog

Before running experiments, validate that all RIDs and versions in config files actually exist in the connected catalog. This catches common errors like typos in RIDs, stale versions, or configs pointing at the wrong catalog.

### Validation Checklist

For each config file, check:

| Config Type | What to Validate | MCP Tool / Resource |
|---|---|---|
| `DatasetSpecConfig(rid=..., version=...)` | RID exists, version exists | `deriva://dataset/{rid}` |
| Asset RID strings `["3WS6"]` | RID exists in an asset table | `validate_rids(rids=[...])` |
| `AssetSpecConfig(rid=...)` | RID exists | `validate_rids(rids=[...])` |
| `workflow_type="Training"` | Workflow type term exists | `deriva://catalog/workflow-types` |

### Validation Workflow

1. **Connect to the catalog** using the same `deriva_ml` config the experiment will use
2. **Read the config files** and extract all RIDs and versions
3. **Validate RIDs** — use `validate_rids` to batch-check that all RIDs exist
4. **Check dataset versions** — for each `DatasetSpecConfig`, read `deriva://dataset/{rid}` and verify the version exists. If the version is older than `current_version`, the config may be using stale data
5. **Report mismatches** — list any RIDs that don't exist, versions that are missing, or versions that are behind current

### Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `Dataset not found: RID=...` | RID doesn't exist in target catalog | Verify RID against correct catalog (dev vs prod) |
| `Version X not found` | Version never created | Use `get_current_version` to find latest, or `increment_dataset_version` |
| Stale version | Data changed since version was created | Call `increment_dataset_version`, then update config |
| Wrong catalog | Config RIDs are from a different catalog | Check `deriva_ml` config group — are you pointing at the right host/catalog? |

### Proactive Validation

After any catalog-modifying action (create_dataset, split_dataset, increment_dataset_version, etc.), proactively:

1. Note the new RID, version, and description
2. Check if existing config files reference the affected entity
3. Offer to update configs if versions are stale or new entities should be added
4. Present changes for approval before modifying files
5. Remind the user to commit config changes before running experiments"""

    @mcp.prompt(
        name="api-naming-conventions",
        description="Reference for DerivaML API naming conventions — when to use lookup_ vs find_ vs list_ vs get_ vs create_ vs add_ method prefixes. Use when choosing the right method name or understanding why a method is named the way it is.",
    )
    def api_naming_conventions_prompt() -> str:
        """api-naming-conventions workflow guide."""
        return """# DerivaML API Naming Conventions

Consistent naming conventions for API methods ensure discoverability and predictable behavior. Use this reference when calling DerivaML tools or writing scripts.

## Method Prefixes

### `lookup_*(identifier)` -- Single Entity by Identifier

Returns a single entity. Raises an error if not found.

| Method | Description |
|--------|-------------|
| `lookup_dataset` | Find dataset by RID |
| `lookup_asset` | Find asset by RID |
| `lookup_term` | Find vocabulary term by name or RID |
| `lookup_workflow` | Find workflow by name or RID |
| `lookup_feature` | Find feature by name |

**Behavior**: Expects exactly one result. Fails loudly if the entity doesn't exist. Use when you have a known identifier and need the entity.

### `find_*(filters)` -- Search with Filters

Returns an iterable of matching entities. Empty result is valid (not an error).

| Method | Description |
|--------|-------------|
| `find_datasets` | Search datasets by type, name, etc. |
| `find_assets` | Search assets by type, metadata |
| `find_features` | Search features by target table, vocabulary |

**Behavior**: Returns zero or more results. Use for search and discovery when you don't know the exact identifier.

### `list_*(context)` -- All Items in Context

Returns all items of a type within a given context.

| Method | Description |
|--------|-------------|
| `list_vocabulary_terms` | All terms in a vocabulary |
| `list_tables` | All tables in a schema |
| `list_assets` | All assets of a type |
| `list_dataset_members` | All members of a dataset |
| `list_dataset_parents` | All parent datasets |
| `list_dataset_children` | All child datasets |
| `list_nested_executions` | All nested executions |
| `list_parent_executions` | All parent executions |

**Behavior**: Returns a complete list. No filtering -- returns everything in scope.

### `get_*(params)` -- Data with Transformation

Returns data in a specific format or with transformation applied.

| Method | Description |
|--------|-------------|
| `get_table` | Get table schema/definition |
| `get_table_sample_data` | Get sample rows from a table |
| `get_record` | Get a specific record by RID |
| `get_dataset_spec` | Get dataset specification |
| `get_execution_info` | Get execution details |
| `get_execution_working_dir` | Get execution working directory path |

**Behavior**: Returns a specific data type or transformed view. Use when you need data in a particular format.

### `create_*(params)` -- New Entity

Creates a new entity and returns it.

| Method | Description |
|--------|-------------|
| `create_dataset` | Create new dataset |
| `create_workflow` | Create new workflow |
| `create_feature` | Create new feature |
| `create_table` | Create new table |
| `create_vocabulary` | Create new vocabulary |
| `create_execution` | Create new execution |
| `create_execution_dataset` | Create dataset from execution outputs |

**Behavior**: Creates and returns the new entity. Fails if entity already exists (where applicable).

### `add_*(target, item)` -- Add to Existing

Adds an item to an existing entity.

| Method | Description |
|--------|-------------|
| `add_dataset_members` | Add members to a dataset |
| `add_dataset_type` | Add a type to a dataset |
| `add_dataset_element_type` | Add element type to dataset |
| `add_dataset_child` | Add child relationship |
| `add_asset_type` | Add type to asset table |
| `add_asset_type_to_asset` | Assign type to specific asset |
| `add_term` | Add term to vocabulary |
| `add_synonym` | Add synonym to term |
| `add_feature_value` | Add feature value |
| `add_feature_value_record` | Add individual feature value record |
| `add_visible_column` | Add column to visible columns |
| `add_visible_foreign_key` | Add FK to visible foreign keys |
| `add_column` | Add column to table |
| `add_nested_execution` | Add nested execution |
| `add_workflow_type` | Add workflow type |

**Behavior**: Modifies an existing entity. Returns None.

### `delete_*` / `remove_*` -- Remove Items

Removes entities or relationships.

| Method | Description |
|--------|-------------|
| `delete_dataset` | Delete a dataset |
| `delete_dataset_members` | Remove members from dataset |
| `delete_dataset_type_term` | Remove type from dataset |
| `delete_feature` | Delete a feature |
| `delete_term` | Delete vocabulary term |
| `remove_asset_type_from_asset` | Remove type from asset |
| `remove_dataset_type` | Remove dataset type |
| `remove_synonym` | Remove synonym from term |
| `remove_visible_column` | Remove from visible columns |
| `remove_visible_foreign_key` | Remove from visible FKs |

**Behavior**: Removes the specified entity or relationship. Returns None.

### `set_*` -- Set/Update Properties

Sets a property on an existing entity.

| Method | Description |
|--------|-------------|
| `set_table_description` | Set table description |
| `set_column_description` | Set column description |
| `set_table_display_name` | Set table display name |
| `set_column_display_name` | Set column display name |
| `set_row_name_pattern` | Set row name display pattern |
| `set_visible_columns` | Set all visible columns |
| `set_visible_foreign_keys` | Set all visible FKs |
| `set_dataset_description` | Set dataset description |
| `set_execution_description` | Set execution description |
| `set_workflow_description` | Set workflow description |
| `set_display_annotation` | Set display annotation |
| `set_table_display` | Set table display config |
| `set_column_display` | Set column display config |
| `set_column_nullok` | Set column nullability |
| `set_default_schema` | Set default schema |
| `set_active_catalog` | Set active catalog |

**Behavior**: Overwrites the specified property. Returns None.

## Parameter Naming

- Use semantic names: `dataset_rid`, `asset_rid`, `execution_rid`
- Table/column parameters: `table_name`, `column_name`, `feature_name`, `vocab_name`
- Boolean parameters: use positive names with `bool` type (e.g., `cache=True`, `dry_run=False`)

## Return Types Summary

| Prefix | Returns |
|--------|---------|
| `lookup_` | Single entity (raises on not found) |
| `find_` | Iterable of entities (may be empty) |
| `list_` | List or dict of entities |
| `get_` | Specific data type |
| `create_` | Created entity |
| `add_` | None |
| `delete_` / `remove_` | None |
| `set_` | None |"""

    @mcp.prompt(
        name="catalog-operations-workflow",
        description="ALWAYS use when performing Deriva catalog operations that modify data (dataset creation, splitting, ETL, feature loading, data import). Generate a committed Python script for full code provenance tracking instead of using interactive MCP tools.",
    )
    def catalog_operations_workflow_prompt() -> str:
        """catalog-operations-workflow workflow guide."""
        return """# Script-Based Workflow for Catalog Operations

For catalog operations that need to be reproducible, auditable, or shareable — dataset creation, splitting, ETL, feature creation, and data loading — use a committed script rather than interactive MCP tools.

## When to Use Scripts vs Interactive MCP

| Situation | Approach |
|-----------|----------|
| One-off exploration, quick queries, checking state | Interactive MCP tools |
| Setting descriptions, display names, annotations | Interactive MCP tools |
| Operations you'll need to reproduce or share | Committed script |
| Dataset creation, splitting, ETL, data loading | Committed script |
| Operations others need to audit or re-run | Committed script |

The key distinction: DerivaML records the git commit hash with every execution. A committed script gives the execution record a code reference that anyone can trace back. Interactive MCP operations have no such reference.

## Workflow: Develop, Test, Commit, Run

### 1. Generate Python Script

Create a script in the `scripts/` directory using the DerivaML Python API:

```python
#!/usr/bin/env python
\"\"\"<Description of what this script does>.\"\"\"

import argparse
from deriva_ml import DerivaML, ExecutionConfiguration

def main():
    parser = argparse.ArgumentParser(description="<Script description>")
    parser.add_argument("--dry-run", action="store_true", help="Test without creating records")
    # Add script-specific arguments
    args = parser.parse_args()

    ml = DerivaML(hostname="...", catalog_id=...)

    config = ExecutionConfiguration(
        workflow_rid="...",
        description="...",
    )

    with ml.create_execution(config, dry_run=args.dry_run) as execution:
        # Perform operations
        ...
        execution.upload_execution_outputs()

if __name__ == "__main__":
    main()
```

Key elements:
- `argparse` for CLI arguments
- `--dry-run` flag for testing without side effects
- `ExecutionConfiguration` context manager for provenance tracking
- `execution.upload_execution_outputs()` to record results

### 2. Test with Dry Run

```bash
python scripts/my_operation.py --dry-run
```

Verify the script works correctly without creating any records in the catalog.

### 3. Commit Script

```bash
git add scripts/my_operation.py
git commit -m "Add script for <operation description>"
```

The script MUST be committed before running for real. This ensures the git commit hash in the execution record points to the actual code that was run.

### 4. Run for Real

```bash
python scripts/my_operation.py
```

The execution record will capture:
- Git commit hash
- Repository URL
- Input datasets and their versions
- Output assets and datasets
- Execution parameters

## Common Script Patterns

### Dataset Creation

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    dataset = execution.create_dataset(
        name="training-v1",
        dataset_type="Training",
        description="Training dataset with 10,000 balanced images.",
    )
    execution.add_dataset_members(dataset.rid, member_rids)
    execution.upload_execution_outputs()
```

### Dataset Splitting

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    splits = execution.split_dataset(
        source_rid="1-ABC4",
        splits={"train": 0.8, "val": 0.1, "test": 0.1},
        stratify_by="Diagnosis",
        group_by="Subject",
    )
    execution.upload_execution_outputs()
```

### Feature Creation and Population

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    feature = execution.create_feature(
        name="Severity",
        target_table="Image",
        vocabulary="Severity_Grade",
        description="Severity grading for chest X-ray findings.",
    )
    for image_rid, severity in annotations.items():
        execution.add_feature_value(feature.name, image_rid, severity)
    execution.upload_execution_outputs()
```

### ETL / Data Loading

```python
with ml.create_execution(config, dry_run=args.dry_run) as execution:
    # Load data from external source
    data = load_external_data(args.source)
    # Transform and insert
    for record in transform(data):
        execution.insert_record("TargetTable", record)
    execution.upload_execution_outputs()
```

## When MCP Tools Are Still Appropriate

Not everything needs a script. Use MCP tools directly for:

- **Exploratory work**: Browsing catalog structure, querying data, checking entity states
- **One-time admin tasks**: Setting descriptions, display names, annotations
- **Read-only operations**: Listing datasets, viewing features, checking versions
- **Quick debugging**: Inspecting specific records, checking execution status

## When to Suggest Scripts

When a user asks to perform a data-modifying operation interactively, suggest:

> "For full provenance tracking, I recommend creating a script that we can commit. This ensures the operation is reproducible and the execution record will reference the exact code. Shall I generate the script?"

Then follow the Develop, Test, Commit, Run workflow."""

    @mcp.prompt(
        name="derivaml-coding-guidelines",
        description="Coding standards and project setup for DerivaML projects — uv/pyproject.toml configuration, Git workflow, Google docstrings, ruff linting, type hints. Use when setting up a new project or establishing development practices.",
    )
    def derivaml_coding_guidelines_prompt() -> str:
        """coding-guidelines workflow guide."""
        return """# Coding Guidelines for DerivaML Projects

This skill covers the recommended workflow, coding standards, and best practices for developing DerivaML-based machine learning projects.

## Repository Setup

### Start with Your Own Repository

Every DerivaML project should live in its own Git repository. Do not develop inside the DerivaML library itself.

```bash
mkdir my-ml-project
cd my-ml-project
git init
uv init
```

### Use `uv` as the Package Manager

DerivaML projects use `uv` for dependency management. Always:

- Define dependencies in `pyproject.toml`.
- Use `uv add` to add dependencies (not `pip install`).
- **Commit `uv.lock`** to version control. This ensures reproducible environments across machines and over time.

```bash
uv add deriva-ml
uv add torch torchvision  # ML framework deps
```

### Typical `pyproject.toml` Structure

```toml
[project]
name = "my-ml-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "deriva-ml>=0.5.0",
]

[project.optional-dependencies]
jupyter = ["jupyterlab", "papermill"]
dev = ["pytest", "ruff"]

[dependency-groups]
jupyter = ["jupyterlab", "papermill"]

[project.scripts]
deriva-ml-run = "my_project.configs:main"
```

## Environment Management

- Use `uv sync` to install the project in development mode.
- Use `uv sync --group=jupyter` when you need Jupyter support.
- Use `uv run` to execute commands within the project environment.
- Never install packages globally for project work.

## Git Workflow

### Branch Strategy

- Use feature branches for all work: `git checkout -b feature/add-segmentation-model`.
- Keep `main` clean and passing.
- Use pull requests for code review.

### Commit Before Running

**Always commit your code before running an experiment.** DerivaML records the git state (commit hash, branch, dirty status) in the execution metadata. If you run with uncommitted changes:

- The execution is marked as having a dirty working tree.
- Reproducing the exact run later becomes difficult or impossible.

### Version Bumping

Use `bump-version` (via the MCP tool or CLI) before production runs:

```bash
uv run bump-version patch  # 0.1.0 -> 0.1.1
```

Versioning conventions:
- **patch**: Bug fixes, small parameter tweaks.
- **minor**: New experiment configurations, new model architectures.
- **major**: Breaking changes to the training pipeline or data format.

Commit the version bump before running.

## Coding Standards

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def train_model(config: ModelConfig, dataset_path: Path) -> dict[str, float]:
    \"\"\"Train the classification model on the provided dataset.

    Args:
        config: Model hyperparameters and architecture configuration.
        dataset_path: Path to the downloaded and extracted dataset.

    Returns:
        Dictionary of metric names to final values, e.g.
        {"accuracy": 0.95, "loss": 0.12}.

    Raises:
        ValueError: If the dataset contains no samples.
    \"\"\"
```

### Type Hints

Use type hints on all function signatures. Use modern Python typing (Python 3.11+):

```python
from pathlib import Path

def load_images(directory: Path, extensions: list[str] | None = None) -> list[Path]:
    ...

def compute_metrics(predictions: dict[str, list[float]]) -> dict[str, float]:
    ...
```

### Code Formatting and Linting

- Use `ruff` for linting and formatting.
- Configure in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

## Semantic Versioning

DerivaML projects follow semantic versioning. The version is recorded in every execution, creating a direct link between code and results.

| Change Type | Version Bump | Example |
|---|---|---|
| Fix a bug in data loading | patch | 0.1.0 -> 0.1.1 |
| Add a new model architecture | minor | 0.1.1 -> 0.2.0 |
| Restructure the config system | major | 0.2.0 -> 1.0.0 |

The version lives in `pyproject.toml` and is managed by the `bump-version` tool.

## Notebook Guidelines

Never commit notebook outputs to Git -- install `nbstripout` to strip them automatically. Keep each notebook focused on one task, and ensure it runs end-to-end with Restart & Run All.

See `setup-notebook-environment` for full environment setup and `run-notebook` for execution tracking.

## Experiments and Data

- Define experiment configs in hydra-zen (see `configure-experiment`)
- Always test with `dry_run=True` before production runs (see `run-experiment`)
- Never commit data files to Git -- store in Deriva catalogs and pin dataset versions (see `dataset-versioning`)
- Wrap all data operations in executions for provenance (see `run-ml-execution`)

## Extensibility

Prefer inheritance and composition over modifying DerivaML library code:

```python
from deriva.ml import DerivaML

class MyProjectML(DerivaML):
    \"\"\"Extended DerivaML with project-specific helpers.\"\"\"

    def load_training_data(self, dataset_rid: str) -> pd.DataFrame:
        ...
```

## Summary Checklist

- [ ] Own repository with `uv` and committed `uv.lock`
- [ ] Feature branches and pull requests
- [ ] Google docstrings and type hints on all public APIs
- [ ] `nbstripout` installed for notebooks
- [ ] No data files in Git -- store in Deriva catalogs
- [ ] Version bumped and committed before production runs"""

    @mcp.prompt(
        name="generate-descriptions",
        description="ALWAYS use when creating any Deriva catalog entity (dataset, execution, feature, table, column, vocabulary, workflow) and the user hasn't provided a description. Auto-generate a meaningful description from context.",
    )
    def generate_descriptions_prompt() -> str:
        """generate-descriptions workflow guide."""
        return """# Generate Descriptions for Catalog Entities

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
5. Create the entity with the approved description"""

    @mcp.prompt(
        name="maintain-experiment-notes",
        description="ALWAYS use after any significant experiment decision — dataset creation, split strategy, feature selection, hyperparameter choice, architecture selection, or catalog structure change. Append the decision and rationale to experiment-decisions.md automatically.",
    )
    def maintain_experiment_notes_prompt() -> str:
        """maintain-experiment-notes workflow guide."""
        return """# Capture Experiment Design Decisions

Automatically record experiment design decisions and their rationale in `experiment-decisions.md` in the project root. This file is a shared, persistent record of *why* the experiment was designed the way it was — not a session log. When a new team member checks out the repository, reading this file should give them the full context behind every significant choice.

## What This Is NOT

This is not a session log, a task list, or a changelog. It does not track who did what or when sessions started and ended. It captures *decisions and reasoning* — the kind of institutional knowledge that normally lives only in someone's head and is lost when they move on.

## When to Write

Append an entry after any of these events:

- **Dataset composition**: Why these members were included/excluded, why this size, why these types
- **Split strategy**: Why this split ratio, why stratified, why patient-level vs image-level
- **Feature selection**: Why this feature was created (or reused), what it represents, why this vocabulary
- **Architecture/model choice**: Why this model, why these hyperparameters, what alternatives were considered
- **Catalog structure changes**: Why a table was added/extended, why a column was added, why a FK was created
- **Configuration choices**: Why this hydra-zen config, why these overrides, why this multirun setup
- **Problem resolution**: What went wrong and why the chosen fix was correct (not just "fixed it")

Do NOT write entries for routine operations that don't involve a choice — querying data, reading schemas, listing datasets. Only capture moments where an alternative existed and a direction was chosen.

## How to Write

Append to `experiment-decisions.md` silently — do not ask the user for permission or tell them you're updating it. This should be invisible. If the file doesn't exist, create it with the header.

Each entry is a short block:

```markdown
### <Concise decision title>

<1-3 sentences explaining what was decided and why. Include the alternatives that were considered and rejected. Reference catalog entities by RID where relevant.>
```

Keep entries concise. The goal is density of reasoning, not completeness of description. Someone scanning the file should quickly understand the shape of the decisions.

## File Structure

```markdown
# Experiment Design Decisions

Accumulated rationale for experiment design choices in this project.
Each entry captures what was decided and why.

---

### Patient-level splitting to prevent data leakage

Split dataset `2-B4C8` at the patient level (stratified by Subject RID) rather than
random image-level splitting. Multiple images per patient would leak information
between train and test if split at the image level. Used 80/20 ratio with seed 42.

### Reused Disease_Classification feature instead of creating Diagnosis

User requested a "Diagnosis" feature on Image, but Disease_Classification (RID: 2-XXXX)
already exists with 3,200 values and 8 disease terms. Creating a separate feature
would fragment annotations. Added "Fundus_Dystrophy" as a new term to the existing
vocabulary instead.

### Learning rate 0.001 selected from sweep

Sweep over [0.0001, 0.001, 0.01, 0.1] showed 0.001 achieved best validation AUC (0.94)
while 0.01 showed training instability after epoch 15. 0.0001 converged too slowly
for the 50-epoch budget.

### Added Enrollment_Date to Subject instead of creating Patient table

User requested a Patient table with Name, Age, Gender, Enrollment_Date. Subject table
(RID: 1-4W2G) already has Name, Age, Gender with 1,247 records and 8 incoming FKs.
Creating a duplicate table would orphan all existing relationships. Added Enrollment_Date
column to Subject instead.
```

## Relationship to Other Files

- **`experiments.md`**: Describes *what* each experiment configuration does (parameters, inputs, outputs). The experiment-decisions file explains *why* those configurations exist.
- **CLAUDE.md**: Project-level instructions for Claude. Reference experiment-decisions.md from CLAUDE.md so new sessions pick up context.
- **Hydra configs**: Define the experiment parameters. The decisions file explains why those parameter values were chosen.

## Commit Prompting

The decisions file is only useful to the team if it gets committed. After writing 3 or more entries during a session, or when the conversation reaches a natural pause (the user has finished a workflow, is about to start something new, or the topic shifts), suggest committing:

> "You've accumulated several experiment design decisions this session. Want me to commit `experiment-decisions.md` so the team has the rationale on record?"

If the user says yes, commit just `experiment-decisions.md` with a message like "Record experiment design decisions" — do not bundle it with unrelated changes. If the user says no or not yet, don't ask again until more entries are added.

Do not prompt after every single entry — that would be annoying. Wait for a batch to accumulate or a natural break in the work.

## Writing Guidelines

- Lead with the decision, not the process that led to it
- Always state what was *rejected* and why — "chose X over Y because Z"
- Reference RIDs for catalog entities so entries are traceable
- Include quantitative evidence when available (accuracy numbers, counts, sizes)
- Keep each entry to 2-5 lines — these should be scannable
- Don't duplicate information that's already in experiment descriptions or config files
- Write in past tense — these are settled decisions, not plans"""

    @mcp.prompt(
        name="semantic-awareness",
        description="ALWAYS use before creating new tables, vocabularies, features, datasets, or workflows in Deriva catalogs. Search for existing entities to prevent duplicates — even if names are misspelled, abbreviated, or use synonyms. Also use when looking up or referencing any catalog entity by name or concept.",
    )
    def semantic_awareness_prompt() -> str:
        """semantic-awareness workflow guide."""
        return """# Catalog Semantic Awareness — Find Before You Create

Before creating ANY new catalog entity (table, vocabulary term, feature, dataset, workflow), search for existing entities that serve the same or similar purpose. Duplicate entities fragment data, confuse users, and undermine the catalog as a single source of truth.

This skill also applies when looking up any entity by name — catalog entities are created by different people at different times, so the same concept often appears under different names, spellings, or structures.

## Why This Matters

Deriva catalogs are shared, long-lived systems. When someone creates a "Diagnosis" feature without noticing that "Disease_Classification" already exists on the same table, data gets split, queries become ambiguous, and downstream consumers don't know which to use. A two-minute search before creation prevents hours of cleanup later.

## The Process

### 1. Parse Semantic Intent

Understand what the user actually needs — not just the literal name, but the underlying concept. "I need a table for patient demographics" might match an existing "Subject" table. "Add a quality label" might match an existing "Image_Quality" feature.

### 2. Expand the Search Term

Before querying, expand the user's term into a set of candidates:

- **Synonyms**: "Patient" → also search "Subject", "Participant", "Individual"
- **Abbreviations**: "DR" → also search "Diabetic_Retinopathy"
- **Spelling variants**: "Xray" → also search "X-ray", "X_ray", "X-Ray"
- **Misspellings**: "Diagnossis" → also search "Diagnosis"; "fundus" → "Fundus"
- **Singular/plural**: "Image" → also search "Images"
- **Formatting variants**: underscores vs spaces vs camelCase, capitalization differences

### 3. Query the Catalog

Use MCP resources to retrieve candidates. Which resources depend on the entity type:

**Tables:**
```
deriva://catalog/schema          # All tables with columns and descriptions
deriva://table/<name>/schema     # Specific table details
```

**Vocabulary terms:**
```
deriva://vocabulary/<vocab_name>              # All terms with descriptions and synonyms
deriva://vocabulary/<vocab_name>/<term_name>  # Lookup by name or synonym
```

**Features:**
```
deriva://table/<table_name>/features          # Features on a target table
deriva://feature/<table_name>/<feature_name>  # Feature details
deriva://catalog/features                     # All features across all tables
```

**Datasets:**
```
deriva://catalog/datasets    # All datasets with types and descriptions
deriva://dataset/<rid>       # Specific dataset details
```

**Workflows:**
```
deriva://catalog/workflows   # All workflows with descriptions
```

For queries that need actual data (counts, specific records, filtering), use the `query_table` or `count_table` MCP tools.

### 4. Score Closeness Across Multiple Signals

For each candidate entity, assess how close it is to what the user is looking for. No single signal is sufficient — weigh them together:

| Signal | What to check | Strong match indicator |
|--------|--------------|----------------------|
| **Name** | Edit distance, synonym match, abbreviation expansion, misspelling tolerance | Edit distance ≤ 2, or known synonym |
| **Description** | Keyword overlap, stated purpose, domain context | 3+ shared domain keywords |
| **Values/Contents** | For vocabs: term overlap. For tables: column overlap. For datasets: member overlap | 50%+ overlap in values or 3+ matching columns |
| **Relationships** | FK count, features referencing it, datasets containing it, record count | Entity with many relationships is well-established |
| **Structure** | Column types, vocabulary references, FK targets | Structural alignment suggests same purpose |

**Thresholds:**
- Match on **1 signal**: Mention as a possibility, low confidence
- Match on **2+ signals**: Present as a likely match
- Match on **name + description + data/relationships**: Almost certainly the same entity

### 5. Decide: Reuse, Extend, or Create

Based on the candidates found, recommend one of these actions. The right action depends on the entity type and what gap exists:

**Tables:**
- Table exists and fully matches → **Reuse as-is**
- Table exists but missing columns → **Add columns** (`add_column`)
- Table exists but missing a relationship → **Add a foreign key**
- No match → **Create new** with a good description

**Vocabulary terms:**
- Term exists under different spelling → **Add a synonym** (`add_synonym`)
- Vocabulary exists but term is missing → **Add the term** (`add_term`)
- Term exists in a different vocabulary → **Point user to the correct vocabulary**
- No match → **Create new term** with a good description

**Features:**
- Feature exists on same table, same purpose → **Reuse the existing feature**
- Feature exists but its vocabulary needs new terms → **Add terms** to the vocabulary (`add_term`)
- Feature exists on a different table → **Clarify intent** — may need a new one or may be targeting wrong table
- No match → **Create new** with a good description

**Datasets:**
- Dataset exists and matches → **Reuse**, possibly with a new version
- Dataset exists but needs more members → **Add members** (`add_dataset_members`)
- Dataset exists but needs a different split → **Split the existing dataset** (`split_dataset`)
- No match → **Create new** with a good description

**Workflows:**
- Workflow exists with same purpose → **Reuse** with different execution parameters
- No match → **Create new** with a good description

### 6. Present Matches with Evidence

When you find close matches, present them with the evidence AND a specific recommended action:

```markdown
I found existing entities that may match what you need:

### "Subject" table (RID: 1-ABC)
- **Description**: "Research participants enrolled in the study"
- **Matching columns**: Name, Age, Gender (3 of 4 requested columns match)
- **Relationships**: 5 FK references from other tables, 2 features, 450 records
- **Gap**: Missing "Enrollment_Date" column

**Recommended action**: Add an "Enrollment_Date" column to the existing Subject table
rather than creating a duplicate "Patient" table. This preserves all existing
relationships and data.
```

Let the user decide. Never create without presenting findings first.

### 7. Create with a Good Description

If no match exists, create the new entity with a clear, searchable description. Future searches depend on descriptions being informative — a vague description like "data" or "labels" makes the entity invisible.

See the `generate-descriptions` skill for templates and detailed guidance.

## Entity-Specific Gotchas

**Tables**: "Subject" vs "Patient" vs "Participant" — these are often the same concept. Check column structure and record count, not just names. A table with 500 records and 5 FK relationships is worth extending, not duplicating.

**Vocabulary terms**: Always search synonyms. "X-ray" might have synonym "Xray" or "radiograph". The right action is usually `add_synonym`, not `add_term`. Use the `deriva://vocabulary/<name>/<term>` resource which matches against synonyms automatically.

**Features**: A feature named "Quality" and one named "Image_Quality" on the same table are almost certainly duplicates. The combination of target table + vocabulary is the strongest duplicate signal. Check how many values already exist — a feature with thousands of values is definitely established.

**Datasets**: Before creating a new training dataset, check member counts on existing datasets. An existing complete dataset with 10,000 members can be split rather than built from scratch. Check children/parents — the needed dataset may already exist as a split.

**Workflows**: Before creating "ResNet Training v2", check if "ResNet Training" already exists — you might just need different execution parameters. Check description and type.

## The Flow

```
User requests creation or lookup
  → Parse semantic intent (what do they actually need?)
  → Expand search terms (synonyms, abbreviations, misspellings)
  → Query catalog resources for candidates
  → Score closeness across multiple signals
  → Assess: reuse, extend, or create?
  → Present matches with evidence and recommended action
  → User decides
  → Execute the chosen action (with good description if creating new)
```"""

    @mcp.prompt(
        name="run-notebook",
        description="Guide for developing or running DerivaML Jupyter notebooks with execution tracking",
    )
    def run_notebook_prompt() -> str:
        """run-notebook workflow guide."""
        return """# Develop and Run a DerivaML Notebook

DerivaML notebooks support full execution tracking and provenance when structured correctly.

## Required Notebook Structure

1. **Imports cell** — All imports in the first code cell
2. **Parameters cell** — Tagged `"parameters"` for papermill injection. Contains all configurable values (host, catalog, dataset RIDs, hyperparameters, `dry_run`)
3. **Config loading** — `ml = DerivaML(host=host, catalog_id=catalog_id, schema=schema)`
4. **Execution context** — Main logic inside `with ml.create_execution(...) as execution:` block
5. **Save execution RID** — Set `DERIVA_ML_SAVE_EXECUTION_RID = execution.rid` after the context block

## Critical Rules

1. **Tag the parameters cell** — Must have `"parameters"` tag for papermill to inject values
2. **Use `create_execution()` context manager** — Provides provenance tracking, auto-status updates
3. **Clear outputs before committing** — Use `nbstripout` or manual clear
4. **Commit before production runs** — Git hash is recorded in the execution record
5. **Test with `dry_run=True`** — Validates pipeline without catalog writes

## Running Notebooks

```bash
# Via CLI (recommended)
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb

# With config overrides
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb assets=my_assets

# Override host/catalog
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --host ml.derivacloud.org --catalog 2

# Show available configs
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --info
```

## MCP Tools

- `inspect_notebook` — View notebook structure, parameters, and tags without running
- `run_notebook` — Execute notebook with parameters and return execution RID

## Pre-Production Checklist

- [ ] Parameters cell tagged `"parameters"`
- [ ] All configurable values in parameters cell
- [ ] Main logic inside `create_execution()` context
- [ ] `DERIVA_ML_SAVE_EXECUTION_RID` set after context
- [ ] Runs end-to-end with Restart & Run All
- [ ] Outputs cleared, code committed, version bumped

For the full guide with environment setup, papermill details, and troubleshooting table, read `references/workflow.md`.

---

# Detailed Guide

# Develop and Run a DerivaML Notebook with Execution Tracking

This skill covers how to structure, develop, and run DerivaML Jupyter notebooks with full execution tracking and provenance.

## Step 0: Verify Environment

Before starting, confirm your environment is set up. See the `setup-notebook-environment` skill for full details. Quick check:

```bash
uv sync --group=jupyter
uv run jupyter kernelspec list  # Should show your project kernel
uv run deriva-globus-auth-utils login --host ml.derivacloud.org --no-browser  # Should confirm auth
```

## Create a Notebook from Template

Start from the project's notebook template if one exists, or create a new notebook with the required structure described below.

## Required Notebook Structure

A DerivaML notebook that supports execution tracking must have these components:

### 1. Imports Cell

The first code cell should contain all imports:

```python
from pathlib import Path
from deriva.ml import DerivaML
from deriva.ml.config import DatasetSpecConfig
```

### 2. Parameters Cell (Tagged "parameters")

The second code cell must be **tagged with `"parameters"`** for papermill injection. This cell contains all configurable values:

```python
# Parameters
host = "ml.derivacloud.org"
catalog_id = "1"
schema = "my_schema"
dataset_rid = "2-B4C8"
dataset_version = 3
learning_rate = 1e-3
batch_size = 32
epochs = 100
dry_run = False
```

**How to tag a cell in JupyterLab**: Click the cell, then go to the sidebar (View > Cell Inspector or click the gear icon) and add `"parameters"` to the cell tags.

When the notebook is run via papermill (by `deriva-ml-run` or the `run_notebook` MCP tool), the values in this cell are replaced by injected parameters. A new cell is inserted immediately after with the injected values.

### 3. Config Loading

Connect to the catalog:

```python
ml = DerivaML(host=host, catalog_id=catalog_id, schema=schema)
```

### 4. Execution Context

Wrap the main computation in an execution context manager. This creates an execution record in the catalog, tracks provenance, and handles cleanup:

```python
with ml.create_execution(
    workflow_rid=workflow_rid,
    datasets=[DatasetSpecConfig(rid=dataset_rid, version=dataset_version)],
    description=f"Training run: lr={learning_rate}, bs={batch_size}, epochs={epochs}",
) as execution:
    # Download data
    dataset_path = execution.download_dataset(dataset_rid)

    # Your training logic here
    model = train(dataset_path, learning_rate, batch_size, epochs)

    # Save outputs to the execution
    execution.save_artifact("model_weights.pt", model.state_dict())
    execution.save_metrics({"accuracy": 0.95, "loss": 0.12})
```

### 5. Save Execution RID

After the execution context closes, save the execution RID for downstream use. The convention is to use the `DERIVA_ML_SAVE_EXECUTION_RID` variable:

```python
DERIVA_ML_SAVE_EXECUTION_RID = execution.rid
print(f"Execution RID: {DERIVA_ML_SAVE_EXECUTION_RID}")
```

This variable is recognized by the DerivaML notebook runner. When the notebook is run programmatically, the runner extracts this value to link the notebook execution to the catalog record.

## Developing the Notebook

### Single Task

Keep each notebook focused on one task: training a model, evaluating results, exploring data, or generating visualizations. Do not combine unrelated tasks.

### Parameterize Everything

Every value that might change between runs should be in the parameters cell:

- Host and catalog connection details.
- Dataset RIDs and versions.
- Model hyperparameters.
- File paths and output locations.
- The `dry_run` flag.

### Clear Outputs Before Committing

Always clear all outputs before committing. If `nbstripout` is installed (see `setup-notebook-environment`), this happens automatically. Otherwise:

- JupyterLab: Kernel > Restart Kernel and Clear All Outputs.
- CLI: `uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb`

### Run End-to-End

Before committing, restart the kernel and run all cells from top to bottom. A notebook that only works with out-of-order cell execution is broken:

- JupyterLab: Kernel > Restart Kernel and Run All Cells.

### Use the Execution Context

Always use `ml.create_execution()` for tracked work. The execution context:

- Creates a timestamped execution record.
- Links the execution to datasets, workflow, and code version.
- Provides methods to save artifacts and metrics.
- Automatically updates status on success or failure.
- Records the git commit hash and branch.

## Commit and Version Before Running

Before running a notebook for production (non-dry-run):

1. Clear outputs and save the notebook.
2. Commit all changes:
   ```bash
   git add notebook.ipynb
   git commit -m "Update training notebook"
   ```
3. Bump the version if needed:
   ```bash
   uv run bump-version patch
   git add pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   ```

## Running the Notebook with Tracking

### Using Hydra Config Defaults (Recommended)

If your project has hydra-zen configs set up, the notebook can be run as part of an experiment:

```bash
uv run deriva-ml-run +experiments=baseline
```

The experiment config can specify the notebook as the task function, with parameters injected from the config.

### With Explicit Host and Catalog

```bash
uv run papermill notebook.ipynb output.ipynb \\
  -p host ml.derivacloud.org \\
  -p catalog_id 1 \\
  -p dataset_rid 2-B4C8 \\
  -p dry_run False
```

### With Parameter Overrides

```bash
uv run papermill notebook.ipynb output.ipynb \\
  -p learning_rate 0.01 \\
  -p epochs 200
```

### From a Parameter File

Create a `params.yaml`:
```yaml
host: ml.derivacloud.org
catalog_id: "1"
dataset_rid: "2-B4C8"
learning_rate: 0.001
epochs: 100
dry_run: false
```

Run:
```bash
uv run papermill notebook.ipynb output.ipynb -f params.yaml
```

### Inspect Before Running

Use the MCP tool to inspect the notebook structure without running it:

- `inspect_notebook` shows the parameters cell, tags, and overall structure.

### Run via MCP Tool

Use the `run_notebook` MCP tool for programmatic execution:

- `run_notebook` runs the notebook with specified parameters and returns the execution RID.

## What Happens During Execution

When a DerivaML notebook runs with execution tracking, these steps occur:

1. **Execution record created**: A new row is inserted in the Execution table with status "Running", linked to the workflow, datasets, and code version.

2. **Data downloaded**: Datasets and assets specified in the config are downloaded to the execution's working directory. Cached assets are symlinked.

3. **Notebook executed**: Papermill runs the notebook cell-by-cell, injecting parameters. Output is captured.

4. **Results uploaded**: Artifacts saved via `execution.save_artifact()` and metrics saved via `execution.save_metrics()` are uploaded to the catalog.

5. **Status updated**: The execution status is set to "Complete" on success or "Failed" on error. The output notebook (with all outputs) is attached to the execution record.

## Complete Workflow Checklist

- [ ] Environment verified (dependencies, kernel, auth)
- [ ] Notebook has imports cell
- [ ] Parameters cell is tagged `"parameters"`
- [ ] All configurable values are in the parameters cell
- [ ] Main logic is inside `ml.create_execution()` context
- [ ] `DERIVA_ML_SAVE_EXECUTION_RID` is set after execution
- [ ] Notebook runs end-to-end with Restart & Run All
- [ ] Outputs cleared before commit
- [ ] Code committed to Git
- [ ] Version bumped if needed
- [ ] Tested with `dry_run=True`
- [ ] Production run completed
- [ ] Execution verified in catalog

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `NameError` on parameter variables | Parameters cell not tagged or not first | Ensure the cell is tagged `"parameters"` and appears early |
| `No kernel named X` | Kernel not installed | Run `uv run deriva-ml-install-kernel` |
| Execution status stuck at "Running" | Notebook crashed without clean exit | Use `update_execution_status` MCP tool to set to "Failed" |
| Outputs still in committed notebook | `nbstripout` not installed | Run `uv run nbstripout --install` |
| `PapermillExecutionError` | A cell raised an exception | Check the output notebook for the traceback |
| `AuthenticationError` during execution | Credentials expired mid-run | Re-authenticate and re-run |
| `DERIVA_ML_SAVE_EXECUTION_RID` not found | Variable not set in notebook | Add the assignment after the execution context block |
| Metrics not appearing in catalog | `save_metrics()` not called inside context | Move the call inside the `with` block |"""

    @mcp.prompt(
        name="setup-notebook-environment",
        description="Set up the environment for running DerivaML Jupyter notebooks — install kernel, uv sync --group=jupyter, configure nbstripout, authenticate with Deriva. Use before developing or running notebooks for the first time.",
    )
    def setup_notebook_environment_prompt() -> str:
        """setup-notebook-environment workflow guide."""
        return """# Set Up Environment for DerivaML Notebooks

This skill walks through every step needed to set up a working environment for developing and running DerivaML Jupyter notebooks.

## Prerequisites

Before starting, ensure you have:

- Python 3.11 or later installed.
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).
- A DerivaML project repository cloned locally.
- A Globus account with access to the target Deriva catalog.

## Step-by-Step Setup

### Step 1: Install Project Dependencies

```bash
uv sync
```

This installs the project and all its core dependencies in an isolated virtual environment.

### Step 2: Install Jupyter Dependencies

```bash
uv sync --group=jupyter
```

This installs JupyterLab, papermill, and any other notebook-related dependencies defined in the project's `pyproject.toml` under `[dependency-groups]`.

If the project does not have a `jupyter` dependency group, add the dependencies manually:

```bash
uv add --group jupyter jupyterlab papermill ipykernel
```

### Step 3: Install nbstripout

```bash
uv run nbstripout --install
```

**Why this matters:** `nbstripout` installs a Git filter that automatically strips notebook outputs (cell outputs, execution counts, metadata) before they are staged for commit. Without it:

- Notebook outputs bloat the repository with binary data (images, large tables).
- Every run creates merge conflicts in output cells.
- Sensitive data (file paths, credentials, intermediate results) may be committed accidentally.

The `--install` flag registers the filter in the repository's `.git/config` so it runs automatically on every `git add`.

Verify it is installed:
```bash
uv run nbstripout --status
```

### Step 4: Install the Jupyter Kernel

```bash
uv run deriva-ml-install-kernel
```

This registers the project's virtual environment as a Jupyter kernel. The kernel name is derived from the project name in `pyproject.toml`.

Alternatively, use the MCP tool:
- `install_jupyter_kernel` to install the kernel programmatically.

To verify the kernel was installed:
```bash
uv run jupyter kernelspec list
```

You should see an entry for your project (e.g., `my-ml-project`).

### Step 5: Authenticate to Deriva

```bash
uv run deriva-globus-auth-utils login --host ml.derivacloud.org
```

Replace `ml.derivacloud.org` with your target host. This opens a browser for Globus authentication and stores credentials locally.

To authenticate to multiple hosts:
```bash
uv run deriva-globus-auth-utils login --host ml.derivacloud.org
uv run deriva-globus-auth-utils login --host ml-dev.derivacloud.org
```

To verify authentication:
```bash
uv run deriva-globus-auth-utils login --host ml.derivacloud.org --no-browser
```

If already authenticated, this will confirm without opening a browser.

### Step 6: Verify the Setup

Start JupyterLab:

```bash
uv run jupyter lab
```

Then verify:

1. **Select the correct kernel**: In JupyterLab, create a new notebook or open an existing one. Select the kernel matching your project name (not the default Python 3 kernel).

2. **Test the import**: In the first cell, run:

```python
from deriva.ml import DerivaML

ml = DerivaML(host="ml.derivacloud.org", catalog_id="1")
print(f"Connected to {ml.host}, catalog {ml.catalog_id}")
```

If this succeeds without errors, your environment is ready.

## Optional: ML Framework Dependencies

Many DerivaML projects need additional ML framework dependencies. These are typically organized as dependency groups:

### PyTorch

```bash
uv sync --group=pytorch
# or if not predefined:
uv add torch torchvision torchaudio
```

### TensorFlow

```bash
uv sync --group=tensorflow
# or if not predefined:
uv add tensorflow
```

### JAX

```bash
uv add jax jaxlib
```

### scikit-learn

```bash
uv add scikit-learn
```

Check the project's `pyproject.toml` for predefined dependency groups before adding packages manually.

## Complete Checklist

- [ ] `uv sync` completed successfully
- [ ] `uv sync --group=jupyter` completed successfully
- [ ] `nbstripout --install` ran (verify with `--status`)
- [ ] Jupyter kernel installed (verify with `jupyter kernelspec list`)
- [ ] Authenticated to Deriva (verify with `login --no-browser`)
- [ ] JupyterLab starts and the project kernel is available
- [ ] `from deriva.ml import DerivaML` imports without error
- [ ] Can connect to the target catalog

## Troubleshooting

### Kernel Not Showing in JupyterLab

**Symptom**: The project kernel does not appear in JupyterLab's kernel list.

**Fix**: Re-install the kernel and restart JupyterLab:
```bash
uv run deriva-ml-install-kernel
# Restart JupyterLab (stop and start again)
uv run jupyter lab
```

If that does not work, install manually:
```bash
uv run python -m ipykernel install --user --name my-project --display-name "My Project"
```

### nbstripout Not Working

**Symptom**: Notebook outputs are still being committed.

**Fix**: Verify the Git filter is installed:
```bash
uv run nbstripout --status
```

If it reports "not installed", re-run:
```bash
uv run nbstripout --install
```

Also check that `.gitattributes` contains:
```
*.ipynb filter=nbstripout
```

### Authentication Errors

**Symptom**: `AuthenticationError` or `401 Unauthorized` when connecting to Deriva.

**Fix**:
1. Re-authenticate:
   ```bash
   uv run deriva-globus-auth-utils login --host ml.derivacloud.org
   ```
2. Ensure you are authenticating to the correct host.
3. Verify your Globus identity has been granted access to the catalog.

### Missing Dependencies

**Symptom**: `ModuleNotFoundError` when importing packages.

**Fix**:
1. Make sure you ran `uv sync` (not just `uv install`).
2. Make sure you are using the correct kernel in JupyterLab (the project kernel, not the default).
3. Check if the missing package is in an optional dependency group:
   ```bash
   uv sync --group=jupyter  # or --group=pytorch, etc.
   ```

### JupyterLab Won't Start

**Symptom**: `jupyter lab` command not found or crashes.

**Fix**:
```bash
uv sync --group=jupyter
uv run jupyter lab
```

Note the `uv run` prefix -- this ensures JupyterLab runs within the project's virtual environment."""

    @mcp.prompt(
        name="troubleshoot-execution",
        description="ALWAYS use when any DerivaML execution fails, errors, gets stuck, or produces unexpected results. Covers authentication errors, missing files, stuck 'Running' status, version mismatches, permission denied, upload timeouts, and dataset download failures.",
    )
    def troubleshoot_execution_prompt() -> str:
        """troubleshoot-execution workflow guide."""
        return """# Troubleshooting DerivaML Executions

This guide covers common problems encountered when running DerivaML executions and their solutions.

---

## Problem: "No Active Execution"

**Symptom**: Tools that require an execution context (like `asset_file_path`, `upload_execution_outputs`) fail with an error about no active execution.

**Cause**: The execution was not properly started, or you are outside the execution context.

**Solution**:
- In Python, always use the context manager pattern:
  ```python
  with ml.execution(workflow_rid="...") as exec:
      # All execution work goes here
  ```
- With MCP tools, ensure you called `start_execution` before attempting execution-scoped operations.
- If the execution was started but the error persists, the execution may have been stopped or may have failed. Check with `get_execution_info`.

---

## Problem: "Files Not Uploaded"

**Symptom**: Execution completes but asset files are not visible in the catalog.

**Cause**: `upload_execution_outputs` was not called, or files were written to the wrong path.

**Solution**:
1. Call `upload_execution_outputs` **after** writing all files but **before** the execution context closes (i.e., inside the `with` block in Python).
2. Ensure files are written to the **exact path** returned by `asset_file_path`. Writing to any other directory will cause the upload to miss those files.
3. Verify the file actually exists at the path before uploading:
   ```python
   path = exec.asset_file_path("MyAssetTable", "output.csv")
   # Write file to `path`
   # Verify: os.path.exists(path) should be True
   ```
4. Check that the execution is still in `Running` status when you attempt the upload. If it was already stopped or failed, uploads will not work.

---

## Problem: "Dataset Not Found"

**Symptom**: Attempting to use a dataset RID returns an error or empty result.

**Cause**: Wrong catalog connection, dataset was deleted, or the RID is incorrect.

**Solution**:
- Verify you are connected to the correct catalog with `connect_catalog` or check the active catalog.
- Check the dataset resources to list available datasets.
- Use `validate_rids` to confirm the RID is valid and belongs to a dataset table.
- If the dataset was recently created, it should be visible immediately -- there is no propagation delay.

---

## Problem: "Invalid RID"

**Symptom**: A tool rejects a RID value or returns "not found".

**Cause**: The RID is malformed, belongs to a different table than expected, or refers to a deleted record.

**Solution**:
- **Tool**: `validate_rids` to check whether the RID exists and what table it belongs to.
- RIDs are case-sensitive alphanumeric strings (e.g., `1-A2B3`). Ensure there are no extra spaces or characters.
- If the RID comes from a different catalog, it will not resolve in the current catalog. Verify you are connected to the right catalog.

---

## Problem: "Permission Denied"

**Symptom**: Operations fail with authentication or authorization errors.

**Cause**: Your credentials have expired or you lack the required role.

**Solution**:
- Re-authenticate using `deriva-globus-auth-utils`:
  ```bash
  deriva-globus-auth-utils login --host <hostname>
  ```
- Check that your user account has the necessary group membership for the operation (read, write, or admin).
- Some operations (like creating tables or modifying schemas) require elevated permissions.

---

## Problem: "Version Mismatch"

**Symptom**: Dataset contents do not match expectations, or a workflow references an outdated dataset version.

**Cause**: The dataset was modified after the version was pinned, or version tracking was not used.

**Solution**:
- Check the dataset's version history through the dataset resources.
- Use `increment_dataset_version` after making changes to a dataset to create a new version snapshot.
- When referencing datasets in workflows, consider pinning to a specific version.
- Use `get_dataset_spec` to see the current dataset specification and version.

---

## Problem: "Feature Not Found"

**Symptom**: Attempting to add feature values fails because the feature does not exist.

**Cause**: The feature was not created, or the name does not match exactly.

**Solution**:
- Check the feature resources to list existing features.
- Feature names are case-sensitive. Verify exact spelling.
- **Tool**: `create_feature` to create the feature if it does not exist.
- Ensure the feature is associated with the correct table.

---

## Problem: "Upload Timeout"

**Symptom**: `upload_execution_outputs` hangs or times out.

**Cause**: Large files, network issues, or server limits.

**Solution**:
- Check your network connectivity.
- For large files, consider breaking them into smaller batches.
- The server may have upload size limits. Check with your catalog administrator.
- Retry the upload -- transient network issues are the most common cause.
- **Tool**: `get_execution_info` to check if partial uploads succeeded.

---

## Problem: "Execution Stuck in Running"

**Symptom**: An execution shows status `Running` but the process has ended or crashed.

**Cause**: The execution context was not properly closed (e.g., crash without cleanup, not using context manager).

**Solution**:
- **Best practice**: Always use the context manager (`with ml.execution(...)`) which automatically handles cleanup on both success and failure.
- To fix a stuck execution manually:
  - **Tool**: `update_execution_status` with the execution RID and status `Failed` (or `Complete` if the work actually finished).
- **Tool**: `get_execution_info` to inspect the execution's current state and metadata.
- For future runs, always use the context manager to prevent this issue.

---

## Problem: "Vocabulary Term Not Found"

**Symptom**: An operation fails because a required vocabulary term does not exist.

**Cause**: The term was not added to the vocabulary, or the name does not match exactly.

**Solution**:
- Check the relevant vocabulary resource to list existing terms.
- Vocabulary term names are case-sensitive.
- **Tool**: `add_term` to add the missing term to the appropriate vocabulary.
- Common vocabularies: `Dataset_Type`, `Asset_Type`, `Workflow_Type`, `Execution_Status`.

---

## General Debugging Tips

### Enable Verbose Logging
When using the Python API, enable verbose logging to see detailed request/response information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Execution State
- **Tool**: `get_execution_info` with the execution RID to see full execution metadata, status, inputs, and outputs.
- **Tool**: `get_execution_working_dir` to find the local working directory and inspect files directly.

### Check Catalog State
- Use the catalog resources to review the current catalog schema, tables, and vocabularies.
- **Tool**: `count_table` to quickly verify data exists in expected tables.

### Review Recent Executions
- Check the recent executions resource to see the latest execution activity, statuses, and any patterns of failure.
- **Tool**: `list_nested_executions` if the execution is part of a larger workflow to see the full execution tree.
- **Tool**: `list_parent_executions` to find the parent execution if this is a nested step.

### Verify Working Directory
- **Tool**: `get_execution_working_dir` returns the local filesystem path for the execution.
- Inspect this directory to verify:
  - Input files were downloaded correctly.
  - Output files were written to the correct locations.
  - No unexpected files or directory structures.

### Clean Up
- **Tool**: `clean_execution_dirs` to remove local execution working directories that are no longer needed, freeing disk space."""
