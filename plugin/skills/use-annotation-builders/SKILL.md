---
name: use-annotation-builders
description: "Write Python scripts using type-safe annotation builder classes (ColumnAnnotation, TableAnnotation, KeyAnnotation) for production Deriva catalog code. Use when writing Python code to configure catalog display, not when using interactive MCP tools."
disable-model-invocation: true
---

# Using Annotation Builder Classes

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

## Reference Resources

For detailed reference material beyond what this skill covers, read these MCP resources:

- `deriva://docs/annotation-contexts` — Complete JSON reference of all valid Chaise annotation contexts and their usage. Read this when you need the full list of contexts or want to understand context inheritance.
- `deriva://docs/annotations` — Full guide to annotation builders and the underlying JSON structure. Read this for advanced pseudo-column source syntax, facet options, or pre-format directives.
- `deriva://docs/chaise/config` — Chaise web UI configuration beyond annotations. Read this for navbar customization, default page sizes, or login configuration.

To inspect current annotations on a specific table or column:
- `deriva://table/{table_name}/annotations` — Display-related annotations currently set on a table
- `deriva://table/{table_name}/column/{column_name}/annotations` — Display-related annotations on a column

## Tips

- Builders produce the same JSON that MCP tools set -- they are two ways to do the same thing.
- Use builders when you need to version-control your catalog configuration in Python scripts.
- Use MCP tools for quick interactive changes.
- Always call `ml.apply_annotations()` (Python) or `apply_annotations()` (MCP) after making changes.
- PseudoColumns are powerful for showing related data without changing the data model.
- Test complex Handlebars patterns with `preview_handlebars_template` before applying them.
