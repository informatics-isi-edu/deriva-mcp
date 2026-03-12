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
    asset_name="Slide_Image",
    columns=[
        {"name": "Magnification", "type": "text", "nullok": true, "comment": "Microscope magnification (e.g., 10x, 40x)"},
        {"name": "Stain", "type": "text", "nullok": true, "comment": "Staining protocol used"},
        {"name": "Width", "type": "int4", "nullok": true, "comment": "Image width in pixels"},
        {"name": "Height", "type": "int4", "nullok": true, "comment": "Image height in pixels"}
    ],
    referenced_tables=["Sample"],
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
get_table(table_name="Sample")

# View sample data (will be empty for new tables)
get_table_sample_data(table_name="Sample", limit=5)

# Count records
count_table(table_name="Sample")
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
    table_name="Subject",
    column_name="Weight_kg",
    column_type="float8",
    nullok=true,
    comment="Subject weight in kilograms"
)
```

## Modifying Column Properties

```
# Make a column required or optional
set_column_nullok(table_name="Subject", column_name="Notes", nullok=true)

# Update column description
set_column_description(table_name="Subject", column_name="Age", description="Age in years at enrollment, rounded down")

# Set column display name
set_column_display_name(table_name="Subject", column_name="Age_At_Enrollment", display_name="Enrollment Age")
```

## Documentation Best Practices

1. **Always comment tables**: The `comment` parameter on `create_table` is shown in the UI and in schema documentation.
2. **Always comment columns**: Column comments appear as tooltips in Chaise and serve as documentation.
3. **Set display names**: Use `set_table_display_name` and `set_column_display_name` for user-friendly names that differ from the technical names.
4. **Set row name patterns**: After creating a table, set a row name pattern so records are identifiable:
   ```
   set_row_name_pattern(table_name="Subject", pattern="{{{Name}}}")
   set_row_name_pattern(table_name="Sample", pattern="{{{Sample_ID}}} ({{{Subject}}})")
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
- Vocabulary tables (created with `create_vocabulary`) are preferred over free-text columns for categorical data -- they enable faceted search and consistent labeling.
