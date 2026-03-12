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

Use `get_table_sample_data(table_name="...")` to preview sample rows.

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
query_table(table_name="Subject", filters={"Species": "Mouse"})
```

### Multiple AND Conditions

```
query_table(
    table_name="Subject",
    filters={"Species": "Mouse", "Status": "Active"}
)
```

This returns rows where Species is "Mouse" AND Status is "Active".

### Count Rows

Use the `count_table` MCP tool to get the number of matching rows without fetching data:

```
count_table(table_name="Subject")
count_table(table_name="Subject", filters={"Species": "Mouse"})
```

## Get Specific Records

### By RID

Use the `get_record` MCP tool to fetch a single record by its RID (Row ID):

```
get_record(table_name="Subject", rid="2-B4C8")
```

This returns the complete record with all columns.

### Validate Known RIDs

Use `validate_rids` to check that dataset, asset, workflow, or execution RIDs exist before using them:

```
validate_rids(dataset_rids=["2-B4C8"], asset_rids=["2-D1E2"])
```

To resolve an unknown RID to its table, read the `deriva://rid/{rid}` resource instead.

## Query Related Data

### Denormalize for ML

Use the `denormalize_dataset` MCP tool to get ML-ready joined data from a dataset:

```
denormalize_dataset(dataset_rid="2-B4C8", include_tables=["Image", "Subject"])
```

This joins the dataset's member tables, resolving foreign keys into human-readable values. The result is a flat table suitable for loading into a DataFrame.

### Query a Single Table

For simpler needs, `query_table` on the relevant table is sufficient:

```
query_table(table_name="Image", filters={"Subject": "2-A1B2"})
```

### Download a Full Dataset

Use the `download_dataset` MCP tool to get a complete local copy of a dataset:

```
download_dataset(dataset_rid="2-B4C8", version="3")
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
query_table(table_name="Image", filters={"Subject": "2-A1B2"})
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
    filters={"Status": "Complete"},
    limit=20
)
```

Note: `query_table` does not support a `sort` parameter. For sorted or complex date range queries, filter results client-side or use the ERMrest API directly.

### Export Data for ML

To get data ready for ML training:

1. **Identify the dataset**: `get_record(table_name="Dataset", rid="2-B4C8")`
2. **Get the members**: `list_dataset_members(dataset_rid="2-B4C8")`
3. **Denormalize**: `denormalize_dataset(dataset_rid="2-B4C8", include_tables=["Image", "Subject"])`
4. **Download assets**: `download_dataset(dataset_rid="2-B4C8", version="3")`

## Historical Queries with Versions

Datasets in Deriva can be versioned. Query specific versions:

```
download_dataset(dataset_rid="2-B4C8", version="3")
```

To see available versions, read:

- `deriva://dataset/{rid}` -- Includes version history.

Always pin to a specific version for reproducible experiments.

## View in Web Interface

To get the Chaise (web UI) URL for any record, use the MCP resource:

- `deriva://chaise-url/{table_or_rid}` -- Pass a table name for the record set view, or a RID for a specific record.

These URLs are useful for sharing records with collaborators or viewing complex relationships that are easier to navigate in the web interface.

## Complete Example Workflow

Here is a typical workflow for exploring and extracting data from a catalog:

1. **Orient yourself**: Read `deriva://catalog/tables` to see what is available.

2. **Explore a table**: Read `deriva://table/Subject/schema` to understand columns, then `get_table_sample_data(table_name="Subject")` for sample rows.

3. **Count records**: `count_table(table_name="Subject")` and `count_table(table_name="Subject", filters={"Species": "Mouse"})`.

4. **Query with filters**: `query_table(table_name="Subject", filters={"Species": "Mouse"}, limit=50)`.

5. **Inspect a specific record**: `get_record(table_name="Subject", rid="2-A1B2")`.

6. **Find related data**: `query_table(table_name="Image", filters={"Subject": "2-A1B2"})`.

7. **Check features**: Read `deriva://table/Image/features`, then `query_table(table_name="Image_Cell_Count", filters={"Image": "2-C3D4"})`.

8. **Get dataset for ML**: `denormalize_dataset(dataset_rid="2-B4C8", include_tables=["Image", "Subject"])` for a flat view, or `download_dataset(dataset_rid="2-B4C8", version="3")` for a full local copy.

9. **Share with a colleague**: Read `deriva://chaise-url/2-A1B2` to get a shareable URL for a specific record, or `deriva://chaise-url/Subject` for the table view.

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
- **Stale data**: Catalog data can change. If you need a stable snapshot, use versioned datasets.
