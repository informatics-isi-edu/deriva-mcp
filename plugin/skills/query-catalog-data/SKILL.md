---
name: query-catalog-data
description: "ALWAYS use this skill when querying, filtering, searching, or browsing data in a Deriva catalog. Triggers on: 'query table', 'find records', 'filter by', 'how many records', 'look up RID', 'what tables exist', 'show me the data', 'explore the catalog', 'get record by RID'."
disable-model-invocation: true
---

# Querying and Exploring Data in a Deriva Catalog

This skill covers how to find, filter, and explore data in a Deriva catalog using MCP tools and resources.

## Discovery Resources

| Resource | Purpose |
|----------|---------|
| `deriva://catalog/tables` | All tables with descriptions and row counts |
| `deriva://catalog/schema` | Full schema with relationships |
| `deriva://table/{name}/schema` | Column names, types, descriptions |
| `deriva://table/{name}/features` | Features on a table |
| `deriva://vocabulary/{name}` | Vocabulary terms |
| `deriva://dataset/{rid}` | Dataset details and versions |
| `deriva://chaise-url/{table_or_rid}` | Web UI link (pass table name or RID) |

## Key Tools

| Tool | Purpose |
|------|---------|
| `query_table` | Query with filters, columns, limit/offset |
| `get_table_sample_data` | Preview sample rows from a table |
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
denormalize_dataset(dataset_rid="2-B4C8", include_tables=["Image", "Subject"])
```

## Tips

- Always use `limit` for large tables to avoid timeouts
- Column names are case-sensitive — check schema first
- Use `denormalize_dataset` to resolve FK RIDs into readable values
- Pin to specific dataset versions for reproducibility

For the full guide with query patterns, feature queries, provenance tracking, and troubleshooting, read `references/workflow.md`.
