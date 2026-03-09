---
name: query-catalog-data
description: "ALWAYS use this skill when querying, filtering, searching, or browsing data in a Deriva catalog. Triggers on: 'query table', 'find records', 'filter by', 'how many records', 'look up RID', 'what tables exist', 'show me the data', 'explore the catalog', 'get record by RID'."
---

# Querying and Exploring Data in a Deriva Catalog

This skill covers how to find, filter, and explore data in a Deriva catalog using MCP tools and resources.

## Discovery Resources

| Resource | Purpose |
|----------|---------|
| `deriva-ml://catalog/tables` | All tables with descriptions and row counts |
| `deriva-ml://catalog/schema` | Full schema with relationships |
| `deriva-ml://table/{name}/schema` | Column names, types, descriptions |
| `deriva-ml://table/{name}/sample` | Sample rows |
| `deriva-ml://table/{name}/features` | Features on a table |
| `deriva-ml://vocabulary/{name}` | Vocabulary terms |
| `deriva-ml://dataset/{rid}` | Dataset details and versions |
| `deriva-ml://chaise-url/{table}/{rid}` | Web UI link |

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
