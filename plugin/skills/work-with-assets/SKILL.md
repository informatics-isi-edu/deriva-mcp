---
name: work-with-assets
description: "Discover, query, and download Deriva assets (files, images, model weights, CSVs) — find asset tables, check provenance, download files, trace which executions created an asset. For uploading assets, see run-ml-execution."
disable-model-invocation: true
---

# Working with Deriva Assets

Assets in DerivaML are managed files (images, CSVs, models, etc.) stored in asset tables. Each asset table has standard columns: `URL`, `Filename`, `Length`, `MD5`, `Description`, plus any custom columns.

## Discovering and Querying Assets

Use the catalog resources and query tools to find assets:

```
# List all tables — asset tables have URL, Filename, Length, MD5 columns
query_table(table_name="Slide_Image", limit=5)

# Search assets with filters
query_table(table_name="Slide_Image", filters={"Subject": "2-A1B2"})

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

# Download a dataset version as a BDBag (within an active execution)
download_execution_dataset(dataset_rid="2-DS01", version="1.0.0")

# Find where downloaded assets are located (uses active execution)
get_execution_working_dir()
```

## Reference Resources

- `deriva://docs/file-assets` — Full guide to asset tables, types, uploading, and provenance tracking. Read this for detailed examples and edge cases beyond what this skill covers.
- `deriva://catalog/asset-tables` — List all asset tables in the catalog
- `deriva://table/{table_name}/assets` — Browse assets in a specific table
- `deriva://asset/{rid}` — Asset details and provenance

## Uploading Assets and Creating Asset Tables

For uploading assets as execution outputs, creating new asset tables, and managing asset types, see the `run-ml-execution` skill. Assets should always be uploaded within an execution context for provenance tracking.
