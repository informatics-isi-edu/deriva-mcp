---
name: work-with-assets
description: "Step-by-step guide to discovering, querying, and downloading Deriva assets (files, images, models, CSVs) - find asset tables, check provenance, download files. For uploading assets and creating asset tables, see run-ml-execution."
---

# Working with Deriva Assets

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

For uploading assets as execution outputs, creating new asset tables, and managing asset types, see the `run-ml-execution` skill. Assets should always be uploaded within an execution context for provenance tracking.
