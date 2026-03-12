---
name: work-with-assets
description: "ALWAYS use this skill when working with file assets in DerivaML — discovering, downloading, uploading, inspecting, or managing images, model weights, CSVs, or any file-based catalog records. Triggers on: 'download asset', 'upload files', 'asset table', 'find images', 'model weights', 'what created this file', 'asset provenance', 'asset types', 'create asset table'."
disable-model-invocation: true
---

# Working with Assets in DerivaML

An asset is a file-based record in a Deriva catalog — it combines a file (stored in Hatrac) with catalog metadata like filename, size, MD5 checksum, and description. Assets live in asset tables, which have standard file-tracking columns plus optional custom metadata. Every asset has a unique RID for stable referencing across the system.

For background on asset tables, types, RIDs, Hatrac storage, caching, and provenance, see `references/concepts.md`.

## Critical Rules

1. **Use RIDs to reference assets** — not filenames or URLs. RIDs are immutable and unique.
2. **Upload within an execution** — assets must be registered with `asset_file_path` and uploaded with `upload_execution_outputs` inside an active execution for provenance tracking.
3. **Download records provenance automatically** — calling `download_asset` within an execution links the asset as an "Input" to that execution.
4. **Create the asset table before uploading** — the table must exist before you can register files for upload to it.

## Workflow Summary

### Discovering and inspecting assets

1. Read `deriva://catalog/asset-tables` or `deriva://catalog/assets` — find asset tables and counts
2. Read `deriva://table/{table_name}/assets` — browse assets in a table
3. Read `deriva://asset/{asset_rid}` — inspect a specific asset (metadata, types, provenance, Chaise URL)
4. `list_asset_executions` — find which execution created or used an asset

### Downloading assets

1. `download_asset` — download a single asset by RID
2. `download_dataset` — download a dataset as a BDBag with all asset files (no execution required)
3. `download_execution_dataset` — same as above but within an active execution (records the dataset as an input for provenance)
4. `restructure_assets` — organize downloaded assets into ML-ready directory layouts

### Creating asset tables

1. `create_asset_table` — define a new asset table with optional custom columns and FK references

### Uploading assets (within an execution)

1. `create_execution` + `start_execution` — start provenance tracking
2. `asset_file_path` — register each output file for upload (returns a path to write to)
3. `upload_execution_outputs` — upload all registered files to Hatrac and catalog
4. `stop_execution` — finalize

### Managing asset types

1. `add_asset_type` — create a new term in the Asset_Type vocabulary
2. `add_asset_type_to_asset` / `remove_asset_type_from_asset` — tag or untag assets

For the full step-by-step guide with MCP tool parameters and Python API examples, see `references/workflow.md`.

## Reference Resources

- `references/concepts.md` — What assets are, asset tables, RIDs, types, Hatrac, caching, provenance
- `references/workflow.md` — Step-by-step MCP and Python API workflows
- `deriva://docs/file-assets` — Full user guide to file assets in DerivaML
- `deriva://catalog/asset-tables` — List all asset tables
- `deriva://catalog/assets` — Asset tables with record counts
- `deriva://table/{table_name}/assets` — Browse assets in a table
- `deriva://asset/{asset_rid}` — Asset details and provenance

## Related Skills

- **`run-ml-execution`** — Full execution lifecycle including asset upload patterns
- **`prepare-training-data`** — Downloading and restructuring assets for ML training
- **`create-dataset`** — Datasets organize assets into versioned collections for reproducibility
