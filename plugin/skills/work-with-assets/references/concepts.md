# Asset Concepts

Background on assets in DerivaML. For the step-by-step guide, see `workflow.md`.

## Table of Contents

- [What is an Asset?](#what-is-an-asset)
- [Asset Tables](#asset-tables)
- [Asset RIDs](#asset-rids)
- [Asset Types](#asset-types)
- [Hatrac Storage](#hatrac-storage)
- [Asset Caching](#asset-caching)
- [Asset Provenance](#asset-provenance)
- [Built-in Asset Tables](#built-in-asset-tables)

---

## What is an Asset?

An asset is a file-based record in a Deriva catalog. Each asset combines a file (stored in Hatrac, Deriva's file storage service) with catalog metadata — filename, size, checksum, description, and any custom columns defined on the asset table.

Common examples include:
- **Images** — microscopy images, X-rays, photographs
- **Model weights** — trained PyTorch/TensorFlow checkpoints
- **Prediction files** — CSVs of model outputs (class probabilities, scores)
- **Segmentation masks** — pixel-level annotation overlays
- **Configuration files** — YAML/JSON experiment configs
- **Plots and figures** — ROC curves, confusion matrices, training loss charts

Assets are the bridge between files on disk and structured catalog data. Unlike raw files, assets have identity (RIDs), provenance (which execution created them), types (vocabulary-based categorization), and relationships (foreign keys to other tables).

## Asset Tables

An asset table is a catalog table with special columns for file management. When you create an asset table, DerivaML automatically adds these **standard asset columns**:

| Column | Type | Description |
|--------|------|-------------|
| `Filename` | text | Original filename (e.g., `model_weights.pt`) |
| `URL` | text | Hatrac storage URL (set automatically on upload) |
| `Length` | int | File size in bytes |
| `MD5` | text | MD5 checksum for integrity verification |
| `Description` | text | Optional human-readable description |

In addition to these standard columns, asset tables can have **custom metadata columns** — for example, an `Image` table might add `Width`, `Height`, and `Format` columns, or a `Model` table might add `Architecture` and `Epochs`.

Asset tables also get automatically created **association tables**:
- An **Asset_Type association** — links each asset to vocabulary terms categorizing it
- An **Execution association** — tracks which executions created or used each asset, with a role of "Input" or "Output"

## Asset RIDs

Every asset has a unique **RID** (Resource IDentifier) — a short, immutable string like `3-JSE4` or `2-IMG1`. RIDs are the primary way to reference assets across the system:

- **In MCP tools**: Pass `asset_rid` to `download_asset`, `list_asset_executions`, etc.
- **In configurations**: Use RIDs in `AssetSpecConfig` to specify execution inputs
- **In provenance**: Execution records reference asset RIDs for inputs and outputs
- **In the web UI**: Each asset's Chaise page URL includes its RID
- **In citation**: `ml.cite(rid)` generates a permanent URL for an asset

RIDs are assigned by the catalog when a record is created and never change. Use RIDs (not filenames or URLs) as the stable identifier for assets.

## Asset Types

Assets are categorized using the **Asset_Type** controlled vocabulary. Each asset can have multiple types (e.g., an image could be both "Training_Data" and "Augmented").

Asset types serve two purposes:
1. **Organization** — filter and browse assets by category in the web UI
2. **Configuration** — specify which types of assets an execution expects

Custom asset types can be created for domain-specific categorization. When you create a new asset table, DerivaML automatically adds the table name as a term in the Asset_Type vocabulary.

## Hatrac Storage

Files are stored in **Hatrac** (HTTP Asset Repository And Catalog), Deriva's file storage service. When you upload an asset, the file goes to Hatrac and the catalog record gets a URL pointing to it. When you download an asset, DerivaML fetches the file from Hatrac using that URL.

You never interact with Hatrac directly — DerivaML handles all file transfers. The `URL` column in asset tables contains the Hatrac path, but you use RIDs and MCP tools for all operations.

## Asset Caching

For large assets that are reused across multiple executions (e.g., pretrained model weights), DerivaML supports **checksum-based caching**. When an asset is specified with `cache=True` in an execution configuration, DerivaML:

1. Checks the local cache directory for a file matching the asset's MD5 checksum
2. If found, creates a symlink to the cached copy instead of re-downloading
3. If not found, downloads the file and stores it in the cache for future use

The cache key is `{rid}_{md5}`, so if an asset's file is updated (new MD5), the old cached copy is not reused.

In hydra-zen configurations:
```python
from deriva_ml.asset.aux_classes import AssetSpecConfig

AssetSpecConfig(rid="6-EPNR", cache=True)   # Cached — reused across executions
AssetSpecConfig(rid="6-EP56", cache=False)   # Not cached — downloaded each time
```

Use caching for large, immutable assets like pretrained weights. Avoid caching for assets that change between runs.

## Asset Provenance

Every asset is linked to the executions that created or used it through an association table with an **Asset_Role** column:

- **Output** — the execution created this asset (e.g., a training run produced model weights)
- **Input** — the execution consumed this asset (e.g., an analysis notebook read prediction CSVs)

This bidirectional tracking means you can answer two key questions:
- "Where did this asset come from?" — find the execution with role "Output"
- "What used this asset?" — find all executions with role "Input"

Provenance is recorded automatically: uploading via `upload_execution_outputs` records "Output" links, and downloading via `download_asset` within an execution records "Input" links.

## Built-in Asset Tables

DerivaML catalogs include two built-in asset tables in the `deriva-ml` schema:

| Table | Purpose |
|-------|---------|
| `Execution_Asset` | General-purpose output files from executions (model weights, predictions, plots). The default target when no specific asset table is specified. |
| `Execution_Metadata` | Auto-managed metadata files (configuration snapshots, logs). Written automatically by the execution framework — you typically don't interact with this directly. |

Domain-specific asset tables (e.g., `Image`, `Model`, `Slide`) are created in the domain schema as needed for each project.
