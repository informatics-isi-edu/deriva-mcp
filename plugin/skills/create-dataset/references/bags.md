# Dataset Bags (BDBags)

## Table of Contents

- [What is a Bag?](#what-is-a-bag)
- [What a Bag Contains](#what-a-bag-contains)
- [How Bag Contents Are Determined](#how-bag-contents-are-determined)
- [Versioning and Reproducibility](#versioning-and-reproducibility)
- [Materialization](#materialization)
- [Caching](#caching)
- [Downloading a Bag](#downloading-a-bag)
- [Previewing Before Download](#previewing-before-download)
- [When Downloads Are Slow or Timing Out](#when-downloads-are-slow-or-timing-out)
- [Working with Bag Contents](#working-with-bag-contents)
- [Hydra-Zen Configuration](#hydra-zen-configuration)

---

## What is a Bag?

A **BDBag** (Big Data Bag) is a self-describing, portable archive of a specific dataset version. It packages everything needed to reproduce a dataset offline: member records, related data, asset files, feature values, and vocabulary terms — all checksummed for integrity.

Bags are the standard way to get data out of a DerivaML catalog for ML training, analysis, or sharing. When you call `download_dataset`, the result is a bag.

## What a Bag Contains

A bag for a specific dataset version includes:

1. **Member records** — All records from registered element types that belong to the dataset (e.g., Image, Subject rows), exported as CSV files per table.
2. **Related records** — Data from tables reachable via foreign key paths from member records (see [How Bag Contents Are Determined](#how-bag-contents-are-determined)).
3. **Nested datasets** — Child datasets are included recursively with all their members.
4. **Feature values** — All feature annotations for dataset members (e.g., Image_Classification labels).
5. **Vocabulary terms** — Controlled vocabulary terms referenced by included records, exported separately.
6. **Asset files** — Binary files (images, model weights, etc.) referenced by member records, fetched when `materialize=True`.
7. **Checksums** — Every file has a cryptographic checksum for integrity verification.

## How Bag Contents Are Determined

The export algorithm uses **FK path traversal** from registered element types to determine what to include.

### Starting points

Only tables registered as **dataset element types** (via `add_dataset_element_type`) that have members in this dataset serve as traversal starting points. Unregistered tables or registered tables with no members are not starting points.

### Traversal rules

From each starting-point record, the export follows foreign key relationships:

- **Both FK directions are followed.** Outgoing FKs (this table references another) and incoming FKs (another table references this one).
- **Vocabulary tables are natural terminators.** Controlled vocabulary terms are collected and exported separately — they don't generate further FK traversal.
- **Feature tables are automatically included.** Feature annotation tables (e.g., `Image_Classification`) for reachable element types are added to the export.
- **Element type boundaries.** A registered element type that has *no members* in this dataset acts as a traversal boundary — the export won't follow FK paths through it. This prevents expensive joins that would return empty results.

### Example

A dataset with `Subject` members where the schema has Subject → Image FK:
- Subject records are exported (starting points)
- Image records referencing those Subjects are included (inbound FK traversal)
- Image_Classification records for those Images are included (feature table)
- Vocabulary terms (e.g., Diagnosis, Species) referenced by any included record are collected separately

If `Image` is also a registered element type but has no members in this dataset, it acts as a boundary and Image records would *not* be traversed through.

## Versioning and Reproducibility

Each bag is tied to a **catalog snapshot** — the exact catalog state at the time the dataset version was created. This means:

- The same dataset RID + version always produces the same data
- Changes made to the catalog after the version was created (new features, updated records) are **not** included in existing versions
- To capture recent changes, call `increment_dataset_version` first, then download the new version

## Materialization

- **`materialize=True`** (default): The bag fetches all referenced asset files from Hatrac storage. Creates a fully self-contained archive.
- **`materialize=False`**: The bag contains only metadata and remote file references. Smaller download, but requires network access to use the assets later.

Use `materialize=False` when you only need the tabular data (record metadata, feature values) and not the actual files.

## Caching

Bags are cached locally by checksum. When you download the same dataset version again, the cached bag is reused without re-downloading. The cache key is `{dataset_rid}_{checksum}`.

The cache location can be configured via the `cache_dir` argument when creating a DerivaML instance. Read the `deriva://storage/cache` resource to see cached bags, and use `clear_cache` to remove all cached data.

## Downloading a Bag

### MCP tool

Call `download_dataset` with `dataset_rid` and `version`.

### Python API

```python
bag = dataset.download_dataset_bag(version="1.0.0")
# or within an execution:
bag = exe.download_dataset_bag(DatasetSpec(rid="2-XXXX", version="1.0.0"))
```

## Previewing Before Download

Call `estimate_bag_size` with `dataset_rid` and `version` to see what a bag will contain before downloading.

Returns row counts and asset file sizes per table. Use this to:
- Verify the bag includes the expected tables
- Decide whether to increase the timeout or use `exclude_tables`
- Estimate disk space needed

## When Downloads Are Slow or Timing Out

Deep FK chains (e.g., Image → Sample → Subject → Study → Institution) can produce expensive server-side joins. Three solutions, in order of preference:

### 1. Increase the download timeout

The default read timeout is 610 seconds (~10 min). For large datasets, call `download_dataset` with `timeout`: `[10, 1800]`. The first value is the connect timeout (rarely needs changing), the second is the read timeout (30 min in this example).

### 2. Exclude tables from the FK graph

Prune tables whose data you don't need by calling `download_dataset` with `exclude_tables` (e.g., `["Study", "Institution"]`). This prevents traversal into those tables entirely.

### 3. Add intermediate records as direct members

Register intermediate tables as element types and add their records as dataset members. This replaces deep FK joins with simpler association lookups.

## Working with Bag Contents

Once downloaded, the bag is a `DatasetBag` object:

```python
# Access tables as DataFrames
images_df = bag.get_table_as_dataframe("Image")
subjects_df = bag.get_table_as_dataframe("Subject")

# Restructure assets for ML frameworks (e.g., PyTorch ImageFolder)
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./ml_data"),
    group_by=["Diagnosis"],
)
```

For full details on restructuring assets, see the `prepare-training-data` skill.

## Hydra-Zen Configuration

Both `timeout` and `exclude_tables` are available on `DatasetSpecConfig`:

```python
from deriva_ml.dataset.aux_classes import DatasetSpecConfig

DatasetSpecConfig(rid="28EA", version="0.4.0", timeout=[10, 1800])
DatasetSpecConfig(rid="28EA", version="0.4.0", exclude_tables=["Study", "Institution"])
```
