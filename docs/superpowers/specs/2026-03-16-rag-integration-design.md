# RAG Integration Across MCP Tools and Resources

**Date:** 2026-03-16
**Status:** Draft
**Repo:** deriva-mcp

## Goal

Integrate the existing RAG system (ChromaDB + schema indexing) deeper into the MCP server so that tools and resources leverage semantic search to improve the user experience. The RAG system currently exists as a standalone set of tools (`rag_search`, `rag_index_schema`, etc.) and backs the `deriva://docs/*` resources. This design extends RAG into tool error handling, entity creation safety, schema freshness, and resource enrichment.

### Design Principles

- **Option A approach**: RAG is a fallback on failure, not a pre-check on every call (except for creation tools where duplicate detection is worth the cost).
- **ACL-aware**: Schema and table contents may be restricted per-user. All RAG queries use the user's `schema_hash` (for schema) or `user_id` (for data) to scope results. No data leaks between users.
- **Two index tiers**: Schema structure is indexed per visibility class (shared among users with identical permissions). Record-level data (datasets, experiments, and optionally other tables) is indexed per user since row-level ACLs may differ.
- **Graceful degradation**: All RAG integration is no-op when RAG is not initialized or no catalog is connected. Tools work identically to today if RAG is unavailable.

## Architecture Overview

Five layers of RAG integration, all sharing a common helper module:

```
LLM / User
    |
    v  MCP Protocol
+-------------------------------------------+
|           DERIVA MCP SERVER               |
|                                           |
|  Layer 1: Auto-Reindex After Mutations    |
|  Layer 2: Error Recovery ("Did you mean?")|
|  Layer 3: Duplicate Detection on Creation |
|  Layer 4: Resource Enrichment (_related_) |
|  Layer 5: Per-User Data Indexing          |
|                                           |
|  +-------------------------------------+ |
|  |     rag/helpers.py (shared module)   | |
|  |  trigger_schema_reindex()            | |
|  |  trigger_data_reindex()              | |
|  |  rag_suggest_entity()                | |
|  |  rag_suggest_record()                | |
|  |  rag_enrich_resource()               | |
|  +-------------------------------------+ |
+-------------------------------------------+
    |
    v
  ChromaDB (deriva_docs collection)
  - Doc chunks (shared across users)
  - Schema chunks (per visibility class)
  - Data chunks (per user)
```

## Shared Helper Module

**New file:** `src/deriva_mcp/rag/helpers.py`

Five public functions. All are no-ops if RAG is not initialized or no catalog is connected.

### `trigger_schema_reindex(conn_info: ConnectionInfo | None) -> None`

- Fire-and-forget: spawns a daemon thread that re-fetches the schema and vocabulary terms, then reindexes.
- **Replaces** the inline `_index_schema_background()` function currently in `tools/catalog.py`. That function is extracted into `helpers.py` so it can be called from multiple tool modules. The connect-time path in `connect_catalog` is updated to call `trigger_schema_reindex(conn_info)` instead of `_index_schema_background(ml, hostname, catalog_id, conn_info)`.
- Used by Layer 1 after schema mutations.
- If `conn_info` is `None` or RAG manager not initialized, silently returns.
- **Debounce**: Uses a timestamp on `conn_info` (`_schema_reindex_at`) to skip if a reindex was triggered within the last 30 seconds. Prevents spawning dozens of threads during batch schema operations (e.g., creating 10 tables in sequence).

### `rag_suggest_entity(query: str, conn_info: ConnectionInfo | None, limit: int = 3) -> list[dict]`

- Searches the user's visibility-class schema index for entities matching the query.
- Returns `[{"name": "Diagnosis_Type", "type": "vocabulary", "relevance": 0.87, "description": "..."}]`.
- Used by Layer 2 (error recovery) and Layer 3 (duplicate detection).
- **ACL enforcement**: uses `conn_info.schema_hash` to build the source filter — never searches across visibility classes or the global doc index.
- Returns empty list if RAG not initialized, no connection, or no `schema_hash`.
- **Dirty flag check**: if `conn_info.schema_dirty` is `True`, triggers a background reindex and clears the flag before searching.

### `rag_enrich_resource(query: str, conn_info: ConnectionInfo | None, limit: int = 3) -> list[dict]`

- Searches the **doc index only** (not the schema index — the resource already contains schema data).
- Returns `[{"title": "Creating Tables", "source": "deriva-ml-docs", "url": "https://...", "relevance": 0.85}]`.
- Filters to results with relevance > 0.7 and deduplicates by URL.
- Returns lightweight objects (title, source, url, relevance) — no full text snippets.
- Used by Layer 4 to append `_related_docs` to resource responses.
- Returns empty list if RAG not available.
- **No dirty flag check** — this function searches only the doc index (GitHub repos), which is independent of catalog schema or data state. Dirty flags are only relevant for `rag_suggest_entity` and `rag_suggest_record`.

### `trigger_data_reindex(conn_info: ConnectionInfo | None) -> None`

- Fire-and-forget: spawns a daemon thread that fetches user-visible records from indexed tables and upserts them into the per-user data index.
- Source key: `data:{hostname}:{catalog_id}:{user_id}`.
- If `conn_info` is `None` or RAG manager not initialized, silently returns.
- Used by Layer 5 after data mutations (dataset creation, execution completion, etc.).
- **Debounce**: Same 30-second debounce as `trigger_schema_reindex`, using `_data_reindex_at` on `conn_info`.

### `rag_suggest_record(query: str, conn_info: ConnectionInfo | None, limit: int = 5) -> list[dict]`

- Searches the user's **data index** for records matching the query (datasets, experiments, and any other indexed tables).
- Returns `[{"name": "Training Set v2", "table": "Dataset", "rid": "1-ABC", "relevance": 0.91, "description": "..."}]`.
- **ACL enforcement**: uses `conn_info.user_id` to scope the search — each user has their own data index keyed by `data:{host}:{catalog}:{user_id}`.
- Used by dataset/experiment tools and resources to help users find records.
- Returns empty list if RAG not initialized, no connection, or no data index exists.

### ConnectionInfo changes

Add fields to `ConnectionInfo` in `connection.py`:

```python
schema_dirty: bool = False       # Set by vocab tools, cleared by helpers on lazy reindex
data_dirty: bool = False         # Set by data mutation tools, cleared by helpers on lazy reindex
_schema_reindex_at: float = 0.0  # Timestamp of last schema reindex trigger (for debounce)
_data_reindex_at: float = 0.0    # Timestamp of last data reindex trigger (for debounce)
```

**Thread safety note**: Simple `bool` flag sets/clears are atomic under CPython's GIL, so no lock is needed for the dirty flags. The debounce timestamps use a check-then-set pattern that is **not** atomic — two concurrent tool calls could both pass the 30-second check and both spawn reindex threads. This is benign: schema/data reindexing is idempotent (the second thread will see the same content and produce the same index). The debounce is best-effort to reduce unnecessary work, not a correctness guarantee.

**Stdio mode note**: In single-user stdio mode, `user_id` defaults to `"default_user"`. All sessions share the same data index, which is correct since there's only one user. The design handles this transparently.

## Layer 1: Auto-Reindex After Schema Mutations

### Tier 1 — Immediate reindex (schema-structural changes)

These tools call `trigger_schema_reindex(conn_info)` after a successful operation:

| Tool | File | What changed |
|------|------|-------------|
| `create_table` | `tools/schema.py` | New table in schema |
| `create_asset_table` | `tools/schema.py` | New asset table |
| `create_vocabulary` | `tools/vocabulary.py` | New vocabulary table |
| `create_feature` | `tools/feature.py` | New feature definition |
| `add_column` | `tools/schema.py` | Table structure changed |
| `delete_feature` | `tools/feature.py` | Feature removed |

The reindex is background/fire-and-forget — it does not block the tool response. The schema hash changes because schema content changed, so the old index gets replaced.

### Tier 2 — Dirty flag (vocabulary term changes)

These tools set `conn_info.schema_dirty = True` after a successful operation:

| Tool | File | What changed |
|------|------|-------------|
| `add_term` | `tools/vocabulary.py` | New vocab term |
| `delete_term` | `tools/vocabulary.py` | Term removed |
| `update_term_description` | `tools/vocabulary.py` | Term description changed |
| `add_synonym` | `tools/vocabulary.py` | Synonym added |
| `remove_synonym` | `tools/vocabulary.py` | Synonym removed |

The actual reindex happens lazily when `rag_suggest_entity()` is next called and sees `schema_dirty=True`. This avoids reindexing 50 times during a batch vocabulary load.

### What does NOT trigger schema reindex

- Metadata-only changes (`set_table_description`, `set_column_display_name`, annotation tools)
- Data operations (`insert_records`, `query_table`)
- Dataset/execution/workflow operations (these trigger **data** reindex instead — see Layer 5)

## Layer 2: Error Recovery with RAG Suggestions

### Target tools

| Tool | File | Parameter(s) | Error pattern |
|------|------|--------------|---------------|
| `query_table` | `tools/data.py` | `table_name` | Table not found |
| `count_table` | `tools/data.py` | `table_name` | Table not found |
| `get_table` | `tools/data.py` | `table_name` | Table not found |
| `insert_records` | `tools/data.py` | `table_name` | Table not found |
| `add_term` | `tools/vocabulary.py` | `vocabulary_name` | Vocabulary table not found |
| `delete_term` | `tools/vocabulary.py` | `vocabulary_name` | Vocabulary table not found |
| `fetch_table_features` | `tools/feature.py` | `table_name`, `feature_name` | Table or feature not found |
| `add_column` | `tools/schema.py` | `table_name` | Table not found |
| `get_table_sample_data` | `tools/annotation.py` | `table_name` | Table not found |

### Pattern

In the `except` block, detect "not found" errors and append RAG suggestions:

```python
except Exception as e:
    error_msg = str(e)
    result = {"status": "error", "message": error_msg}

    if _is_not_found_error(error_msg):
        conn_info = conn_manager.get_active_connection_info()
        suggestions = rag_suggest_entity(table_name, conn_info)
        if suggestions:
            result["suggestions"] = suggestions
            result["hint"] = f"Did you mean: {suggestions[0]['name']}?"

    return json.dumps(result)
```

### `_is_not_found_error(message: str) -> bool`

A utility in `helpers.py` that checks if an error message indicates an **entity not found** condition. Matches patterns specific to table/vocabulary/feature lookups:
- `"table not found"`, `"no such table"`
- `"does not exist"` (when preceded by a table/entity name)
- `"could not find"`, `"not found in schema"`

Deliberately excludes generic patterns like bare `"KeyError"` to avoid false positives (e.g., a missing column in a record dict should not trigger RAG suggestions). The heuristic is intentionally conservative — it's better to miss a suggestion than to show irrelevant ones.

Works with existing error patterns from `deriva-ml` and ERMrest without requiring changes to those libraries.

### Response format (when suggestions found)

```json
{
  "status": "error",
  "message": "Table 'Diagnoiss' not found in schema 'isa'",
  "suggestions": [
    {"name": "Diagnosis", "type": "table", "relevance": 0.92, "description": "Patient diagnosis records"},
    {"name": "Diagnosis_Type", "type": "vocabulary", "relevance": 0.85, "description": "Types of diagnoses"}
  ],
  "hint": "Did you mean: Diagnosis?"
}
```

When `rag_suggest_entity` returns empty, the response is unchanged from current behavior.

## Layer 3: Duplicate Detection on Creation

### Target tools

| Tool | File | What it creates |
|------|------|----------------|
| `create_table` | `tools/schema.py` | Domain table |
| `create_asset_table` | `tools/schema.py` | Asset table |
| `create_vocabulary` | `tools/vocabulary.py` | Vocabulary table |
| `create_feature` | `tools/feature.py` | Feature definition |

### Pattern

Pre-check at the top of the tool, before calling into `deriva-ml`:

```python
# Layer 3: Check for semantic near-duplicates
conn_info = conn_manager.get_active_connection_info()
similar = rag_suggest_entity(vocabulary_name, conn_info, limit=3)
warnings = [
    s for s in similar
    if s["relevance"] > 0.8 and s["name"].lower() != vocabulary_name.lower()
]

# ... proceed with creation ...

# Attach warnings to success response
if warnings:
    result["similar_existing"] = warnings
    result["warning"] = (
        f"Created '{vocabulary_name}', but similar entities exist: "
        f"{', '.join(w['name'] for w in warnings)}. "
        f"Verify this isn't a duplicate."
    )
```

### Design decisions

- **Warning, not blocking.** The tool still creates the entity. The LLM can decide whether to proceed or roll back using the corresponding deletion tool (`delete_catalog` is too heavy, but individual entities can be removed — e.g., drop and recreate a table, or `delete_term` for vocabulary terms). There is no single-step "undo create" tool.
- **Relevance threshold of 0.8.** Below this, matches are too fuzzy. Above this, they're likely meaningful near-duplicates. This threshold (and the 0.7 threshold for `rag_enrich_resource`) should be defined as named constants in `helpers.py` (e.g., `DUPLICATE_RELEVANCE_THRESHOLD = 0.8`, `ENRICHMENT_RELEVANCE_THRESHOLD = 0.7`) so they can be tuned based on experience with the ONNX MiniLM model.
- **Exact match excluded.** If someone creates "Diagnosis" and "Diagnosis" already exists, that's an ERMrest conflict error, not a RAG concern.

### Response format (when near-duplicates found)

```json
{
  "status": "created",
  "name": "Diagnosis",
  "schema": "isa",
  "similar_existing": [
    {"name": "Diagnosis_Type", "type": "vocabulary", "relevance": 0.88, "description": "Types of diagnoses"}
  ],
  "warning": "Created 'Diagnosis', but similar entities exist: Diagnosis_Type. Verify this isn't a duplicate."
}
```

When no near-duplicates are found, the response is unchanged.

## Layer 4: Resource Enrichment

### Target resources

| Resource | Query for RAG | Why |
|----------|--------------|-----|
| `deriva://catalog/schema` | `"catalog schema tables columns"` | Links to table creation docs, FK guide |
| `deriva://catalog/datasets` | `"creating managing datasets"` | Dataset workflow docs |
| `deriva://catalog/features` | `"defining using features"` | Feature definition docs |
| `deriva://catalog/vocabularies` | `"controlled vocabularies terms"` | Vocabulary management docs |
| `deriva://catalog/workflows` | `"workflows executions"` | Workflow/execution docs |
| `deriva://table/{name}/schema` | `"table {name} columns foreign keys"` | Context-specific docs for that table type |
| `deriva://table/{name}/features` | `"features {name}"` | How to add/query feature values |
| `deriva://dataset/{rid}` | `"dataset members versions"` | Dataset versioning, bag export docs |

### What does NOT get enrichment

- Config templates (`deriva://config/*`) — static code samples
- Storage resources (`deriva://storage/*`) — filesystem info
- Doc resources (`deriva://docs/*`) — already RAG-backed (would be circular)
- Citation/RID resources — utility lookups
- Annotation resources — too granular

### Pattern

At the end of a resource function, before returning:

```python
conn_info = conn_manager.get_active_connection_info()
related = rag_enrich_resource("catalog schema tables columns", conn_info)
if related:
    result["_related_docs"] = related

return json.dumps(result, indent=2)
```

### Parameterized resources get contextual queries

For `deriva://table/{name}/schema`, the query adapts to the table type:

```python
query_parts = [f"table {table_name}"]
if table_info.get("is_vocabulary"):
    query_parts.append("vocabulary controlled terms")
elif table_info.get("is_asset"):
    query_parts.append("asset file upload management")
query = " ".join(query_parts)
```

### Response format

```json
{
  "tables": ["Image", "Subject"],
  "_related_docs": [
    {
      "title": "Creating Tables",
      "source": "deriva-ml-docs",
      "url": "https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/tables.md",
      "relevance": 0.89
    }
  ]
}
```

## Layer 5: Per-User Data Indexing

### Problem

The schema index captures **structure** (tables, columns, FKs, features, vocabulary terms) but not **data records**. Users exploring a catalog often ask "what datasets exist?", "which experiment used the training data?", or "find the dataset with lung images". Today, these questions require the LLM to call `query_table("Dataset")` and scan through results. With data indexing, RAG can answer these questions semantically.

### ACL constraint

Dataset and experiment records are subject to per-user row-level ACLs. Two users connected to the same catalog may see different datasets. Unlike schema (where a small number of visibility classes exist), row-level ACL combinations can be unique per user. Therefore, the data index is **keyed per user**, not per visibility class.

### Source key pattern

```
data:{hostname}:{catalog_id}:{user_id}
```

Example: `data:dev.eye-ai.org:52:a1b2c3d4e5f6`

This is distinct from the schema source key (`schema:{host}:{catalog}:{schema_hash}`). The `user_id` comes from `conn_info.user_id`, which is derived from the user's authentication credential at connect time.

### What gets indexed

**Default indexed tables:**

| Table | What's indexed | Why |
|-------|---------------|-----|
| `Dataset` | Name, description, types, version, RID | Core exploration — "find the training dataset" |
| `Execution` | Description, workflow name, status, datasets used, RID | "Which experiment used dataset X?" |

**Extensible to additional tables:**

The data indexer accepts a list of table names to index. The default is `["Dataset", "Execution"]`, but this can be extended via configuration or a future `rag_index_table_data` tool. Any domain table can be added (e.g., "Subject", "Image") as long as its records are accessible to the connected user.

When indexing additional tables:
- Fetch records visible to the current user via `query_table`
- Convert each record to a markdown snippet (name/description fields + key metadata)
- Chunk and embed in the per-user data source

### Indexing flow

**At connect time** (after schema indexing completes):

```
1. conn_info.user_id is set from credentials
2. Check if data:{host}:{catalog}:{user_id} source exists in ChromaDB
3. If exists and not stale (< 1 hour old), skip
4. Otherwise, background thread:
   a. Fetch Dataset records visible to this user
   b. Fetch Execution records visible to this user
   c. Convert to markdown, chunk, embed
   d. Upsert as source data:{host}:{catalog}:{user_id}
```

**Staleness check**: Data records change more frequently than schema. We use a simple time-based staleness check (1 hour default) rather than content hashing, since computing a hash over all records is expensive. The staleness timestamp is stored in ChromaDB chunk metadata (`indexed_at` field, same pattern as schema chunks) — on index, the newest `indexed_at` across all chunks for the user's data source is compared against the current time. The `data_dirty` flag on `conn_info` can also force a refresh regardless of staleness.

### Record-to-markdown conversion

Each record becomes a small markdown document:

```markdown
## Dataset: Training Set v2 (RID: 1-ABC)

**Description:** 500 annotated lung CT images for model training
**Types:** Training
**Version:** 0.4.0
**Created:** 2026-03-10
```

```markdown
## Execution: Lung Segmentation Run 3 (RID: 2-DEF)

**Workflow:** Lung Segmentation Pipeline
**Status:** Completed
**Input Datasets:** Training Set v2 (1-ABC), Validation Set (1-XYZ)
**Description:** Third training run with augmented data
**Created:** 2026-03-12
```

These are small enough that each record is typically a single chunk (under 800 tokens).

### Data mutation triggers (dirty flag)

These tools set `conn_info.data_dirty = True`:

| Tool | File | What changed |
|------|------|-------------|
| `create_dataset` | `tools/dataset.py` | New dataset |
| `delete_dataset` | `tools/dataset.py` | Dataset removed |
| `set_dataset_description` | `tools/dataset.py` | Dataset metadata changed |
| `add_dataset_type` | `tools/dataset.py` | Dataset type changed |
| `remove_dataset_type` | `tools/dataset.py` | Dataset type changed |
| `add_dataset_members` | `tools/dataset.py` | Dataset composition changed |
| `delete_dataset_members` | `tools/dataset.py` | Dataset composition changed |
| `create_execution` | `tools/execution.py` | New execution |
| `stop_execution` | `tools/execution.py` | Execution completed |
| `update_execution_status` | `tools/execution.py` | Execution status changed |
| `create_execution_dataset` | `tools/execution.py` | New output dataset |
| `insert_records` | `tools/data.py` | Records added to a user-configured indexed table (see note) |

**Note on `insert_records`:** The default indexed tables (Dataset, Execution) are managed by dedicated tools and blocked from `insert_records` by `_MANAGED_TABLES`. The dirty flag on `insert_records` only fires when the user has extended the indexed table list (via configuration or `rag_index_table_data`) to include domain tables that accept direct inserts (e.g., "Subject", "Image"). If the target table is not in the indexed table list, no dirty flag is set.

The actual reindex happens lazily when `rag_suggest_record()` is next called and sees `data_dirty=True`.

### Integration into tools and resources

**Layer 2 extension — Error recovery for dataset/experiment lookups:**

When a dataset or execution tool fails on a bad RID or name, `rag_suggest_record()` provides alternatives:

```json
{
  "status": "error",
  "message": "Dataset '1-ZZZ' not found",
  "suggestions": [
    {"name": "Training Set v2", "table": "Dataset", "rid": "1-ABC", "relevance": 0.89}
  ],
  "hint": "Did you mean: Training Set v2 (1-ABC)?"
}
```

Target tools for data-level error recovery:

| Tool | File | What's searched |
|------|------|----------------|
| `add_dataset_members` | `tools/dataset.py` | Dataset by RID/name |
| `delete_dataset_members` | `tools/dataset.py` | Dataset by RID/name |
| `set_dataset_description` | `tools/dataset.py` | Dataset by RID/name |
| `add_dataset_type` | `tools/dataset.py` | Dataset by RID/name |
| `download_dataset` | `tools/dataset.py` | Dataset by RID/name |
| `split_dataset` | `tools/dataset.py` | Dataset by RID/name |
| `denormalize_dataset` | `tools/dataset.py` | Dataset by RID/name |
| `restore_execution` | `tools/execution.py` | Execution by RID |
| `download_execution_dataset` | `tools/execution.py` | Execution/dataset by RID |

**Layer 4 extension — Resource enrichment with data context:**

Dataset and execution resources gain `_related_data` alongside `_related_docs`:

| Resource | Data query | What it adds |
|----------|-----------|-------------|
| `deriva://catalog/datasets` | `"datasets"` | Not needed (already listing all) |
| `deriva://dataset/{rid}` | `"dataset {name} {description}"` | Related datasets, executions that used it |
| `deriva://catalog/executions` | `"executions experiments"` | Not needed (already listing all) |
| `deriva://execution/{rid}` | `"execution {workflow} {description}"` | Related executions, input/output datasets |

```json
{
  "rid": "1-ABC",
  "name": "Training Set v2",
  "_related_data": [
    {"name": "Lung Segmentation Run 3", "table": "Execution", "rid": "2-DEF", "relevance": 0.87, "relationship": "used this dataset"}
  ],
  "_related_docs": [
    {"title": "Dataset Versioning", "source": "deriva-ml-docs", "url": "https://...", "relevance": 0.82}
  ]
}
```

### `rag_search` tool update

The existing `rag_search` tool should also search the per-user data index when connected. Update `rag_search` in `tools/rag.py` to include a third search alongside docs and schema:

```python
data_results = []
if include_data:  # new parameter, default True
    data_source = data_source_name(conn_info.hostname, conn_info.catalog_id, conn_info.user_id)
    data_results = manager.search(query=query, limit=limit, source=data_source)

all_results = doc_results + schema_results + data_results
all_results.sort(key=lambda r: r.get("relevance", 0), reverse=True)
```

**Result shape note**: Data results include `rid` and `table` fields that doc/schema results lack. Doc results include `github_url` and `repo` fields that data results lack. The merged result list has a union shape — consumers should check for field presence rather than assuming a fixed schema. Each result always has: `text`, `relevance`, `source`, `doc_type`. Additional fields vary by source type.

## Skill Deprecation: `deriva:semantic-awareness`

Layer 3 (duplicate detection) supersedes the `deriva:semantic-awareness` skill in the `deriva-skills` repository. The skill currently works at the LLM prompt level — it tells the LLM to call `rag_search` before creation tools. Layer 3 moves this check into the tool implementation itself, making it:

- **Automatic** — cannot be skipped by the LLM
- **Lower token cost** — one tool call instead of two
- **Client-agnostic** — works for any MCP client, not just Claude with the skill installed

**Recommendation:** Simplify the skill to a one-liner noting that the MCP server handles duplicate detection natively. Keep the skill (rather than removing it) so that users on older MCP server versions without Layer 3 still get the behavioral guardrail.

## Files Changed

| File | Changes |
|------|---------|
| `src/deriva_mcp/rag/helpers.py` | **New.** Five helper functions + `_is_not_found_error`. |
| `src/deriva_mcp/rag/data.py` | **New.** Per-user data indexing: `index_user_data()`, `data_source_name()`, record-to-markdown conversion. |
| `src/deriva_mcp/connection.py` | Add `schema_dirty: bool = False` and `data_dirty: bool = False` to `ConnectionInfo`. |
| `src/deriva_mcp/tools/schema.py` | Layer 1 reindex on `create_table`, `create_asset_table`, `add_column`. Layer 2 error recovery on `add_column`. Layer 3 duplicate check on `create_table`, `create_asset_table`. |
| `src/deriva_mcp/tools/vocabulary.py` | Layer 1 reindex on `create_vocabulary`. Layer 1 dirty flag on `add_term`, `delete_term`, `update_term_description`, `add_synonym`, `remove_synonym`. Layer 2 error recovery on `add_term`, `delete_term`. Layer 3 duplicate check on `create_vocabulary`. |
| `src/deriva_mcp/tools/feature.py` | Layer 1 reindex on `create_feature`, `delete_feature`. Layer 2 error recovery on `fetch_table_features`. Layer 3 duplicate check on `create_feature`. |
| `src/deriva_mcp/tools/data.py` | Layer 2 error recovery on `query_table`, `count_table`, `get_table`, `insert_records`. Layer 5 data dirty flag on `insert_records`. |
| `src/deriva_mcp/tools/dataset.py` | Layer 5 data dirty flag on `create_dataset`, `delete_dataset`, `set_dataset_description`, `add_dataset_type`, `remove_dataset_type`, `add_dataset_members`, `delete_dataset_members`. Layer 2 data-level error recovery on dataset RID lookups. |
| `src/deriva_mcp/tools/execution.py` | Layer 5 data dirty flag on `create_execution`, `stop_execution`, `update_execution_status`, `create_execution_dataset`. Layer 2 data-level error recovery on execution RID lookups. |
| `src/deriva_mcp/tools/rag.py` | Update `rag_search` to include per-user data index results (new `include_data` parameter). |
| `src/deriva_mcp/tools/catalog.py` | Trigger data indexing in background after connect (alongside schema indexing). |
| `src/deriva_mcp/tools/annotation.py` | Layer 2 error recovery on `get_table_sample_data`. |
| `src/deriva_mcp/resources.py` | Layer 4 enrichment on ~8 catalog/table/dataset resources. Layer 5 `_related_data` on dataset/execution resources. |

## Testing Strategy

- **Unit tests for helpers.py**: Mock `RAGManager.search()` and verify `rag_suggest_entity` scopes by `schema_hash`, `rag_suggest_record` scopes by `user_id`, returns empty when RAG unavailable, and handles dirty flags.
- **Unit tests for data.py (rag module)**: Verify record-to-markdown conversion, data source naming, staleness check logic.
- **Unit tests for each layer**: Mock `rag_suggest_entity` / `rag_suggest_record` / `rag_enrich_resource` in tool tests. Verify suggestions appear in error responses (Layer 2), warnings appear in creation responses (Layer 3), `_related_docs` appears in resource responses (Layer 4), and `_related_data` appears in dataset/execution resources (Layer 5).
- **ACL isolation test**: Mock two users with different `user_id` values. Verify their data indexes are independent — User A's datasets don't appear in User B's `rag_suggest_record` results.
- **Integration test**: Connect to a test catalog, create a vocabulary and dataset, verify the schema and data reindex triggers (Layers 1/5), then query with a typo and verify suggestions (Layer 2).
- **Graceful degradation test**: Set RAG manager to `None`, run all tools, verify they behave identically to the current codebase (no errors, no suggestions, no enrichment).
- **Debounce test**: Trigger `trigger_schema_reindex` 10 times in rapid succession, verify only 1-2 background threads are spawned.
- **False positive test**: Verify `_is_not_found_error` rejects error messages that match generic patterns but not entity-not-found (e.g., `"KeyError: 'missing_column'"` should NOT trigger suggestions).
