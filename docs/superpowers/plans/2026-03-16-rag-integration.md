# RAG Integration Across MCP Tools and Resources — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the existing RAG system deeper into all MCP tools and resources so that schema mutations auto-reindex, tool errors suggest alternatives, entity creation detects duplicates, catalog resources link related docs, and per-user data records are semantically searchable.

**Architecture:** Five integration layers share a common helper module (`rag/helpers.py`). Layer 1 keeps the RAG index fresh via auto-reindex triggers. Layer 2 catches "not found" errors and suggests alternatives. Layer 3 pre-checks creation tools for semantic duplicates. Layer 4 appends related documentation links to catalog resources. Layer 5 adds per-user data indexing for datasets and executions.

**Tech Stack:** Python 3.11+, ChromaDB with ONNX MiniLM-L6-v2 embeddings, FastMCP, pytest with unittest.mock.

**Spec:** `docs/superpowers/specs/2026-03-16-rag-integration-design.md`

---

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `src/deriva_mcp/rag/helpers.py` | Shared RAG helper functions (5 public + 1 private) | **New** |
| `src/deriva_mcp/rag/data.py` | Per-user data indexing (record-to-markdown, staleness) | **New** |
| `src/deriva_mcp/connection.py` | Add dirty flags + debounce timestamps to ConnectionInfo | Modify |
| `src/deriva_mcp/tools/catalog.py` | Replace inline `_index_schema_background` with helper call; trigger data indexing | Modify |
| `src/deriva_mcp/tools/schema.py` | Layers 1, 2, 3 on create_table, create_asset_table, add_column | Modify |
| `src/deriva_mcp/tools/vocabulary.py` | Layers 1, 2, 3 on create_vocabulary, add_term, delete_term, etc. | Modify |
| `src/deriva_mcp/tools/feature.py` | Layers 1, 2, 3 on create_feature, delete_feature, fetch_table_features | Modify |
| `src/deriva_mcp/tools/data.py` | Layer 2 on query_table, count_table, get_table, insert_records; Layer 5 dirty flag | Modify |
| `src/deriva_mcp/tools/dataset.py` | Layer 2 error recovery; Layer 5 dirty flags | Modify |
| `src/deriva_mcp/tools/execution.py` | Layer 2 error recovery; Layer 5 dirty flags | Modify |
| `src/deriva_mcp/tools/annotation.py` | Layer 2 on get_table_sample_data | Modify |
| `src/deriva_mcp/tools/rag.py` | Update rag_search to include per-user data index | Modify |
| `src/deriva_mcp/resources.py` | Layer 4 enrichment; Layer 5 _related_data | Modify |
| `tests/test_rag_helpers.py` | Unit tests for helpers.py | **New** |
| `tests/test_rag_data.py` | Unit tests for data.py | **New** |
| `tests/test_rag_integration_layers.py` | Tests for layers 1-5 across tools | **New** |
| `deriva-skills/.../semantic-awareness/SKILL.md` | Deprecation note | Modify (separate repo) |

---

## Chunk 1: Foundation — ConnectionInfo Fields + Shared Helpers

### Task 1: Add dirty flags and debounce timestamps to ConnectionInfo

**Files:**
- Modify: `src/deriva_mcp/connection.py:90-108`

- [ ] **Step 1: Read the current ConnectionInfo**

The dataclass is at line 90. We need to add 4 new fields after `schema_hash`.

- [ ] **Step 2: Add the new fields**

```python
# In ConnectionInfo dataclass, after schema_hash:
schema_dirty: bool = False       # Vocab changes → lazy reindex
data_dirty: bool = False         # Data mutations → lazy reindex
_schema_reindex_at: float = 0.0  # Debounce timestamp
_data_reindex_at: float = 0.0    # Debounce timestamp
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `pytest tests/test_connection.py -v`
Expected: All existing tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/deriva_mcp/connection.py
git commit -m "feat: add dirty flags and debounce timestamps to ConnectionInfo"
```

### Task 2: Create `rag/helpers.py` — constants and `_is_not_found_error`

**Files:**
- Create: `src/deriva_mcp/rag/helpers.py`
- Create: `tests/test_rag_helpers.py`

- [ ] **Step 1: Write the failing test for `_is_not_found_error`**

```python
# tests/test_rag_helpers.py
"""Tests for RAG helper functions."""

import pytest


class TestIsNotFoundError:
    """Tests for the _is_not_found_error heuristic."""

    def test_table_not_found(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("Table 'Diagnoiss' not found in schema 'isa'") is True

    def test_no_such_table(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("no such table: Subject_Info") is True

    def test_does_not_exist(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("Table 'FooBar' does not exist") is True

    def test_could_not_find(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("could not find table Image_Data") is True

    def test_not_found_in_schema(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("'BadName' not found in schema") is True

    def test_generic_key_error_rejected(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("KeyError: 'missing_column'") is False

    def test_permission_error_rejected(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("Permission denied: cannot access table") is False

    def test_empty_string(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("") is False

    def test_connection_error_rejected(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("Connection refused") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_helpers.py::TestIsNotFoundError -v`
Expected: FAIL with ImportError (module doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/deriva_mcp/rag/helpers.py
"""Shared RAG helper functions for MCP tool and resource integration.

Five public helpers used across Layers 1-5 of the RAG integration:
- trigger_schema_reindex: Fire-and-forget schema reindex (Layer 1)
- trigger_data_reindex: Fire-and-forget data reindex (Layer 5)
- rag_suggest_entity: Search schema index for "did you mean?" (Layers 2, 3)
- rag_suggest_record: Search data index for record suggestions (Layer 5)
- rag_enrich_resource: Search doc index for related docs (Layer 4)

All functions are no-ops when RAG is not initialized or no catalog is connected.
"""

from __future__ import annotations

import logging
import re
import time
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deriva_mcp.connection import ConnectionInfo

logger = logging.getLogger("deriva-mcp")

# ---------------------------------------------------------------------------
# Named constants for relevance thresholds
# ---------------------------------------------------------------------------
DUPLICATE_RELEVANCE_THRESHOLD = 0.8   # Layer 3: near-duplicate detection
ENRICHMENT_RELEVANCE_THRESHOLD = 0.7  # Layer 4: resource enrichment
DEBOUNCE_SECONDS = 30.0               # Minimum seconds between reindex triggers

# ---------------------------------------------------------------------------
# Not-found error detection patterns
# ---------------------------------------------------------------------------
_NOT_FOUND_PATTERNS = [
    re.compile(r"table.+not found", re.IGNORECASE),
    re.compile(r"no such table", re.IGNORECASE),
    re.compile(r"does not exist", re.IGNORECASE),
    re.compile(r"could not find", re.IGNORECASE),
    re.compile(r"not found in schema", re.IGNORECASE),
]


def _is_not_found_error(message: str) -> bool:
    """Check if an error message indicates an entity-not-found condition.

    Matches patterns specific to table/vocabulary/feature lookups.
    Deliberately excludes generic patterns like bare "KeyError" to
    avoid false positives.
    """
    if not message:
        return False
    return any(p.search(message) for p in _NOT_FOUND_PATTERNS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_helpers.py::TestIsNotFoundError -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_mcp/rag/helpers.py tests/test_rag_helpers.py
git commit -m "feat: add _is_not_found_error heuristic to rag/helpers.py"
```

### Task 3: Implement `trigger_schema_reindex` in helpers.py

**Files:**
- Modify: `src/deriva_mcp/rag/helpers.py`
- Modify: `tests/test_rag_helpers.py`

- [ ] **Step 1: Write failing test for trigger_schema_reindex**

```python
# Append to tests/test_rag_helpers.py
from unittest.mock import MagicMock, patch
import time


class TestTriggerSchemaReindex:
    """Tests for trigger_schema_reindex."""

    def _make_conn_info(self):
        conn_info = MagicMock()
        conn_info.ml_instance = MagicMock()
        conn_info.hostname = "test.example.org"
        conn_info.catalog_id = "1"
        conn_info.schema_hash = "abc123"
        conn_info._schema_reindex_at = 0.0
        conn_info.schema_dirty = False
        return conn_info

    def test_noop_when_conn_info_none(self):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        # Should not raise
        trigger_schema_reindex(None)

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_noop_when_rag_not_initialized(self, mock_get_rag):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        mock_get_rag.return_value = None
        trigger_schema_reindex(self._make_conn_info())
        # No error, no thread spawned

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_debounce_skips_rapid_calls(self, mock_get_rag):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        mock_get_rag.return_value = MagicMock()
        conn_info = self._make_conn_info()
        conn_info._schema_reindex_at = time.time()  # Just triggered
        trigger_schema_reindex(conn_info)
        # Should be debounced — no thread started

    @patch("deriva_mcp.rag.helpers.threading.Thread")
    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_spawns_thread_when_stale(self, mock_get_rag, mock_thread_cls):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        mock_get_rag.return_value = MagicMock()
        conn_info = self._make_conn_info()
        conn_info._schema_reindex_at = 0.0  # Never triggered
        trigger_schema_reindex(conn_info)
        mock_thread_cls.assert_called_once()
        mock_thread_cls.return_value.start.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_helpers.py::TestTriggerSchemaReindex -v`
Expected: FAIL (function doesn't exist yet).

- [ ] **Step 3: Implement trigger_schema_reindex**

Add to `src/deriva_mcp/rag/helpers.py`:

```python
def trigger_schema_reindex(conn_info: ConnectionInfo | None) -> None:
    """Fire-and-forget background schema reindex.

    Spawns a daemon thread that re-fetches the schema and vocabulary
    terms, then reindexes. Debounces to at most once per DEBOUNCE_SECONDS.

    No-op if conn_info is None or RAG is not initialized.
    """
    if conn_info is None:
        return

    from deriva_mcp.rag import get_rag_manager
    manager = get_rag_manager()
    if manager is None:
        return

    # Debounce: skip if reindex was triggered recently
    now = time.time()
    if now - conn_info._schema_reindex_at < DEBOUNCE_SECONDS:
        return

    conn_info._schema_reindex_at = now

    def _do_reindex():
        try:
            from deriva_mcp.rag.schema import compute_schema_hash

            ml = conn_info.ml_instance
            schema_info = ml.model.get_schema_description()

            # Fetch vocabulary terms for all vocabulary tables
            vocab_terms: dict[str, list[dict[str, str]]] = {}
            schemas = schema_info.get("schemas", {})
            for schema_data in schemas.values():
                tables = schema_data.get("tables", {})
                for table_name, table_info in tables.items():
                    if table_info.get("is_vocabulary"):
                        try:
                            terms = ml.list_vocabulary_terms(table_name)
                            vocab_terms[table_name] = [
                                {
                                    "Name": t.name,
                                    "Description": t.description or "",
                                    "Synonyms": list(t.synonyms) if t.synonyms else [],
                                }
                                for t in terms
                            ]
                        except Exception:
                            pass

            schema_hash = compute_schema_hash(schema_info, vocab_terms)
            conn_info.schema_hash = schema_hash

            result = manager.index_catalog_schema(
                schema_info, conn_info.hostname, conn_info.catalog_id,
                vocabulary_terms=vocab_terms,
            )
            status = result.get("status", "unknown")
            if status == "indexed":
                logger.info(
                    f"Reindexed schema for {conn_info.hostname}:{conn_info.catalog_id} "
                    f"(visibility class {schema_hash}): "
                    f"{result.get('chunks_created', 0)} chunks"
                )
        except Exception as e:
            logger.warning(f"Background schema reindex failed: {e}")

    thread = threading.Thread(target=_do_reindex, daemon=True, name="schema-rag-reindex")
    thread.start()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_helpers.py::TestTriggerSchemaReindex -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_mcp/rag/helpers.py tests/test_rag_helpers.py
git commit -m "feat: add trigger_schema_reindex to rag/helpers.py"
```

### Task 4: Implement `rag_suggest_entity` in helpers.py

**Files:**
- Modify: `src/deriva_mcp/rag/helpers.py`
- Modify: `tests/test_rag_helpers.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_rag_helpers.py
class TestRagSuggestEntity:
    """Tests for rag_suggest_entity."""

    def _make_conn_info(self, schema_hash="abc123", schema_dirty=False):
        conn_info = MagicMock()
        conn_info.hostname = "test.example.org"
        conn_info.catalog_id = "1"
        conn_info.schema_hash = schema_hash
        conn_info.schema_dirty = schema_dirty
        conn_info._schema_reindex_at = 0.0
        conn_info.ml_instance = MagicMock()
        return conn_info

    def test_returns_empty_when_no_conn(self):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        assert rag_suggest_entity("Diagnosis", None) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_returns_empty_when_rag_not_init(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        mock_get_rag.return_value = None
        assert rag_suggest_entity("Diagnosis", self._make_conn_info()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_returns_empty_when_no_schema_hash(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        mock_get_rag.return_value = MagicMock()
        conn_info = self._make_conn_info(schema_hash=None)
        assert rag_suggest_entity("Diagnosis", conn_info) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_searches_schema_index(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {
                "text": "## isa.Diagnosis (vocabulary)\nDiagnosis terms",
                "relevance": 0.92,
                "source": "schema:test.example.org:1:abc123",
                "section_heading": "isa.Diagnosis",
                "doc_type": "catalog-schema",
            }
        ]
        mock_get_rag.return_value = mock_manager
        conn_info = self._make_conn_info()
        results = rag_suggest_entity("Diagnoiss", conn_info, limit=3)
        assert len(results) == 1
        assert results[0]["name"] == "Diagnosis"
        assert results[0]["relevance"] == 0.92

    @patch("deriva_mcp.rag.helpers.trigger_schema_reindex")
    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_triggers_reindex_when_dirty(self, mock_get_rag, mock_reindex):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        mock_manager = MagicMock()
        mock_manager.search.return_value = []
        mock_get_rag.return_value = mock_manager
        conn_info = self._make_conn_info(schema_dirty=True)
        rag_suggest_entity("Foo", conn_info)
        mock_reindex.assert_called_once_with(conn_info)
        assert conn_info.schema_dirty is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_helpers.py::TestRagSuggestEntity -v`
Expected: FAIL (function doesn't exist).

- [ ] **Step 3: Implement rag_suggest_entity**

Add to `src/deriva_mcp/rag/helpers.py`:

```python
def rag_suggest_entity(
    query: str,
    conn_info: ConnectionInfo | None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Search the user's visibility-class schema index for entity suggestions.

    Returns a list of dicts: [{"name": ..., "type": ..., "relevance": ..., "description": ...}]
    ACL enforcement: uses conn_info.schema_hash to scope search.
    """
    if conn_info is None:
        return []

    from deriva_mcp.rag import get_rag_manager
    manager = get_rag_manager()
    if manager is None:
        return []

    schema_hash = getattr(conn_info, "schema_hash", None)
    if not schema_hash:
        return []

    # Lazy reindex if dirty
    if getattr(conn_info, "schema_dirty", False):
        trigger_schema_reindex(conn_info)
        conn_info.schema_dirty = False

    from deriva_mcp.rag.schema import schema_source_name
    source = schema_source_name(conn_info.hostname, conn_info.catalog_id, schema_hash)

    results = manager.search(query=query, limit=limit, source=source)

    # Parse section headings to extract entity names and types
    suggestions = []
    for r in results:
        heading = r.get("section_heading", "")
        # Headings are like "isa.Diagnosis (vocabulary)" or "isa.Image (asset)"
        name = heading.split(".")[-1].split(" (")[0].strip() if "." in heading else heading.split(" (")[0].strip()
        entity_type = "unknown"
        if "(vocabulary)" in heading:
            entity_type = "vocabulary"
        elif "(asset)" in heading:
            entity_type = "asset"
        elif "(association)" in heading:
            entity_type = "association"
        else:
            entity_type = "table"

        description = r.get("text", "")[:200]

        if name:
            suggestions.append({
                "name": name,
                "type": entity_type,
                "relevance": r.get("relevance", 0),
                "description": description,
            })

    return suggestions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_helpers.py::TestRagSuggestEntity -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_mcp/rag/helpers.py tests/test_rag_helpers.py
git commit -m "feat: add rag_suggest_entity to rag/helpers.py"
```

### Task 5: Implement `rag_enrich_resource` in helpers.py

**Files:**
- Modify: `src/deriva_mcp/rag/helpers.py`
- Modify: `tests/test_rag_helpers.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_rag_helpers.py
class TestRagEnrichResource:
    """Tests for rag_enrich_resource."""

    def test_returns_empty_when_no_conn(self):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        assert rag_enrich_resource("tables", None) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_returns_empty_when_rag_not_init(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        mock_get_rag.return_value = None
        assert rag_enrich_resource("tables", MagicMock()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_filters_by_relevance(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "Creating Tables", "relevance": 0.9, "github_url": "https://example.com/tables",
             "source": "docs", "section_heading": "Creating Tables"},
            {"text": "Unrelated", "relevance": 0.5, "github_url": "https://example.com/other",
             "source": "docs", "section_heading": "Unrelated"},
        ]
        mock_get_rag.return_value = mock_manager
        results = rag_enrich_resource("tables", MagicMock())
        assert len(results) == 1
        assert results[0]["title"] == "Creating Tables"
        assert results[0]["relevance"] == 0.9

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_deduplicates_by_url(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "A", "relevance": 0.9, "github_url": "https://example.com/same",
             "source": "docs", "section_heading": "A"},
            {"text": "B", "relevance": 0.85, "github_url": "https://example.com/same",
             "source": "docs", "section_heading": "B"},
        ]
        mock_get_rag.return_value = mock_manager
        results = rag_enrich_resource("query", MagicMock())
        assert len(results) == 1  # Deduplicated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_helpers.py::TestRagEnrichResource -v`
Expected: FAIL.

- [ ] **Step 3: Implement rag_enrich_resource**

Add to `src/deriva_mcp/rag/helpers.py`:

```python
def rag_enrich_resource(
    query: str,
    conn_info: ConnectionInfo | None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Search the doc index for related documentation links.

    Returns lightweight objects: [{"title": ..., "source": ..., "url": ..., "relevance": ...}]
    Filters to relevance > ENRICHMENT_RELEVANCE_THRESHOLD and deduplicates by URL.
    Searches ONLY the doc index (not schema/data). No dirty flag check.
    """
    if conn_info is None:
        return []

    from deriva_mcp.rag import get_rag_manager
    manager = get_rag_manager()
    if manager is None:
        return []

    # Search doc index only (exclude schema and data sources)
    results = manager.search(query=query, limit=limit * 2)  # Fetch extra for filtering

    # Filter: only doc results (not schema/data), above relevance threshold
    seen_urls: set[str] = set()
    enriched: list[dict[str, Any]] = []

    for r in results:
        doc_type = r.get("doc_type", "")
        if doc_type == "catalog-schema" or doc_type == "catalog-data":
            continue  # Skip schema/data results

        relevance = r.get("relevance", 0)
        if relevance < ENRICHMENT_RELEVANCE_THRESHOLD:
            continue

        url = r.get("github_url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        enriched.append({
            "title": r.get("section_heading", "") or r.get("path", ""),
            "source": r.get("source", ""),
            "url": url,
            "relevance": relevance,
        })

        if len(enriched) >= limit:
            break

    return enriched
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_helpers.py::TestRagEnrichResource -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_mcp/rag/helpers.py tests/test_rag_helpers.py
git commit -m "feat: add rag_enrich_resource to rag/helpers.py"
```

### Task 6: Implement `trigger_data_reindex` and `rag_suggest_record` in helpers.py

**Files:**
- Modify: `src/deriva_mcp/rag/helpers.py`
- Create: `src/deriva_mcp/rag/data.py`
- Create: `tests/test_rag_data.py`
- Modify: `tests/test_rag_helpers.py`

- [ ] **Step 1: Write failing tests for data.py**

```python
# tests/test_rag_data.py
"""Tests for per-user data indexing."""

import pytest
from unittest.mock import MagicMock


class TestDataSourceName:
    def test_format(self):
        from deriva_mcp.rag.data import data_source_name
        assert data_source_name("dev.example.org", "52", "user123") == "data:dev.example.org:52:user123"

    def test_default_user(self):
        from deriva_mcp.rag.data import data_source_name
        assert data_source_name("host", "1", "default_user") == "data:host:1:default_user"


class TestRecordToMarkdown:
    def test_dataset_record(self):
        from deriva_mcp.rag.data import dataset_record_to_markdown
        record = {
            "RID": "1-ABC",
            "Description": "500 annotated lung CT images",
            "RCT": "2026-03-10T00:00:00",
        }
        md = dataset_record_to_markdown(record, types=["Training"], version="0.4.0")
        assert "1-ABC" in md
        assert "500 annotated lung CT images" in md
        assert "Training" in md
        assert "0.4.0" in md

    def test_execution_record(self):
        from deriva_mcp.rag.data import execution_record_to_markdown
        record = {
            "RID": "2-DEF",
            "Description": "Third training run",
            "RCT": "2026-03-12T00:00:00",
        }
        md = execution_record_to_markdown(record, workflow_name="Lung Seg", status="Completed")
        assert "2-DEF" in md
        assert "Third training run" in md
        assert "Lung Seg" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_data.py -v`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Create `rag/data.py`**

```python
# src/deriva_mcp/rag/data.py
"""Per-user data indexing for RAG.

Indexes dataset and execution records per-user so RAG can answer
questions like "find the training dataset" or "which experiment used
dataset X?".

Data chunks are stored in the same ChromaDB collection as docs/schema,
distinguished by ``doc_type="catalog-data"`` and a source name of
``data:{hostname}:{catalog_id}:{user_id}``.

ACL constraint: Each user gets their own data index keyed by user_id,
since row-level ACLs may differ between users.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("deriva-mcp")

# Default tables to index for data search
DEFAULT_INDEXED_TABLES = ["Dataset", "Execution"]

# Staleness threshold: re-index if older than this (seconds)
DATA_STALENESS_SECONDS = 3600  # 1 hour


def data_source_name(hostname: str, catalog_id: str | int, user_id: str) -> str:
    """Build the RAG source name for a user's data index."""
    return f"data:{hostname}:{catalog_id}:{user_id}"


def dataset_record_to_markdown(
    record: dict[str, Any],
    types: list[str] | None = None,
    version: str | None = None,
) -> str:
    """Convert a dataset record to a markdown snippet for embedding."""
    rid = record.get("RID", "")
    desc = record.get("Description", "") or ""
    created = record.get("RCT", "")
    parts = [f"## Dataset: {desc[:80]} (RID: {rid})"]
    if desc:
        parts.append(f"**Description:** {desc}")
    if types:
        parts.append(f"**Types:** {', '.join(types)}")
    if version:
        parts.append(f"**Version:** {version}")
    if created:
        parts.append(f"**Created:** {created[:10]}")
    return "\n".join(parts)


def execution_record_to_markdown(
    record: dict[str, Any],
    workflow_name: str | None = None,
    status: str | None = None,
    input_datasets: list[str] | None = None,
) -> str:
    """Convert an execution record to a markdown snippet for embedding."""
    rid = record.get("RID", "")
    desc = record.get("Description", "") or ""
    created = record.get("RCT", "")
    parts = [f"## Execution: {desc[:80]} (RID: {rid})"]
    if workflow_name:
        parts.append(f"**Workflow:** {workflow_name}")
    if status:
        parts.append(f"**Status:** {status}")
    if input_datasets:
        parts.append(f"**Input Datasets:** {', '.join(input_datasets)}")
    if desc:
        parts.append(f"**Description:** {desc}")
    if created:
        parts.append(f"**Created:** {created[:10]}")
    return "\n".join(parts)


def index_user_data(
    ml: Any,
    hostname: str,
    catalog_id: str | int,
    user_id: str,
    collection: Any,
    chunk_size_target: int = 800,
    indexed_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Index user-visible dataset and execution records into ChromaDB.

    Args:
        ml: DerivaML instance (for fetching records).
        hostname: Catalog hostname.
        catalog_id: Catalog ID.
        user_id: User identifier for index isolation.
        collection: ChromaDB collection.
        chunk_size_target: Target tokens per chunk (unused for now — records are small).
        indexed_tables: Tables to index (default: DEFAULT_INDEXED_TABLES).

    Returns:
        Dict with indexing statistics.
    """
    source = data_source_name(hostname, catalog_id, user_id)
    tables = indexed_tables or DEFAULT_INDEXED_TABLES

    # Check staleness
    if _is_data_index_fresh(collection, source):
        return {"source": source, "status": "fresh", "chunks_created": 0}

    # Remove old data chunks for this user
    _remove_data_chunks(collection, source)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    for table_name in tables:
        try:
            if table_name == "Dataset":
                _index_datasets(ml, source, now, ids, documents, metadatas)
            elif table_name == "Execution":
                _index_executions(ml, source, now, ids, documents, metadatas)
            else:
                _index_generic_table(ml, table_name, source, now, ids, documents, metadatas)
        except Exception as e:
            logger.warning(f"Failed to index {table_name} for {source}: {e}")

    if not ids:
        return {"source": source, "status": "empty", "chunks_created": 0}

    # Batch upsert
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(f"Indexed {len(ids)} data records for {source}")
    return {"source": source, "status": "indexed", "chunks_created": len(ids)}


def _index_datasets(ml, source, now, ids, documents, metadatas):
    """Index all datasets visible to the user."""
    for ds in ml.find_datasets():
        rid = ds.dataset_rid
        md = dataset_record_to_markdown(
            {"RID": rid, "Description": ds.description, "RCT": ""},
            types=ds.dataset_types,
            version=str(ds.current_version) if ds.current_version else None,
        )
        ids.append(f"{source}:Dataset:{rid}")
        documents.append(md)
        metadatas.append({
            "source": source,
            "doc_type": "catalog-data",
            "table": "Dataset",
            "rid": rid,
            "indexed_at": now,
        })


def _index_executions(ml, source, now, ids, documents, metadatas):
    """Index all executions visible to the user."""
    try:
        rows = list(ml.get_table_as_dict("Execution"))
    except Exception:
        return

    for row in rows:
        rid = row.get("RID", "")
        md = execution_record_to_markdown(
            row,
            workflow_name=row.get("Workflow", ""),
            status=row.get("Status", ""),
        )
        ids.append(f"{source}:Execution:{rid}")
        documents.append(md)
        metadatas.append({
            "source": source,
            "doc_type": "catalog-data",
            "table": "Execution",
            "rid": rid,
            "indexed_at": now,
        })


def _index_generic_table(ml, table_name, source, now, ids, documents, metadatas):
    """Index records from any table by converting to markdown."""
    try:
        rows = list(ml.get_table_as_dict(table_name))
    except Exception:
        return

    for row in rows:
        rid = row.get("RID", "")
        desc = row.get("Description", "") or row.get("Name", "") or str(row)[:200]
        md = f"## {table_name}: {desc[:80]} (RID: {rid})\n"
        for k, v in row.items():
            if v and k not in ("RID", "RCT", "RMT", "RCB", "RMB"):
                md += f"**{k}:** {v}\n"
        ids.append(f"{source}:{table_name}:{rid}")
        documents.append(md)
        metadatas.append({
            "source": source,
            "doc_type": "catalog-data",
            "table": table_name,
            "rid": rid,
            "indexed_at": now,
        })


def _is_data_index_fresh(collection: Any, source: str) -> bool:
    """Check if the data index is fresh (< DATA_STALENESS_SECONDS old)."""
    try:
        results = collection.get(
            where={"source": source},
            include=["metadatas"],
            limit=1,
        )
        if results and results["metadatas"]:
            indexed_at_str = results["metadatas"][0].get("indexed_at", "")
            if indexed_at_str:
                indexed_at = datetime.fromisoformat(indexed_at_str)
                age = (datetime.now(timezone.utc) - indexed_at).total_seconds()
                return age < DATA_STALENESS_SECONDS
    except Exception:
        pass
    return False


def _remove_data_chunks(collection: Any, source: str) -> int:
    """Remove all data chunks for a user from the collection."""
    try:
        results = collection.get(
            where={"source": source},
            include=[],
        )
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            return len(results["ids"])
    except Exception as e:
        logger.warning(f"Failed to delete data chunks for {source}: {e}")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag_data.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Add trigger_data_reindex and rag_suggest_record to helpers.py**

Add to `src/deriva_mcp/rag/helpers.py`:

```python
def trigger_data_reindex(conn_info: ConnectionInfo | None) -> None:
    """Fire-and-forget background data reindex.

    Spawns a daemon thread that fetches user-visible records from
    indexed tables and upserts them into the per-user data index.
    Debounces to at most once per DEBOUNCE_SECONDS.
    """
    if conn_info is None:
        return

    from deriva_mcp.rag import get_rag_manager
    manager = get_rag_manager()
    if manager is None:
        return

    now = time.time()
    if now - conn_info._data_reindex_at < DEBOUNCE_SECONDS:
        return

    conn_info._data_reindex_at = now

    def _do_reindex():
        try:
            from deriva_mcp.rag.data import index_user_data
            ml = conn_info.ml_instance
            # Access the collection from the manager
            manager._ensure_initialized()
            index_user_data(
                ml=ml,
                hostname=conn_info.hostname,
                catalog_id=conn_info.catalog_id,
                user_id=conn_info.user_id,
                collection=manager._collection,
            )
        except Exception as e:
            logger.warning(f"Background data reindex failed: {e}")

    thread = threading.Thread(target=_do_reindex, daemon=True, name="data-rag-reindex")
    thread.start()


def rag_suggest_record(
    query: str,
    conn_info: ConnectionInfo | None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search the user's data index for record suggestions.

    Returns: [{"name": ..., "table": ..., "rid": ..., "relevance": ..., "description": ...}]
    ACL enforcement: uses conn_info.user_id to scope search.
    """
    if conn_info is None:
        return []

    from deriva_mcp.rag import get_rag_manager
    manager = get_rag_manager()
    if manager is None:
        return []

    user_id = getattr(conn_info, "user_id", None)
    if not user_id:
        return []

    # Lazy reindex if dirty
    if getattr(conn_info, "data_dirty", False):
        trigger_data_reindex(conn_info)
        conn_info.data_dirty = False

    from deriva_mcp.rag.data import data_source_name
    source = data_source_name(conn_info.hostname, conn_info.catalog_id, user_id)

    results = manager.search(query=query, limit=limit, source=source)

    suggestions = []
    for r in results:
        meta = r.get("source", "")
        text = r.get("text", "")
        heading = r.get("section_heading", "")

        # Parse name from heading "## Dataset: Training Set v2 (RID: 1-ABC)"
        name = heading
        rid = ""
        if "(RID:" in heading:
            parts = heading.split("(RID:")
            name = parts[0].strip().split(":", 1)[-1].strip() if ":" in parts[0] else parts[0].strip()
            rid = parts[1].replace(")", "").strip()

        # Get table from metadata
        table = ""
        # The metadata is embedded in the chunk metadata via ChromaDB
        # We can parse from the heading prefix
        if heading.startswith("## Dataset:"):
            table = "Dataset"
        elif heading.startswith("## Execution:"):
            table = "Execution"

        suggestions.append({
            "name": name,
            "table": table,
            "rid": rid,
            "relevance": r.get("relevance", 0),
            "description": text[:200],
        })

    return suggestions
```

- [ ] **Step 6: Write tests for trigger_data_reindex and rag_suggest_record**

```python
# Append to tests/test_rag_helpers.py
class TestTriggerDataReindex:
    def test_noop_when_conn_info_none(self):
        from deriva_mcp.rag.helpers import trigger_data_reindex
        trigger_data_reindex(None)  # Should not raise


class TestRagSuggestRecord:
    def test_returns_empty_when_no_conn(self):
        from deriva_mcp.rag.helpers import rag_suggest_record
        assert rag_suggest_record("training", None) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_returns_empty_when_no_user_id(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_suggest_record
        mock_get_rag.return_value = MagicMock()
        conn_info = MagicMock()
        conn_info.user_id = None
        assert rag_suggest_record("training", conn_info) == []
```

- [ ] **Step 7: Run all helper tests**

Run: `pytest tests/test_rag_helpers.py tests/test_rag_data.py -v`
Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
git add src/deriva_mcp/rag/data.py src/deriva_mcp/rag/helpers.py tests/test_rag_data.py tests/test_rag_helpers.py
git commit -m "feat: add data indexing and suggest_record/trigger_data_reindex"
```

### Task 7: Update catalog.py to use helpers

**Files:**
- Modify: `src/deriva_mcp/tools/catalog.py`

- [ ] **Step 1: Replace `_index_schema_background` with `trigger_schema_reindex`**

In `tools/catalog.py`, replace the `_index_schema_background` function (lines 21-99) and its call in `connect_catalog` with a call to the new helper. Also add data indexing.

The `_index_schema_background` function at the top of the file should be replaced with a simple import-and-call pattern. In `connect_catalog`, change:

```python
# OLD (lines 163-164):
active_conn_info = conn_manager.get_active_connection_info()
_index_schema_background(ml, resolved_hostname, catalog_id, active_conn_info)

# NEW:
active_conn_info = conn_manager.get_active_connection_info()
from deriva_mcp.rag.helpers import trigger_schema_reindex, trigger_data_reindex
trigger_schema_reindex(active_conn_info)
trigger_data_reindex(active_conn_info)
```

Delete the entire `_index_schema_background` function (lines 21-99).

- [ ] **Step 2: Run existing catalog tests**

Run: `pytest tests/test_catalog.py -v`
Expected: All PASS (the background indexing is fire-and-forget, tests don't depend on it).

- [ ] **Step 3: Commit**

```bash
git add src/deriva_mcp/tools/catalog.py
git commit -m "refactor: replace inline schema indexing with rag/helpers calls"
```

---

## Chunk 2: Layer 1 — Auto-Reindex After Mutations

### Task 8: Layer 1 on schema tools (create_table, create_asset_table, add_column)

**Files:**
- Modify: `src/deriva_mcp/tools/schema.py`

- [ ] **Step 1: Add reindex trigger to create_table**

After the successful `ml.create_table(table_def, schema=schema)` call (line 135), before the return, add:

```python
            table = ml.create_table(table_def, schema=schema)

            # Layer 1: Trigger schema reindex after structural change
            from deriva_mcp.rag.helpers import trigger_schema_reindex
            trigger_schema_reindex(conn_manager.get_active_connection_info())

            return json.dumps({
```

- [ ] **Step 2: Add reindex trigger to create_asset_table**

After `ml.create_asset(...)` (line 204), add the same trigger.

- [ ] **Step 3: Add reindex trigger to add_column**

After `handle.add_column(...)` (line 513), add the same trigger.

- [ ] **Step 4: Run schema tests**

Run: `pytest tests/test_schema.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_mcp/tools/schema.py
git commit -m "feat: Layer 1 — auto-reindex after schema mutations"
```

### Task 9: Layer 1 on vocabulary tools

**Files:**
- Modify: `src/deriva_mcp/tools/vocabulary.py`

- [ ] **Step 1: Add reindex trigger to create_vocabulary (immediate)**

After `ml.create_vocabulary(...)` (line 93), add:

```python
            # Layer 1: Trigger schema reindex after structural change
            from deriva_mcp.rag.helpers import trigger_schema_reindex
            trigger_schema_reindex(conn_manager.get_active_connection_info())
```

- [ ] **Step 2: Add dirty flag to add_term, delete_term, update_term_description, add_synonym, remove_synonym**

After each successful operation in these 5 tools, add:

```python
            # Layer 1: Mark schema dirty for lazy reindex
            conn_info = conn_manager.get_active_connection_info()
            if conn_info:
                conn_info.schema_dirty = True
```

For `add_term`, add it after line 56 (after `return json.dumps({...})` in the success case).
For `delete_term`, after `ml.delete_term(...)`.
For `update_term_description`, after `term.description = description`.
For `add_synonym`, after `term.synonyms = ...`.
For `remove_synonym`, after `term.synonyms = ...`.

- [ ] **Step 3: Run vocabulary tests**

Run: `pytest tests/test_vocabulary.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/deriva_mcp/tools/vocabulary.py
git commit -m "feat: Layer 1 — reindex/dirty-flag on vocabulary mutations"
```

### Task 10: Layer 1 on feature tools (create_feature, delete_feature)

**Files:**
- Modify: `src/deriva_mcp/tools/feature.py`

- [ ] **Step 1: Add reindex triggers**

After `ml.create_feature(...)` (line 103) and after `ml.delete_feature(...)` (line 138), add:

```python
            from deriva_mcp.rag.helpers import trigger_schema_reindex
            trigger_schema_reindex(conn_manager.get_active_connection_info())
```

- [ ] **Step 2: Run feature tests**

Run: `pytest tests/test_feature.py -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add src/deriva_mcp/tools/feature.py
git commit -m "feat: Layer 1 — auto-reindex after feature mutations"
```

---

## Chunk 3: Layer 2 — Error Recovery with RAG Suggestions

### Task 11: Layer 2 on data tools (query_table, count_table, get_table, insert_records)

**Files:**
- Modify: `src/deriva_mcp/tools/data.py`
- Create: `tests/test_rag_integration_layers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_integration_layers.py
"""Tests for RAG integration layers across MCP tools."""

import json
import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import _create_tool_capture


class TestLayer2DataTools:
    """Layer 2: Error recovery on data tools."""

    def _make_conn_manager_with_rag(self):
        """Create a mock conn_manager whose tools raise 'not found' errors."""
        from deriva_ml import DerivaMLException
        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.model.name_to_table.side_effect = Exception("Table 'Diagnoiss' not found in schema 'isa'")
        conn_manager.get_active_or_raise.return_value = mock_ml

        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.hostname = "test.example.org"
        mock_conn_info.catalog_id = "1"
        mock_conn_info.schema_dirty = False
        conn_manager.get_active_connection_info.return_value = mock_conn_info
        return conn_manager

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_query_table_suggests_on_not_found(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis (vocabulary)", "relevance": 0.92,
             "source": "schema:test:1:abc", "section_heading": "isa.Diagnosis",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = self._make_conn_manager_with_rag()
        from deriva_mcp.tools.data import register_data_tools
        mcp, tools = _create_tool_capture()
        register_data_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["query_table"]("Diagnoiss")))
        assert result["status"] == "error"
        assert "suggestions" in result
        assert result["hint"].startswith("Did you mean:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer2DataTools -v`
Expected: FAIL (no suggestions in error response yet).

- [ ] **Step 3: Add Layer 2 to data tools**

In `tools/data.py`, modify the `except` blocks in `query_table`, `count_table`, `get_table`, and `insert_records`. For each tool, change the except block:

```python
        except Exception as e:
            logger.error(f"Failed to query table: {e}")
            error_msg = str(e)
            result = {"status": "error", "message": error_msg}

            # Layer 2: Suggest alternatives on entity-not-found errors
            from deriva_mcp.rag.helpers import _is_not_found_error, rag_suggest_entity
            if _is_not_found_error(error_msg):
                conn_info = conn_manager.get_active_connection_info()
                suggestions = rag_suggest_entity(table_name, conn_info)
                if suggestions:
                    result["suggestions"] = suggestions
                    result["hint"] = f"Did you mean: {suggestions[0]['name']}?"

            return json.dumps(result)
```

Apply this pattern to: `query_table`, `count_table`, `get_table`, `insert_records`.

- [ ] **Step 4: Add Layer 5 conditional data dirty flag to `insert_records`**

In `insert_records`, after the successful insert (but before the return), add a conditional dirty flag that fires only when the target table is one of the indexed data tables:

```python
            # Layer 5: Mark data dirty when inserting into indexed tables
            from deriva_mcp.rag.data import DEFAULT_INDEXED_TABLES
            if table_name in DEFAULT_INDEXED_TABLES or table_name in ("Dataset", "Execution"):
                conn_info = conn_manager.get_active_connection_info()
                if conn_info:
                    conn_info.data_dirty = True
```

**Important:** This check must respect the existing `_MANAGED_TABLES` guard — Dataset and Execution inserts are blocked by that dict, but other tables that may be added to `DEFAULT_INDEXED_TABLES` in the future should trigger the dirty flag. Place this code after the `_MANAGED_TABLES` check and the actual insert succeeds.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer2DataTools -v`
Expected: PASS.

- [ ] **Step 6: Run existing data tests for regression**

Run: `pytest tests/test_data.py -v`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/deriva_mcp/tools/data.py tests/test_rag_integration_layers.py
git commit -m "feat: Layer 2 — error recovery with RAG suggestions on data tools; Layer 5 dirty flag on insert_records"
```

### Task 12: Layer 2 on vocabulary tools (add_term, delete_term)

**Files:**
- Modify: `src/deriva_mcp/tools/vocabulary.py`

- [ ] **Step 1: Add Layer 2 to add_term except block**

In the outer `except` block of `add_term` (line 69), add RAG suggestions:

```python
            logger.error(f"Failed to add term: {e}")
            error_msg = str(e)
            result = {"status": "error", "message": error_msg}

            from deriva_mcp.rag.helpers import _is_not_found_error, rag_suggest_entity
            if _is_not_found_error(error_msg):
                conn_info = conn_manager.get_active_connection_info()
                suggestions = rag_suggest_entity(vocabulary_name, conn_info)
                if suggestions:
                    result["suggestions"] = suggestions
                    result["hint"] = f"Did you mean: {suggestions[0]['name']}?"

            return json.dumps(result)
```

- [ ] **Step 2: Add Layer 2 to delete_term except block**

Same pattern for `delete_term`.

- [ ] **Step 3: Run vocab tests**

Run: `pytest tests/test_vocabulary.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/deriva_mcp/tools/vocabulary.py
git commit -m "feat: Layer 2 — error recovery on vocabulary tools"
```

### Task 13: Layer 2 on feature tools (fetch_table_features)

**Files:**
- Modify: `src/deriva_mcp/tools/feature.py`

- [ ] **Step 1: Add Layer 2 to fetch_table_features except block**

In the except block (line 517), add:

```python
            from deriva_mcp.rag.helpers import _is_not_found_error, rag_suggest_entity
            if _is_not_found_error(error_msg):
                conn_info = conn_manager.get_active_connection_info()
                suggestions = rag_suggest_entity(table_name, conn_info)
                if suggestions:
                    result["suggestions"] = suggestions
                    result["hint"] = f"Did you mean: {suggestions[0]['name']}?"
```

- [ ] **Step 2: Run feature tests**

Run: `pytest tests/test_feature.py -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add src/deriva_mcp/tools/feature.py
git commit -m "feat: Layer 2 — error recovery on feature tools"
```

### Task 14: Layer 2 on schema tools (add_column) and annotation tools (get_table_sample_data)

**Files:**
- Modify: `src/deriva_mcp/tools/schema.py`
- Modify: `src/deriva_mcp/tools/annotation.py`

- [ ] **Step 1: Add Layer 2 to add_column except block**

In `tools/schema.py`, modify the `add_column` except block (line 527):

```python
            from deriva_mcp.rag.helpers import _is_not_found_error, rag_suggest_entity
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

- [ ] **Step 2: Add Layer 2 to get_table_sample_data except block**

In `tools/annotation.py`, find `get_table_sample_data` and apply the same pattern.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_schema.py tests/test_annotation.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/deriva_mcp/tools/schema.py src/deriva_mcp/tools/annotation.py
git commit -m "feat: Layer 2 — error recovery on add_column and get_table_sample_data"
```

### Task 15: Layer 2 on dataset and execution tools (data-level error recovery)

**Files:**
- Modify: `src/deriva_mcp/tools/dataset.py`
- Modify: `src/deriva_mcp/tools/execution.py`

- [ ] **Step 1: Add data-level error recovery to dataset tools**

For each of: `add_dataset_members`, `delete_dataset_members`, `set_dataset_description`, `add_dataset_type`, `download_dataset`, `split_dataset`, `denormalize_dataset` — in the except block, add:

```python
            from deriva_mcp.rag.helpers import _is_not_found_error, rag_suggest_record
            error_msg = str(e)
            result = {"status": "error", "message": error_msg}
            if _is_not_found_error(error_msg):
                conn_info = conn_manager.get_active_connection_info()
                suggestions = rag_suggest_record(dataset_rid, conn_info)
                if suggestions:
                    result["suggestions"] = suggestions
                    result["hint"] = f"Did you mean: {suggestions[0]['name']} ({suggestions[0]['rid']})?"
            return json.dumps(result)
```

- [ ] **Step 2: Add data-level error recovery to execution tools**

For `restore_execution` and `download_execution_dataset`, apply the same pattern using `rag_suggest_record` with the execution/dataset RID.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_dataset.py tests/test_execution.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/deriva_mcp/tools/dataset.py src/deriva_mcp/tools/execution.py
git commit -m "feat: Layer 2 — data-level error recovery on dataset/execution tools"
```

---

## Chunk 4: Layer 3 — Duplicate Detection on Creation

### Task 16: Layer 3 on create_table and create_asset_table

**Files:**
- Modify: `src/deriva_mcp/tools/schema.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_rag_integration_layers.py
class TestLayer3DuplicateDetection:
    """Layer 3: Duplicate detection on creation tools."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_create_table_warns_on_similar(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis_Type (vocabulary)", "relevance": 0.88,
             "source": "schema:test:1:abc", "section_heading": "isa.Diagnosis_Type",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.create_table.return_value = MagicMock(name="Diagnosis", schema=MagicMock(name="isa"),
                                                       columns=[])
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.schema_dirty = False
        mock_conn_info._schema_reindex_at = 0.0
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.tools.schema import register_schema_tools
        mcp, tools = _create_tool_capture()
        register_schema_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["create_table"]("Diagnosis")))
        assert result["status"] == "created"
        assert "similar_existing" in result
        assert "warning" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer3DuplicateDetection -v`
Expected: FAIL.

- [ ] **Step 3: Add Layer 3 pre-check to create_table**

In `tools/schema.py`, at the top of `create_table` (after `ml = conn_manager.get_active_or_raise()`), add:

```python
            # Layer 3: Check for semantic near-duplicates
            from deriva_mcp.rag.helpers import rag_suggest_entity, DUPLICATE_RELEVANCE_THRESHOLD
            conn_info = conn_manager.get_active_connection_info()
            similar = rag_suggest_entity(table_name, conn_info, limit=3)
            dup_warnings = [
                s for s in similar
                if s["relevance"] > DUPLICATE_RELEVANCE_THRESHOLD
                and s["name"].lower() != table_name.lower()
            ]
```

Then after the successful return dict, before `return json.dumps(...)`:

```python
            result = {
                "status": "created",
                "table_name": table.name,
                "schema": table.schema.name,
                "columns": [c.name for c in table.columns],
            }

            # Layer 3: Attach warnings if near-duplicates found
            if dup_warnings:
                result["similar_existing"] = dup_warnings
                result["warning"] = (
                    f"Created '{table_name}', but similar entities exist: "
                    f"{', '.join(w['name'] for w in dup_warnings)}. "
                    f"Verify this isn't a duplicate."
                )

            return json.dumps(result)
```

Apply the same pattern to `create_asset_table`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer3DuplicateDetection -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_mcp/tools/schema.py tests/test_rag_integration_layers.py
git commit -m "feat: Layer 3 — duplicate detection on create_table/create_asset_table"
```

### Task 17: Layer 3 on create_vocabulary and create_feature

**Files:**
- Modify: `src/deriva_mcp/tools/vocabulary.py`
- Modify: `src/deriva_mcp/tools/feature.py`
- Modify: `tests/test_rag_integration_layers.py`

- [ ] **Step 1: Write failing tests for create_vocabulary and create_feature duplicate detection**

```python
# Append to tests/test_rag_integration_layers.py

class TestLayer3CreateVocabulary:
    """Layer 3: Duplicate detection on create_vocabulary."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_create_vocabulary_warns_on_similar(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis_Type (vocabulary)", "relevance": 0.88,
             "source": "schema:test:1:abc", "section_heading": "isa.Diagnosis_Type",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.create_vocabulary.return_value = MagicMock()
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.schema_dirty = False
        mock_conn_info._schema_reindex_at = 0.0
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.tools.vocabulary import register_vocabulary_tools
        mcp, tools = _create_tool_capture()
        register_vocabulary_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["create_vocabulary"]("Diagnosis")))
        assert result["status"] == "created"
        assert "similar_existing" in result
        assert "warning" in result


class TestLayer3CreateFeature:
    """Layer 3: Duplicate detection on create_feature."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_create_feature_warns_on_similar(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Image_Quality (feature)", "relevance": 0.85,
             "source": "schema:test:1:abc", "section_heading": "isa.Image_Quality",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.create_feature.return_value = MagicMock()
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.schema_dirty = False
        mock_conn_info._schema_reindex_at = 0.0
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.tools.feature import register_feature_tools
        mcp, tools = _create_tool_capture()
        register_feature_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["create_feature"]("Image_Score", "Image")))
        assert result["status"] == "created"
        assert "similar_existing" in result
        assert "warning" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer3CreateVocabulary tests/test_rag_integration_layers.py::TestLayer3CreateFeature -v`
Expected: FAIL (no `similar_existing` or `warning` in response yet).

- [ ] **Step 3: Add Layer 3 to create_vocabulary**

Same pattern as create_table: pre-check with `rag_suggest_entity(vocabulary_name, ...)`, attach `similar_existing` and `warning` fields on success.

- [ ] **Step 4: Add Layer 3 to create_feature**

Same pattern using `rag_suggest_entity(feature_name, ...)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer3CreateVocabulary tests/test_rag_integration_layers.py::TestLayer3CreateFeature -v`
Expected: All PASS.

- [ ] **Step 6: Run existing vocab and feature tests for regression**

Run: `pytest tests/test_vocabulary.py tests/test_feature.py -v`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/deriva_mcp/tools/vocabulary.py src/deriva_mcp/tools/feature.py tests/test_rag_integration_layers.py
git commit -m "feat: Layer 3 — duplicate detection on create_vocabulary/create_feature"
```

---

## Chunk 5: Layer 4 — Resource Enrichment

### Task 18: Add _related_docs to catalog resources

**Files:**
- Modify: `src/deriva_mcp/resources.py`
- Modify: `tests/test_rag_integration_layers.py`

- [ ] **Step 1: Write the failing test for Layer 4 enrichment**

```python
# Append to tests/test_rag_integration_layers.py
from tests.conftest import _create_resource_capture


class TestLayer4ResourceEnrichment:
    """Layer 4: _related_docs appears in resource responses."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_catalog_schema_includes_related_docs(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "Creating Tables Guide", "relevance": 0.92,
             "github_url": "https://example.com/tables-guide",
             "source": "docs", "section_heading": "Creating Tables",
             "doc_type": "documentation"},
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.model.get_schema_description.return_value = {"schemas": {}}
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.domain_schemas = ["isa"]
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.resources import register_resources
        mcp, resources = _create_resource_capture()
        register_resources(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(resources["get_catalog_schema"]()))
        assert "_related_docs" in result
        assert len(result["_related_docs"]) >= 1
        assert result["_related_docs"][0]["title"] == "Creating Tables"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer4ResourceEnrichment -v`
Expected: FAIL (no `_related_docs` in response yet).

- [ ] **Step 3: Implement enrichment on `get_catalog_schema`**

Before the `return json.dumps(schema_info, indent=2)` in `get_catalog_schema()` (around line 449), add:

```python
            # Layer 4: Enrich with related docs
            from deriva_mcp.rag.helpers import rag_enrich_resource
            conn_info = conn_manager.get_active_connection_info()
            related = rag_enrich_resource("catalog schema tables columns", conn_info)
            if related:
                schema_info["_related_docs"] = related
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer4ResourceEnrichment -v`
Expected: PASS.

- [ ] **Step 5: Add enrichment to remaining catalog resources**

Apply the same pattern to:
- `get_catalog_datasets`: query `"creating managing datasets"`
- `get_catalog_features`: query `"defining using features"`
- `get_catalog_vocabularies`: query `"controlled vocabularies terms"`
- `get_catalog_workflows`: query `"workflows executions"`

For each, add the enrichment block before the return statement.

- [ ] **Step 6: Add enrichment to parameterized resources**

For `get_table_schema` (line 1119): use contextual query based on table type:

```python
            # Layer 4: Contextual enrichment
            from deriva_mcp.rag.helpers import rag_enrich_resource
            conn_info = conn_manager.get_active_connection_info()
            query_parts = [f"table {table_name}"]
            # Add context based on table type (check the result dict)
            related = rag_enrich_resource(" ".join(query_parts), conn_info)
            if related:
                result["_related_docs"] = related
```

For `get_table_features` (line 891): query `f"features {table_name}"`.
For `get_dataset_details` (line 663): query `"dataset members versions"`.

- [ ] **Step 7: Run resource tests**

Run: `pytest tests/test_resources.py -v`
Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
git add src/deriva_mcp/resources.py tests/test_rag_integration_layers.py
git commit -m "feat: Layer 4 — resource enrichment with _related_docs"
```

---

## Chunk 6: Layer 5 — Per-User Data Indexing Integration

### Task 19: Layer 5 dirty flags on dataset tools

**Files:**
- Modify: `src/deriva_mcp/tools/dataset.py`

- [ ] **Step 1: Add data_dirty flag to dataset mutation tools**

After each successful operation in these tools, add:

```python
            conn_info = conn_manager.get_active_connection_info()
            if conn_info:
                conn_info.data_dirty = True
```

Apply to: `create_dataset`, `delete_dataset`, `set_dataset_description`, `add_dataset_type`, `remove_dataset_type`, `add_dataset_members`, `delete_dataset_members`.

- [ ] **Step 2: Run dataset tests**

Run: `pytest tests/test_dataset.py -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add src/deriva_mcp/tools/dataset.py
git commit -m "feat: Layer 5 — data dirty flags on dataset mutation tools"
```

### Task 20: Layer 5 dirty flags on execution tools

**Files:**
- Modify: `src/deriva_mcp/tools/execution.py`

- [ ] **Step 1: Add data_dirty flag to execution mutation tools**

Apply to: `create_execution`, `stop_execution`, `update_execution_status`, `create_execution_dataset`.

- [ ] **Step 2: Run execution tests**

Run: `pytest tests/test_execution.py -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add src/deriva_mcp/tools/execution.py
git commit -m "feat: Layer 5 — data dirty flags on execution mutation tools"
```

### Task 21: Layer 5 — _related_data on dataset/execution resources

**Files:**
- Modify: `src/deriva_mcp/resources.py`
- Modify: `tests/test_rag_integration_layers.py`

- [ ] **Step 1: Write the failing test for Layer 5 _related_data**

```python
# Append to tests/test_rag_integration_layers.py

class TestLayer5RelatedData:
    """Layer 5: _related_data appears in dataset/execution resource responses."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_dataset_details_includes_related_data(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## Dataset: Validation Set (RID: 2-XYZ)", "relevance": 0.85,
             "source": "data:test:1:user1", "section_heading": "## Dataset: Validation Set (RID: 2-XYZ)",
             "doc_type": "catalog-data"},
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        # Mock find_datasets to return a dataset
        mock_ds = MagicMock()
        mock_ds.dataset_rid = "1-ABC"
        mock_ds.description = "Training dataset for lung segmentation"
        mock_ds.dataset_types = ["Training"]
        mock_ds.current_version = "1.0"
        mock_ds.current_version_rid = "V1"
        mock_ml.find_datasets.return_value = [mock_ds]
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.user_id = "user1"
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        mock_conn_info.data_dirty = False
        mock_conn_info._data_reindex_at = 0.0
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.resources import register_resources
        mcp, resources = _create_resource_capture()
        register_resources(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(resources["get_dataset_details"]("1-ABC")))
        assert "_related_data" in result
        assert len(result["_related_data"]) >= 1

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_related_data_excludes_self(self, mock_get_rag):
        """Self-references (same RID) must be filtered out."""
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## Dataset: Same (RID: 1-ABC)", "relevance": 0.99,
             "source": "data:test:1:user1", "section_heading": "## Dataset: Same (RID: 1-ABC)",
             "doc_type": "catalog-data"},
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ds = MagicMock()
        mock_ds.dataset_rid = "1-ABC"
        mock_ds.description = "Training dataset"
        mock_ds.dataset_types = ["Training"]
        mock_ds.current_version = "1.0"
        mock_ds.current_version_rid = "V1"
        mock_ml.find_datasets.return_value = [mock_ds]
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.user_id = "user1"
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        mock_conn_info.data_dirty = False
        mock_conn_info._data_reindex_at = 0.0
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.resources import register_resources
        mcp, resources = _create_resource_capture()
        register_resources(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(resources["get_dataset_details"]("1-ABC")))
        # Self should be excluded, so _related_data should be empty or absent
        related = result.get("_related_data", [])
        assert all(r.get("rid") != "1-ABC" for r in related)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer5RelatedData -v`
Expected: FAIL (no `_related_data` in response yet).

- [ ] **Step 3: Add _related_data to get_dataset_details**

In `get_dataset_details` (line 663), before the return, add:

```python
            # Layer 5: Add related data from per-user index
            from deriva_mcp.rag.helpers import rag_suggest_record
            conn_info = conn_manager.get_active_connection_info()
            if conn_info and ds.description:
                related_data = rag_suggest_record(
                    f"dataset {ds.description}", conn_info, limit=3
                )
                # Exclude self from results
                related_data = [r for r in related_data if r.get("rid") != dataset_rid]
                if related_data:
                    result["_related_data"] = related_data
```

(Where `result` is the dict being built before `json.dumps`.)

- [ ] **Step 4: Add _related_data to get_execution_details**

Same pattern for `get_execution_details` (line 1639):

```python
            from deriva_mcp.rag.helpers import rag_suggest_record
            conn_info = conn_manager.get_active_connection_info()
            if conn_info and exe.description:
                related_data = rag_suggest_record(
                    f"execution {exe.description}", conn_info, limit=3
                )
                related_data = [r for r in related_data if r.get("rid") != execution_rid]
                if related_data:
                    result["_related_data"] = related_data
```

- [ ] **Step 5: Run Layer 5 tests to verify they pass**

Run: `pytest tests/test_rag_integration_layers.py::TestLayer5RelatedData -v`
Expected: All PASS.

- [ ] **Step 6: Run resource tests for regression**

Run: `pytest tests/test_resources.py -v`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/deriva_mcp/resources.py tests/test_rag_integration_layers.py
git commit -m "feat: Layer 5 — _related_data on dataset/execution resources"
```

### Task 22: Update rag_search to include data index

**Files:**
- Modify: `src/deriva_mcp/tools/rag.py`

- [ ] **Step 1: Add `include_data` parameter to rag_search**

In `tools/rag.py`, modify the `rag_search` function signature to add `include_data: bool = True`. Then after the schema results merge (line 100), add:

```python
        # Also search per-user data index if connected
        data_results = []
        if include_data:
            conn_info = conn_manager.get_active_connection_info()
            if conn_info:
                from deriva_mcp.rag.data import data_source_name
                data_source = data_source_name(
                    conn_info.hostname, conn_info.catalog_id, conn_info.user_id
                )
                data_results = manager.search(query=query, limit=limit, source=data_source)

        # Merge and re-rank by relevance
        all_results = doc_results + schema_results + data_results
        all_results.sort(key=lambda r: r.get("relevance", 0), reverse=True)
        all_results = all_results[:limit]
```

Also update the docstring to mention the `include_data` parameter.

- [ ] **Step 2: Run rag tool tests**

Run: `pytest tests/test_rag_tools.py -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add src/deriva_mcp/tools/rag.py
git commit -m "feat: Layer 5 — rag_search includes per-user data index"
```

---

## Chunk 7: Cross-Cutting Tests, Graceful Degradation + Skill Update

### Task 23: ACL isolation test — two users with independent data indexes

**Files:**
- Modify: `tests/test_rag_integration_layers.py`

- [ ] **Step 1: Write the ACL isolation test**

```python
# Append to tests/test_rag_integration_layers.py

class TestACLIsolation:
    """Verify that two users have independent data indexes."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_two_users_independent_data_indexes(self, mock_get_rag):
        """User A's data index must not leak into User B's search results."""
        from deriva_mcp.rag.helpers import rag_suggest_record

        mock_manager = MagicMock()
        mock_get_rag.return_value = mock_manager

        # User A searches — manager.search is called with user A's data source
        conn_info_a = MagicMock()
        conn_info_a.user_id = "user_alice"
        conn_info_a.hostname = "test.example.org"
        conn_info_a.catalog_id = "1"
        conn_info_a.data_dirty = False
        conn_info_a._data_reindex_at = 0.0

        mock_manager.search.return_value = [
            {"text": "Alice's dataset", "relevance": 0.9,
             "source": "data:test.example.org:1:user_alice",
             "section_heading": "## Dataset: Alice Training (RID: A-1)",
             "doc_type": "catalog-data"}
        ]
        results_a = rag_suggest_record("training", conn_info_a)
        # Verify search was scoped to Alice's data source
        call_args = mock_manager.search.call_args
        assert "data:test.example.org:1:user_alice" in str(call_args)

        # User B searches — should use a different source key
        conn_info_b = MagicMock()
        conn_info_b.user_id = "user_bob"
        conn_info_b.hostname = "test.example.org"
        conn_info_b.catalog_id = "1"
        conn_info_b.data_dirty = False
        conn_info_b._data_reindex_at = 0.0

        mock_manager.search.return_value = []
        results_b = rag_suggest_record("training", conn_info_b)
        call_args_b = mock_manager.search.call_args
        assert "data:test.example.org:1:user_bob" in str(call_args_b)

        # The two source keys must be different
        from deriva_mcp.rag.data import data_source_name
        assert data_source_name("test.example.org", "1", "user_alice") != \
               data_source_name("test.example.org", "1", "user_bob")

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_schema_isolation_by_visibility_class(self, mock_get_rag):
        """Two users with different schema_hash see different schema indexes."""
        from deriva_mcp.rag.helpers import rag_suggest_entity

        mock_manager = MagicMock()
        mock_get_rag.return_value = mock_manager

        # User A with schema_hash "hash_a"
        conn_a = MagicMock()
        conn_a.schema_hash = "hash_a"
        conn_a.hostname = "test"
        conn_a.catalog_id = "1"
        conn_a.schema_dirty = False
        conn_a._schema_reindex_at = 0.0
        conn_a.ml_instance = MagicMock()

        mock_manager.search.return_value = []
        rag_suggest_entity("Diagnosis", conn_a)
        call_a = mock_manager.search.call_args
        assert "hash_a" in str(call_a)

        # User B with schema_hash "hash_b"
        conn_b = MagicMock()
        conn_b.schema_hash = "hash_b"
        conn_b.hostname = "test"
        conn_b.catalog_id = "1"
        conn_b.schema_dirty = False
        conn_b._schema_reindex_at = 0.0
        conn_b.ml_instance = MagicMock()

        rag_suggest_entity("Diagnosis", conn_b)
        call_b = mock_manager.search.call_args
        assert "hash_b" in str(call_b)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_rag_integration_layers.py::TestACLIsolation -v`
Expected: All PASS (these are unit tests that verify source key scoping — they should pass once the helper functions exist from earlier tasks).

- [ ] **Step 3: Commit**

```bash
git add tests/test_rag_integration_layers.py
git commit -m "test: ACL isolation — verify independent data/schema indexes per user"
```

### Task 24: Integration test — end-to-end scenario

**Files:**
- Modify: `tests/test_rag_integration_layers.py`

- [ ] **Step 1: Write the integration test**

This test exercises the full flow: connect → create entity → verify reindex triggered → query with typo → verify suggestion.

```python
# Append to tests/test_rag_integration_layers.py

class TestEndToEndIntegration:
    """Integration test: connect, create, reindex, query-with-typo, suggest."""

    @patch("deriva_mcp.rag.helpers.threading.Thread")
    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_create_table_triggers_reindex_and_error_suggests(self, mock_get_rag, mock_thread_cls):
        """Full flow: create_table triggers reindex; query_table with typo gets suggestion."""
        mock_manager = MagicMock()
        mock_get_rag.return_value = mock_manager

        # Setup conn_manager
        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_table = MagicMock()
        mock_table.name = "Diagnosis"
        mock_table.schema = MagicMock()
        mock_table.schema.name = "isa"
        mock_table.columns = []
        mock_ml.create_table.return_value = mock_table
        conn_manager.get_active_or_raise.return_value = mock_ml

        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.schema_dirty = False
        mock_conn_info._schema_reindex_at = 0.0
        mock_conn_info._data_reindex_at = 0.0
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        # Phase 1: create_table → should trigger reindex
        mock_manager.search.return_value = []  # No duplicates
        from deriva_mcp.tools.schema import register_schema_tools
        mcp, tools = _create_tool_capture()
        register_schema_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["create_table"]("Diagnosis")))
        assert result["status"] == "created"

        # Verify reindex was triggered (Thread was called)
        assert mock_thread_cls.call_count >= 1

        # Phase 2: query_table with typo → should get suggestion
        mock_ml.model.name_to_table.side_effect = Exception(
            "Table 'Diagnoiss' not found in schema 'isa'"
        )
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis (table)", "relevance": 0.92,
             "source": "schema:test:1:abc123",
             "section_heading": "isa.Diagnosis",
             "doc_type": "catalog-schema"}
        ]

        from deriva_mcp.tools.data import register_data_tools
        mcp2, tools2 = _create_tool_capture()
        register_data_tools(mcp2, conn_manager)

        result2 = json.loads(asyncio.run(tools2["query_table"]("Diagnoiss")))
        assert result2["status"] == "error"
        assert "suggestions" in result2
        assert result2["hint"].startswith("Did you mean:")
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_rag_integration_layers.py::TestEndToEndIntegration -v`
Expected: PASS (this test exercises the full Layer 1 → Layer 2 flow using mocks).

- [ ] **Step 3: Commit**

```bash
git add tests/test_rag_integration_layers.py
git commit -m "test: end-to-end integration test for create → reindex → query-with-typo → suggest"
```

### Task 25: Add graceful degradation and debounce tests

**Files:**
- Modify: `tests/test_rag_helpers.py`

- [ ] **Step 1: Add graceful degradation test**

```python
# Append to tests/test_rag_helpers.py
class TestGracefulDegradation:
    """Verify all helpers are no-ops when RAG is unavailable."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_suggest_entity_noop(self, _):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        assert rag_suggest_entity("test", MagicMock()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_suggest_record_noop(self, _):
        from deriva_mcp.rag.helpers import rag_suggest_record
        assert rag_suggest_record("test", MagicMock()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_enrich_resource_noop(self, _):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        assert rag_enrich_resource("test", MagicMock()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_trigger_schema_reindex_noop(self, _):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        trigger_schema_reindex(MagicMock())  # Should not raise

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_trigger_data_reindex_noop(self, _):
        from deriva_mcp.rag.helpers import trigger_data_reindex
        trigger_data_reindex(MagicMock())  # Should not raise


class TestDebounce:
    """Verify debounce prevents rapid reindex."""

    @patch("deriva_mcp.rag.helpers.threading.Thread")
    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_rapid_calls_debounced(self, mock_get_rag, mock_thread_cls):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        mock_get_rag.return_value = MagicMock()

        conn_info = MagicMock()
        conn_info._schema_reindex_at = 0.0

        # First call should trigger
        trigger_schema_reindex(conn_info)
        assert mock_thread_cls.call_count == 1

        # Rapid second call should be debounced
        trigger_schema_reindex(conn_info)
        assert mock_thread_cls.call_count == 1  # Still 1
```

- [ ] **Step 2: Run all helper tests**

Run: `pytest tests/test_rag_helpers.py -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_rag_helpers.py
git commit -m "test: add graceful degradation and debounce tests"
```

### Task 26: Update the `deriva:semantic-awareness` skill

**Files:**
- Modify: `/Users/carl/GitHub/deriva-skills/skills/semantic-awareness/SKILL.md`

- [ ] **Step 1: Add deprecation note to the top of the skill**

After the frontmatter, add a prominent note:

```markdown
> **Note (2026-03-16):** The Deriva MCP server now performs automatic duplicate detection (Layer 3) on all creation tools (`create_table`, `create_asset_table`, `create_vocabulary`, `create_feature`). When a near-duplicate entity is detected, the tool response includes a `similar_existing` field with suggestions and a warning message. This skill remains available as a behavioral guardrail for users on older MCP server versions that lack built-in duplicate detection.
```

- [ ] **Step 2: Commit in the skills repo**

```bash
cd /Users/carl/GitHub/deriva-skills
git add skills/semantic-awareness/SKILL.md
git commit -m "docs: note that MCP server now handles duplicate detection natively"
```

### Task 27: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All PASS.

- [ ] **Step 2: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix: address any remaining test failures"
```
