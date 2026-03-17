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

# Import get_rag_manager at module level so tests can patch it via
# `patch("deriva_mcp.rag.helpers.get_rag_manager")`.
# The actual chromadb initialization is deferred until first use.
from deriva_mcp.rag import get_rag_manager

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
    """Check if an error message indicates an entity-not-found condition."""
    if not message:
        return False
    return any(p.search(message) for p in _NOT_FOUND_PATTERNS)


def trigger_schema_reindex(conn_info: ConnectionInfo | None) -> None:
    """Fire-and-forget background schema reindex.
    Debounces to at most once per DEBOUNCE_SECONDS.
    No-op if conn_info is None or RAG is not initialized.
    """
    if conn_info is None:
        return

    manager = get_rag_manager()
    if manager is None:
        return

    now = time.time()
    if now - conn_info._schema_reindex_at < DEBOUNCE_SECONDS:
        return

    conn_info._schema_reindex_at = now

    def _do_reindex():
        try:
            from deriva_mcp.rag.schema import compute_schema_hash
            ml = conn_info.ml_instance
            schema_info = ml.model.get_schema_description()
            vocab_terms: dict[str, list[dict[str, str]]] = {}
            schemas = schema_info.get("schemas", {})
            for schema_data in schemas.values():
                tables = schema_data.get("tables", {})
                for table_name, table_info in tables.items():
                    if table_info.get("is_vocabulary"):
                        try:
                            terms = ml.list_vocabulary_terms(table_name)
                            vocab_terms[table_name] = [
                                {"Name": t.name, "Description": t.description or "",
                                 "Synonyms": list(t.synonyms) if t.synonyms else []}
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


def rag_suggest_entity(
    query: str,
    conn_info: ConnectionInfo | None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Search the user's visibility-class schema index for entity suggestions.
    Returns: [{"name": ..., "type": ..., "relevance": ..., "description": ...}]
    ACL enforcement: uses conn_info.schema_hash to scope search.
    """
    if conn_info is None:
        return []
    manager = get_rag_manager()
    if manager is None:
        return []
    schema_hash = getattr(conn_info, "schema_hash", None)
    if not schema_hash:
        return []
    if getattr(conn_info, "schema_dirty", False):
        trigger_schema_reindex(conn_info)
        conn_info.schema_dirty = False
    from deriva_mcp.rag.schema import schema_source_name
    source = schema_source_name(conn_info.hostname, conn_info.catalog_id, schema_hash)
    results = manager.search(query=query, limit=limit, source=source)
    suggestions = []
    for r in results:
        heading = r.get("section_heading", "")
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
            suggestions.append({"name": name, "type": entity_type, "relevance": r.get("relevance", 0), "description": description})
    return suggestions


def rag_enrich_resource(
    query: str,
    conn_info: ConnectionInfo | None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Search the doc index for related documentation links.
    Filters to relevance > ENRICHMENT_RELEVANCE_THRESHOLD and deduplicates by URL.
    """
    if conn_info is None:
        return []
    manager = get_rag_manager()
    if manager is None:
        return []
    results = manager.search(query=query, limit=limit * 2)
    seen_urls: set[str] = set()
    enriched: list[dict[str, Any]] = []
    for r in results:
        doc_type = r.get("doc_type", "")
        if doc_type == "catalog-schema" or doc_type == "catalog-data":
            continue
        relevance = r.get("relevance", 0)
        if relevance < ENRICHMENT_RELEVANCE_THRESHOLD:
            continue
        url = r.get("github_url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        enriched.append({"title": r.get("section_heading", "") or r.get("path", ""),
                         "source": r.get("source", ""), "url": url, "relevance": relevance})
        if len(enriched) >= limit:
            break
    return enriched


def trigger_data_reindex(conn_info: ConnectionInfo | None) -> None:
    """Fire-and-forget background data reindex. Debounces to at most once per DEBOUNCE_SECONDS."""
    if conn_info is None:
        return
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
            manager._ensure_initialized()
            index_user_data(ml=ml, hostname=conn_info.hostname, catalog_id=conn_info.catalog_id,
                          user_id=conn_info.user_id, collection=manager._collection)
        except Exception as e:
            logger.warning(f"Background data reindex failed: {e}")

    thread = threading.Thread(target=_do_reindex, daemon=True, name="data-rag-reindex")
    thread.start()


def rag_suggest_record(
    query: str, conn_info: ConnectionInfo | None, limit: int = 5,
) -> list[dict[str, Any]]:
    """Search the user's data index for record suggestions.
    Returns: [{"name": ..., "table": ..., "rid": ..., "relevance": ..., "description": ...}]
    """
    if conn_info is None:
        return []
    manager = get_rag_manager()
    if manager is None:
        return []
    user_id = getattr(conn_info, "user_id", None)
    if not user_id:
        return []
    if getattr(conn_info, "data_dirty", False):
        trigger_data_reindex(conn_info)
        conn_info.data_dirty = False
    from deriva_mcp.rag.data import data_source_name
    source = data_source_name(conn_info.hostname, conn_info.catalog_id, user_id)
    results = manager.search(query=query, limit=limit, source=source)
    suggestions = []
    for r in results:
        heading = r.get("section_heading", "")
        # Strip markdown heading prefix if present
        clean_heading = re.sub(r"^#+\s*", "", heading)
        name = clean_heading
        rid = ""
        if "(RID:" in clean_heading:
            parts = clean_heading.split("(RID:")
            name = parts[0].strip().split(":", 1)[-1].strip() if ":" in parts[0] else parts[0].strip()
            rid = parts[1].replace(")", "").strip()
        table = ""
        if clean_heading.startswith("Dataset:"):
            table = "Dataset"
        elif clean_heading.startswith("Execution:"):
            table = "Execution"
        suggestions.append({"name": name, "table": table, "rid": rid,
                           "relevance": r.get("relevance", 0), "description": r.get("text", "")[:200]})
    return suggestions
