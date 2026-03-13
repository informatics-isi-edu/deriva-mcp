"""Catalog schema indexing for RAG.

Converts a Deriva catalog schema into markdown documents suitable for
chunking and embedding. Each table becomes a markdown document with
columns, foreign keys, features, and vocabulary terms.

Schema chunks are stored in the same ChromaDB collection as documentation
chunks, distinguished by ``doc_type="catalog-schema"`` and a source name
of ``schema:{hostname}:{catalog_id}``.

Access control is inherited from ERMrest: if a user can connect to the
catalog (and thus read /schema), they can see all schema chunks. The
source-level filter in ``rag_search`` ensures users only see schema
chunks for their active catalog connection.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from deriva_mcp.rag.chunker import chunk_markdown

logger = logging.getLogger("deriva-mcp")


def schema_source_name(hostname: str, catalog_id: str | int) -> str:
    """Build the RAG source name for a catalog schema."""
    return f"schema:{hostname}:{catalog_id}"


def _schema_hash(schema_info: dict[str, Any]) -> str:
    """Compute a stable hash of the schema for change detection."""
    # Sort keys for determinism
    canonical = json.dumps(schema_info, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _table_to_markdown(
    schema_name: str,
    table_name: str,
    table_info: dict[str, Any],
    vocabulary_terms: list[dict[str, str]] | None = None,
) -> str:
    """Render a single table as a markdown document.

    Args:
        schema_name: Schema containing the table.
        table_name: Table name.
        table_info: Table info dict from ``get_schema_description()``.
        vocabulary_terms: Optional list of term dicts with 'Name' and 'Description'
            keys. Included for vocabulary tables so RAG can answer
            "what values are available?" questions.
    """
    parts: list[str] = []

    # Table classification
    tags = []
    if table_info.get("is_vocabulary"):
        tags.append("vocabulary")
    if table_info.get("is_asset"):
        tags.append("asset")
    if table_info.get("is_association"):
        tags.append("association")
    tag_str = f" ({', '.join(tags)})" if tags else ""

    parts.append(f"## {schema_name}.{table_name}{tag_str}")

    if table_info.get("comment"):
        parts.append(table_info["comment"])

    # Columns
    columns = table_info.get("columns", [])
    if columns:
        parts.append("### Columns")
        col_lines = []
        for col in columns:
            null_str = "" if col.get("nullok", True) else " NOT NULL"
            comment = f" — {col['comment']}" if col.get("comment") else ""
            col_lines.append(f"- **{col['name']}** ({col['type']}{null_str}){comment}")
        parts.append("\n".join(col_lines))

    # Foreign keys
    fks = table_info.get("foreign_keys", [])
    if fks:
        parts.append("### Foreign Keys")
        fk_lines = []
        for fk in fks:
            cols = ", ".join(fk.get("columns", []))
            ref = fk.get("referenced_table", "")
            ref_cols = ", ".join(fk.get("referenced_columns", []))
            fk_lines.append(f"- {cols} → {ref} ({ref_cols})")
        parts.append("\n".join(fk_lines))

    # Features
    features = table_info.get("features", [])
    if features:
        parts.append("### Features")
        feat_lines = []
        for f in features:
            feat_lines.append(f"- **{f['name']}** (table: {f['feature_table']})")
        parts.append("\n".join(feat_lines))

    # Vocabulary terms (for vocabulary tables)
    if vocabulary_terms:
        parts.append("### Terms")
        term_lines = []
        for term in vocabulary_terms:
            name = term.get("Name", "")
            desc = term.get("Description", "")
            if desc:
                term_lines.append(f"- **{name}** — {desc}")
            else:
                term_lines.append(f"- **{name}**")
        parts.append("\n".join(term_lines))

    return "\n\n".join(parts)


def schema_to_markdown(
    schema_info: dict[str, Any],
    vocabulary_terms: dict[str, list[dict[str, str]]] | None = None,
) -> str:
    """Convert a full catalog schema description to a markdown document.

    Args:
        schema_info: Output of ``ml.model.get_schema_description()``.
        vocabulary_terms: Optional mapping of ``table_name`` to a list of
            term dicts (each with ``Name`` and ``Description`` keys).
            Vocabulary terms are included in the markdown so RAG can
            answer questions like "what diagnosis types exist?"

    Returns:
        A single markdown string covering all tables.
    """
    vocab_terms = vocabulary_terms or {}
    parts: list[str] = []

    domain_schemas = schema_info.get("domain_schemas", [])
    default_schema = schema_info.get("default_schema", "")

    parts.append(f"# Catalog Schema")
    parts.append(f"Domain schemas: {', '.join(domain_schemas)}. Default: {default_schema}.")

    schemas = schema_info.get("schemas", {})
    for schema_name, schema_data in schemas.items():
        tables = schema_data.get("tables", {})
        for table_name, table_info in sorted(tables.items()):
            terms = vocab_terms.get(table_name)
            parts.append(_table_to_markdown(schema_name, table_name, table_info, terms))

    return "\n\n".join(parts)


def index_catalog_schema(
    schema_info: dict[str, Any],
    hostname: str,
    catalog_id: str | int,
    collection: Any,
    chunk_size_target: int = 800,
    overlap_sentences: int = 1,
    vocabulary_terms: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Index a catalog schema into the RAG collection.

    Converts the schema to markdown, chunks it, and upserts into ChromaDB.
    Uses the schema hash for change detection — returns early if unchanged.

    Args:
        schema_info: Output of ``ml.model.get_schema_description()``.
        hostname: Catalog hostname.
        catalog_id: Catalog ID.
        collection: ChromaDB collection.
        chunk_size_target: Target tokens per chunk.
        overlap_sentences: Sentence overlap between chunks.
        vocabulary_terms: Optional mapping of vocabulary table names to their
            term lists (each term has ``Name`` and ``Description``).

    Returns:
        Dict with indexing statistics.
    """
    source = schema_source_name(hostname, catalog_id)
    # Include vocab terms in hash so term changes trigger re-indexing
    hash_input = {**schema_info, "_vocab_terms": vocabulary_terms or {}}
    current_hash = _schema_hash(hash_input)

    # Check if already indexed with same hash
    try:
        existing = collection.get(
            where={"source": source},
            include=["metadatas"],
            limit=1,
        )
        if existing and existing["metadatas"]:
            stored_hash = existing["metadatas"][0].get("schema_hash", "")
            if stored_hash == current_hash:
                return {
                    "source": source,
                    "status": "unchanged",
                    "schema_hash": current_hash,
                }
    except Exception:
        pass  # Proceed with indexing

    # Remove old chunks for this catalog
    _remove_schema_chunks(collection, source)

    # Convert to markdown and chunk
    markdown = schema_to_markdown(schema_info, vocabulary_terms=vocabulary_terms)
    chunks = chunk_markdown(
        markdown,
        chunk_size_target=chunk_size_target,
        overlap_sentences=overlap_sentences,
    )

    if not chunks:
        return {"source": source, "status": "empty", "chunks_created": 0}

    # Prepare batch
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"{source}:{chunk.chunk_index}"
        ids.append(chunk_id)
        documents.append(chunk.text)
        metadatas.append({
            "source": source,
            "doc_type": "catalog-schema",
            "section_heading": chunk.section_heading,
            "heading_hierarchy": " > ".join(chunk.heading_hierarchy) if chunk.heading_hierarchy else "",
            "chunk_index": chunk.chunk_index,
            "schema_hash": current_hash,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        })

    # Batch upsert
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(f"Indexed schema for {source}: {len(ids)} chunks (hash={current_hash})")

    return {
        "source": source,
        "status": "indexed",
        "chunks_created": len(ids),
        "schema_hash": current_hash,
    }


def _remove_schema_chunks(collection: Any, source: str) -> int:
    """Remove all schema chunks for a catalog from the collection."""
    try:
        results = collection.get(
            where={"source": source},
            include=[],
        )
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            return len(results["ids"])
    except Exception as e:
        logger.warning(f"Failed to delete schema chunks for {source}: {e}")
    return 0
