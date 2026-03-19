"""Catalog schema indexing for RAG.

Converts a Deriva catalog schema into markdown documents suitable for
chunking and embedding. Each table becomes a markdown document with
columns, foreign keys, features, and vocabulary terms.

Schema chunks are stored in the same ChromaDB collection as documentation
chunks, distinguished by ``doc_type="catalog-schema"`` and a source name
of ``schema:{hostname}:{catalog_id}:{schema_hash}``.

**Multi-user visibility isolation:**

ERMrest returns an identity-dependent schema — each user sees only the
tables, columns, keys, and policies they have permission to enumerate.
Two users with different permissions get different ``/schema`` responses.

We use the schema hash as a **visibility fingerprint**: users whose
``/schema`` responses hash to the same value share an index.  With RBAC
and a small number of distinct permission profiles, this typically
produces only 2–3 indexes per catalog (e.g. one for regular users, one
for admins) rather than one per user.

At connect time:
1. Fetch the user's ``/schema`` and compute its hash.
2. If an index with that hash already exists, reuse it (no-op).
3. Otherwise, build a new index from this user's schema view.

At search time, ``rag_search`` resolves the current user's schema hash
to the matching source key so results are scoped to their visibility
class.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from deriva_mcp.rag.chunker import chunk_markdown

logger = logging.getLogger("deriva-mcp")


def schema_source_name(hostname: str, catalog_id: str | int, schema_hash: str | None = None) -> str:
    """Build the RAG source name for a catalog schema.

    When ``schema_hash`` is provided, the source name includes it to
    isolate indexes by visibility class.  Without a hash, returns a
    prefix suitable for searching across all visibility classes for a
    catalog.
    """
    base = f"schema:{hostname}:{catalog_id}"
    if schema_hash:
        return f"{base}:{schema_hash}"
    return base


def schema_source_prefix(hostname: str, catalog_id: str | int) -> str:
    """Return the source name prefix for a catalog (matches all visibility classes)."""
    return f"schema:{hostname}:{catalog_id}:"


def _schema_hash(schema_info: dict[str, Any]) -> str:
    """Compute a stable hash of the schema for change detection."""
    # Sort keys for determinism
    canonical = json.dumps(schema_info, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _table_to_markdown(
    schema_name: str,
    table_name: str,
    table_info: dict[str, Any],
    vocabulary_terms: list[dict[str, Any]] | None = None,
    feature_details: list[dict[str, Any]] | None = None,
) -> str:
    """Render a single table as a markdown document.

    Args:
        schema_name: Schema containing the table.
        table_name: Table name.
        table_info: Table info dict from ``get_schema_description()``.
        vocabulary_terms: Optional list of term dicts with ``Name``,
            ``Description``, and optionally ``Synonyms`` (list of strings).
            Included for vocabulary tables so RAG can answer
            "what values are available?" questions.
        feature_details: Optional list of enriched feature dicts with
            ``name``, ``comment``, ``term_columns``, ``asset_columns``,
            ``value_columns``. Included so RAG can answer "what columns
            does this feature have?" questions.
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

    # Features — use enriched details when available, fall back to basic schema info
    if feature_details:
        parts.append("### Features")
        feat_lines = []
        for f in feature_details:
            name = f.get("name", "")
            comment = f.get("comment", "")
            feat_table = f.get("feature_table", "")
            header = f"- **{name}**"
            if feat_table:
                header += f" (table: {feat_table})"
            if comment:
                header += f" — {comment}"
            feat_lines.append(header)

            # Term columns (vocabulary-based values)
            for col in f.get("term_columns", []):
                col_line = f"  - Term: **{col['name']}**"
                if col.get("vocabulary"):
                    col_line += f" (vocabulary: {col['vocabulary']})"
                if col.get("comment"):
                    col_line += f" — {col['comment']}"
                feat_lines.append(col_line)

            # Asset columns
            for col in f.get("asset_columns", []):
                col_line = f"  - Asset: **{col['name']}**"
                if col.get("comment"):
                    col_line += f" — {col['comment']}"
                feat_lines.append(col_line)

            # Value/metadata columns
            for col in f.get("value_columns", []):
                col_line = f"  - Value: **{col['name']}**"
                if col.get("type"):
                    col_line += f" ({col['type']})"
                if col.get("comment"):
                    col_line += f" — {col['comment']}"
                feat_lines.append(col_line)

        parts.append("\n".join(feat_lines))
    else:
        # Fall back to basic feature info from schema
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
            synonyms = term.get("Synonyms", []) or []
            line = f"- **{name}**"
            if desc:
                line += f" — {desc}"
            if synonyms:
                line += f" (synonyms: {', '.join(synonyms)})"
            term_lines.append(line)
        parts.append("\n".join(term_lines))

    return "\n\n".join(parts)


def schema_to_markdown(
    schema_info: dict[str, Any],
    vocabulary_terms: dict[str, list[dict[str, Any]]] | None = None,
    feature_details: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Convert a full catalog schema description to a markdown document.

    Args:
        schema_info: Output of ``ml.model.get_schema_description()``.
        vocabulary_terms: Optional mapping of ``table_name`` to a list of
            term dicts (each with ``Name``, ``Description``, and optionally
            ``Synonyms`` keys). Vocabulary terms are included in the
            markdown so RAG can answer questions like "what diagnosis
            types exist?"
        feature_details: Optional mapping of ``table_name`` to enriched
            feature definitions with columns, descriptions, and vocabulary
            references.

    Returns:
        A single markdown string covering all tables.
    """
    vocab_terms = vocabulary_terms or {}
    feat_details = feature_details or {}
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
            features = feat_details.get(table_name)
            parts.append(_table_to_markdown(schema_name, table_name, table_info, terms, features))

    return "\n\n".join(parts)


def find_schema_source(
    hostname: str,
    catalog_id: str | int,
    schema_hash: str,
    collection: Any,
) -> str | None:
    """Find the source name for an existing index matching this schema hash.

    Returns the full source name if an index exists for this visibility
    class, or ``None`` if one needs to be created.
    """
    source = schema_source_name(hostname, catalog_id, schema_hash)
    try:
        existing = collection.get(
            where={"source": source},
            include=[],
            limit=1,
        )
        if existing and existing["ids"]:
            return source
    except Exception:
        pass
    return None


def compute_schema_hash(
    schema_info: dict[str, Any],
    vocabulary_terms: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Compute the visibility-class hash for a schema.

    Includes vocabulary terms so term changes trigger re-indexing.
    """
    hash_input = {**schema_info, "_vocab_terms": vocabulary_terms or {}}
    return _schema_hash(hash_input)


def index_catalog_schema(
    schema_info: dict[str, Any],
    hostname: str,
    catalog_id: str | int,
    collection: Any,
    chunk_size_target: int = 800,
    overlap_sentences: int = 1,
    vocabulary_terms: dict[str, list[dict[str, Any]]] | None = None,
    feature_details: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Index a catalog schema into the RAG collection.

    Converts the schema to markdown, chunks it, and upserts into ChromaDB.
    The schema hash serves as both a change-detection key and a
    visibility-class fingerprint — users whose ``/schema`` responses
    produce the same hash share the same index.

    Args:
        schema_info: Output of ``ml.model.get_schema_description()``.
        hostname: Catalog hostname.
        catalog_id: Catalog ID.
        collection: ChromaDB collection.
        chunk_size_target: Target tokens per chunk.
        overlap_sentences: Sentence overlap between chunks.
        vocabulary_terms: Optional mapping of vocabulary table names to their
            term lists (each term has ``Name``, ``Description``, and
            optionally ``Synonyms``).
        feature_details: Optional mapping of table names to enriched feature
            definitions with columns, descriptions, and vocabulary references.

    Returns:
        Dict with indexing statistics.
    """
    current_hash = compute_schema_hash(schema_info, vocabulary_terms)
    source = schema_source_name(hostname, catalog_id, current_hash)

    # Check if already indexed with this hash (same visibility class + same content)
    if find_schema_source(hostname, catalog_id, current_hash, collection):
        return {
            "source": source,
            "status": "unchanged",
            "schema_hash": current_hash,
        }

    # Remove old chunks for this specific visibility class (if hash changed)
    _remove_schema_chunks(collection, source)

    # Convert to markdown and chunk
    markdown = schema_to_markdown(schema_info, vocabulary_terms=vocabulary_terms, feature_details=feature_details)
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
