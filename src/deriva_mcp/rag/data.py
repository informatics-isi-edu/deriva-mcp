# src/deriva_mcp/rag/data.py
"""Per-user data indexing for RAG.

Indexes dataset and execution records per-user so RAG can answer
questions like "find the training dataset" or "which experiment used dataset X?".

Data chunks are stored in the same ChromaDB collection as docs/schema,
distinguished by doc_type="catalog-data" and a source name of
data:{hostname}:{catalog_id}:{user_id}.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("deriva-mcp")

DEFAULT_INDEXED_TABLES = ["Dataset", "Execution"]
DATA_STALENESS_SECONDS = 3600


def data_source_name(hostname: str, catalog_id: str | int, user_id: str) -> str:
    """Build the RAG source name for a user's data index."""
    return f"data:{hostname}:{catalog_id}:{user_id}"


def dataset_record_to_markdown(
    record: dict[str, Any], types: list[str] | None = None, version: str | None = None,
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
    record: dict[str, Any], workflow_name: str | None = None,
    status: str | None = None, input_datasets: list[str] | None = None,
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
    ml: Any, hostname: str, catalog_id: str | int, user_id: str,
    collection: Any, chunk_size_target: int = 800,
    indexed_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Index user-visible dataset and execution records into ChromaDB."""
    source = data_source_name(hostname, catalog_id, user_id)
    tables = indexed_tables or DEFAULT_INDEXED_TABLES

    if _is_data_index_fresh(collection, source):
        return {"source": source, "status": "fresh", "chunks_created": 0}

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

    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])

    logger.info(f"Indexed {len(ids)} data records for {source}")
    return {"source": source, "status": "indexed", "chunks_created": len(ids)}


def _index_datasets(ml, source, now, ids, documents, metadatas):
    for ds in ml.find_datasets():
        rid = ds.dataset_rid
        md = dataset_record_to_markdown(
            {"RID": rid, "Description": ds.description, "RCT": ""},
            types=ds.dataset_types,
            version=str(ds.current_version) if ds.current_version else None,
        )
        ids.append(f"{source}:Dataset:{rid}")
        documents.append(md)
        metadatas.append({"source": source, "doc_type": "catalog-data", "table": "Dataset", "rid": rid, "indexed_at": now})


def _index_executions(ml, source, now, ids, documents, metadatas):
    try:
        rows = list(ml.get_table_as_dict("Execution"))
    except Exception:
        return
    for row in rows:
        rid = row.get("RID", "")
        md = execution_record_to_markdown(row, workflow_name=row.get("Workflow", ""), status=row.get("Status", ""))
        ids.append(f"{source}:Execution:{rid}")
        documents.append(md)
        metadatas.append({"source": source, "doc_type": "catalog-data", "table": "Execution", "rid": rid, "indexed_at": now})


def _index_generic_table(ml, table_name, source, now, ids, documents, metadatas):
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
        metadatas.append({"source": source, "doc_type": "catalog-data", "table": table_name, "rid": rid, "indexed_at": now})


def _is_data_index_fresh(collection: Any, source: str) -> bool:
    try:
        results = collection.get(where={"source": source}, include=["metadatas"], limit=1)
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
    try:
        results = collection.get(where={"source": source}, include=[])
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            return len(results["ids"])
    except Exception as e:
        logger.warning(f"Failed to delete data chunks for {source}: {e}")
    return 0
