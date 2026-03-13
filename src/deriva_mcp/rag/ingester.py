"""Ingestion pipeline for RAG documentation indexing.

Orchestrates the full pipeline: crawl -> fetch -> chunk -> embed -> store.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from deriva_mcp.rag.chunker import chunk_markdown
from deriva_mcp.rag.config import SourceConfig
from deriva_mcp.rag.crawler import crawl_repo, fetch_file_content

logger = logging.getLogger("deriva-mcp")


def _make_chunk_id(source_name: str, path: str, chunk_idx: int) -> str:
    """Generate a deterministic chunk ID."""
    return f"{source_name}:{path}:{chunk_idx}"


def ingest_source(
    source: SourceConfig,
    collection: Any,  # chromadb.Collection
    chunk_size_target: int = 800,
    overlap_sentences: int = 1,
    github_token: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Full ingestion: crawl a source, fetch files, chunk, and store in ChromaDB.

    Args:
        source: Source configuration
        collection: ChromaDB collection to upsert into
        chunk_size_target: Target tokens per chunk
        overlap_sentences: Sentence overlap between chunks
        github_token: Optional GitHub token
        progress_callback: Optional callback(message, percent_complete)

    Returns:
        Dict with ingestion statistics
    """
    stats = {
        "source": source.name,
        "files_processed": 0,
        "chunks_created": 0,
        "errors": [],
    }

    if progress_callback:
        progress_callback(f"Crawling {source.name}...", 0.0)

    # Crawl repository
    try:
        crawl_result = crawl_repo(source, github_token=github_token)
    except Exception as e:
        stats["errors"].append(f"Crawl failed: {e}")
        return stats

    if crawl_result.unchanged:
        stats["status"] = "unchanged"
        if progress_callback:
            progress_callback(f"{source.name}: no changes detected", 100.0)
        return stats

    total_files = len(crawl_result.files)
    if total_files == 0:
        stats["status"] = "no_files"
        return stats

    # Process each file
    all_ids: list[str] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    for i, file_info in enumerate(crawl_result.files):
        if progress_callback:
            pct = ((i + 1) / total_files) * 90.0  # Reserve last 10% for upsert
            progress_callback(f"Processing {file_info.path}...", pct)

        try:
            content = fetch_file_content(source, file_info.path, github_token=github_token)
        except Exception as e:
            stats["errors"].append(f"Fetch failed for {file_info.path}: {e}")
            continue

        # Chunk the document
        chunks = chunk_markdown(content, chunk_size_target=chunk_size_target, overlap_sentences=overlap_sentences)

        for chunk in chunks:
            chunk_id = _make_chunk_id(source.name, file_info.path, chunk.chunk_index)
            metadata = {
                "source": source.name,
                "repo": f"{source.repo_owner}/{source.repo_name}",
                "path": file_info.path,
                "section_heading": chunk.section_heading,
                "heading_hierarchy": " > ".join(chunk.heading_hierarchy) if chunk.heading_hierarchy else "",
                "doc_type": source.doc_type,
                "chunk_index": chunk.chunk_index,
                "commit_sha": crawl_result.tree_sha,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }

            all_ids.append(chunk_id)
            all_documents.append(chunk.text)
            all_metadatas.append(metadata)

        stats["files_processed"] += 1

    # Batch upsert to ChromaDB
    if all_ids:
        if progress_callback:
            progress_callback(f"Upserting {len(all_ids)} chunks...", 90.0)

        # ChromaDB has a batch size limit, process in chunks of 500
        batch_size = 500
        for batch_start in range(0, len(all_ids), batch_size):
            batch_end = batch_start + batch_size
            collection.upsert(
                ids=all_ids[batch_start:batch_end],
                documents=all_documents[batch_start:batch_end],
                metadatas=all_metadatas[batch_start:batch_end],
            )

    stats["chunks_created"] = len(all_ids)
    stats["tree_sha"] = crawl_result.tree_sha
    stats["status"] = "completed"

    # Update source state
    source.last_indexed_sha = crawl_result.tree_sha
    source.last_indexed_at = datetime.now(timezone.utc)

    if progress_callback:
        progress_callback(
            f"{source.name}: indexed {stats['files_processed']} files, {stats['chunks_created']} chunks",
            100.0,
        )

    return stats


def update_source(
    source: SourceConfig,
    collection: Any,  # chromadb.Collection
    previous_file_shas: dict[str, str],
    chunk_size_target: int = 800,
    overlap_sentences: int = 1,
    github_token: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Incremental update: only process changed files.

    Args:
        source: Source configuration
        collection: ChromaDB collection
        previous_file_shas: Dict of {path: sha} from previous index
        chunk_size_target: Target tokens per chunk
        overlap_sentences: Sentence overlap between chunks
        github_token: Optional GitHub token
        progress_callback: Optional callback(message, percent_complete)

    Returns:
        Dict with update statistics
    """
    stats = {
        "source": source.name,
        "files_added": 0,
        "files_modified": 0,
        "files_deleted": 0,
        "chunks_upserted": 0,
        "chunks_deleted": 0,
        "errors": [],
    }

    if progress_callback:
        progress_callback(f"Checking {source.name} for changes...", 0.0)

    # Crawl with previous state for diff
    try:
        crawl_result = crawl_repo(source, previous_files=previous_file_shas, github_token=github_token)
    except Exception as e:
        stats["errors"].append(f"Crawl failed: {e}")
        return stats

    if crawl_result.unchanged:
        stats["status"] = "unchanged"
        if progress_callback:
            progress_callback(f"{source.name}: no changes", 100.0)
        return stats

    changed_paths = crawl_result.added + crawl_result.modified
    total_work = len(changed_paths) + len(crawl_result.deleted)

    if total_work == 0:
        stats["status"] = "no_changes"
        return stats

    work_done = 0

    # Delete chunks for removed files
    for path in crawl_result.deleted:
        _delete_chunks_for_path(collection, source.name, path)
        stats["files_deleted"] += 1
        work_done += 1
        if progress_callback:
            progress_callback(f"Deleted: {path}", (work_done / total_work) * 90.0)

    # Process added and modified files
    all_ids: list[str] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    for path in changed_paths:
        work_done += 1
        if progress_callback:
            progress_callback(f"Processing: {path}", (work_done / total_work) * 90.0)

        # For modified files, delete old chunks first
        if path in crawl_result.modified:
            _delete_chunks_for_path(collection, source.name, path)
            stats["files_modified"] += 1
        else:
            stats["files_added"] += 1

        try:
            content = fetch_file_content(source, path, github_token=github_token)
        except Exception as e:
            stats["errors"].append(f"Fetch failed for {path}: {e}")
            continue

        chunks = chunk_markdown(content, chunk_size_target=chunk_size_target, overlap_sentences=overlap_sentences)

        for chunk in chunks:
            chunk_id = _make_chunk_id(source.name, path, chunk.chunk_index)
            metadata = {
                "source": source.name,
                "repo": f"{source.repo_owner}/{source.repo_name}",
                "path": path,
                "section_heading": chunk.section_heading,
                "heading_hierarchy": " > ".join(chunk.heading_hierarchy) if chunk.heading_hierarchy else "",
                "doc_type": source.doc_type,
                "chunk_index": chunk.chunk_index,
                "commit_sha": crawl_result.tree_sha,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            all_ids.append(chunk_id)
            all_documents.append(chunk.text)
            all_metadatas.append(metadata)

    # Batch upsert
    if all_ids:
        if progress_callback:
            progress_callback(f"Upserting {len(all_ids)} chunks...", 90.0)

        batch_size = 500
        for batch_start in range(0, len(all_ids), batch_size):
            batch_end = batch_start + batch_size
            collection.upsert(
                ids=all_ids[batch_start:batch_end],
                documents=all_documents[batch_start:batch_end],
                metadatas=all_metadatas[batch_start:batch_end],
            )

    stats["chunks_upserted"] = len(all_ids)
    stats["tree_sha"] = crawl_result.tree_sha
    stats["status"] = "completed"

    # Update source state
    source.last_indexed_sha = crawl_result.tree_sha
    source.last_indexed_at = datetime.now(timezone.utc)

    if progress_callback:
        progress_callback(
            f"{source.name}: +{stats['files_added']} ~{stats['files_modified']} -{stats['files_deleted']}",
            100.0,
        )

    return stats


def _delete_chunks_for_path(collection: Any, source_name: str, path: str) -> int:
    """Delete all chunks for a specific file path from the collection.

    Returns the number of chunks deleted.
    """
    try:
        # Query for chunks matching this source and path
        results = collection.get(
            where={"$and": [{"source": source_name}, {"path": path}]},
            include=[],
        )
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            return len(results["ids"])
    except Exception as e:
        logger.warning(f"Failed to delete chunks for {source_name}:{path}: {e}")
    return 0
