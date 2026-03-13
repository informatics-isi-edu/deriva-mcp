"""RAG (Retrieval-Augmented Generation) documentation service.

Provides semantic search across indexed documentation from
Deriva ecosystem repositories using ChromaDB and fastembed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from deriva_mcp.rag.config import RAGConfig, SourceConfig, load_sources, save_sources

logger = logging.getLogger("deriva-mcp")


class RAGManager:
    """Manages the RAG documentation index.

    Provides initialization, search, ingestion, and lifecycle management
    for the ChromaDB-backed documentation index.
    """

    def __init__(self, config: RAGConfig | None = None):
        self._config = config or RAGConfig()
        self._collection = None
        self._client = None
        self._sources: list[SourceConfig] = []
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize ChromaDB and load source configurations.

        Creates the persistent ChromaDB client and collection,
        loads source configs from disk (or defaults).
        """
        if self._initialized:
            return

        # Ensure persistence directory exists
        self._config.persist_dir.mkdir(parents=True, exist_ok=True)
        self._config.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(path=str(self._config.chroma_dir))

        # Create or get collection with ChromaDB's built-in ONNX MiniLM-L6-v2 embeddings
        embedding_fn = ONNXMiniLM_L6_V2()
        self._collection = self._client.get_or_create_collection(
            name=self._config.collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Load source configurations
        self._sources = load_sources(self._config.sources_path)

        self._initialized = True
        count = self._collection.count()
        logger.info(
            f"RAG manager initialized: {count} chunks in index, "
            f"{len(self._sources)} sources configured"
        )

        # Auto-ingest if index is empty or auto_update_on_start is set
        if count == 0 or self._config.auto_update_on_start:
            action = "auto-updating" if count > 0 else "auto-populating empty"
            logger.info(f"RAG: {action} index in background thread")
            thread = threading.Thread(target=self._background_ingest, daemon=True)
            thread.start()

    def _background_ingest(self) -> None:
        """Run full ingestion in a background thread."""
        try:
            result = self.ingest_source()
            total = result.get("total_chunks", 0)
            logger.info(f"RAG background ingestion complete: {total} chunks indexed")
        except Exception as e:
            logger.error(f"RAG background ingestion failed: {e}")

    def _ensure_initialized(self) -> None:
        """Ensure the manager is initialized, raising if not."""
        if not self._initialized:
            raise RuntimeError("RAG manager not initialized. Call initialize() first.")

    def search(
        self,
        query: str,
        limit: int | None = None,
        source: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the documentation index.

        Args:
            query: Search query text
            limit: Maximum number of results (default from config)
            source: Filter by source name (e.g. "deriva-ml-docs")
            doc_type: Filter by document type (e.g. "api-reference")

        Returns:
            List of result dicts with text, metadata, and relevance score
        """
        self._ensure_initialized()

        if limit is None:
            limit = self._config.default_search_limit

        # Build where filter
        where_filter = None
        conditions = []
        if source:
            conditions.append({"source": source})
        if doc_type:
            conditions.append({"doc_type": doc_type})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        try:
            query_params: dict[str, Any] = {
                "query_texts": [query],
                "n_results": limit,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                query_params["where"] = where_filter

            results = self._collection.query(**query_params)
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

        # Format results
        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to 0-1 relevance score
                relevance = max(0.0, 1.0 - (distance / 2.0))

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""

                # Build GitHub URL from metadata
                repo = metadata.get("repo", "")
                path = metadata.get("path", "")
                github_url = ""
                if repo and path:
                    # Find the source config to get branch
                    branch = "main"
                    for s in self._sources:
                        if s.name == metadata.get("source"):
                            branch = s.branch
                            break
                    github_url = f"https://github.com/{repo}/blob/{branch}/{path}"

                formatted.append({
                    "id": doc_id,
                    "text": document,
                    "relevance": round(relevance, 4),
                    "source": metadata.get("source", ""),
                    "repo": repo,
                    "path": path,
                    "section_heading": metadata.get("section_heading", ""),
                    "heading_hierarchy": metadata.get("heading_hierarchy", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "github_url": github_url,
                })

        return formatted

    def ingest_source(
        self,
        source_name: str | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Full ingestion of one or all sources.

        Args:
            source_name: Specific source to ingest, or None for all
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with ingestion statistics
        """
        self._ensure_initialized()
        from deriva_mcp.rag.ingester import ingest_source

        results = []
        sources_to_ingest = self._sources
        if source_name:
            sources_to_ingest = [s for s in self._sources if s.name == source_name]
            if not sources_to_ingest:
                return {"error": f"Source not found: {source_name}"}

        total = len(sources_to_ingest)
        for i, source in enumerate(sources_to_ingest):

            def source_progress(msg: str, pct: float) -> None:
                if progress_callback:
                    # Scale progress across all sources
                    overall_pct = ((i / total) + (pct / 100.0 / total)) * 100.0
                    progress_callback(msg, overall_pct)

            stats = ingest_source(
                source=source,
                collection=self._collection,
                chunk_size_target=self._config.chunk_size_target,
                overlap_sentences=self._config.chunk_overlap_sentences,
                progress_callback=source_progress,
            )
            results.append(stats)

        # Save updated source configs
        save_sources(self._config.sources_path, self._sources)

        return {
            "sources_processed": len(results),
            "results": results,
            "total_chunks": self._collection.count(),
        }

    def update_source(
        self,
        source_name: str | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Incremental update of one or all sources.

        Args:
            source_name: Specific source to update, or None for all
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with update statistics
        """
        self._ensure_initialized()
        from deriva_mcp.rag.ingester import update_source

        results = []
        sources_to_update = self._sources
        if source_name:
            sources_to_update = [s for s in self._sources if s.name == source_name]
            if not sources_to_update:
                return {"error": f"Source not found: {source_name}"}

        total = len(sources_to_update)
        for i, source in enumerate(sources_to_update):
            # Build previous file SHA map from existing chunks
            previous_file_shas = self._get_file_shas_for_source(source.name)

            def source_progress(msg: str, pct: float) -> None:
                if progress_callback:
                    overall_pct = ((i / total) + (pct / 100.0 / total)) * 100.0
                    progress_callback(msg, overall_pct)

            stats = update_source(
                source=source,
                collection=self._collection,
                previous_file_shas=previous_file_shas,
                chunk_size_target=self._config.chunk_size_target,
                overlap_sentences=self._config.chunk_overlap_sentences,
                progress_callback=source_progress,
            )
            results.append(stats)

        # Save updated source configs
        save_sources(self._config.sources_path, self._sources)

        return {
            "sources_processed": len(results),
            "results": results,
            "total_chunks": self._collection.count(),
        }

    def _get_file_shas_for_source(self, source_name: str) -> dict[str, str]:
        """Get {path: commit_sha} for all indexed files from a source."""
        try:
            results = self._collection.get(
                where={"source": source_name},
                include=["metadatas"],
            )
            file_shas: dict[str, str] = {}
            if results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    path = metadata.get("path", "")
                    sha = metadata.get("commit_sha", "")
                    if path and sha:
                        file_shas[path] = sha
            return file_shas
        except Exception:
            return {}

    def get_status(self) -> dict[str, Any]:
        """Get index status information.

        Returns:
            Dict with index stats, source list, and last update times
        """
        self._ensure_initialized()

        sources_info = []
        for source in self._sources:
            # Count chunks for this source
            try:
                source_results = self._collection.get(
                    where={"source": source.name},
                    include=[],
                )
                chunk_count = len(source_results["ids"]) if source_results and source_results["ids"] else 0
            except Exception:
                chunk_count = 0

            sources_info.append({
                "name": source.name,
                "repo": f"{source.repo_owner}/{source.repo_name}",
                "branch": source.branch,
                "path_prefix": source.path_prefix,
                "doc_type": source.doc_type,
                "chunk_count": chunk_count,
                "last_indexed_sha": source.last_indexed_sha,
                "last_indexed_at": source.last_indexed_at.isoformat() if source.last_indexed_at else None,
            })

        # Find indexed catalog schemas (source names starting with "schema:")
        schema_sources = []
        try:
            all_results = self._collection.get(
                where={"doc_type": "catalog-schema"},
                include=["metadatas"],
            )
            if all_results and all_results["metadatas"]:
                # Group by source
                schema_counts: dict[str, int] = {}
                schema_hashes: dict[str, str] = {}
                for metadata in all_results["metadatas"]:
                    src = metadata.get("source", "")
                    schema_counts[src] = schema_counts.get(src, 0) + 1
                    if "schema_hash" in metadata:
                        schema_hashes[src] = metadata["schema_hash"]
                for src, count in sorted(schema_counts.items()):
                    schema_sources.append({
                        "name": src,
                        "doc_type": "catalog-schema",
                        "chunk_count": count,
                        "schema_hash": schema_hashes.get(src, ""),
                    })
        except Exception:
            pass

        return {
            "initialized": self._initialized,
            "total_chunks": self._collection.count() if self._collection else 0,
            "persist_dir": str(self._config.persist_dir),
            "collection_name": self._config.collection_name,
            "sources": sources_info,
            "catalog_schemas": schema_sources,
        }

    def add_source(
        self,
        name: str,
        repo_owner: str,
        repo_name: str,
        branch: str = "main",
        path_prefix: str = "docs/",
        include_patterns: list[str] | None = None,
        doc_type: str = "user-guide",
    ) -> dict[str, Any]:
        """Register a new documentation source.

        Args:
            name: Unique source name
            repo_owner: GitHub repo owner
            repo_name: GitHub repo name
            branch: Git branch (default "main")
            path_prefix: Path prefix to filter files
            include_patterns: File patterns to include (default ["*.md"])
            doc_type: Document type tag

        Returns:
            Dict confirming the new source
        """
        self._ensure_initialized()

        # Check for duplicates
        for s in self._sources:
            if s.name == name:
                return {"error": f"Source already exists: {name}"}

        source = SourceConfig(
            name=name,
            source_type="github_repo",
            repo_owner=repo_owner,
            repo_name=repo_name,
            branch=branch,
            path_prefix=path_prefix,
            include_patterns=include_patterns or ["*.md"],
            doc_type=doc_type,
        )

        self._sources.append(source)
        save_sources(self._config.sources_path, self._sources)

        return {
            "status": "created",
            "source": source.to_dict(),
            "message": f"Source '{name}' registered. Run rag_ingest(source_name='{name}') to index it.",
        }

    def remove_source(self, name: str) -> dict[str, Any]:
        """Remove a documentation source and its indexed chunks.

        Args:
            name: Source name to remove

        Returns:
            Dict with removal status
        """
        self._ensure_initialized()

        source = None
        for s in self._sources:
            if s.name == name:
                source = s
                break

        if source is None:
            return {"error": f"Source not found: {name}"}

        # Delete all chunks for this source
        chunks_deleted = 0
        try:
            results = self._collection.get(
                where={"source": name},
                include=[],
            )
            if results and results["ids"]:
                self._collection.delete(ids=results["ids"])
                chunks_deleted = len(results["ids"])
        except Exception as e:
            logger.warning(f"Failed to delete chunks for source {name}: {e}")

        # Remove source config
        self._sources = [s for s in self._sources if s.name != name]
        save_sources(self._config.sources_path, self._sources)

        return {
            "status": "removed",
            "source": name,
            "chunks_deleted": chunks_deleted,
        }

    def index_catalog_schema(
        self,
        schema_info: dict[str, Any],
        hostname: str,
        catalog_id: str | int,
        vocabulary_terms: dict[str, list[dict[str, str]]] | None = None,
    ) -> dict[str, Any]:
        """Index a catalog's schema for RAG search.

        Converts the schema to markdown, chunks it, and stores it in the
        same collection as documentation. Uses schema hashing for change
        detection — returns immediately if the schema hasn't changed.

        Args:
            schema_info: Output of ``ml.model.get_schema_description()``.
            hostname: Catalog hostname.
            catalog_id: Catalog ID.
            vocabulary_terms: Optional mapping of vocabulary table names to their
                term lists. Included in the index so RAG can answer questions
                about available vocabulary values.

        Returns:
            Dict with indexing statistics.
        """
        self._ensure_initialized()
        from deriva_mcp.rag.schema import index_catalog_schema

        return index_catalog_schema(
            schema_info=schema_info,
            hostname=hostname,
            catalog_id=catalog_id,
            collection=self._collection,
            chunk_size_target=self._config.chunk_size_target,
            overlap_sentences=self._config.chunk_overlap_sentences,
            vocabulary_terms=vocabulary_terms,
        )

    def remove_catalog_schema(self, hostname: str, catalog_id: str | int) -> dict[str, Any]:
        """Remove indexed schema chunks for a catalog.

        Args:
            hostname: Catalog hostname.
            catalog_id: Catalog ID.

        Returns:
            Dict with removal status.
        """
        self._ensure_initialized()
        from deriva_mcp.rag.schema import _remove_schema_chunks, schema_source_name

        source = schema_source_name(hostname, catalog_id)
        deleted = _remove_schema_chunks(self._collection, source)
        return {"source": source, "chunks_deleted": deleted}

    def shutdown(self) -> None:
        """Clean shutdown of the RAG manager."""
        if self._initialized:
            logger.info("Shutting down RAG manager")
            # ChromaDB PersistentClient auto-persists, nothing special needed
            self._initialized = False


# Global singleton
_rag_manager: RAGManager | None = None


def init_rag_manager(config: RAGConfig | None = None) -> RAGManager:
    """Initialize the global RAG manager.

    Args:
        config: RAG configuration. Uses defaults if None.

    Returns:
        The initialized RAGManager instance.
    """
    global _rag_manager

    if _rag_manager is not None:
        logger.debug("RAG manager already initialized, ignoring reinit")
        return _rag_manager

    _rag_manager = RAGManager(config)
    _rag_manager.initialize()
    return _rag_manager


def get_rag_manager() -> RAGManager | None:
    """Get the global RAG manager instance, or None if not initialized."""
    return _rag_manager
