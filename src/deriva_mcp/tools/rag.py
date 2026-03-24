"""MCP tools for RAG documentation search and management.

Provides semantic search across indexed Deriva documentation,
with tools for ingestion, incremental updates, and source management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-mcp")


def _get_rag_or_error() -> dict | None:
    """Get the RAG manager or return an error dict if not initialized."""
    from deriva_mcp.rag import get_rag_manager

    manager = get_rag_manager()
    if manager is None:
        return {"error": "RAG service not initialized. Check server logs for details."}
    return None


def register_rag_tools(mcp_server: "FastMCP", conn_manager: "ConnectionManager") -> None:
    """Register RAG documentation tools with the MCP server."""

    @mcp_server.tool()
    def rag_search(
        query: str,
        limit: int = 10,
        source: str | None = None,
        doc_type: str | None = None,
        include_schema: bool = True,
        include_data: bool = True,
    ) -> dict:
        """Search Deriva documentation, catalog schema, and catalog data using semantic similarity.

        Searches across three categories of indexed content:

        1. **Catalog schema** (``doc_type="catalog-schema"``): The connected
           catalog's structure, including:
           - Tables with columns, types, nullability, and comments
           - Foreign key relationships between tables
           - Feature definitions with term columns, asset columns, and value columns
           - Vocabulary terms with names, descriptions, and synonyms
           Use this to understand what data exists, how tables relate, what
           features are defined, and what vocabulary values are available.

        2. **Catalog data** (``doc_type="catalog-data"``): User-specific records
           from the connected catalog, including:
           - Datasets with types, versions, and descriptions
           - Executions with workflow names, status, and inputs
           Use this to find specific datasets by purpose or trace experiment
           provenance.

        3. **Documentation** (``doc_type="user-guide"``, ``"api-reference"``,
           ``"sdk-reference"``): Indexed docs from Deriva ecosystem repositories
           (deriva-ml, ermrest, chaise, deriva-py). Use this for API usage
           questions and how-to guidance.

        **Prefer this tool over reading raw resources** for catalog exploration.
        Use ``doc_type`` to focus results on the category you need.

        Args:
            query: Natural language search query (e.g., "how to create a dataset")
            limit: Maximum number of results to return (default 10)
            source: Filter by source name (e.g., "deriva-ml-docs", "ermrest-docs")
            doc_type: Filter by document type. Key values:
                - ``"catalog-schema"``: Tables, columns, FKs, features, vocab terms
                - ``"catalog-data"``: Datasets and executions in the catalog
                - ``"user-guide"``: DerivaML and Chaise documentation
                - ``"api-reference"``: ERMrest API documentation
                - ``"sdk-reference"``: deriva-py SDK documentation
            include_schema: If True (default), include catalog schema results when
                          connected. Set to False to search only documentation.
            include_data: If True (default), include per-user data index results
                        (datasets, executions) when connected. Set to False to
                        exclude user data from search results.

        Returns:
            Dict with search results including text snippets, relevance scores,
            source metadata, and GitHub URLs.

        Examples:
            # Explore catalog structure — tables, columns, relationships
            rag_search("Image tables and features", doc_type="catalog-schema")

            # Find vocabulary terms by meaning
            rag_search("classification categories", doc_type="catalog-schema")

            # Find feature definitions and their columns
            rag_search("diagnosis label confidence", doc_type="catalog-schema")

            # Find datasets by description, type, or purpose
            rag_search("training split labeled", doc_type="catalog-data")

            # Find executions by workflow or status
            rag_search("training experiment results", doc_type="catalog-data")

            # Search Deriva API documentation
            rag_search("how to create a dataset", include_schema=False, include_data=False)

            # Search everything (docs + schema + data) — the default
            rag_search("how are images classified")
        """
        error = _get_rag_or_error()
        if error:
            return error

        from deriva_mcp.rag import get_rag_manager

        manager = get_rag_manager()

        # If caller specified a source or doc_type filter, use it directly
        if source or doc_type:
            results = manager.search(query=query, limit=limit, source=source, doc_type=doc_type)
            return {
                "query": query,
                "result_count": len(results),
                "results": results,
            }

        # Default: search docs, and optionally include catalog schema
        doc_results = manager.search(query=query, limit=limit)

        schema_results = []
        if include_schema:
            conn_info = conn_manager.get_active_connection_info()
            if conn_info:
                # Resolve the user's visibility-class source name.
                # The schema hash was stored on the connection info at connect time.
                schema_hash = getattr(conn_info, "schema_hash", None)
                if schema_hash:
                    from deriva_mcp.rag.schema import schema_source_name

                    catalog_source = schema_source_name(
                        conn_info.hostname, conn_info.catalog_id, schema_hash
                    )
                    schema_results = manager.search(
                        query=query, limit=limit, source=catalog_source
                    )

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

        return {
            "query": query,
            "result_count": len(all_results),
            "results": all_results,
        }

    @mcp_server.tool()
    def rag_ingest(source_name: str | None = None) -> dict:
        """Full crawl and index of documentation sources.

        Crawls GitHub repositories, fetches all documentation files,
        chunks them, and indexes them for semantic search. This is a
        long-running operation that runs in the background.

        Args:
            source_name: Specific source to ingest (e.g., "deriva-ml-docs").
                        If None, ingests all configured sources.

        Returns:
            Dict with task ID for tracking progress, or immediate results
            if the operation completes quickly.
        """
        error = _get_rag_or_error()
        if error:
            return error

        from deriva_mcp.rag import get_rag_manager
        from deriva_mcp.tasks import TaskProgress, TaskType, get_task_manager

        manager = get_rag_manager()
        task_manager = get_task_manager()

        def run_ingest(progress_updater=None, **kwargs):
            source = kwargs.get("source_name")

            def progress_callback(msg: str, pct: float):
                if progress_updater:
                    progress_updater(
                        TaskProgress(
                            current_step=msg,
                            percent_complete=pct,
                            message=msg,
                        )
                    )

            return manager.ingest_source(
                source_name=source,
                progress_callback=progress_callback,
            )

        task = task_manager.create_task(
            user_id="default_user",
            task_type=TaskType.RAG_INGEST,
            task_fn=run_ingest,
            parameters={"source_name": source_name},
        )

        return {
            "task_id": task.task_id,
            "status": "started",
            "message": f"Ingestion started for {'all sources' if not source_name else source_name}. "
            f"Use get_task_status('{task.task_id}') to check progress.",
        }

    @mcp_server.tool()
    def rag_update(source_name: str | None = None) -> dict:
        """Incremental update of documentation index.

        Checks for changed files in source repositories and only
        re-indexes files that have been added, modified, or deleted.
        Much faster than full ingestion when few files have changed.

        Args:
            source_name: Specific source to update (e.g., "deriva-ml-docs").
                        If None, updates all configured sources.

        Returns:
            Dict with task ID for tracking progress.
        """
        error = _get_rag_or_error()
        if error:
            return error

        from deriva_mcp.rag import get_rag_manager
        from deriva_mcp.tasks import TaskProgress, TaskType, get_task_manager

        manager = get_rag_manager()
        task_manager = get_task_manager()

        def run_update(progress_updater=None, **kwargs):
            source = kwargs.get("source_name")

            def progress_callback(msg: str, pct: float):
                if progress_updater:
                    progress_updater(
                        TaskProgress(
                            current_step=msg,
                            percent_complete=pct,
                            message=msg,
                        )
                    )

            return manager.update_source(
                source_name=source,
                progress_callback=progress_callback,
            )

        task = task_manager.create_task(
            user_id="default_user",
            task_type=TaskType.RAG_UPDATE,
            task_fn=run_update,
            parameters={"source_name": source_name},
        )

        return {
            "task_id": task.task_id,
            "status": "started",
            "message": f"Update started for {'all sources' if not source_name else source_name}. "
            f"Use get_task_status('{task.task_id}') to check progress.",
        }

    @mcp_server.tool()
    def rag_status() -> dict:
        """Get the status of the RAG documentation index.

        Returns information about the index including total chunks,
        configured sources, and last update times.

        Returns:
            Dict with index status, source configurations, and statistics.
        """
        error = _get_rag_or_error()
        if error:
            return error

        from deriva_mcp.rag import get_rag_manager

        manager = get_rag_manager()
        return manager.get_status()

    @mcp_server.tool()
    def rag_add_source(
        name: str,
        repo_owner: str,
        repo_name: str,
        branch: str = "main",
        path_prefix: str = "docs/",
        include_patterns: list[str] | None = None,
        doc_type: str = "user-guide",
    ) -> dict:
        """Register a new documentation source for RAG indexing.

        After adding a source, run rag_ingest(source_name=name) to index it.

        Args:
            name: Unique name for this source (e.g., "my-project-docs")
            repo_owner: GitHub repository owner (e.g., "informatics-isi-edu")
            repo_name: GitHub repository name (e.g., "deriva-ml")
            branch: Git branch to index (default "main")
            path_prefix: Only index files under this path (default "docs/")
            include_patterns: File patterns to include (default ["*.md"])
            doc_type: Document type tag for filtering (default "user-guide")

        Returns:
            Dict confirming the source was added.
        """
        error = _get_rag_or_error()
        if error:
            return error

        from deriva_mcp.rag import get_rag_manager

        manager = get_rag_manager()
        return manager.add_source(
            name=name,
            repo_owner=repo_owner,
            repo_name=repo_name,
            branch=branch,
            path_prefix=path_prefix,
            include_patterns=include_patterns,
            doc_type=doc_type,
        )

    @mcp_server.tool()
    def rag_remove_source(name: str) -> dict:
        """Remove a documentation source and its indexed chunks.

        This deletes all indexed chunks for the source and removes
        it from the configuration.

        Args:
            name: Name of the source to remove (e.g., "my-project-docs")

        Returns:
            Dict with removal status and number of chunks deleted.
        """
        error = _get_rag_or_error()
        if error:
            return error

        from deriva_mcp.rag import get_rag_manager

        manager = get_rag_manager()
        return manager.remove_source(name)

    @mcp_server.tool()
    def rag_index_schema() -> dict:
        """Re-index the connected catalog's schema for RAG search.

        Fetches the current schema from the connected catalog and indexes
        it for semantic search. This happens automatically on connect_catalog,
        but can be called manually after schema changes (e.g., after creating
        tables, adding columns, or creating features).

        Uses schema hashing — returns immediately if unchanged.

        Returns:
            Dict with indexing statistics (status, chunks_created, schema_hash).
        """
        error = _get_rag_or_error()
        if error:
            return error

        conn_info = conn_manager.get_active_connection_info()
        if not conn_info:
            return {"error": "No active catalog connection. Run connect_catalog first."}

        from deriva_mcp.rag import get_rag_manager
        from deriva_mcp.rag.schema import compute_schema_hash

        manager = get_rag_manager()
        ml = conn_info.ml_instance
        schema_info = ml.model.get_schema_description()

        # Fetch vocabulary terms for richer indexing
        vocab_terms: dict = {}
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

        # Update the connection's schema hash for visibility-class search
        schema_hash = compute_schema_hash(schema_info, vocab_terms)
        conn_info.schema_hash = schema_hash

        return manager.index_catalog_schema(
            schema_info, conn_info.hostname, conn_info.catalog_id,
            vocabulary_terms=vocab_terms,
        )
