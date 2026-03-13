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
    ) -> dict:
        """Search Deriva documentation using semantic similarity.

        Searches across indexed documentation from Deriva ecosystem repositories
        (deriva-ml, ermrest, chaise, deriva-py) using vector embeddings.

        Args:
            query: Natural language search query (e.g., "how to create a dataset")
            limit: Maximum number of results to return (default 10)
            source: Filter by source name (e.g., "deriva-ml-docs", "ermrest-docs")
            doc_type: Filter by document type (e.g., "api-reference", "user-guide", "sdk-reference")

        Returns:
            Dict with search results including text snippets, relevance scores,
            source metadata, and GitHub URLs.
        """
        error = _get_rag_or_error()
        if error:
            return error

        from deriva_mcp.rag import get_rag_manager

        manager = get_rag_manager()
        results = manager.search(query=query, limit=limit, source=source, doc_type=doc_type)

        return {
            "query": query,
            "result_count": len(results),
            "results": results,
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
