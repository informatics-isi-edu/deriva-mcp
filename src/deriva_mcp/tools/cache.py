"""MCP tools for managing the tabular result cache."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_mcp.connection import ConnectionManager

logger = logging.getLogger(__name__)


def register_cache_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register cache management tools."""

    @mcp.tool()
    async def list_cached_results() -> str:
        """List all cached tabular query results.

        Returns metadata for each cached result including the tool that
        produced it, parameters, row count, age, and cache key. Use the
        cache_key with query_cached_result to re-query with different
        sort/filter/pagination.

        Returns:
            JSON with list of cached result entries.
        """
        try:
            conn_info = conn_manager.get_active_connection_info()
            if not conn_info or not conn_info.result_cache:
                return json.dumps({"status": "ok", "results": [], "message": "No result cache available"})

            entries = conn_info.result_cache.list_cached()
            return json.dumps({
                "status": "ok",
                "results": [e.to_summary() for e in entries],
                "count": len(entries),
            })
        except Exception as e:
            logger.error(f"Failed to list cached results: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def query_cached_result(
        cache_key: str,
        sort_by: str | None = None,
        sort_desc: bool = False,
        filter_col: str | None = None,
        filter_val: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> str:
        """Re-query a cached tabular result with different sort/filter/pagination.

        Use list_cached_results to find available cache keys. This tool lets
        you paginate, sort, and filter previously computed results without
        re-executing the original query.

        Args:
            cache_key: The cache key from a previous query result.
            sort_by: Column name to sort by (e.g., "Image.CDR").
            sort_desc: Sort descending if True.
            filter_col: Column name to filter on.
            filter_val: Value to filter for (substring match, case-insensitive).
            limit: Maximum rows to return (default: 100).
            offset: Number of rows to skip for pagination.

        Returns:
            JSON with columns, rows, count, and total_count.
        """
        try:
            conn_info = conn_manager.get_active_connection_info()
            if not conn_info or not conn_info.result_cache:
                return json.dumps({"status": "error", "message": "No result cache available"})

            result = conn_info.result_cache.query(
                cache_key,
                sort_by=sort_by,
                sort_desc=sort_desc,
                filter_col=filter_col,
                filter_val=filter_val,
                limit=limit,
                offset=offset,
            )
            if result is None:
                return json.dumps({"status": "error", "message": f"Cache key '{cache_key}' not found or expired"})

            return json.dumps({
                "columns": result.columns,
                "rows": result.rows,
                "count": result.count,
                "total_count": result.total_count,
                "cache_key": result.cache_key,
                "source": result.source,
                "tool_name": result.tool_name,
                "from_cache": True,
            })
        except Exception as e:
            logger.error(f"Failed to query cached result: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def invalidate_cache(
        cache_key: str | None = None,
        source: str | None = None,
    ) -> str:
        """Invalidate cached tabular query results.

        Args:
            cache_key: Invalidate a specific cached result.
            source: Invalidate all results from this source ("bag" or "catalog").
            If neither is provided, invalidates all cached results.

        Returns:
            JSON with the number of entries invalidated.
        """
        try:
            conn_info = conn_manager.get_active_connection_info()
            if not conn_info or not conn_info.result_cache:
                return json.dumps({"status": "ok", "invalidated": 0, "message": "No result cache available"})

            count = conn_info.result_cache.invalidate(
                cache_key=cache_key,
                source=source,
            )
            return json.dumps({
                "status": "ok",
                "invalidated": count,
            })
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return json.dumps({"status": "error", "message": str(e)})
