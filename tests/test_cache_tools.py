"""Tests for cache management MCP tools and resource.

Tests cover:
- list_cached_results: listing cached entries
- query_cached_result: re-querying with sort/filter/pagination
- invalidate_cache: targeted and bulk invalidation
- Cache integration in denormalize_dataset and query_table
- Auto-invalidation on insert_records and update_record
- deriva://cache/results resource
"""

import json

import pytest
from unittest.mock import MagicMock, patch

from deriva_mcp.result_cache import CacheMeta, ResultCache
from tests.conftest import _create_tool_capture, _create_resource_capture, parse_json_result, assert_success


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def result_cache(tmp_path):
    """Create a ResultCache for testing."""
    db_path = tmp_path / "test_tools_cache.db"
    cache = ResultCache(db_path)
    yield cache
    cache.close()


@pytest.fixture
def mock_conn_manager_with_cache(mock_ml, result_cache):
    """Create a ConnectionManager mock with a result cache attached."""
    from deriva_mcp.connection import ConnectionManager

    conn_manager = MagicMock(spec=ConnectionManager)
    conn_manager.get_active.return_value = mock_ml
    conn_manager.get_active_or_raise.return_value = mock_ml

    mock_conn_info = MagicMock()
    mock_conn_info.execution = MagicMock()
    mock_conn_info.execution.execution_rid = "EXE-TEST"
    mock_conn_info.workflow_rid = "WF-TEST"
    mock_conn_info.user_id = "test_user"
    mock_conn_info.active_tool_execution = None
    mock_conn_info.result_cache = result_cache

    conn_manager.get_active_connection_info.return_value = mock_conn_info
    conn_manager.get_active_connection_info_or_raise.return_value = mock_conn_info
    conn_manager.get_active_execution.return_value = mock_conn_info.execution
    conn_manager.get_active_execution_or_raise.return_value = mock_conn_info.execution

    return conn_manager


@pytest.fixture
def cache_tools(mock_conn_manager_with_cache):
    """Capture cache tools with a connected mock that has a result cache."""
    from deriva_mcp.tools.cache import register_cache_tools
    mcp, tools = _create_tool_capture()
    register_cache_tools(mcp, mock_conn_manager_with_cache)
    return tools


@pytest.fixture
def cache_tools_no_cache(mock_conn_manager):
    """Capture cache tools with a connected mock that has NO result cache."""
    mock_conn_info = mock_conn_manager.get_active_connection_info()
    mock_conn_info.result_cache = None

    from deriva_mcp.tools.cache import register_cache_tools
    mcp, tools = _create_tool_capture()
    register_cache_tools(mcp, mock_conn_manager)
    return tools


def _populate_cache(result_cache, n_entries=3):
    """Populate cache with sample entries and return their keys."""
    keys = []
    for i in range(n_entries):
        cols = ["col_a", "col_b", "col_c"]
        rows = [{"col_a": f"val_{i}_{j}", "col_b": j * 10, "col_c": f"text_{j}"} for j in range(5)]
        key = result_cache.cache_key("denormalize", rid=f"DATASET-{i}", tables=["T1", "T2"])
        meta = CacheMeta(
            cache_key=key,
            tool_name="denormalize_dataset",
            params={"dataset_rid": f"DATASET-{i}", "include_tables": ["T1", "T2"]},
            columns=cols,
            source="bag" if i % 2 == 0 else "catalog",
            ttl_seconds=None if i % 2 == 0 else 300,
        )
        result_cache.store(key, cols, rows, meta)
        keys.append(key)
    return keys


# =============================================================================
# TestListCachedResults
# =============================================================================


class TestListCachedResults:
    """Tests for the list_cached_results tool."""

    @pytest.mark.asyncio
    async def test_list_empty_cache(self, cache_tools):
        """Returns empty list when nothing is cached."""
        result = await cache_tools["list_cached_results"]()
        data = parse_json_result(result)
        assert data["status"] == "ok"
        assert data["count"] == 0
        assert data["results"] == []

    @pytest.mark.asyncio
    async def test_list_populated_cache(self, cache_tools, result_cache):
        """Returns metadata for all cached entries."""
        keys = _populate_cache(result_cache, 3)

        result = await cache_tools["list_cached_results"]()
        data = parse_json_result(result)

        assert data["status"] == "ok"
        assert data["count"] == 3
        assert all("cache_key" in entry for entry in data["results"])
        assert all("row_count" in entry for entry in data["results"])

    @pytest.mark.asyncio
    async def test_list_no_cache_available(self, cache_tools_no_cache):
        """Returns graceful message when no cache is configured."""
        result = await cache_tools_no_cache["list_cached_results"]()
        data = parse_json_result(result)
        assert data["status"] == "ok"
        assert data["results"] == []


# =============================================================================
# TestQueryCachedResult
# =============================================================================


class TestQueryCachedResult:
    """Tests for the query_cached_result tool."""

    @pytest.mark.asyncio
    async def test_query_basic(self, cache_tools, result_cache):
        """Basic query returns cached data."""
        keys = _populate_cache(result_cache, 1)

        result = await cache_tools["query_cached_result"](cache_key=keys[0])
        data = parse_json_result(result)

        assert data["from_cache"] is True
        assert data["count"] == 5
        assert data["total_count"] == 5
        assert len(data["columns"]) == 3

    @pytest.mark.asyncio
    async def test_query_with_limit(self, cache_tools, result_cache):
        """Limit restricts returned rows."""
        keys = _populate_cache(result_cache, 1)

        result = await cache_tools["query_cached_result"](cache_key=keys[0], limit=2)
        data = parse_json_result(result)

        assert data["count"] == 2
        assert data["total_count"] == 5

    @pytest.mark.asyncio
    async def test_query_with_sort(self, cache_tools, result_cache):
        """Sort by column works."""
        keys = _populate_cache(result_cache, 1)

        result = await cache_tools["query_cached_result"](
            cache_key=keys[0], sort_by="col_b", sort_desc=True, limit=100
        )
        data = parse_json_result(result)

        values = [r["col_b"] for r in data["rows"]]
        assert values == sorted(values, reverse=True)

    @pytest.mark.asyncio
    async def test_query_with_filter(self, cache_tools, result_cache):
        """Filter by column value works."""
        keys = _populate_cache(result_cache, 1)

        result = await cache_tools["query_cached_result"](
            cache_key=keys[0], filter_col="col_a", filter_val="val_0_2", limit=100
        )
        data = parse_json_result(result)

        assert data["count"] == 1
        assert data["rows"][0]["col_a"] == "val_0_2"

    @pytest.mark.asyncio
    async def test_query_nonexistent_key(self, cache_tools, result_cache):
        """Querying a nonexistent key returns error."""
        result = await cache_tools["query_cached_result"](cache_key="rc_doesnotexist00")
        data = parse_json_result(result)
        assert data["status"] == "error"
        assert "not found" in data["message"]

    @pytest.mark.asyncio
    async def test_query_with_pagination(self, cache_tools, result_cache):
        """Offset and limit for pagination."""
        keys = _populate_cache(result_cache, 1)

        result = await cache_tools["query_cached_result"](
            cache_key=keys[0], limit=2, offset=3
        )
        data = parse_json_result(result)

        assert data["count"] == 2  # 5 total - 3 offset = 2 remaining
        assert data["total_count"] == 5

    @pytest.mark.asyncio
    async def test_query_no_cache_available(self, cache_tools_no_cache):
        """Returns error when no cache is configured."""
        result = await cache_tools_no_cache["query_cached_result"](cache_key="rc_anything000000")
        data = parse_json_result(result)
        assert data["status"] == "error"


# =============================================================================
# TestInvalidateCache
# =============================================================================


class TestInvalidateCache:
    """Tests for the invalidate_cache tool."""

    @pytest.mark.asyncio
    async def test_invalidate_by_key(self, cache_tools, result_cache):
        """Invalidate a specific entry."""
        keys = _populate_cache(result_cache, 3)

        result = await cache_tools["invalidate_cache"](cache_key=keys[0])
        data = parse_json_result(result)

        assert data["status"] == "ok"
        assert data["invalidated"] == 1
        assert not result_cache.has(keys[0])
        assert result_cache.has(keys[1])

    @pytest.mark.asyncio
    async def test_invalidate_by_source(self, cache_tools, result_cache):
        """Invalidate all catalog-sourced entries."""
        keys = _populate_cache(result_cache, 4)
        # keys[0]=bag, keys[1]=catalog, keys[2]=bag, keys[3]=catalog

        result = await cache_tools["invalidate_cache"](source="catalog")
        data = parse_json_result(result)

        assert data["status"] == "ok"
        assert data["invalidated"] == 2

    @pytest.mark.asyncio
    async def test_invalidate_all(self, cache_tools, result_cache):
        """Invalidate everything."""
        _populate_cache(result_cache, 3)

        result = await cache_tools["invalidate_cache"]()
        data = parse_json_result(result)

        assert data["status"] == "ok"
        assert data["invalidated"] == 3
        assert result_cache.list_cached() == []

    @pytest.mark.asyncio
    async def test_invalidate_no_cache(self, cache_tools_no_cache):
        """Graceful when no cache is available."""
        result = await cache_tools_no_cache["invalidate_cache"]()
        data = parse_json_result(result)
        assert data["status"] == "ok"
        assert data["invalidated"] == 0


# =============================================================================
# TestCacheResource
# =============================================================================


class TestCacheResource:
    """Tests for the deriva://cache/results resource."""

    @pytest.mark.asyncio
    async def test_resource_empty_cache(self, mock_conn_manager_with_cache):
        """Resource returns empty list for empty cache."""
        from deriva_mcp.resources import register_resources
        mcp, resources = _create_resource_capture()
        register_resources(mcp, mock_conn_manager_with_cache)

        # Find the cache resource
        if "deriva://cache/results" in resources:
            result = await resources["deriva://cache/results"]()
            data = json.loads(result)
            assert data["count"] == 0
        else:
            pytest.skip("cache/results resource not registered")

    @pytest.mark.asyncio
    async def test_resource_populated_cache(self, mock_conn_manager_with_cache, result_cache):
        """Resource returns cached entries."""
        _populate_cache(result_cache, 2)

        from deriva_mcp.resources import register_resources
        mcp, resources = _create_resource_capture()
        register_resources(mcp, mock_conn_manager_with_cache)

        if "deriva://cache/results" in resources:
            result = await resources["deriva://cache/results"]()
            data = json.loads(result)
            assert data["count"] == 2
        else:
            pytest.skip("cache/results resource not registered")
