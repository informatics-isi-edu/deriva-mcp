"""Tests for the ResultCache class.

Tests cover all public methods of ResultCache:
- cache_key: deterministic key generation
- has: existence and expiry checking
- store: table creation and data insertion
- query: retrieval with sort, filter, pagination
- list_cached: listing non-expired entries
- invalidate: targeted and bulk invalidation
- close: lifecycle management
"""

import time
from pathlib import Path

import pytest

from deriva_mcp.result_cache import CacheMeta, CacheResult, ResultCache


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cache_db(tmp_path):
    """Create a ResultCache backed by a temporary SQLite database."""
    db_path = tmp_path / "test_cache.db"
    cache = ResultCache(db_path)
    yield cache
    cache.close()


@pytest.fixture
def sample_rows():
    """Sample tabular data for testing."""
    return [
        {"Image.RID": "1-AAA", "Image.Filename": "img001.jpg", "Subject.Gender": "F", "Clinical.CDR": 0.7, "Clinical.IOP": 14},
        {"Image.RID": "1-BBB", "Image.Filename": "img002.jpg", "Subject.Gender": "M", "Clinical.CDR": 0.3, "Clinical.IOP": 18},
        {"Image.RID": "1-CCC", "Image.Filename": "img003.jpg", "Subject.Gender": "F", "Clinical.CDR": 0.9, "Clinical.IOP": 11},
        {"Image.RID": "1-DDD", "Image.Filename": "img004.jpg", "Subject.Gender": "M", "Clinical.CDR": 0.5, "Clinical.IOP": 22},
        {"Image.RID": "1-EEE", "Image.Filename": "img005.jpg", "Subject.Gender": "F", "Clinical.CDR": None, "Clinical.IOP": 16},
    ]


@pytest.fixture
def sample_columns():
    """Column names for sample data."""
    return ["Image.RID", "Image.Filename", "Subject.Gender", "Clinical.CDR", "Clinical.IOP"]


def _store_sample(cache, rows, columns, source="bag", ttl=None, tool="denormalize_dataset", key=None):
    """Helper to store sample data and return the cache key."""
    if key is None:
        key = cache.cache_key("denormalize", dataset_rid="4-411G", tables=["Image", "Subject"], version="2.10.0")
    meta = CacheMeta(
        cache_key=key,
        tool_name=tool,
        params={"dataset_rid": "4-411G", "include_tables": ["Image", "Subject"]},
        columns=columns,
        source=source,
        ttl_seconds=ttl,
    )
    cache.store(key, columns, rows, meta)
    return key


# =============================================================================
# TestCacheKey
# =============================================================================


class TestCacheKey:
    """Tests for cache key generation."""

    def test_deterministic(self):
        """Same parameters always produce the same key."""
        key1 = ResultCache.cache_key("denormalize", dataset_rid="4-411G", tables=["Image", "Subject"])
        key2 = ResultCache.cache_key("denormalize", dataset_rid="4-411G", tables=["Image", "Subject"])
        assert key1 == key2

    def test_prefix(self):
        """Keys start with 'rc_' prefix."""
        key = ResultCache.cache_key("denormalize", dataset_rid="X")
        assert key.startswith("rc_")

    def test_different_params_different_keys(self):
        """Different parameters produce different keys."""
        key1 = ResultCache.cache_key("denormalize", dataset_rid="4-411G")
        key2 = ResultCache.cache_key("denormalize", dataset_rid="4-222A")
        assert key1 != key2

    def test_different_tools_different_keys(self):
        """Different tool names produce different keys."""
        key1 = ResultCache.cache_key("denormalize", dataset_rid="X")
        key2 = ResultCache.cache_key("query", dataset_rid="X")
        assert key1 != key2

    def test_list_order_independent(self):
        """Lists of strings are sorted for deterministic keys."""
        key1 = ResultCache.cache_key("denormalize", tables=["Image", "Subject"])
        key2 = ResultCache.cache_key("denormalize", tables=["Subject", "Image"])
        assert key1 == key2

    def test_none_params_ignored(self):
        """None-valued parameters are excluded from key computation."""
        key1 = ResultCache.cache_key("query", table_name="Image")
        key2 = ResultCache.cache_key("query", table_name="Image", version=None)
        assert key1 == key2


# =============================================================================
# TestStore
# =============================================================================


class TestStore:
    """Tests for storing tabular results."""

    def test_store_creates_table(self, cache_db, sample_rows, sample_columns):
        """Storing creates a SQLite table with data."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        assert cache_db.has(key)

    def test_store_records_metadata(self, cache_db, sample_rows, sample_columns):
        """Stored metadata is retrievable."""
        key = _store_sample(cache_db, sample_rows, sample_columns, source="bag")
        meta = cache_db.get_meta(key)

        assert meta is not None
        assert meta.tool_name == "denormalize_dataset"
        assert meta.source == "bag"
        assert meta.row_count == 5
        assert meta.columns == sample_columns

    def test_store_replaces_existing(self, cache_db, sample_columns):
        """Storing with the same key replaces existing data."""
        rows1 = [{"Image.RID": "1-AAA", "Image.Filename": "old.jpg", "Subject.Gender": "F", "Clinical.CDR": 0.1, "Clinical.IOP": 10}]
        rows2 = [{"Image.RID": "1-BBB", "Image.Filename": "new.jpg", "Subject.Gender": "M", "Clinical.CDR": 0.2, "Clinical.IOP": 20}]

        key = _store_sample(cache_db, rows1, sample_columns)
        _store_sample(cache_db, rows2, sample_columns, key=key)

        result = cache_db.query(key)
        assert result.count == 1
        assert result.rows[0]["Image.RID"] == "1-BBB"

    def test_store_empty_rows(self, cache_db, sample_columns):
        """Storing zero rows creates an empty cached table."""
        key = _store_sample(cache_db, [], sample_columns)
        meta = cache_db.get_meta(key)
        assert meta.row_count == 0

    def test_store_with_null_values(self, cache_db, sample_rows, sample_columns):
        """Null values are stored correctly."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, limit=10)
        # Row 5 (1-EEE) has CDR=None
        eee_row = [r for r in result.rows if r["Image.RID"] == "1-EEE"][0]
        assert eee_row["Clinical.CDR"] is None


# =============================================================================
# TestHas
# =============================================================================


class TestHas:
    """Tests for existence and expiry checking."""

    def test_has_returns_true_for_stored(self, cache_db, sample_rows, sample_columns):
        """has() returns True for a stored, non-expired entry."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        assert cache_db.has(key)

    def test_has_returns_false_for_unknown(self, cache_db):
        """has() returns False for a key that was never stored."""
        assert not cache_db.has("rc_nonexistent12345")

    def test_has_returns_false_for_expired(self, cache_db, sample_rows, sample_columns):
        """has() returns False and cleans up expired entries."""
        key = _store_sample(cache_db, sample_rows, sample_columns, ttl=1)
        assert cache_db.has(key)  # Not expired yet

        time.sleep(1.1)
        assert not cache_db.has(key)  # Now expired

    def test_bag_source_never_expires(self, cache_db, sample_rows, sample_columns):
        """Bag-sourced entries (ttl=None) never expire."""
        key = _store_sample(cache_db, sample_rows, sample_columns, source="bag", ttl=None)
        # Can't actually test infinity, but verify it doesn't expire instantly
        meta = cache_db.get_meta(key)
        assert meta.ttl_seconds is None
        assert not meta.is_expired()


# =============================================================================
# TestQuery
# =============================================================================


class TestQuery:
    """Tests for querying cached results."""

    def test_query_returns_all_rows(self, cache_db, sample_rows, sample_columns):
        """Basic query returns all stored rows."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, limit=100)

        assert result is not None
        assert result.count == 5
        assert result.total_count == 5
        assert result.columns == sample_columns
        assert result.cache_key == key

    def test_query_with_limit(self, cache_db, sample_rows, sample_columns):
        """Limit restricts returned rows."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, limit=2)

        assert result.count == 2
        assert result.total_count == 5

    def test_query_with_offset(self, cache_db, sample_rows, sample_columns):
        """Offset skips initial rows."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, limit=100, offset=3)

        assert result.count == 2  # 5 total - 3 offset = 2 remaining

    def test_query_sort_ascending(self, cache_db, sample_rows, sample_columns):
        """Sort ascending by a numeric column."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, sort_by="Clinical.IOP", limit=100)

        iops = [r["Clinical.IOP"] for r in result.rows if r["Clinical.IOP"] is not None]
        assert iops == sorted(iops)

    def test_query_sort_descending(self, cache_db, sample_rows, sample_columns):
        """Sort descending by a numeric column."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, sort_by="Clinical.IOP", sort_desc=True, limit=100)

        iops = [r["Clinical.IOP"] for r in result.rows if r["Clinical.IOP"] is not None]
        assert iops == sorted(iops, reverse=True)

    def test_query_filter(self, cache_db, sample_rows, sample_columns):
        """Filter by substring match on a column."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, filter_col="Subject.Gender", filter_val="F", limit=100)

        assert result.count == 3  # Three F rows
        assert result.total_count == 3
        assert all(r["Subject.Gender"] == "F" for r in result.rows)

    def test_query_filter_and_sort(self, cache_db, sample_rows, sample_columns):
        """Filter and sort can be combined."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(
            key,
            filter_col="Subject.Gender",
            filter_val="F",
            sort_by="Clinical.IOP",
            limit=100,
        )

        assert result.count == 3
        iops = [r["Clinical.IOP"] for r in result.rows]
        assert iops == sorted(iops)

    def test_query_nonexistent_key(self, cache_db):
        """Querying a nonexistent key returns None."""
        result = cache_db.query("rc_nonexistent12345")
        assert result is None

    def test_query_expired_key(self, cache_db, sample_rows, sample_columns):
        """Querying an expired key returns None and cleans up."""
        key = _store_sample(cache_db, sample_rows, sample_columns, ttl=1)
        time.sleep(1.1)
        result = cache_db.query(key)
        assert result is None

    def test_query_preserves_original_column_names(self, cache_db, sample_rows, sample_columns):
        """Column names in results use original dot notation, not sanitized names."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, limit=1)

        assert "Image.RID" in result.rows[0]
        assert "Image__RID" not in result.rows[0]

    def test_query_result_metadata(self, cache_db, sample_rows, sample_columns):
        """CacheResult includes source and tool metadata."""
        key = _store_sample(cache_db, sample_rows, sample_columns, source="bag", tool="denormalize_dataset")
        result = cache_db.query(key, limit=1)

        assert result.source == "bag"
        assert result.tool_name == "denormalize_dataset"


# =============================================================================
# TestListCached
# =============================================================================


class TestListCached:
    """Tests for listing cached entries."""

    def test_list_empty(self, cache_db):
        """Empty cache returns empty list."""
        assert cache_db.list_cached() == []

    def test_list_returns_all_entries(self, cache_db, sample_rows, sample_columns):
        """Lists all non-expired entries."""
        key1 = cache_db.cache_key("denormalize", rid="A")
        key2 = cache_db.cache_key("query", table="B")

        _store_sample(cache_db, sample_rows, sample_columns, key=key1)
        _store_sample(cache_db, sample_rows[:2], sample_columns, key=key2, tool="query_table")

        entries = cache_db.list_cached()
        assert len(entries) == 2

    def test_list_excludes_expired(self, cache_db, sample_rows, sample_columns):
        """Expired entries are excluded and cleaned up."""
        key1 = cache_db.cache_key("denormalize", rid="A")
        key2 = cache_db.cache_key("query", table="B")

        _store_sample(cache_db, sample_rows, sample_columns, key=key1, ttl=1)
        _store_sample(cache_db, sample_rows, sample_columns, key=key2, ttl=None)  # permanent

        time.sleep(1.1)
        entries = cache_db.list_cached()
        assert len(entries) == 1
        assert entries[0].cache_key == key2

    def test_list_ordered_by_recency(self, cache_db, sample_rows, sample_columns):
        """Entries are ordered newest-first."""
        key1 = cache_db.cache_key("denormalize", rid="first")
        key2 = cache_db.cache_key("denormalize", rid="second")

        _store_sample(cache_db, sample_rows, sample_columns, key=key1)
        time.sleep(0.1)
        _store_sample(cache_db, sample_rows, sample_columns, key=key2)

        entries = cache_db.list_cached()
        assert entries[0].cache_key == key2  # Newer first

    def test_to_summary(self, cache_db, sample_rows, sample_columns):
        """CacheMeta.to_summary returns human-readable dict."""
        key = _store_sample(cache_db, sample_rows, sample_columns, source="bag")
        meta = cache_db.get_meta(key)
        summary = meta.to_summary()

        assert summary["cache_key"] == key
        assert summary["tool_name"] == "denormalize_dataset"
        assert summary["row_count"] == 5
        assert summary["source"] == "bag"
        assert summary["ttl"] == "permanent"
        assert summary["expired"] is False
        assert "ago" in summary["age"]


# =============================================================================
# TestInvalidate
# =============================================================================


class TestInvalidate:
    """Tests for cache invalidation."""

    def test_invalidate_by_key(self, cache_db, sample_rows, sample_columns):
        """Invalidate a specific entry by key."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        assert cache_db.has(key)

        count = cache_db.invalidate(cache_key=key)
        assert count == 1
        assert not cache_db.has(key)

    def test_invalidate_by_source(self, cache_db, sample_rows, sample_columns):
        """Invalidate all entries from a specific source."""
        key1 = cache_db.cache_key("denormalize", rid="bag1")
        key2 = cache_db.cache_key("query", table="cat1")
        key3 = cache_db.cache_key("denormalize", rid="bag2")

        _store_sample(cache_db, sample_rows, sample_columns, key=key1, source="bag")
        _store_sample(cache_db, sample_rows, sample_columns, key=key2, source="catalog")
        _store_sample(cache_db, sample_rows, sample_columns, key=key3, source="bag")

        count = cache_db.invalidate(source="catalog")
        assert count == 1
        assert cache_db.has(key1)  # bag entries survive
        assert not cache_db.has(key2)  # catalog entry removed
        assert cache_db.has(key3)  # bag entries survive

    def test_invalidate_all(self, cache_db, sample_rows, sample_columns):
        """Invalidate with no args clears everything."""
        key1 = cache_db.cache_key("denormalize", rid="A")
        key2 = cache_db.cache_key("query", table="B")

        _store_sample(cache_db, sample_rows, sample_columns, key=key1, source="bag")
        _store_sample(cache_db, sample_rows, sample_columns, key=key2, source="catalog")

        count = cache_db.invalidate()
        assert count == 2
        assert cache_db.list_cached() == []


# =============================================================================
# TestLifecycle
# =============================================================================


class TestLifecycle:
    """Tests for cache lifecycle management."""

    def test_persistence_across_reopen(self, tmp_path, sample_rows, sample_columns):
        """Bag-sourced data persists after close and reopen."""
        db_path = tmp_path / "persist.db"

        # Store data
        cache1 = ResultCache(db_path)
        key = _store_sample(cache1, sample_rows, sample_columns, source="bag", ttl=None)
        cache1.close()

        # Reopen and verify
        cache2 = ResultCache(db_path)
        assert cache2.has(key)
        result = cache2.query(key, limit=10)
        assert result.count == 5
        cache2.close()

    def test_db_path_property(self, cache_db, tmp_path):
        """db_path property returns the database path."""
        assert cache_db.db_path == tmp_path / "test_cache.db"

    def test_creates_parent_directories(self, tmp_path):
        """ResultCache creates parent directories if they don't exist."""
        db_path = tmp_path / "deep" / "nested" / "cache.db"
        cache = ResultCache(db_path)
        assert db_path.exists()
        cache.close()


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_special_characters_in_column_names(self, cache_db):
        """Columns with dots, dashes, spaces are handled."""
        cols = ["Table.Col-Name", "Table.Col Name", "Table.Col.Sub"]
        rows = [{"Table.Col-Name": "a", "Table.Col Name": "b", "Table.Col.Sub": "c"}]
        key = _store_sample(cache_db, rows, cols, key="rc_special_cols123")

        result = cache_db.query(key, limit=10)
        assert result.count == 1
        assert result.rows[0]["Table.Col-Name"] == "a"

    def test_large_batch_insert(self, cache_db):
        """Rows are inserted in batches (stress test for >1000 rows)."""
        cols = ["id", "value"]
        rows = [{"id": str(i), "value": float(i)} for i in range(2500)]
        key = _store_sample(cache_db, rows, cols, key="rc_largebatch12345")

        result = cache_db.query(key, limit=3000)
        assert result.count == 2500

    def test_mixed_types_in_column(self, cache_db):
        """Columns with mixed types (int and None) handled gracefully."""
        cols = ["name", "score"]
        rows = [
            {"name": "Alice", "score": 95},
            {"name": "Bob", "score": None},
            {"name": "Charlie", "score": 87},
        ]
        key = _store_sample(cache_db, rows, cols, key="rc_mixedtypes12345")

        result = cache_db.query(key, sort_by="score", limit=10)
        assert result.count == 3

    def test_filter_no_matches(self, cache_db, sample_rows, sample_columns):
        """Filter that matches nothing returns empty result."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, filter_col="Subject.Gender", filter_val="X", limit=100)

        assert result.count == 0
        assert result.total_count == 0

    def test_sort_by_nonexistent_column(self, cache_db, sample_rows, sample_columns):
        """Sort by nonexistent column falls back to default order."""
        key = _store_sample(cache_db, sample_rows, sample_columns)
        result = cache_db.query(key, sort_by="Nonexistent.Col", limit=100)
        # Should not error, falls back to _rowid_ order
        assert result.count == 5
