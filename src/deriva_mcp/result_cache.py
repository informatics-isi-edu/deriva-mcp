"""Per-connection SQLite result cache for tabular query results.

Caches results from denormalize_dataset, query_table, fetch_table_features, etc.
in a queryable SQLite database so repeated access is instant and results can be
re-queried with different sort/filter/pagination without re-executing the original query.

Bag-sourced results never expire (immutable snapshots). Catalog-sourced results
have bounded TTL with auto-invalidation on data mutations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Columns to skip when inferring types (system columns are rarely useful for analysis)
_SYSTEM_COLUMNS = {"RCT", "RMT", "RCB", "RMB"}


@dataclass
class CacheMeta:
    """Metadata for a cached result entry."""

    cache_key: str
    tool_name: str
    params: dict
    columns: list[str]
    source: str  # "bag" or "catalog"
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int | None = None  # None = never expire
    row_count: int = 0

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    def age_seconds(self) -> float:
        """How old this entry is in seconds."""
        return time.time() - self.created_at

    def to_summary(self) -> dict[str, Any]:
        """Return a human-readable summary for listing."""
        age = self.age_seconds()
        if age < 60:
            age_str = f"{age:.0f}s ago"
        elif age < 3600:
            age_str = f"{age / 60:.1f}m ago"
        else:
            age_str = f"{age / 3600:.1f}h ago"

        return {
            "cache_key": self.cache_key,
            "tool_name": self.tool_name,
            "params": self.params,
            "row_count": self.row_count,
            "columns": self.columns,
            "source": self.source,
            "age": age_str,
            "ttl": f"{self.ttl_seconds}s" if self.ttl_seconds else "permanent",
            "expired": self.is_expired(),
        }


@dataclass
class CacheResult:
    """Result from querying a cached table."""

    columns: list[str]
    rows: list[dict[str, Any]]
    count: int  # rows returned (after limit/offset)
    total_count: int  # total rows in cached table
    cache_key: str
    source: str
    tool_name: str
    params: dict


def _infer_sqlite_type(value: Any) -> str:
    """Infer SQLite column type from a Python value."""
    if value is None:
        return "TEXT"
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _sanitize_col_name(col: str, index: int) -> str:
    """Sanitize a column name for use as a SQLite column identifier.

    Uses an index suffix to guarantee uniqueness even when different original
    names would collide after character replacement (e.g., "Col-Name" and
    "Col Name" both become "Col_Name" without the suffix).
    """
    return f"c{index}"


class ResultCache:
    """Per-connection SQLite store for caching tabular query results.

    Each cached result is stored as a separate table in a SQLite database.
    A metadata table (_cache_meta) tracks what each cached table represents,
    when it was created, its TTL, and its source.

    Usage::

        cache = ResultCache(Path("~/.deriva-ml/result_cache/my_cache.db"))

        key = cache.cache_key("denormalize", dataset_rid="4-411G",
                              tables=["Image", "Subject"], version="2.10.0")

        if not cache.has(key):
            rows = expensive_computation()
            columns = list(rows[0].keys())
            cache.store(key, columns, rows, CacheMeta(
                cache_key=key,
                tool_name="denormalize_dataset",
                params={"dataset_rid": "4-411G", ...},
                columns=columns,
                source="bag",
            ))

        result = cache.query(key, sort_by="Image.CDR", limit=20)
    """

    _BATCH_SIZE = 1000
    _META_TABLE = "_cache_meta"

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._ensure_meta_table()
        # Map from cache_key → {original_col: sanitized_col}
        self._col_maps: dict[str, dict[str, str]] = {}

    def _ensure_meta_table(self) -> None:
        """Create the metadata table if it doesn't exist."""
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._META_TABLE} (
                cache_key TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                params_json TEXT NOT NULL,
                columns_json TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl_seconds INTEGER,
                row_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Cache key generation
    # ------------------------------------------------------------------

    @staticmethod
    def cache_key(tool_name: str, **params: Any) -> str:
        """Generate a deterministic cache key from tool name and parameters.

        Returns a string like ``rc_a1b2c3d4e5f67890`` (16 hex chars).
        """
        # Sort params for determinism; convert lists to sorted tuples
        normalized: dict[str, Any] = {}
        for k, v in sorted(params.items()):
            if isinstance(v, list):
                normalized[k] = sorted(v) if all(isinstance(i, str) for i in v) else v
            elif v is not None:
                normalized[k] = v

        key_str = f"{tool_name}:{json.dumps(normalized, sort_keys=True)}"
        digest = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return f"rc_{digest}"

    # ------------------------------------------------------------------
    # Cache status
    # ------------------------------------------------------------------

    def has(self, cache_key: str) -> bool:
        """Check if a non-expired cache entry exists for this key."""
        meta = self.get_meta(cache_key)
        if meta is None:
            return False
        if meta.is_expired():
            self.invalidate(cache_key=cache_key)
            return False
        return True

    def get_meta(self, cache_key: str) -> CacheMeta | None:
        """Get metadata for a cached entry, or None if not found."""
        row = self._conn.execute(
            f"SELECT * FROM {self._META_TABLE} WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return CacheMeta(
            cache_key=row["cache_key"],
            tool_name=row["tool_name"],
            params=json.loads(row["params_json"]),
            columns=json.loads(row["columns_json"]),
            source=row["source"],
            created_at=row["created_at"],
            ttl_seconds=row["ttl_seconds"],
            row_count=row["row_count"],
        )

    # ------------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------------

    def store(
        self,
        cache_key: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        meta: CacheMeta,
    ) -> None:
        """Store a tabular result as a new SQLite table.

        If a table with this cache_key already exists, it is replaced.
        """
        table_name = cache_key  # rc_... is already safe for SQLite

        # Build column name mapping (original → sanitized)
        col_map = {col: _sanitize_col_name(col, i) for i, col in enumerate(columns)}
        self._col_maps[cache_key] = col_map

        # Infer column types from first row
        type_map: dict[str, str] = {}
        if rows:
            for col in columns:
                type_map[col] = _infer_sqlite_type(rows[0].get(col))
        else:
            type_map = {col: "TEXT" for col in columns}

        # Drop existing table if any
        self._conn.execute(f"DROP TABLE IF EXISTS [{table_name}]")

        # Create table
        col_defs = ", ".join(
            f"[{col_map[col]}] {type_map[col]}" for col in columns
        )
        # Add a rowid alias for ordering
        create_sql = f"CREATE TABLE [{table_name}] (_rowid_ INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})"
        self._conn.execute(create_sql)

        # Insert rows in batches
        if rows:
            sanitized_cols = [col_map[c] for c in columns]
            placeholders = ", ".join("?" for _ in columns)
            col_names = ", ".join(f"[{sc}]" for sc in sanitized_cols)
            insert_sql = f"INSERT INTO [{table_name}] ({col_names}) VALUES ({placeholders})"

            for i in range(0, len(rows), self._BATCH_SIZE):
                batch = rows[i : i + self._BATCH_SIZE]
                values = [
                    tuple(
                        str(row.get(col)) if row.get(col) is not None and not isinstance(row.get(col), (int, float, bool)) else row.get(col)
                        for col in columns
                    )
                    for row in batch
                ]
                self._conn.executemany(insert_sql, values)

        # Store metadata
        meta.row_count = len(rows)
        meta.columns = columns
        meta.cache_key = cache_key
        self._conn.execute(
            f"""INSERT OR REPLACE INTO {self._META_TABLE}
            (cache_key, tool_name, params_json, columns_json, source, created_at, ttl_seconds, row_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cache_key,
                meta.tool_name,
                json.dumps(meta.params),
                json.dumps(meta.columns),
                meta.source,
                meta.created_at,
                meta.ttl_seconds,
                meta.row_count,
            ),
        )
        self._conn.commit()
        logger.info(
            f"Cached {len(rows)} rows for {meta.tool_name} as {cache_key} "
            f"(source={meta.source}, ttl={meta.ttl_seconds})"
        )

    # ------------------------------------------------------------------
    # Query cached results
    # ------------------------------------------------------------------

    def query(
        self,
        cache_key: str,
        sort_by: str | None = None,
        sort_desc: bool = False,
        filter_col: str | None = None,
        filter_val: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> CacheResult | None:
        """Query a cached result with optional sort, filter, and pagination.

        Args:
            cache_key: The cache key to query.
            sort_by: Column name to sort by (original name, e.g. "Image.CDR").
            sort_desc: Sort descending if True.
            filter_col: Column name to filter on.
            filter_val: Value to filter for (substring match, case-insensitive).
            limit: Max rows to return.
            offset: Number of rows to skip.

        Returns:
            CacheResult with the query results, or None if not cached.
        """
        meta = self.get_meta(cache_key)
        if meta is None:
            return None
        if meta.is_expired():
            self.invalidate(cache_key=cache_key)
            return None

        table_name = cache_key
        col_map = self._col_maps.get(cache_key)
        if col_map is None:
            # Rebuild col_map from stored columns
            col_map = {col: _sanitize_col_name(col, i) for i, col in enumerate(meta.columns)}
            self._col_maps[cache_key] = col_map

        # Reverse map: sanitized → original
        reverse_map = {v: k for k, v in col_map.items()}

        # Build query
        sanitized_cols = [col_map[c] for c in meta.columns]
        select_cols = ", ".join(f"[{sc}]" for sc in sanitized_cols)
        sql = f"SELECT {select_cols} FROM [{table_name}]"
        params: list[Any] = []

        # Filter
        if filter_col and filter_val is not None:
            san_filter = col_map.get(filter_col)
            if san_filter:
                sql += f" WHERE [{san_filter}] LIKE ?"
                params.append(f"%{filter_val}%")

        # Count total matching rows
        count_sql = f"SELECT COUNT(*) FROM [{table_name}]"
        if filter_col and filter_val is not None:
            san_filter = col_map.get(filter_col)
            if san_filter:
                count_sql += f" WHERE [{san_filter}] LIKE ?"
                total_count = self._conn.execute(count_sql, [f"%{filter_val}%"]).fetchone()[0]
            else:
                total_count = meta.row_count
        else:
            total_count = meta.row_count

        # Sort
        if sort_by:
            san_sort = col_map.get(sort_by)
            if san_sort:
                direction = "DESC" if sort_desc else "ASC"
                sql += f" ORDER BY [{san_sort}] {direction}"
        else:
            sql += " ORDER BY _rowid_"

        # Pagination
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        # Execute
        cursor = self._conn.execute(sql, params)
        rows = []
        for row in cursor:
            # Map sanitized column names back to original names
            row_dict = {}
            for san_col, value in zip(sanitized_cols, row):
                original_col = reverse_map.get(san_col, san_col)
                row_dict[original_col] = value
            rows.append(row_dict)

        return CacheResult(
            columns=meta.columns,
            rows=rows,
            count=len(rows),
            total_count=total_count,
            cache_key=cache_key,
            source=meta.source,
            tool_name=meta.tool_name,
            params=meta.params,
        )

    # ------------------------------------------------------------------
    # Listing and invalidation
    # ------------------------------------------------------------------

    def list_cached(self) -> list[CacheMeta]:
        """List all non-expired cache entries."""
        rows = self._conn.execute(
            f"SELECT * FROM {self._META_TABLE} ORDER BY created_at DESC"
        ).fetchall()

        result = []
        expired_keys = []
        for row in rows:
            meta = CacheMeta(
                cache_key=row["cache_key"],
                tool_name=row["tool_name"],
                params=json.loads(row["params_json"]),
                columns=json.loads(row["columns_json"]),
                source=row["source"],
                created_at=row["created_at"],
                ttl_seconds=row["ttl_seconds"],
                row_count=row["row_count"],
            )
            if meta.is_expired():
                expired_keys.append(meta.cache_key)
            else:
                result.append(meta)

        # Lazy cleanup of expired entries
        for key in expired_keys:
            self._drop_cached_table(key)

        return result

    def invalidate(
        self,
        cache_key: str | None = None,
        source: str | None = None,
    ) -> int:
        """Invalidate cache entries.

        Args:
            cache_key: Invalidate a specific entry.
            source: Invalidate all entries from this source ("bag" or "catalog").
            If both are None, invalidate everything.

        Returns:
            Number of entries invalidated.
        """
        if cache_key:
            # Single entry
            self._drop_cached_table(cache_key)
            return 1
        elif source:
            # All entries from a source
            rows = self._conn.execute(
                f"SELECT cache_key FROM {self._META_TABLE} WHERE source = ?",
                (source,),
            ).fetchall()
            for row in rows:
                self._drop_cached_table(row["cache_key"])
            return len(rows)
        else:
            # Everything
            rows = self._conn.execute(
                f"SELECT cache_key FROM {self._META_TABLE}"
            ).fetchall()
            for row in rows:
                self._drop_cached_table(row["cache_key"])
            return len(rows)

    def _drop_cached_table(self, cache_key: str) -> None:
        """Drop a cached table and its metadata entry."""
        self._conn.execute(f"DROP TABLE IF EXISTS [{cache_key}]")
        self._conn.execute(
            f"DELETE FROM {self._META_TABLE} WHERE cache_key = ?",
            (cache_key,),
        )
        self._col_maps.pop(cache_key, None)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return self._db_path
