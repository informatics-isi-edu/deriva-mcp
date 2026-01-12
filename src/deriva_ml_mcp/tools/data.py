"""Data query and manipulation tools for DerivaML MCP server.

These tools enable querying and inserting records in catalog tables.
Use these to explore data, find specific records, and add new entries.

**Querying Data**:
- query_table(): Fetch records from any table with optional filtering
- count_table(): Count records in a table

**Inserting Data**:
- insert_records(): Add new records to a table

For asset tables, use the execution workflow (asset_file_path + upload_execution_outputs)
to properly track provenance.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-ml-mcp")


def register_data_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register data query and manipulation tools with the MCP server."""

    @mcp.tool()
    async def query_table(
        table_name: str,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> str:
        """Query records from a table with optional column selection and filtering.

        Fetches data from any table in the domain or ML schema. Use filters
        to narrow results. For large tables, use limit/offset for pagination.

        Args:
            table_name: Name of the table to query (e.g., "Image", "Subject", "Dataset").
            columns: List of column names to return. Default: all columns.
            filters: Dictionary of {column: value} equality filters.
            limit: Maximum records to return (default: 100, max: 1000).
            offset: Number of records to skip for pagination.

        Returns:
            JSON with records array and total_count.

        Examples:
            query_table("Image") -> first 100 images
            query_table("Image", columns=["RID", "Filename"], limit=10)
            query_table("Subject", filters={"Species": "Human"})
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            pb = ml.pathBuilder()
            path = pb.schemas[table.schema.name].tables[table_name]

            # Apply filters
            if filters:
                for col, val in filters.items():
                    path = path.filter(getattr(path, col) == val)

            # Fetch with limit
            limit = min(limit, 1000)  # Cap at 1000
            entities = path.entities()

            # Get total count before limiting
            # Note: This is approximate for filtered queries
            all_records = list(entities.fetch(limit=limit, offset=offset))

            # Select columns if specified
            if columns:
                all_records = [
                    {k: v for k, v in rec.items() if k in columns}
                    for rec in all_records
                ]

            return json.dumps({
                "table": table_name,
                "records": all_records,
                "count": len(all_records),
                "limit": limit,
                "offset": offset,
            })
        except Exception as e:
            logger.error(f"Failed to query table: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def count_table(
        table_name: str,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Count records in a table, optionally with filters.

        Args:
            table_name: Name of the table to count.
            filters: Dictionary of {column: value} equality filters.

        Returns:
            JSON with table name and count.

        Examples:
            count_table("Image") -> total image count
            count_table("Subject", filters={"Species": "Human"}) -> count of human subjects
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            pb = ml.pathBuilder()
            path = pb.schemas[table.schema.name].tables[table_name]

            # Apply filters
            if filters:
                for col, val in filters.items():
                    path = path.filter(getattr(path, col) == val)

            # Count using aggregates
            count = len(list(path.entities().fetch()))

            return json.dumps({
                "table": table_name,
                "count": count,
                "filters": filters,
            })
        except Exception as e:
            logger.error(f"Failed to count table: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def insert_records(
        table_name: str,
        records: list[dict[str, Any]],
    ) -> str:
        """Insert new records into a table.

        For asset tables (with file uploads), use the execution workflow instead:
        asset_file_path() + upload_execution_outputs() for proper provenance.

        Args:
            table_name: Name of the table to insert into.
            records: List of dictionaries with column values.

        Returns:
            JSON with inserted_count and record RIDs.

        Example:
            insert_records("Subject", [{"Name": "Patient A", "Age": 45}])
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            pb = ml.pathBuilder()
            path = pb.schemas[table.schema.name].tables[table_name]

            # Insert records
            result = path.insert(records)
            inserted = list(result)

            return json.dumps({
                "status": "inserted",
                "table": table_name,
                "inserted_count": len(inserted),
                "rids": [r.get("RID") for r in inserted],
            })
        except Exception as e:
            logger.error(f"Failed to insert records: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_record(
        table_name: str,
        rid: str,
    ) -> str:
        """Get a single record by its RID.

        Args:
            table_name: Name of the table containing the record.
            rid: The RID of the record to fetch.

        Returns:
            JSON with the complete record or error if not found.

        Example:
            get_record("Image", "1-ABC") -> full image record
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            pb = ml.pathBuilder()
            path = pb.schemas[table.schema.name].tables[table_name]

            # Filter by RID
            records = list(path.filter(path.RID == rid).entities().fetch())

            if not records:
                return json.dumps({
                    "status": "not_found",
                    "message": f"No record with RID {rid} in table {table_name}",
                })

            return json.dumps({
                "table": table_name,
                "rid": rid,
                "record": records[0],
            })
        except Exception as e:
            logger.error(f"Failed to get record: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def update_record(
        table_name: str,
        rid: str,
        updates: dict[str, Any],
    ) -> str:
        """Update fields in an existing record.

        Args:
            table_name: Name of the table containing the record.
            rid: The RID of the record to update.
            updates: Dictionary of {column: new_value} updates.

        Returns:
            JSON with update status.

        Example:
            update_record("Subject", "1-ABC", {"Age": 46, "Status": "Active"})
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            pb = ml.pathBuilder()
            path = pb.schemas[table.schema.name].tables[table_name]

            # Get current record
            records = list(path.filter(path.RID == rid).entities().fetch())
            if not records:
                return json.dumps({
                    "status": "not_found",
                    "message": f"No record with RID {rid} in table {table_name}",
                })

            # Apply updates
            record = records[0]
            record.update(updates)

            # Update in database
            path.update([record])

            return json.dumps({
                "status": "updated",
                "table": table_name,
                "rid": rid,
                "updated_fields": list(updates.keys()),
            })
        except Exception as e:
            logger.error(f"Failed to update record: {e}")
            return json.dumps({"status": "error", "message": str(e)})
