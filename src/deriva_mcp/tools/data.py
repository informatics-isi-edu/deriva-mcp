"""Data query and manipulation tools for DerivaML MCP server.

These tools enable querying and inserting records in catalog tables.
Use these to explore data, find specific records, and add new entries.

**Previewing Data**:
- preview_table(): Preview records from any table with optional filtering (max 100 rows)

**Inserting Data**:
- insert_records(): Add new records to a table

For asset tables, use the DerivaML Python API execution workflow for asset
staging and upload to properly track provenance.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-mcp")

# Tables managed by dedicated tools - insert_records should not be used for these
_MANAGED_TABLES = {
    # ML schema tables with dedicated management
    "Dataset": "Use create_dataset() to create datasets",
    "Dataset_Version": "Managed automatically by dataset versioning",
    "Execution": "Use create_execution() to create executions",
    "Workflow": "Use create_workflow() to create workflows",
    # Vocabulary tables
    "Dataset_Type": "Use add_term('Dataset_Type', ...) or create_dataset_type_term()",
    "Asset_Type": "Use add_asset_type() to add asset types",
    "Workflow_Type": "Use add_workflow_type() to add workflow types",
    "Feature_Name": "Use create_feature() which registers the feature name",
    # Association tables
    "Dataset_Dataset": "Use add_dataset_child() to nest datasets",
    "Dataset_Dataset_Type": "Use create_dataset(dataset_types=[...]) or add_dataset_type()",
}

# Patterns for dynamically named managed tables
_MANAGED_TABLE_PATTERNS = [
    ("_Dataset_", "Use add_dataset_members() to add records to datasets"),
    ("_Feature", "Use add_feature_value() to add feature values"),
]


def register_data_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register data query and manipulation tools with the MCP server."""

    @mcp.tool()
    async def preview_table(
        table_name: str,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> str:
        """Preview records from a table with optional column selection and filtering.

        Returns a sample of records for understanding data structure and content.
        For bulk data access, use the DerivaML Python API directly.

        Args:
            table_name: Name of the table to preview (e.g., "Image", "Subject", "Dataset").
            columns: List of column names to return. Default: all columns.
            filters: Dictionary of {column: value} equality filters.
            limit: Maximum records to return (default: 25, max: 100).
            offset: Number of records to skip.

        Returns:
            JSON with records array, count, and table name.

        Examples:
            preview_table("Image") -> first 25 images
            preview_table("Image", columns=["RID", "Filename"], limit=10)
            preview_table("Subject", filters={"Species": "Human"})
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
            limit = min(limit, 100)  # Hard cap at 100 rows
            entities = path.entities()

            # Fetch records with offset by fetching (offset + limit) and slicing
            # Note: Deriva's fetch() doesn't support offset directly
            fetch_limit = offset + limit if offset > 0 else limit
            all_records = list(entities.fetch(limit=fetch_limit))
            if offset > 0:
                all_records = all_records[offset:]

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
    async def insert_records(
        table_name: str,
        records: list[dict[str, Any]],
    ) -> str:
        """Insert new records into a domain table.

        **IMPORTANT**: This tool is for domain-specific tables only (e.g., Subject,
        Image metadata). Do NOT use for:
        - Datasets → use create_dataset(), add_dataset_members()
        - Features → use add_feature_value()
        - Vocabularies → use add_term()
        - Executions → use create_execution()
        - Workflows → use create_workflow()
        - Assets with files → use the DerivaML Python API execution workflow

        Args:
            table_name: Name of the domain table to insert into.
            records: List of dictionaries with column values.

        Returns:
            JSON with inserted_count and record RIDs.

        Example:
            insert_records("Subject", [{"Name": "Patient A", "Age": 45}])
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)

            # Check if this is a managed table
            if table_name in _MANAGED_TABLES:
                return json.dumps({
                    "status": "error",
                    "message": f"Cannot use insert_records for '{table_name}'. {_MANAGED_TABLES[table_name]}",
                    "table": table_name,
                })

            # Check for managed table patterns (dataset member tables, feature tables)
            for pattern, guidance in _MANAGED_TABLE_PATTERNS:
                if pattern in table_name:
                    return json.dumps({
                        "status": "error",
                        "message": f"Cannot use insert_records for '{table_name}'. {guidance}",
                        "table": table_name,
                    })

            # Check if it's a vocabulary table
            if ml.model.is_vocabulary(table):
                return json.dumps({
                    "status": "error",
                    "message": f"Cannot use insert_records for vocabulary table '{table_name}'. Use add_term('{table_name}', term_name, description) instead.",
                    "table": table_name,
                })

            # Check if it's an asset table (should use execution workflow)
            if ml.model.is_asset(table):
                return json.dumps({
                    "status": "error",
                    "message": f"Cannot use insert_records for asset table '{table_name}'. Use the DerivaML Python API execution workflow for asset staging and upload with proper provenance tracking.",
                    "table": table_name,
                })

            # Check if table is in ML schema (generally managed)
            if table.schema.name == ml.ml_schema:
                return json.dumps({
                    "status": "error",
                    "message": f"Cannot use insert_records for ML schema table '{table_name}'. Use the dedicated tool for this table type.",
                    "table": table_name,
                })

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
