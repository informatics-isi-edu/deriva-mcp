"""Data query and manipulation tools for DerivaML MCP server.

These tools enable querying and inserting records in catalog tables.
Use these to explore data, find specific records, and add new entries.

**Querying Data**:
- get_table(): Get all records from a table (simple, no filtering)
- query_table(): Fetch records from any table with optional filtering
- count_table(): Count records in a table

**Inserting Data**:
- insert_records(): Add new records to a table

**Exporting Data**:
- export_entity(): Export an entity by RID as CSV, JSON, or BDBag

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
    """Register data query and manipulation tools with the MCP server.

    These tools work with both DerivaML and plain ERMrest catalogs.
    """

    @mcp.tool()
    async def get_table(
        table_name: str,
        limit: int = 1000,
    ) -> str:
        """Get all records from a catalog table.

        Retrieves all contents of a table from the catalog. This is a simple
        way to get table data without filtering. For filtered queries, use
        query_table() instead.

        Works with both DerivaML and plain ERMrest catalogs.

        Args:
            table_name: Table name, either unqualified (e.g., "Image") or qualified
                with schema (e.g., "isa.dataset"). Use qualified names when a table
                exists in multiple schemas.
            limit: Maximum records to return (default: 1000).

        Returns:
            JSON with table name and records array.

        Examples:
            get_table("Image") -> all images in catalog
            get_table("isa.dataset", limit=100) -> first 100 datasets from isa schema
        """
        try:
            conn_info = conn_manager.get_active_or_raise()

            if conn_info.is_derivaml:
                # Use DerivaML's optimized method
                ml = conn_info.ml_instance
                rows = []
                for i, row in enumerate(ml.get_table_as_dict(table_name)):
                    if i >= limit:
                        break
                    rows.append(row)
            else:
                # Use pathbuilder for plain ERMrest
                table = conn_manager.find_table(table_name)
                pb = conn_info.get_pathbuilder()
                path = pb.schemas[table.schema.name].tables[table.name]
                rows = list(path.entities().fetch(limit=limit))

            return json.dumps(
                {
                    "table": table_name,
                    "records": rows,
                    "count": len(rows),
                    "limit": limit,
                }
            )
        except Exception as e:
            logger.error(f"Failed to get table: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def query_table(
        table_name: str,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> str:
        """Query records from a table with optional column selection and filtering.

        Fetches data from any table in the catalog. Use filters to narrow results.
        For large tables, use limit/offset for pagination.

        Works with both DerivaML and plain ERMrest catalogs.

        Args:
            table_name: Table name, either unqualified (e.g., "Image") or qualified
                with schema (e.g., "isa.dataset"). Use qualified names when a table
                exists in multiple schemas.
            columns: List of column names to return. Default: all columns.
            filters: Dictionary of {column: value} equality filters.
            limit: Maximum records to return (default: 100, max: 1000).
            offset: Number of records to skip for pagination.

        Returns:
            JSON with records array and total_count.

        Examples:
            query_table("Image") -> first 100 images
            query_table("isa.dataset", columns=["RID", "title"], limit=10)
            query_table("Subject", filters={"Species": "Human"})
        """
        try:
            conn_info = conn_manager.get_active_or_raise()
            table = conn_manager.find_table(table_name)
            pb = conn_info.get_pathbuilder()
            path = pb.schemas[table.schema.name].tables[table.name]

            # Apply filters
            if filters:
                for col, val in filters.items():
                    path = path.filter(getattr(path, col) == val)

            # Fetch with limit
            limit = min(limit, 1000)  # Cap at 1000
            entities = path.entities()

            # Fetch records with offset by fetching (offset + limit) and slicing
            # Note: Deriva's fetch() doesn't support offset directly
            fetch_limit = offset + limit if offset > 0 else limit
            all_records = list(entities.fetch(limit=fetch_limit))
            if offset > 0:
                all_records = all_records[offset:]

            # Select columns if specified
            if columns:
                all_records = [{k: v for k, v in rec.items() if k in columns} for rec in all_records]

            return json.dumps(
                {
                    "table": table_name,
                    "records": all_records,
                    "count": len(all_records),
                    "limit": limit,
                    "offset": offset,
                }
            )
        except Exception as e:
            logger.error(f"Failed to query table: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def count_table(
        table_name: str,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Count records in a table, optionally with filters.

        Works with both DerivaML and plain ERMrest catalogs.

        Args:
            table_name: Table name, either unqualified (e.g., "Image") or qualified
                with schema (e.g., "isa.dataset").
            filters: Dictionary of {column: value} equality filters.

        Returns:
            JSON with table name and count.

        Examples:
            count_table("Image") -> total image count
            count_table("isa.dataset", filters={"released": true}) -> count of released datasets
        """
        try:
            conn_info = conn_manager.get_active_or_raise()
            table = conn_manager.find_table(table_name)
            pb = conn_info.get_pathbuilder()
            path = pb.schemas[table.schema.name].tables[table.name]

            # Apply filters
            if filters:
                for col, val in filters.items():
                    path = path.filter(getattr(path, col) == val)

            # Count using aggregates
            count = len(list(path.entities().fetch()))

            return json.dumps(
                {
                    "table": table_name,
                    "count": count,
                    "filters": filters,
                }
            )
        except Exception as e:
            logger.error(f"Failed to count table: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def insert_records(
        table_name: str,
        records: list[dict[str, Any]],
    ) -> str:
        """Insert new records into a domain table.

        Works with both DerivaML and plain ERMrest catalogs.

        **For DerivaML catalogs**: Do NOT use for managed tables:
        - Datasets → use create_dataset(), add_dataset_members()
        - Features → use add_feature_value()
        - Vocabularies → use add_term()
        - Executions → use create_execution()
        - Workflows → use create_workflow()
        - Assets with files → use asset_file_path() + upload_execution_outputs()

        **For plain ERMrest catalogs**: Can insert into any table.

        Args:
            table_name: Table name, either unqualified (e.g., "Subject") or qualified
                with schema (e.g., "isa.subject").
            records: List of dictionaries with column values.

        Returns:
            JSON with inserted_count and record RIDs.

        Example:
            insert_records("Subject", [{"Name": "Patient A", "Age": 45}])
        """
        try:
            conn_info = conn_manager.get_active_or_raise()
            table = conn_manager.find_table(table_name)

            # For DerivaML catalogs, apply restrictions on managed tables
            if conn_info.is_derivaml:
                ml = conn_info.ml_instance

                # Check if this is a managed table
                if table_name in _MANAGED_TABLES:
                    return json.dumps(
                        {
                            "status": "error",
                            "message": f"Cannot use insert_records for '{table_name}'. {_MANAGED_TABLES[table_name]}",
                            "table": table_name,
                        }
                    )

                # Check for managed table patterns (dataset member tables, feature tables)
                for pattern, guidance in _MANAGED_TABLE_PATTERNS:
                    if pattern in table_name:
                        return json.dumps(
                            {
                                "status": "error",
                                "message": f"Cannot use insert_records for '{table_name}'. {guidance}",
                                "table": table_name,
                            }
                        )

                # Check if it's a vocabulary table
                if ml.model.is_vocabulary(table):
                    return json.dumps(
                        {
                            "status": "error",
                            "message": f"Cannot use insert_records for vocabulary table '{table_name}'. Use add_term('{table_name}', term_name, description) instead.",
                            "table": table_name,
                        }
                    )

                # Check if it's an asset table (should use execution workflow)
                if ml.model.is_asset(table):
                    return json.dumps(
                        {
                            "status": "error",
                            "message": f"Cannot use insert_records for asset table '{table_name}'. Use asset_file_path() + upload_execution_outputs() for proper provenance tracking.",
                            "table": table_name,
                        }
                    )

                # Check if table is in ML schema (generally managed)
                if table.schema.name == ml.ml_schema:
                    return json.dumps(
                        {
                            "status": "error",
                            "message": f"Cannot use insert_records for ML schema table '{table_name}'. Use the dedicated tool for this table type.",
                            "table": table_name,
                        }
                    )

            pb = conn_info.get_pathbuilder()
            path = pb.schemas[table.schema.name].tables[table.name]

            # Insert records
            result = path.insert(records)
            inserted = list(result)

            return json.dumps(
                {
                    "status": "inserted",
                    "table": table_name,
                    "inserted_count": len(inserted),
                    "rids": [r.get("RID") for r in inserted],
                }
            )
        except Exception as e:
            logger.error(f"Failed to insert records: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_record(
        table_name: str,
        rid: str,
    ) -> str:
        """Get a single record by its RID.

        Works with both DerivaML and plain ERMrest catalogs.

        Args:
            table_name: Table name, either unqualified (e.g., "Image") or qualified
                with schema (e.g., "isa.dataset").
            rid: The RID of the record to fetch.

        Returns:
            JSON with the complete record or error if not found.

        Example:
            get_record("Image", "1-ABC") -> full image record
        """
        try:
            conn_info = conn_manager.get_active_or_raise()
            table = conn_manager.find_table(table_name)
            pb = conn_info.get_pathbuilder()
            path = pb.schemas[table.schema.name].tables[table.name]

            # Filter by RID
            records = list(path.filter(path.RID == rid).entities().fetch())

            if not records:
                return json.dumps(
                    {
                        "status": "not_found",
                        "message": f"No record with RID {rid} in table {table_name}",
                    }
                )

            return json.dumps(
                {
                    "table": table_name,
                    "rid": rid,
                    "record": records[0],
                }
            )
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

        Works with both DerivaML and plain ERMrest catalogs.

        Args:
            table_name: Table name, either unqualified (e.g., "Subject") or qualified
                with schema (e.g., "isa.subject").
            rid: The RID of the record to update.
            updates: Dictionary of {column: new_value} updates.

        Returns:
            JSON with update status.

        Example:
            update_record("Subject", "1-ABC", {"Age": 46, "Status": "Active"})
        """
        try:
            conn_info = conn_manager.get_active_or_raise()
            table = conn_manager.find_table(table_name)
            pb = conn_info.get_pathbuilder()
            path = pb.schemas[table.schema.name].tables[table.name]

            # Get current record
            records = list(path.filter(path.RID == rid).entities().fetch())
            if not records:
                return json.dumps(
                    {
                        "status": "not_found",
                        "message": f"No record with RID {rid} in table {table_name}",
                    }
                )

            # Apply updates
            record = records[0]
            record.update(updates)

            # Update in database
            path.update([record])

            return json.dumps(
                {
                    "status": "updated",
                    "table": table_name,
                    "rid": rid,
                    "updated_fields": list(updates.keys()),
                }
            )
        except Exception as e:
            logger.error(f"Failed to update record: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def export_entity(
        rid: str,
        output_dir: str | None = None,
        export_format: str = "bag",
        template_name: str | None = None,
        export_spec: dict[str, Any] | None = None,
        include_schema: bool = True,
    ) -> str:
        """Export an entity by RID, similar to the Chaise export button.

        Exports an entity from the connected catalog using either:
        1. The export annotation defined on the table (default)
        2. A specific template from the annotation (via template_name)
        3. A custom export specification (via export_spec)

        Works with both DerivaML and plain ERMrest catalogs.

        Args:
            rid: The RID of the entity to export.
            output_dir: Directory for output files. Defaults to current directory.
            export_format: Export format - "bag", "csv", or "json". Default is "bag".
                - "bag": Creates a BDBag archive with metadata and fetch references
                - "csv": Creates CSV file(s) with the data
                - "json": Creates JSON file(s) with the data
            template_name: Optional name of specific export template to use from
                the table's export annotation.
            export_spec: Optional custom export specification. If provided, overrides
                the table's export annotation. Can be in either:
                - Export annotation format (with "templates" array)
                - DerivaDownload config format (with "catalog"/"query_processors")
                Use {RID} placeholder in query_path for RID substitution.
            include_schema: If True (default), include the catalog schema in bag
                exports. This is useful for understanding the data model.

        Returns:
            JSON with export results:
            - status: "success" or "error"
            - path: Path to the exported file or bag
            - format: The export format used
            - rid: The exported RID
            - table: The table name
            - schema: The schema name

        Examples:
            # Export using table's default export annotation
            export_entity("3-KFBY")

            # Export as CSV
            export_entity("3-KFBY", export_format="csv")

            # Export using specific template
            export_entity("3-KFBY", template_name="BDBag")

            # Export with custom spec
            export_entity("3-KFBY", export_spec={
                "catalog": {
                    "query_processors": [{
                        "processor": "csv",
                        "processor_params": {
                            "query_path": "/entity/isa:dataset/RID={RID}",
                            "output_path": "dataset.csv"
                        }
                    }]
                }
            })
        """
        try:
            from deriva.core.export import export_entity as _export_entity

            conn_info = conn_manager.get_active_or_raise()
            hostname = conn_info.hostname
            catalog_id = conn_info.catalog_id

            # Get credentials from connection
            credentials = conn_info.credentials

            result = _export_entity(
                hostname=hostname,
                catalog_id=catalog_id,
                rid=rid,
                output_dir=output_dir,
                export_format=export_format,
                template_name=template_name,
                export_spec=export_spec,
                credentials=credentials,
                include_schema=include_schema,
            )

            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to export entity: {e}")
            return json.dumps({"status": "error", "message": str(e)})
