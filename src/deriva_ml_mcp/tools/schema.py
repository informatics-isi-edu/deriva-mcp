"""Schema manipulation tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-ml-mcp")


def register_schema_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register schema manipulation tools with the MCP server."""

    @mcp.tool()
    async def create_table(
        table_name: str,
        columns: list[dict] | None = None,
        comment: str = "",
    ) -> str:
        """Create a new table in the domain schema.

        Args:
            table_name: Name for the new table.
            columns: Column definitions, each with:
                - name: Column name (required)
                - type: One of "text", "int4", "int8", "float4", "float8", "boolean", "date", "timestamptz", "jsonb"
                - nullok: Allow nulls (default: True)
                - comment: Column description
            comment: Description of the table's purpose.

        Returns:
            JSON with status, table_name, schema, columns.

        Example:
            create_table("Subject", [{"name": "Name", "type": "text"}, {"name": "Age", "type": "int4"}])
        """
        try:
            from deriva_ml import TableDefinition, ColumnDefinition, BuiltinTypes

            ml = conn_manager.get_active_or_raise()

            col_defs = []
            if columns:
                type_map = {
                    "text": BuiltinTypes.text,
                    "int4": BuiltinTypes.int4,
                    "int8": BuiltinTypes.int8,
                    "float4": BuiltinTypes.float4,
                    "float8": BuiltinTypes.float8,
                    "boolean": BuiltinTypes.boolean,
                    "date": BuiltinTypes.date,
                    "timestamptz": BuiltinTypes.timestamptz,
                    "jsonb": BuiltinTypes.jsonb,
                }
                for col in columns:
                    col_type = type_map.get(col.get("type", "text"), BuiltinTypes.text)
                    col_defs.append(ColumnDefinition(
                        name=col["name"],
                        type=col_type,
                        nullok=col.get("nullok", True),
                        comment=col.get("comment", ""),
                    ))

            table_def = TableDefinition(
                table_name=table_name,
                column_defs=col_defs,
                comment=comment,
            )
            table = ml.create_table(table_def)
            return json.dumps({
                "status": "created",
                "table_name": table.name,
                "schema": table.schema.name,
                "columns": [c.name for c in table.columns],
            })
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def create_asset_table(
        asset_name: str,
        columns: list[dict] | None = None,
        referenced_tables: list[str] | None = None,
        comment: str = "",
    ) -> str:
        """Create a new asset table for file management with automatic URL/checksum tracking.

        Asset tables automatically include: URL, Filename, Length, MD5, Description.
        They integrate with executions for provenance tracking.

        Args:
            asset_name: Name for the asset table (e.g., "Image", "Model", "Checkpoint").
            columns: Additional columns beyond standard asset columns.
            referenced_tables: Tables this asset should have foreign keys to.
            comment: Description of the asset table's purpose.

        Returns:
            JSON with status, table_name, schema, columns.

        Example:
            create_asset_table("Image", [{"name": "Width", "type": "int4"}], ["Subject"])
        """
        try:
            from deriva_ml import ColumnDefinition, BuiltinTypes

            ml = conn_manager.get_active_or_raise()

            col_defs = []
            if columns:
                type_map = {
                    "text": BuiltinTypes.text,
                    "int4": BuiltinTypes.int4,
                    "int8": BuiltinTypes.int8,
                    "float4": BuiltinTypes.float4,
                    "float8": BuiltinTypes.float8,
                    "boolean": BuiltinTypes.boolean,
                    "date": BuiltinTypes.date,
                    "timestamptz": BuiltinTypes.timestamptz,
                }
                for col in columns:
                    col_type = type_map.get(col.get("type", "text"), BuiltinTypes.text)
                    col_defs.append(ColumnDefinition(
                        name=col["name"],
                        type=col_type,
                        nullok=col.get("nullok", True),
                        comment=col.get("comment", ""),
                    ))

            ref_tables = []
            if referenced_tables:
                for table_name in referenced_tables:
                    ref_tables.append(ml.model.name_to_table(table_name))

            table = ml.create_asset(
                asset_name=asset_name,
                column_defs=col_defs if col_defs else None,
                referenced_tables=ref_tables if ref_tables else None,
                comment=comment,
            )
            return json.dumps({
                "status": "created",
                "table_name": table.name,
                "schema": table.schema.name,
                "columns": [c.name for c in table.columns],
            })
        except Exception as e:
            logger.error(f"Failed to create asset table: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_assets(asset_table: str) -> str:
        """List all assets in an asset table with their metadata.

        Args:
            asset_table: Name of the asset table (e.g., "Image", "Model").

        Returns:
            JSON array of {RID, Filename, URL, Length, MD5, Asset_Type}.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            assets = ml.list_assets(asset_table)
            result = []
            for asset in assets:
                result.append({
                    "RID": asset.get("RID"),
                    "Filename": asset.get("Filename"),
                    "URL": asset.get("URL"),
                    "Length": asset.get("Length"),
                    "MD5": asset.get("MD5"),
                    "Asset_Type": asset.get("Asset_Type", []),
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list assets: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_tables() -> str:
        """List all tables in the domain schema with their properties.

        Returns:
            JSON array of {name, comment, is_vocabulary, is_asset, column_count}.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            tables = []
            for table in ml.model.schemas[ml.domain_schema].tables.values():
                tables.append({
                    "name": table.name,
                    "comment": table.comment or "",
                    "is_vocabulary": ml.model.is_vocabulary(table),
                    "is_asset": ml.model.is_asset(table),
                    "column_count": len(list(table.columns)),
                })
            return json.dumps(tables)
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_table_schema(table_name: str) -> str:
        """Get the full schema of a table including all columns and their types.

        Args:
            table_name: Name of the table to describe.

        Returns:
            JSON with name, schema, comment, columns (with type info), is_vocabulary, is_asset.

        Example:
            get_table_schema("Image") -> shows all Image table columns and types
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            columns = []
            for col in table.columns:
                columns.append({
                    "name": col.name,
                    "type": str(col.type.typename),
                    "nullok": col.nullok,
                    "comment": col.comment or "",
                })
            return json.dumps({
                "name": table.name,
                "schema": table.schema.name,
                "comment": table.comment or "",
                "columns": columns,
                "is_vocabulary": ml.model.is_vocabulary(table),
                "is_asset": ml.model.is_asset(table),
            })
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_asset_types() -> str:
        """List available asset types from the Asset_Type vocabulary.

        Returns:
            JSON array of {name, description} for each asset type.
        """
        try:
            from deriva_ml import MLVocab

            ml = conn_manager.get_active_or_raise()
            terms = ml.list_vocabulary_terms(MLVocab.asset_type)
            result = []
            for term in terms:
                result.append({
                    "name": term.name,
                    "description": term.description,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list asset types: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_asset_type(type_name: str, description: str) -> str:
        """Add a new asset type to the Asset_Type vocabulary.

        Args:
            type_name: Name for the asset type.
            description: What this asset type represents.

        Returns:
            JSON with status, name, description, rid.

        Example:
            add_asset_type("Segmentation Mask", "Binary mask images for segmentation")
        """
        try:
            from deriva_ml import MLVocab

            ml = conn_manager.get_active_or_raise()
            term = ml.add_term(
                table=MLVocab.asset_type,
                term_name=type_name,
                description=description,
                exists_ok=True,
            )
            return json.dumps({
                "status": "created",
                "name": term.name,
                "description": term.description,
                "rid": term.rid,
            })
        except Exception as e:
            logger.error(f"Failed to add asset type: {e}")
            return json.dumps({"status": "error", "message": str(e)})
