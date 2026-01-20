"""Schema manipulation tools for DerivaML MCP server.

This module provides tools for manipulating catalog schema including:
- Creating and configuring tables
- Adding and modifying columns
- Managing table/column annotations and display settings
"""

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
        foreign_keys: list[dict] | None = None,
        comment: str = "",
    ) -> str:
        """Create a new table in the domain schema.

        This tool creates a standard table (not an asset table). For tables that store
        files with automatic URL/checksum tracking, use create_asset_table instead.

        **Process Overview**:
        1. Define columns with names, types, and constraints
        2. Optionally define foreign keys to reference other tables
        3. The table is created in the domain schema
        4. The navigation bar is automatically updated

        Args:
            table_name: Name for the new table (e.g., "Subject", "Experiment", "Protocol").
            columns: Column definitions, each dict with:
                - name (str, required): Column name
                - type (str): One of "text", "int2", "int4", "int8", "float4", "float8",
                  "boolean", "date", "timestamp", "timestamptz", "json", "jsonb", "markdown"
                  (default: "text")
                - nullok (bool): Allow null values (default: True)
                - comment (str): Column description
            foreign_keys: Foreign key definitions, each dict with:
                - column (str, required): Column name in this table (must also be in columns list)
                - referenced_table (str, required): Name of the table to reference
                - referenced_column (str): Column in referenced table (default: "RID")
                - on_delete (str): Action on delete - "NO ACTION", "CASCADE", "SET NULL" (default: "NO ACTION")
            comment: Description of the table's purpose.

        Returns:
            JSON with status, table_name, schema, columns.

        Examples:
            Simple table:
                create_table("Subject", [
                    {"name": "Name", "type": "text", "nullok": false},
                    {"name": "Age", "type": "int4"},
                    {"name": "Notes", "type": "markdown"}
                ])

            Table with foreign key:
                create_table("Sample", [
                    {"name": "Name", "type": "text", "nullok": false},
                    {"name": "Subject", "type": "text", "nullok": false},
                    {"name": "Collection_Date", "type": "date"}
                ], foreign_keys=[
                    {"column": "Subject", "referenced_table": "Subject", "on_delete": "CASCADE"}
                ])
        """
        try:
            from deriva_ml import BuiltinTypes, ColumnDefinition, ForeignKeyDefinition, TableDefinition

            ml = conn_manager.get_active_or_raise()

            type_map = {
                "text": BuiltinTypes.text,
                "int2": BuiltinTypes.int2,
                "int4": BuiltinTypes.int4,
                "int8": BuiltinTypes.int8,
                "float4": BuiltinTypes.float4,
                "float8": BuiltinTypes.float8,
                "boolean": BuiltinTypes.boolean,
                "date": BuiltinTypes.date,
                "timestamp": BuiltinTypes.timestamp,
                "timestamptz": BuiltinTypes.timestamptz,
                "json": BuiltinTypes.json,
                "jsonb": BuiltinTypes.jsonb,
                "markdown": BuiltinTypes.markdown,
            }

            col_defs = []
            if columns:
                for col in columns:
                    col_type = type_map.get(col.get("type", "text"), BuiltinTypes.text)
                    col_defs.append(ColumnDefinition(
                        name=col["name"],
                        type=col_type,
                        nullok=col.get("nullok", True),
                        comment=col.get("comment", ""),
                    ))

            fkey_defs = []
            if foreign_keys:
                for fk in foreign_keys:
                    fkey_defs.append(ForeignKeyDefinition(
                        colnames=[fk["column"]],
                        pk_sname=ml.domain_schema,
                        pk_tname=fk["referenced_table"],
                        pk_colnames=[fk.get("referenced_column", "RID")],
                        on_delete=fk.get("on_delete", "NO ACTION"),
                    ))

            table_def = TableDefinition(
                name=table_name,
                column_defs=col_defs,
                fkey_defs=fkey_defs,
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
            from deriva_ml import BuiltinTypes, ColumnDefinition

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
            JSON array of {asset_rid, filename, url, length, md5, asset_types, asset_table}.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            assets = ml.list_assets(asset_table)
            result = []
            for asset in assets:
                result.append({
                    "asset_rid": asset.asset_rid,
                    "filename": asset.filename,
                    "url": asset.url,
                    "length": asset.length,
                    "md5": asset.md5,
                    "asset_types": asset.asset_types,
                    "asset_table": asset.asset_table,
                    "description": asset.description,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list assets: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def lookup_asset(asset_rid: str) -> str:
        """Look up an asset by its RID from any asset table.

        Returns detailed information about a specific asset including its
        metadata, types, and the execution that created it (if tracked).

        Args:
            asset_rid: RID of the asset to look up (e.g., "3JSE").

        Returns:
            JSON with asset details:
            - asset_rid: The asset's RID
            - asset_table: Which table contains this asset (e.g., "Image", "Model")
            - filename: Original filename
            - url: URL to download the file
            - length: File size in bytes
            - md5: MD5 checksum
            - description: Asset description
            - asset_types: List of asset type vocabulary terms
            - execution_rid: RID of execution that created this asset (if tracked)
            - chaise_url: URL to view in Chaise web interface

        Example:
            lookup_asset("3JSE") -> detailed asset information
        """
        try:
            ml = conn_manager.get_active_or_raise()
            asset = ml.lookup_asset(asset_rid)

            return json.dumps({
                "asset_rid": asset.asset_rid,
                "asset_table": asset.asset_table,
                "filename": asset.filename,
                "url": asset.url,
                "length": asset.length,
                "md5": asset.md5,
                "description": asset.description,
                "asset_types": asset.asset_types,
                "execution_rid": asset.execution_rid,
                "chaise_url": asset.get_chaise_url(),
            })
        except Exception as e:
            logger.error(f"Failed to lookup asset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_asset_executions(asset_rid: str, asset_role: str | None = None) -> str:
        """List all executions associated with an asset.

        Given an asset RID, returns a list of executions that created or used
        the asset, along with the role (Input/Output) in each execution. This
        is useful for provenance tracking - finding which execution created an
        asset or which executions used it as input.

        Args:
            asset_rid: RID of the asset to look up.
            asset_role: Optional filter: "Input" or "Output". If omitted, returns all.

        Returns:
            JSON array of execution records showing which executions are associated
            with this asset. Each record includes execution_rid, workflow_rid,
            status, and description.

        Example:
            list_asset_executions("3JSE") -> finds all executions that created/used this asset
            list_asset_executions("3JSE", "Output") -> finds only the execution that created it
        """
        try:
            ml = conn_manager.get_active_or_raise()
            executions = ml.list_asset_executions(asset_rid, asset_role=asset_role)

            result = []
            for exe in executions:
                result.append({
                    "execution_rid": exe.execution_rid,
                    "workflow_rid": exe.workflow_rid,
                    "status": exe.status.value if hasattr(exe.status, 'value') else str(exe.status),
                    "description": exe.description,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list asset executions: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def find_assets(
        asset_table: str | None = None,
        asset_type: str | None = None,
    ) -> str:
        """Find assets in the catalog with optional filtering.

        Search for assets across all tables or filter by table name and/or
        asset type.

        Args:
            asset_table: Optional table name to search (e.g., "Image", "Model").
                If omitted, searches all asset tables.
            asset_type: Optional asset type to filter by (e.g., "Training_Data").
                If omitted, returns assets of all types.

        Returns:
            JSON array of assets with {asset_rid, asset_table, filename,
            asset_types, execution_rid}.

        Example:
            find_assets() -> all assets in catalog
            find_assets("Model") -> all Model assets
            find_assets(asset_type="Training_Data") -> assets of this type
        """
        try:
            ml = conn_manager.get_active_or_raise()
            assets = ml.find_assets(asset_table=asset_table, asset_type=asset_type)

            result = []
            for asset in assets:
                result.append({
                    "asset_rid": asset.asset_rid,
                    "asset_table": asset.asset_table,
                    "filename": asset.filename,
                    "asset_types": asset.asset_types,
                    "execution_rid": asset.execution_rid,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to find assets: {e}")
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

    @mcp.tool()
    async def add_asset_type_to_asset(asset_rid: str, type_name: str) -> str:
        """Add an asset type to a specific asset.

        Associates an asset with a type from the Asset_Type vocabulary. An asset
        can have multiple types.

        Args:
            asset_rid: RID of the asset to modify.
            type_name: Name of the asset type to add (must exist in Asset_Type vocab).

        Returns:
            JSON with status, asset_rid, and updated types list.

        Example:
            add_asset_type_to_asset("3JSE", "Training_Data")
        """
        try:
            ml = conn_manager.get_active_or_raise()
            asset = ml.lookup_asset(asset_rid)
            asset.add_asset_type(type_name)

            return json.dumps({
                "status": "added",
                "asset_rid": asset_rid,
                "type_name": type_name,
                "asset_types": asset.asset_types,
            })
        except Exception as e:
            logger.error(f"Failed to add asset type to asset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def remove_asset_type_from_asset(asset_rid: str, type_name: str) -> str:
        """Remove an asset type from a specific asset.

        Removes the association between an asset and a type.

        Args:
            asset_rid: RID of the asset to modify.
            type_name: Name of the asset type to remove.

        Returns:
            JSON with status, asset_rid, and updated types list.

        Example:
            remove_asset_type_from_asset("3JSE", "Training_Data")
        """
        try:
            ml = conn_manager.get_active_or_raise()
            asset = ml.lookup_asset(asset_rid)
            asset.remove_asset_type(type_name)

            return json.dumps({
                "status": "removed",
                "asset_rid": asset_rid,
                "type_name": type_name,
                "asset_types": asset.asset_types,
            })
        except Exception as e:
            logger.error(f"Failed to remove asset type from asset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_schema_description(include_system_columns: bool = False) -> str:
        """Get a complete description of the catalog schema structure.

        Returns the full data model including all tables, columns, foreign keys,
        and relationships in both the domain and ML schemas. Use this to understand
        how data is organized before querying or creating new structures.

        Args:
            include_system_columns: Include RID, RCT, RMT, RCB, RMB columns (default: False).

        Returns:
            JSON with complete schema structure:
            - domain_schema: Name of the domain schema
            - ml_schema: Name of the ML schema (deriva-ml)
            - schemas: Dict of schema definitions, each containing:
              - tables: Dict of table definitions with columns, foreign_keys, features

        Example:
            get_schema_description() -> full catalog structure for understanding data model
        """
        try:
            ml = conn_manager.get_active_or_raise()
            schema_info = ml.model.get_schema_description(
                include_system_columns=include_system_columns
            )
            return json.dumps(schema_info)
        except Exception as e:
            logger.error(f"Failed to get schema description: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    # -------------------------------------------------------------------------
    # Table Manipulation Tools
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def set_table_description(table_name: str, description: str) -> str:
        """Set or update the description (comment) for a table.

        Args:
            table_name: Name of the table to update.
            description: New description for the table.

        Returns:
            JSON with status, table_name, description.

        Example:
            set_table_description("Image", "Medical images for analysis")
        """
        try:
            from deriva_ml.model.handles import TableHandle

            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            handle = TableHandle(table)
            handle.description = description

            return json.dumps({
                "status": "updated",
                "table_name": table_name,
                "description": description,
            })
        except Exception as e:
            logger.error(f"Failed to set table description: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_table_display_name(table_name: str, display_name: str) -> str:
        """Set the display name shown in the UI for a table.

        Args:
            table_name: Name of the table to update.
            display_name: Human-readable name to display in the UI.

        Returns:
            JSON with status, table_name, display_name.

        Example:
            set_table_display_name("Image", "Medical Images")
        """
        try:
            from deriva_ml.model.handles import TableHandle

            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            handle = TableHandle(table)
            handle.set_display_name(display_name)

            return json.dumps({
                "status": "updated",
                "table_name": table_name,
                "display_name": display_name,
            })
        except Exception as e:
            logger.error(f"Failed to set table display name: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_row_name_pattern(table_name: str, pattern: str) -> str:
        """Set the pattern used to display row names in the UI.

        The pattern uses Handlebars syntax with triple braces for column values.

        Args:
            table_name: Name of the table to update.
            pattern: Handlebars template (e.g., "{{{Name}}}" or "{{{FirstName}}} {{{LastName}}}").

        Returns:
            JSON with status, table_name, pattern.

        Examples:
            set_row_name_pattern("Subject", "{{{Name}}}")
            set_row_name_pattern("Image", "{{{Filename}}} ({{{RID}}})")
        """
        try:
            from deriva_ml.model.handles import TableHandle

            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            handle = TableHandle(table)
            handle.set_row_name_pattern(pattern)

            return json.dumps({
                "status": "updated",
                "table_name": table_name,
                "pattern": pattern,
            })
        except Exception as e:
            logger.error(f"Failed to set row name pattern: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_visible_columns(
        table_name: str,
        columns: list[str],
        context: str = "*",
    ) -> str:
        """Set which columns are visible in a table display context.

        Controls which columns appear when viewing the table in the UI.
        Different contexts show different views of the data.

        Args:
            table_name: Name of the table to configure.
            columns: List of column names to display (in order).
            context: Display context - "*" (all), "compact", "detailed", "entry", or "filter".

        Returns:
            JSON with status, table_name, columns, context.

        Example:
            set_visible_columns("Subject", ["RID", "Name", "Age", "Notes"])
            set_visible_columns("Subject", ["Name", "Age"], context="compact")
        """
        try:
            from deriva_ml.model.handles import TableHandle

            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            handle = TableHandle(table)
            handle.set_visible_columns(columns, context=context)

            return json.dumps({
                "status": "updated",
                "table_name": table_name,
                "columns": columns,
                "context": context,
            })
        except Exception as e:
            logger.error(f"Failed to set visible columns: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_column(
        table_name: str,
        column_name: str,
        column_type: str = "text",
        nullok: bool = True,
        default: str | None = None,
        comment: str | None = None,
    ) -> str:
        """Add a new column to an existing table.

        Args:
            table_name: Name of the table to modify.
            column_name: Name for the new column.
            column_type: Data type - one of "text", "int2", "int4", "int8", "float4",
                "float8", "boolean", "date", "timestamp", "timestamptz", "json",
                "jsonb", "markdown" (default: "text").
            nullok: Whether NULL values are allowed (default: True).
            default: Default value for new rows (optional).
            comment: Description of the column (optional).

        Returns:
            JSON with status, table_name, column_name, column_type.

        Example:
            add_column("Subject", "Age", "int4", nullok=True, comment="Subject age in years")
        """
        try:
            from deriva_ml.core.enums import BuiltinTypes
            from deriva_ml.model.handles import TableHandle

            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            handle = TableHandle(table)

            type_map = {
                "text": BuiltinTypes.text,
                "int2": BuiltinTypes.int2,
                "int4": BuiltinTypes.int4,
                "int8": BuiltinTypes.int8,
                "float4": BuiltinTypes.float4,
                "float8": BuiltinTypes.float8,
                "boolean": BuiltinTypes.boolean,
                "date": BuiltinTypes.date,
                "timestamp": BuiltinTypes.timestamp,
                "timestamptz": BuiltinTypes.timestamptz,
                "json": BuiltinTypes.json,
                "jsonb": BuiltinTypes.jsonb,
                "markdown": BuiltinTypes.markdown,
            }
            col_type = type_map.get(column_type, BuiltinTypes.text)

            col = handle.add_column(
                name=column_name,
                column_type=col_type,
                nullok=nullok,
                default=default,
                comment=comment,
            )

            return json.dumps({
                "status": "created",
                "table_name": table_name,
                "column_name": col.name,
                "column_type": column_type,
            })
        except Exception as e:
            logger.error(f"Failed to add column: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    # -------------------------------------------------------------------------
    # Column Manipulation Tools
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def set_column_description(
        table_name: str,
        column_name: str,
        description: str,
    ) -> str:
        """Set or update the description (comment) for a column.

        Args:
            table_name: Name of the table containing the column.
            column_name: Name of the column to update.
            description: New description for the column.

        Returns:
            JSON with status, table_name, column_name, description.

        Example:
            set_column_description("Subject", "Age", "Subject age in years at enrollment")
        """
        try:
            from deriva_ml.model.handles import TableHandle

            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            handle = TableHandle(table)
            col = handle.column(column_name)
            col.description = description

            return json.dumps({
                "status": "updated",
                "table_name": table_name,
                "column_name": column_name,
                "description": description,
            })
        except Exception as e:
            logger.error(f"Failed to set column description: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_column_display_name(
        table_name: str,
        column_name: str,
        display_name: str,
    ) -> str:
        """Set the display name shown in the UI for a column.

        Args:
            table_name: Name of the table containing the column.
            column_name: Name of the column to update.
            display_name: Human-readable name to display in the UI.

        Returns:
            JSON with status, table_name, column_name, display_name.

        Example:
            set_column_display_name("Subject", "DOB", "Date of Birth")
        """
        try:
            from deriva_ml.model.handles import TableHandle

            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            handle = TableHandle(table)
            col = handle.column(column_name)
            col.set_display_name(display_name)

            return json.dumps({
                "status": "updated",
                "table_name": table_name,
                "column_name": column_name,
                "display_name": display_name,
            })
        except Exception as e:
            logger.error(f"Failed to set column display name: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_column_nullok(
        table_name: str,
        column_name: str,
        nullok: bool,
    ) -> str:
        """Set whether a column allows NULL values.

        Args:
            table_name: Name of the table containing the column.
            column_name: Name of the column to update.
            nullok: True to allow NULL values, False to require values.

        Returns:
            JSON with status, table_name, column_name, nullok.

        Note:
            Setting nullok=False will fail if the column contains NULL values.

        Example:
            set_column_nullok("Subject", "Name", False)  # Make Name required
        """
        try:
            from deriva_ml.model.handles import TableHandle

            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            handle = TableHandle(table)
            col = handle.column(column_name)
            col.set_nullok(nullok)

            return json.dumps({
                "status": "updated",
                "table_name": table_name,
                "column_name": column_name,
                "nullok": nullok,
            })
        except Exception as e:
            logger.error(f"Failed to set column nullok: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_table_columns(
        table_name: str,
        include_system: bool = False,
    ) -> str:
        """Get detailed information about all columns in a table.

        Args:
            table_name: Name of the table to describe.
            include_system: Include system columns (RID, RCT, etc.) (default: False).

        Returns:
            JSON array of column details: {name, type, nullok, description,
            display_name, is_system}.

        Example:
            get_table_columns("Subject") -> all user columns
            get_table_columns("Subject", include_system=True) -> all columns
        """
        try:
            from deriva_ml.model.handles import TableHandle

            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            handle = TableHandle(table)

            if include_system:
                columns = list(handle.all_columns)
            else:
                columns = handle.user_columns

            result = []
            for col in columns:
                result.append({
                    "name": col.name,
                    "type": col._column.type.typename,
                    "nullok": col.nullok,
                    "description": col.description,
                    "display_name": col.get_display_name(),
                    "is_system": col.is_system_column,
                })

            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to get table columns: {e}")
            return json.dumps({"status": "error", "message": str(e)})
