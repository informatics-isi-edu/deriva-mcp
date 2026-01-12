"""Catalog connection and management tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-ml-mcp")


def register_catalog_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register catalog management tools with the MCP server."""

    @mcp.tool()
    async def connect_catalog(
        hostname: str,
        catalog_id: str,
        domain_schema: str | None = None,
    ) -> str:
        """Connect to an existing DerivaML catalog. Must be called before using other tools.

        Args:
            hostname: Server hostname (e.g., "dev.eye-ai.org", "www.atlas-d2k.org").
            catalog_id: Catalog ID number (e.g., "1", "52").
            domain_schema: Schema name for domain tables. Auto-detected if omitted.

        Returns:
            JSON with status, hostname, catalog_id, domain_schema, project_name.

        Example:
            connect_catalog("dev.eye-ai.org", "52") -> connects to eye-ai catalog
        """
        try:
            ml = conn_manager.connect(hostname, catalog_id, domain_schema)
            return json.dumps({
                "status": "connected",
                "hostname": hostname,
                "catalog_id": catalog_id,
                "domain_schema": ml.domain_schema,
                "project_name": ml.project_name,
            })
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

    @mcp.tool()
    async def disconnect_catalog() -> str:
        """Disconnect from the currently active catalog."""
        if conn_manager.disconnect():
            return json.dumps({"status": "disconnected"})
        return json.dumps({"status": "no_active_connection"})

    @mcp.tool()
    async def list_connections() -> str:
        """List all open catalog connections and show which is active."""
        connections = conn_manager.list_connections()
        return json.dumps(connections)

    @mcp.tool()
    async def set_active_catalog(hostname: str, catalog_id: str) -> str:
        """Switch the active catalog when multiple catalogs are connected.

        Args:
            hostname: Server hostname of the catalog to activate.
            catalog_id: Catalog ID to activate.
        """
        if conn_manager.set_active(hostname, catalog_id):
            return json.dumps({
                "status": "success",
                "active_catalog": f"{hostname}:{catalog_id}",
            })
        return json.dumps({
            "status": "error",
            "message": f"No connection found for {hostname}:{catalog_id}",
        })

    @mcp.tool()
    async def get_catalog_info() -> str:
        """Get details about the active catalog: hostname, schemas, project name."""
        try:
            ml = conn_manager.get_active_or_raise()
            return json.dumps({
                "hostname": ml.host_name,
                "catalog_id": ml.catalog_id,
                "domain_schema": ml.domain_schema,
                "ml_schema": ml.ml_schema,
                "project_name": ml.project_name,
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_users() -> str:
        """List all users who have access to the active catalog."""
        try:
            ml = conn_manager.get_active_or_raise()
            users = ml.user_list()
            return json.dumps(users)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_chaise_url(table_or_rid: str) -> str:
        """Get a web UI URL for viewing a table or specific record in Chaise.

        Args:
            table_or_rid: Either a table name (e.g., "Image", "Dataset") or
                a RID (e.g., "1-ABC", "W-XYZ") for a specific record.

        Returns:
            JSON with url field containing the Chaise web interface URL.

        Examples:
            get_chaise_url("Image") -> URL to browse all images
            get_chaise_url("1-ABC") -> URL to view specific record 1-ABC
        """
        try:
            ml = conn_manager.get_active_or_raise()
            # First try as a table name
            try:
                url = ml.chaise_url(table_or_rid)
                return json.dumps({"url": url, "table_or_rid": table_or_rid})
            except Exception:
                # If not a table name, try as a RID
                result = ml.resolve_rid(table_or_rid)
                schema_name = result.table.schema.name
                table_name = result.table.name
                base_url = f"https://{ml.host_name}/chaise/record/#{ml.catalog_id}"
                url = f"{base_url}/{schema_name}:{table_name}/RID={result.rid}"
                return json.dumps({"url": url, "table_or_rid": table_or_rid})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def resolve_rid(rid: str) -> str:
        """Find which table a RID belongs to and get its Chaise URL.

        Args:
            rid: Resource Identifier (e.g., "1-ABC", "W-XYZ").

        Returns:
            JSON with rid, schema, table, and url fields.

        Example:
            resolve_rid("1-ABC") -> {"schema": "domain", "table": "Image", ...}
        """
        try:
            ml = conn_manager.get_active_or_raise()
            result = ml.resolve_rid(rid)
            schema_name = result.table.schema.name
            table_name = result.table.name
            base_url = f"https://{ml.host_name}/chaise/record/#{ml.catalog_id}"
            url = f"{base_url}/{schema_name}:{table_name}/RID={result.rid}"
            return json.dumps({
                "rid": rid,
                "schema": schema_name,
                "table": table_name,
                "url": url,
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def create_catalog(
        hostname: str,
        project_name: str,
    ) -> str:
        """Create a new DerivaML catalog with all ML schema tables.

        Creates a fresh catalog with Dataset, Execution, Workflow, Feature, and
        vocabulary tables. Automatically connects to the new catalog.

        Args:
            hostname: Server hostname (e.g., "localhost", "deriva.example.org").
            project_name: Name for the project, becomes the domain schema name.

        Returns:
            JSON with status, hostname, catalog_id, domain_schema, project_name.

        Example:
            create_catalog("localhost", "my_ml_project")
        """
        try:
            from deriva.core.ermrest_model import Schema
            from deriva_ml.schema import create_ml_catalog

            # Create the catalog with deriva-ml schema
            catalog = create_ml_catalog(hostname, project_name)

            # Create the domain schema (project_name becomes the domain schema)
            model = catalog.getCatalogModel()
            model.create_schema(Schema.define(project_name))

            # Connect to the newly created catalog
            ml = conn_manager.connect(hostname, str(catalog.catalog_id), project_name)

            return json.dumps({
                "status": "created",
                "hostname": hostname,
                "catalog_id": str(catalog.catalog_id),
                "domain_schema": ml.domain_schema,
                "project_name": project_name,
            })
        except Exception as e:
            logger.error(f"Failed to create catalog: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

    @mcp.tool()
    async def delete_catalog(
        hostname: str,
        catalog_id: str,
    ) -> str:
        """PERMANENTLY DELETE a catalog and all its data. Cannot be undone.

        Args:
            hostname: Server hostname.
            catalog_id: ID of the catalog to delete.

        Returns:
            JSON with deletion status.
        """
        try:
            from deriva.core import DerivaServer

            # Disconnect if this is the active catalog
            active = conn_manager.get_active()
            if active and str(active.catalog_id) == str(catalog_id):
                conn_manager.disconnect()

            # Delete the catalog
            server = DerivaServer("https", hostname)
            server.delete(f"/ermrest/catalog/{catalog_id}")

            return json.dumps({
                "status": "deleted",
                "hostname": hostname,
                "catalog_id": catalog_id,
            })
        except Exception as e:
            logger.error(f"Failed to delete catalog: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

    @mcp.tool()
    async def apply_catalog_annotations(
        navbar_brand_text: str = "ML Data Browser",
        head_title: str = "Catalog ML",
    ) -> str:
        """Apply catalog-level annotations to initialize the Chaise web interface.

        Chaise is Deriva's web-based data browser. This method sets up annotations that
        control how Chaise displays and organizes the catalog, including the navigation
        bar and display settings.

        **Navigation Bar Structure**:
        Creates a navigation bar with organized dropdown menus:
        - **User Info**: Users, Groups, and RID Lease tables
        - **Deriva-ML**: Core ML tables (Workflow, Execution, Dataset, Dataset_Version, etc.)
        - **WWW**: Web content tables (Page, File)
        - **{Domain Schema}**: All domain-specific tables (excludes vocabularies/associations)
        - **Vocabulary**: All controlled vocabulary tables from ML and domain schemas
        - **Assets**: All asset tables from ML and domain schemas
        - **Catalog Registry**: Link to ermrest registry
        - **Documentation**: Links to ML docs and instructions

        **Display Settings**:
        - Underscores in names displayed as spaces
        - System columns (RID) shown in views
        - Default landing page set to Dataset table
        - Faceted search and record deletion enabled

        **Bulk Upload**:
        Configures drag-and-drop file upload for asset tables.

        **When to call**: After creating the domain schema and all tables. The menus are
        dynamically built from the current schema structure.

        Args:
            navbar_brand_text: Text in the navigation bar brand area (default: "ML Data Browser").
            head_title: Browser tab title (default: "Catalog ML").

        Returns:
            JSON with status and applied settings.

        Example workflow:
            1. create_catalog("localhost", "my_project")
            2. create_vocabulary("Species", "Types of species")
            3. create_asset("Image", ...)
            4. apply_catalog_annotations("My ML Project", "ML Catalog")
        """
        try:
            ml = conn_manager.get_active_or_raise()
            ml.apply_catalog_annotations(
                navbar_brand_text=navbar_brand_text,
                head_title=head_title,
            )
            return json.dumps({
                "status": "success",
                "navbar_brand_text": navbar_brand_text,
                "head_title": head_title,
                "message": "Catalog annotations applied. Navigation bar and display settings configured.",
            })
        except Exception as e:
            logger.error(f"Failed to apply catalog annotations: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })
