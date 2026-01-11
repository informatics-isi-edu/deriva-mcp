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
        """Connect to a DerivaML catalog.

        Establishes a connection to a Deriva catalog with ML schema support.
        The connection will be set as the active connection for subsequent operations.

        Args:
            hostname: Hostname of the Deriva server (e.g., 'deriva.example.org')
            catalog_id: Catalog identifier (either numeric ID or catalog name)
            domain_schema: Optional domain schema name. If not specified, will auto-detect.

        Returns:
            Connection status message with catalog information.
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
        """Disconnect from the active DerivaML catalog.

        Closes the connection to the currently active catalog.

        Returns:
            Disconnection status message.
        """
        if conn_manager.disconnect():
            return json.dumps({"status": "disconnected"})
        return json.dumps({"status": "no_active_connection"})

    @mcp.tool()
    async def list_connections() -> str:
        """List all active catalog connections.

        Returns information about all currently connected catalogs,
        including which one is the active connection.

        Returns:
            JSON array of connection information.
        """
        connections = conn_manager.list_connections()
        return json.dumps(connections)

    @mcp.tool()
    async def set_active_catalog(hostname: str, catalog_id: str) -> str:
        """Set the active catalog connection.

        When multiple catalogs are connected, this sets which one
        is used for subsequent operations.

        Args:
            hostname: Server hostname of the connection to activate.
            catalog_id: Catalog identifier of the connection to activate.

        Returns:
            Status message indicating success or failure.
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
        """Get information about the active catalog.

        Returns detailed information about the currently connected catalog,
        including schema names, project info, and available tables.

        Returns:
            JSON object with catalog information.
        """
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
        """List users with access to the catalog.

        Returns a list of users who have access to the currently
        connected catalog.

        Returns:
            JSON array of user information (ID and Full_Name).
        """
        try:
            ml = conn_manager.get_active_or_raise()
            users = ml.user_list()
            return json.dumps(users)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_chaise_url(table_name: str) -> str:
        """Get the Chaise web interface URL for a table.

        Generates a URL to view/edit the specified table in the
        Deriva Chaise web interface.

        Args:
            table_name: Name of the table to get the URL for.

        Returns:
            URL string or error message.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            url = ml.chaise_url(table_name)
            return json.dumps({"url": url})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
