"""Connection management for DerivaML MCP server.

This module handles DerivaML catalog connections, maintaining
active connections and providing access to DerivaML instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deriva_ml import DerivaML, DerivaMLException

if TYPE_CHECKING:
    pass

logger = logging.getLogger("deriva-ml-mcp")


@dataclass
class ConnectionInfo:
    """Information about an active DerivaML connection."""

    hostname: str
    catalog_id: str | int
    domain_schema: str | None
    ml_instance: DerivaML


class ConnectionManager:
    """Manages DerivaML catalog connections.

    Maintains a registry of active connections and provides
    methods to connect, disconnect, and access DerivaML instances.
    """

    def __init__(self) -> None:
        self._connections: dict[str, ConnectionInfo] = {}
        self._active_connection: str | None = None

    def _connection_key(self, hostname: str, catalog_id: str | int) -> str:
        """Generate a unique key for a connection."""
        return f"{hostname}:{catalog_id}"

    def connect(
        self,
        hostname: str,
        catalog_id: str | int,
        domain_schema: str | None = None,
        set_active: bool = True,
    ) -> DerivaML:
        """Connect to a DerivaML catalog.

        Args:
            hostname: Hostname of the Deriva server.
            catalog_id: Catalog identifier.
            domain_schema: Optional domain schema name.
            set_active: If True, set this as the active connection.

        Returns:
            DerivaML instance for the catalog.

        Raises:
            DerivaMLException: If connection fails.
        """
        key = self._connection_key(hostname, catalog_id)

        # Return existing connection if available
        if key in self._connections:
            if set_active:
                self._active_connection = key
            logger.info(f"Reusing existing connection to {key}")
            return self._connections[key].ml_instance

        # Create new connection
        logger.info(f"Connecting to {hostname}, catalog {catalog_id}")
        try:
            ml = DerivaML(
                hostname=hostname,
                catalog_id=catalog_id,
                domain_schema=domain_schema,
                check_auth=True,
            )
            self._connections[key] = ConnectionInfo(
                hostname=hostname,
                catalog_id=catalog_id,
                domain_schema=domain_schema,
                ml_instance=ml,
            )
            if set_active:
                self._active_connection = key
            logger.info(f"Successfully connected to {key}")
            return ml
        except Exception as e:
            logger.error(f"Failed to connect to {key}: {e}")
            raise DerivaMLException(f"Failed to connect to {hostname}:{catalog_id}: {e}")

    def disconnect(self, hostname: str | None = None, catalog_id: str | int | None = None) -> bool:
        """Disconnect from a catalog.

        Args:
            hostname: Hostname to disconnect from. If None, uses active connection.
            catalog_id: Catalog to disconnect from. If None, uses active connection.

        Returns:
            True if disconnected successfully.
        """
        if hostname is None and catalog_id is None:
            key = self._active_connection
        else:
            key = self._connection_key(hostname or "", catalog_id or "")

        if key and key in self._connections:
            del self._connections[key]
            if self._active_connection == key:
                self._active_connection = None
            logger.info(f"Disconnected from {key}")
            return True
        return False

    def get_active(self) -> DerivaML | None:
        """Get the active DerivaML instance.

        Returns:
            Active DerivaML instance or None if no active connection.
        """
        if self._active_connection and self._active_connection in self._connections:
            return self._connections[self._active_connection].ml_instance
        return None

    def get_active_or_raise(self) -> DerivaML:
        """Get the active DerivaML instance or raise an error.

        Returns:
            Active DerivaML instance.

        Raises:
            DerivaMLException: If no active connection.
        """
        ml = self.get_active()
        if ml is None:
            raise DerivaMLException(
                "No active catalog connection. Use 'connect' tool to connect to a catalog first."
            )
        return ml

    def get_connection(self, hostname: str, catalog_id: str | int) -> DerivaML | None:
        """Get a specific connection.

        Args:
            hostname: Server hostname.
            catalog_id: Catalog identifier.

        Returns:
            DerivaML instance or None if not connected.
        """
        key = self._connection_key(hostname, catalog_id)
        if key in self._connections:
            return self._connections[key].ml_instance
        return None

    def list_connections(self) -> list[dict[str, Any]]:
        """List all active connections.

        Returns:
            List of connection information dictionaries.
        """
        return [
            {
                "hostname": info.hostname,
                "catalog_id": info.catalog_id,
                "domain_schema": info.domain_schema,
                "is_active": key == self._active_connection,
            }
            for key, info in self._connections.items()
        ]

    def set_active(self, hostname: str, catalog_id: str | int) -> bool:
        """Set the active connection.

        Args:
            hostname: Server hostname.
            catalog_id: Catalog identifier.

        Returns:
            True if connection exists and was set as active.
        """
        key = self._connection_key(hostname, catalog_id)
        if key in self._connections:
            self._active_connection = key
            return True
        return False
