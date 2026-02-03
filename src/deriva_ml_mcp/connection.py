"""Connection management for Deriva MCP server.

This module handles Deriva catalog connections, maintaining
active connections and providing access to DerivaML instances.

When connecting to a catalog, an MCP workflow and execution are automatically
created to track all operations performed through the MCP server.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deriva_ml import DerivaML, DerivaMLException

if TYPE_CHECKING:
    pass

logger = logging.getLogger("deriva-mcp")

# Workflow type for MCP server operations
MCP_WORKFLOW_TYPE = "Deriva MCP"


def get_mcp_workflow_info() -> dict[str, str | bool]:
    """Get workflow metadata from environment variables.

    Returns:
        Dictionary with workflow_name, workflow_type, version, git_commit, in_docker.
    """
    return {
        "workflow_name": os.environ.get("DERIVA_MCP_WORKFLOW_NAME", "Deriva MCP Server"),
        "workflow_type": os.environ.get("DERIVA_MCP_WORKFLOW_TYPE", MCP_WORKFLOW_TYPE),
        "version": os.environ.get("DERIVA_MCP_VERSION", ""),
        "git_commit": os.environ.get("DERIVA_MCP_GIT_COMMIT", ""),
        "in_docker": os.environ.get("DERIVA_MCP_IN_DOCKER", "false").lower() == "true",
    }


@dataclass
class ConnectionInfo:
    """Information about an active DerivaML connection.

    Each connection has an associated workflow and execution for tracking
    all MCP operations performed on the catalog.
    """

    hostname: str
    catalog_id: str | int
    domain_schemas: set[str] | None
    ml_instance: DerivaML
    workflow_rid: str | None = None
    execution: Any = None  # Execution object from deriva_ml


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

    def _ensure_mcp_workflow_type(self, ml: DerivaML) -> None:
        """Ensure the 'DerivaML MCP' workflow type exists in the catalog.

        Creates the workflow type term if it doesn't exist.
        """
        from deriva_ml import MLVocab

        try:
            ml.lookup_term(MLVocab.workflow_type, MCP_WORKFLOW_TYPE)
            logger.debug(f"Workflow type '{MCP_WORKFLOW_TYPE}' already exists")
        except Exception:
            logger.info(f"Creating workflow type '{MCP_WORKFLOW_TYPE}'")
            ml.add_term(
                table=MLVocab.workflow_type,
                term_name=MCP_WORKFLOW_TYPE,
                description="Operations performed through the DerivaML MCP Server",
                exists_ok=True,
            )

    def _create_mcp_execution(self, ml: DerivaML) -> tuple[str | None, Any]:
        """Create a workflow and execution for MCP operations.

        Returns:
            Tuple of (workflow_rid, execution) or (None, None) if creation fails.
        """
        from deriva_ml.execution import ExecutionConfiguration

        try:
            # Ensure the MCP workflow type exists
            self._ensure_mcp_workflow_type(ml)

            # Get workflow info from environment
            workflow_info = get_mcp_workflow_info()

            # Build workflow description
            description_parts = ["MCP Server Session"]
            if workflow_info["in_docker"]:
                description_parts.append("(Docker)")
            if workflow_info["version"]:
                description_parts.append(f"v{workflow_info['version']}")
            if workflow_info["git_commit"]:
                description_parts.append(f"[{workflow_info['git_commit'][:12]}]")

            description = " ".join(description_parts)

            # Create the workflow
            workflow = ml.create_workflow(
                name=workflow_info["workflow_name"],
                workflow_type=workflow_info["workflow_type"],
                description=description,
            )
            workflow_rid = ml.add_workflow(workflow)
            logger.info(f"Created MCP workflow: {workflow_rid}")

            # Create execution configuration
            config = ExecutionConfiguration(
                workflow=workflow,
                description=f"MCP session on {ml.host_name}:{ml.catalog_id}",
            )

            # Create and start execution
            execution = ml.create_execution(config)
            execution.__enter__()  # Start the execution context
            logger.info(f"Created MCP execution: {execution.execution_rid}")

            return workflow_rid, execution

        except Exception as e:
            logger.warning(f"Failed to create MCP workflow/execution: {e}")
            # Don't fail the connection if workflow creation fails
            return None, None

    def connect(
        self,
        hostname: str,
        catalog_id: str | int,
        domain_schemas: set[str] | None = None,
        set_active: bool = True,
    ) -> DerivaML:
        """Connect to a DerivaML catalog.

        Args:
            hostname: Hostname of the Deriva server.
            catalog_id: Catalog identifier.
            domain_schemas: Optional set of domain schema names. Auto-detected if omitted.
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
                domain_schemas=domain_schemas,
                check_auth=True,
            )

            # Create MCP workflow and execution for tracking operations
            workflow_rid, execution = self._create_mcp_execution(ml)

            self._connections[key] = ConnectionInfo(
                hostname=hostname,
                catalog_id=catalog_id,
                domain_schemas=domain_schemas,
                ml_instance=ml,
                workflow_rid=workflow_rid,
                execution=execution,
            )
            if set_active:
                self._active_connection = key
            logger.info(f"Successfully connected to {key}")
            return ml
        except Exception as e:
            logger.error(f"Failed to connect to {key}: {e}")
            raise DerivaMLException(f"Failed to connect to {hostname}:{catalog_id}: {e}")

    def disconnect(
        self,
        hostname: str | None = None,
        catalog_id: str | int | None = None,
        upload_outputs: bool = True,
    ) -> bool:
        """Disconnect from a catalog.

        Properly closes the MCP execution context and optionally uploads
        any registered outputs.

        Args:
            hostname: Hostname to disconnect from. If None, uses active connection.
            catalog_id: Catalog to disconnect from. If None, uses active connection.
            upload_outputs: If True, upload any registered execution outputs.

        Returns:
            True if disconnected successfully.
        """
        if hostname is None and catalog_id is None:
            key = self._active_connection
        else:
            key = self._connection_key(hostname or "", catalog_id or "")

        if key and key in self._connections:
            conn_info = self._connections[key]

            # Close the execution context if one exists
            if conn_info.execution is not None:
                try:
                    conn_info.execution.__exit__(None, None, None)
                    logger.info(f"Closed MCP execution for {key}")

                    # Upload outputs if requested
                    if upload_outputs:
                        try:
                            conn_info.execution.upload_execution_outputs(clean_folder=True)
                            logger.info(f"Uploaded MCP execution outputs for {key}")
                        except Exception as e:
                            logger.warning(f"Failed to upload execution outputs: {e}")
                except Exception as e:
                    logger.warning(f"Failed to close execution: {e}")

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

    def get_active_execution(self) -> Any | None:
        """Get the active execution context.

        Returns:
            Active Execution object or None if no active connection or no execution.
        """
        if self._active_connection and self._active_connection in self._connections:
            return self._connections[self._active_connection].execution
        return None

    def get_active_execution_or_raise(self) -> Any:
        """Get the active execution context or raise an error.

        Returns:
            Active Execution object.

        Raises:
            DerivaMLException: If no active connection or no execution.
        """
        execution = self.get_active_execution()
        if execution is None:
            raise DerivaMLException(
                "No active execution context. Connect to a catalog first."
            )
        return execution

    def get_active_connection_info(self) -> ConnectionInfo | None:
        """Get the active connection info including workflow and execution.

        Returns:
            ConnectionInfo or None if no active connection.
        """
        if self._active_connection and self._active_connection in self._connections:
            return self._connections[self._active_connection]
        return None

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
                "domain_schemas": list(info.domain_schemas) if info.domain_schemas else None,
                "is_active": key == self._active_connection,
                "workflow_rid": info.workflow_rid,
                "execution_rid": info.execution.execution_rid if info.execution else None,
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
