"""Connection management for Deriva MCP server.

This module handles catalog connections for both DerivaML catalogs and plain
ERMrest catalogs. It maintains active connections and provides access to either
DerivaML instances or raw ErmrestCatalog instances depending on catalog type.

When connecting to a DerivaML catalog, an MCP workflow and execution are
automatically created to track all operations performed through the MCP server.

For plain ERMrest catalogs, basic catalog operations are available without
the ML-specific tracking features.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from deriva.core import DerivaServer, ErmrestCatalog, get_credential

if TYPE_CHECKING:
    from deriva_ml import DerivaML

logger = logging.getLogger("deriva-ml-mcp")

# Workflow type for MCP server operations
MCP_WORKFLOW_TYPE = "DerivaML MCP"

# ML schema name for DerivaML catalogs
ML_SCHEMA_NAME = "deriva-ml"


class CatalogType(str, Enum):
    """Type of catalog connection."""

    DERIVAML = "derivaml"  # Full DerivaML catalog with ML schema
    ERMREST = "ermrest"  # Plain ERMrest catalog without ML schema


def get_mcp_workflow_info() -> dict[str, str | bool]:
    """Get workflow metadata from environment variables.

    Returns:
        Dictionary with workflow_name, workflow_type, version, git_commit, in_docker.
    """
    return {
        "workflow_name": os.environ.get("DERIVAML_MCP_WORKFLOW_NAME", "DerivaML MCP Server"),
        "workflow_type": os.environ.get("DERIVAML_MCP_WORKFLOW_TYPE", MCP_WORKFLOW_TYPE),
        "version": os.environ.get("DERIVAML_MCP_VERSION", ""),
        "git_commit": os.environ.get("DERIVAML_MCP_GIT_COMMIT", ""),
        "in_docker": os.environ.get("DERIVAML_MCP_IN_DOCKER", "false").lower() == "true",
    }


def check_is_derivaml_catalog(catalog: ErmrestCatalog) -> dict[str, Any]:
    """Check if a catalog is a DerivaML catalog with the ML schema.

    Args:
        catalog: ErmrestCatalog instance to check.

    Returns:
        Dictionary with:
        - is_derivaml: True if catalog has the deriva-ml schema
        - ml_schema: Name of the ML schema if found
        - domain_schema: Auto-detected domain schema name if DerivaML
        - missing_tables: List of missing required tables (if partial)
        - validation_report: Detailed validation results (if DerivaML)
    """
    model = catalog.getCatalogModel()

    # Check for ML schema
    if ML_SCHEMA_NAME not in model.schemas:
        return {
            "is_derivaml": False,
            "ml_schema": None,
            "domain_schema": None,
            "reason": f"Schema '{ML_SCHEMA_NAME}' not found",
        }

    ml_schema = model.schemas[ML_SCHEMA_NAME]

    # Check for core ML tables
    required_tables = ["Dataset", "Execution", "Workflow", "Dataset_Version"]
    missing_tables = [t for t in required_tables if t not in ml_schema.tables]

    if missing_tables:
        return {
            "is_derivaml": False,
            "ml_schema": ML_SCHEMA_NAME,
            "domain_schema": None,
            "missing_tables": missing_tables,
            "reason": f"Missing required tables: {missing_tables}",
        }

    # Auto-detect domain schema (first non-system schema that isn't deriva-ml)
    system_schemas = {"public", "deriva-ml", "WWW", "_acl_admin"}
    domain_schema = None
    for schema_name in model.schemas:
        if schema_name not in system_schemas:
            domain_schema = schema_name
            break

    return {
        "is_derivaml": True,
        "ml_schema": ML_SCHEMA_NAME,
        "domain_schema": domain_schema,
    }


@dataclass
class ConnectionInfo:
    """Information about an active catalog connection.

    Supports both DerivaML catalogs and plain ERMrest catalogs.
    For DerivaML catalogs, includes workflow and execution tracking.
    """

    hostname: str
    catalog_id: str | int
    catalog_type: CatalogType
    catalog: ErmrestCatalog  # Always available
    ml_instance: "DerivaML | None" = None  # Only for DerivaML catalogs
    domain_schema: str | None = None
    workflow_rid: str | None = None
    execution: Any = None  # Execution object from deriva_ml

    @property
    def is_derivaml(self) -> bool:
        """Check if this is a DerivaML catalog connection."""
        return self.catalog_type == CatalogType.DERIVAML

    def get_model(self):
        """Get the catalog model (works for both catalog types)."""
        return self.catalog.getCatalogModel()

    def get_pathbuilder(self):
        """Get a path builder for queries (works for both catalog types)."""
        return self.catalog.getPathBuilder()


class ConnectionManager:
    """Manages catalog connections for both DerivaML and plain ERMrest catalogs.

    Maintains a registry of active connections and provides methods to connect,
    disconnect, and access catalog instances. Automatically detects whether a
    catalog is DerivaML-compliant and adjusts available features accordingly.
    """

    def __init__(self) -> None:
        self._connections: dict[str, ConnectionInfo] = {}
        self._active_connection: str | None = None

    def _connection_key(self, hostname: str, catalog_id: str | int) -> str:
        """Generate a unique key for a connection."""
        return f"{hostname}:{catalog_id}"

    def _ensure_mcp_workflow_type(self, ml: "DerivaML") -> None:
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

    def _create_mcp_execution(self, ml: "DerivaML") -> tuple[str | None, Any]:
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
        domain_schema: str | None = None,
        set_active: bool = True,
        require_derivaml: bool = False,
    ) -> ConnectionInfo:
        """Connect to a catalog (DerivaML or plain ERMrest).

        Automatically detects whether the catalog is DerivaML-compliant. For
        DerivaML catalogs, creates an ML instance and MCP execution tracking.
        For plain ERMrest catalogs, provides basic catalog access.

        Args:
            hostname: Hostname of the Deriva server.
            catalog_id: Catalog identifier.
            domain_schema: Optional domain schema name (auto-detected for DerivaML).
            set_active: If True, set this as the active connection.
            require_derivaml: If True, raise error if catalog is not DerivaML.

        Returns:
            ConnectionInfo with catalog access details.

        Raises:
            DerivaMLException: If require_derivaml=True and catalog is not DerivaML,
                              or if connection fails.
        """
        from deriva_ml import DerivaML, DerivaMLException

        key = self._connection_key(hostname, catalog_id)

        # Return existing connection if available
        if key in self._connections:
            if set_active:
                self._active_connection = key
            logger.info(f"Reusing existing connection to {key}")
            return self._connections[key]

        # Create new connection - first connect to ERMrest
        logger.info(f"Connecting to {hostname}, catalog {catalog_id}")
        try:
            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            catalog = server.connect_ermrest(catalog_id)

            # Check if this is a DerivaML catalog
            check_result = check_is_derivaml_catalog(catalog)

            if check_result["is_derivaml"]:
                # Full DerivaML connection
                detected_domain = domain_schema or check_result.get("domain_schema")
                ml = DerivaML(
                    hostname=hostname,
                    catalog_id=catalog_id,
                    domain_schema=detected_domain,
                    check_auth=True,
                )

                # Create MCP workflow and execution for tracking operations
                workflow_rid, execution = self._create_mcp_execution(ml)

                conn_info = ConnectionInfo(
                    hostname=hostname,
                    catalog_id=catalog_id,
                    catalog_type=CatalogType.DERIVAML,
                    catalog=catalog,
                    ml_instance=ml,
                    domain_schema=ml.domain_schema,
                    workflow_rid=workflow_rid,
                    execution=execution,
                )
                logger.info(f"Connected to DerivaML catalog {key}")

            else:
                # Plain ERMrest connection
                if require_derivaml:
                    reason = check_result.get("reason", "Not a DerivaML catalog")
                    raise DerivaMLException(f"Catalog {hostname}:{catalog_id} is not DerivaML compliant: {reason}")

                conn_info = ConnectionInfo(
                    hostname=hostname,
                    catalog_id=catalog_id,
                    catalog_type=CatalogType.ERMREST,
                    catalog=catalog,
                    ml_instance=None,
                    domain_schema=domain_schema,
                    workflow_rid=None,
                    execution=None,
                )
                logger.info(f"Connected to ERMrest catalog {key} (non-DerivaML)")

            self._connections[key] = conn_info
            if set_active:
                self._active_connection = key

            return conn_info

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

        For DerivaML catalogs, properly closes the MCP execution context and
        optionally uploads any registered outputs.

        Args:
            hostname: Hostname to disconnect from. If None, uses active connection.
            catalog_id: Catalog to disconnect from. If None, uses active connection.
            upload_outputs: If True, upload any registered execution outputs (DerivaML only).

        Returns:
            True if disconnected successfully.
        """
        if hostname is None and catalog_id is None:
            key = self._active_connection
        else:
            key = self._connection_key(hostname or "", catalog_id or "")

        if key and key in self._connections:
            conn_info = self._connections[key]

            # Close the execution context if one exists (DerivaML only)
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

    def get_active(self) -> ConnectionInfo | None:
        """Get the active connection info.

        Returns:
            Active ConnectionInfo or None if no active connection.
        """
        if self._active_connection and self._active_connection in self._connections:
            return self._connections[self._active_connection]
        return None

    def get_active_or_raise(self) -> ConnectionInfo:
        """Get the active connection info or raise an error.

        Returns:
            Active ConnectionInfo.

        Raises:
            DerivaMLException: If no active connection.
        """
        from deriva_ml import DerivaMLException

        conn = self.get_active()
        if conn is None:
            raise DerivaMLException("No active catalog connection. Use 'connect_catalog' tool to connect first.")
        return conn

    def get_active_ml(self) -> "DerivaML | None":
        """Get the active DerivaML instance (if connected to DerivaML catalog).

        Returns:
            Active DerivaML instance or None if not connected or not DerivaML.
        """
        conn = self.get_active()
        if conn and conn.is_derivaml:
            return conn.ml_instance
        return None

    def get_active_ml_or_raise(self) -> "DerivaML":
        """Get the active DerivaML instance or raise an error.

        Returns:
            Active DerivaML instance.

        Raises:
            DerivaMLException: If no active connection or not a DerivaML catalog.
        """
        from deriva_ml import DerivaMLException

        ml = self.get_active_ml()
        if ml is None:
            conn = self.get_active()
            if conn is None:
                raise DerivaMLException("No active catalog connection. Use 'connect_catalog' tool first.")
            else:
                raise DerivaMLException(
                    f"Catalog {conn.hostname}:{conn.catalog_id} is not a DerivaML catalog. "
                    "This operation requires a DerivaML catalog with the ML schema."
                )
        return ml

    def get_active_execution(self) -> Any | None:
        """Get the active execution context (DerivaML only).

        Returns:
            Active Execution object or None if no active connection or no execution.
        """
        conn = self.get_active()
        if conn:
            return conn.execution
        return None

    def get_active_execution_or_raise(self) -> Any:
        """Get the active execution context or raise an error.

        Returns:
            Active Execution object.

        Raises:
            DerivaMLException: If no active connection or no execution.
        """
        from deriva_ml import DerivaMLException

        execution = self.get_active_execution()
        if execution is None:
            conn = self.get_active()
            if conn is None:
                raise DerivaMLException("No active catalog connection. Connect to a catalog first.")
            elif not conn.is_derivaml:
                raise DerivaMLException("Execution tracking requires a DerivaML catalog.")
            else:
                raise DerivaMLException("No active execution context. This shouldn't happen for DerivaML catalogs.")
        return execution

    def get_active_connection_info(self) -> ConnectionInfo | None:
        """Get the active connection info including workflow and execution.

        Returns:
            ConnectionInfo or None if no active connection.
        """
        return self.get_active()

    def get_active_connection(self) -> "DerivaML | None":
        """Get the active DerivaML instance (if connected to DerivaML catalog).

        DEPRECATED: Use get_active_ml() for DerivaML operations or
        get_active() for ConnectionInfo that works with both catalog types.

        For backward compatibility, this returns the DerivaML instance if
        connected to a DerivaML catalog, or None otherwise.

        Returns:
            Active DerivaML instance or None if not connected or not DerivaML.
        """
        return self.get_active_ml()

    def get_connection(self, hostname: str, catalog_id: str | int) -> ConnectionInfo | None:
        """Get a specific connection.

        Args:
            hostname: Server hostname.
            catalog_id: Catalog identifier.

        Returns:
            ConnectionInfo or None if not connected.
        """
        key = self._connection_key(hostname, catalog_id)
        return self._connections.get(key)

    def list_connections(self) -> list[dict[str, Any]]:
        """List all active connections.

        Returns:
            List of connection information dictionaries.
        """
        return [
            {
                "hostname": info.hostname,
                "catalog_id": info.catalog_id,
                "catalog_type": info.catalog_type.value,
                "is_derivaml": info.is_derivaml,
                "domain_schema": info.domain_schema,
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

    def require_derivaml(self, operation: str = "This operation") -> "DerivaML":
        """Require that the active catalog is DerivaML and return the ML instance.

        Convenience method for tools that require DerivaML functionality.

        Args:
            operation: Description of the operation for error messages.

        Returns:
            DerivaML instance.

        Raises:
            DerivaMLException: If not connected or not a DerivaML catalog.
        """
        from deriva_ml import DerivaMLException

        conn = self.get_active()
        if conn is None:
            raise DerivaMLException("No active catalog connection. Use 'connect_catalog' tool first.")
        if not conn.is_derivaml:
            raise DerivaMLException(
                f"{operation} requires a DerivaML catalog. "
                f"Catalog {conn.hostname}:{conn.catalog_id} is a plain ERMrest catalog. "
                "Use 'is_derivaml_catalog' to check catalog type."
            )
        return conn.ml_instance
