"""Connection management for Deriva MCP server.

This module handles Deriva catalog connections, maintaining
active connections and providing access to DerivaML instances.

Multi-user isolation:
- Each connection is keyed by (user_id, hostname, catalog_id) so two users
  connecting to the same catalog get separate ConnectionInfo objects.
- The active connection is tracked per async context using contextvars,
  so concurrent requests in HTTP transport mode don't interfere.
- User identity is derived from credentials at connect time and stored
  on ConnectionInfo for use by other modules (background tasks, executions).

HTTP transport fallback:
- With streamable HTTP transport, each HTTP request gets a fresh async
  context, so contextvar values set during connect_catalog are lost in
  subsequent requests. To handle this, the ConnectionManager uses a
  three-tier fallback when resolving the active connection:
  1. ContextVar (works in stdio and same-request context)
  2. Instance-level ``_last_active_key`` (survives across HTTP requests)
  3. Sole connection (if exactly one connection exists, use it)

When connecting to a catalog, an MCP workflow and execution are automatically
created to track all operations performed through the MCP server.
"""

from __future__ import annotations

import contextvars
import hashlib
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

# Per-request active connection tracking.
# In HTTP transport, each async request gets its own contextvar scope,
# preventing one user's connect() from overwriting another's active connection.
# In stdio transport, there's only one context so this behaves like a simple variable.
_active_connection_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_connection", default=None
)


def derive_user_id(credential: dict | None) -> str:
    """Derive a user identifier from Deriva credentials.

    Uses a hash of the webauthn cookie value for privacy.
    Falls back to "default_user" for single-user (stdio) mode.

    Args:
        credential: Credential dict from DerivaML (contains 'cookie' key).

    Returns:
        A string identifying the user.
    """
    if credential:
        cookie = credential.get("cookie", "")
        if "webauthn=" in cookie:
            webauthn = cookie.split("webauthn=")[1].split(";")[0]
            return hashlib.sha256(webauthn.encode()).hexdigest()[:16]
    return "default_user"


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
    all MCP operations performed on the catalog. The user_id field
    identifies who owns this connection for multi-user isolation.
    """

    hostname: str
    catalog_id: str | int
    domain_schemas: set[str] | None
    ml_instance: DerivaML
    user_id: str = "default_user"
    workflow_rid: str | None = None
    execution: Any = None  # MCP session execution from deriva_ml
    active_tool_execution: Any = None  # User-created execution via create_execution tool


class ConnectionManager:
    """Manages DerivaML catalog connections with multi-user isolation.

    Connections are keyed by (user_id, hostname, catalog_id) so different
    users connecting to the same catalog get separate state. The active
    connection is tracked per-request using contextvars, which prevents
    concurrent HTTP requests from interfering with each other.

    In stdio mode (one process per client), there's only one user and
    one context, so this behaves identically to a simple instance variable.

    For HTTP transport (streamable HTTP), each request gets a fresh async
    context so contextvar values are lost between requests. To handle this,
    the manager maintains a ``_last_active_key`` instance variable as a
    fallback, and will also auto-resolve when exactly one connection exists.
    See ``_resolve_active_key()`` for the full fallback chain.
    """

    def __init__(self) -> None:
        self._connections: dict[str, ConnectionInfo] = {}
        self._last_active_key: str | None = None

    @property
    def _active_connection(self) -> str | None:
        """Get the active connection key for the current request context."""
        return _active_connection_var.get()

    @_active_connection.setter
    def _active_connection(self, value: str | None) -> None:
        """Set the active connection key for the current request context."""
        _active_connection_var.set(value)
        if value is not None:
            self._last_active_key = value

    def _connection_key(self, hostname: str, catalog_id: str | int, user_id: str = "") -> str:
        """Generate a unique key for a connection.

        Includes user_id so two users on the same catalog get separate entries.
        """
        return f"{user_id}:{hostname}:{catalog_id}"

    def _resolve_active_key(self) -> str | None:
        """Resolve the active connection key with fallbacks for HTTP transport.

        With streamable HTTP transport, each request gets a fresh async context
        so the contextvar is lost. This method tries three strategies in order:

        1. ContextVar (works in stdio and same-request context)
        2. Last active key (instance var, survives across HTTP requests)
        3. Sole connection (if exactly one exists)

        Returns:
            The resolved connection key, or None if no connection can be determined.
        """
        key = self._active_connection
        if key and key in self._connections:
            return key
        if self._last_active_key and self._last_active_key in self._connections:
            return self._last_active_key
        if len(self._connections) == 1:
            return next(iter(self._connections.keys()))
        return None

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
        default_schema: str | None = None,
        set_active: bool = True,
    ) -> DerivaML:
        """Connect to a DerivaML catalog.

        Args:
            hostname: Hostname of the Deriva server.
            catalog_id: Catalog identifier.
            domain_schemas: Optional set of domain schema names. Auto-detected if omitted.
            default_schema: Optional default schema for table creation and lookups.
            set_active: If True, set this as the active connection.

        Returns:
            DerivaML instance for the catalog.

        Raises:
            DerivaMLException: If connection fails.
        """
        # Create new connection
        logger.info(f"Connecting to {hostname}, catalog {catalog_id}")
        try:
            ml = DerivaML(
                hostname=hostname,
                catalog_id=catalog_id,
                domain_schemas=domain_schemas,
                default_schema=default_schema,
                check_auth=True,
            )

            # Derive user identity from the connection's credentials
            user_id = derive_user_id(ml.credential)
            key = self._connection_key(hostname, catalog_id, user_id)

            # Return existing connection if available for this user
            if key in self._connections:
                if set_active:
                    self._active_connection = key
                logger.info(f"Reusing existing connection to {key}")
                return self._connections[key].ml_instance

            # Create MCP workflow and execution for tracking operations
            workflow_rid, execution = self._create_mcp_execution(ml)

            self._connections[key] = ConnectionInfo(
                hostname=hostname,
                catalog_id=catalog_id,
                domain_schemas=domain_schemas,
                ml_instance=ml,
                user_id=user_id,
                workflow_rid=workflow_rid,
                execution=execution,
            )
            if set_active:
                self._active_connection = key
            logger.info(f"Successfully connected to {key}")
            return ml
        except DerivaMLException:
            raise
        except Exception as e:
            logger.error(f"Failed to connect to {hostname}:{catalog_id}: {e}")
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
            key = self._resolve_active_key()
        else:
            # Find the connection key for this user+host+catalog
            conn_info = self._find_connection(hostname or "", catalog_id or "")
            key = None
            if conn_info:
                for k, v in self._connections.items():
                    if v is conn_info:
                        key = k
                        break

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
            if self._last_active_key == key:
                self._last_active_key = None
            logger.info(f"Disconnected from {key}")
            return True
        return False

    def _find_connection(self, hostname: str, catalog_id: str | int) -> ConnectionInfo | None:
        """Find a connection by hostname and catalog_id (any user).

        Used for disconnect when we don't have user_id handy.
        Prefers the active connection's user if available.
        """
        # Try active connection first (with fallback for HTTP transport)
        active = self._resolve_active_key()
        if active and active in self._connections:
            info = self._connections[active]
            if info.hostname == hostname and str(info.catalog_id) == str(catalog_id):
                return info

        # Fall back to any matching connection
        for info in self._connections.values():
            if info.hostname == hostname and str(info.catalog_id) == str(catalog_id):
                return info
        return None

    def get_active(self) -> DerivaML | None:
        """Get the active DerivaML instance.

        Uses ``_resolve_active_key()`` to find the active connection,
        falling back through instance-level and sole-connection strategies
        when the contextvar is lost (e.g., across HTTP requests).

        Returns:
            Active DerivaML instance or None if no active connection.
        """
        key = self._resolve_active_key()
        if key:
            return self._connections[key].ml_instance
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

        Uses ``_resolve_active_key()`` to find the active connection,
        falling back through instance-level and sole-connection strategies
        when the contextvar is lost (e.g., across HTTP requests).

        Returns:
            Active Execution object or None if no active connection or no execution.
        """
        key = self._resolve_active_key()
        if key:
            return self._connections[key].execution
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

        Uses ``_resolve_active_key()`` to find the active connection,
        falling back through instance-level and sole-connection strategies
        when the contextvar is lost (e.g., across HTTP requests).

        Returns:
            ConnectionInfo or None if no active connection.
        """
        key = self._resolve_active_key()
        if key:
            return self._connections[key]
        return None

    def get_active_connection_info_or_raise(self) -> ConnectionInfo:
        """Get the active connection info or raise an error.

        Returns:
            ConnectionInfo for the active connection.

        Raises:
            DerivaMLException: If no active connection.
        """
        info = self.get_active_connection_info()
        if info is None:
            raise DerivaMLException(
                "No active catalog connection. Use 'connect' tool to connect to a catalog first."
            )
        return info

    def get_connection(self, hostname: str, catalog_id: str | int) -> DerivaML | None:
        """Get a specific connection.

        Args:
            hostname: Server hostname.
            catalog_id: Catalog identifier.

        Returns:
            DerivaML instance or None if not connected.
        """
        conn = self._find_connection(hostname, catalog_id)
        return conn.ml_instance if conn else None

    def list_connections(self) -> list[dict[str, Any]]:
        """List all active connections.

        Returns:
            List of connection information dictionaries.
        """
        active_key = self._resolve_active_key()
        return [
            {
                "hostname": info.hostname,
                "catalog_id": info.catalog_id,
                "domain_schemas": list(info.domain_schemas) if info.domain_schemas else None,
                "is_active": key == active_key,
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
        conn = self._find_connection(hostname, catalog_id)
        if conn:
            for key, info in self._connections.items():
                if info is conn:
                    self._active_connection = key
                    return True
        return False
