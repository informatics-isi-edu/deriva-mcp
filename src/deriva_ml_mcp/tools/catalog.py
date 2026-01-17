"""Catalog connection and management tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from deriva.core import DerivaServer, get_credential

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-ml-mcp")


def _get_catalog_registry(hostname: str) -> list[dict[str, Any]]:
    """Query the catalog registry (catalog 0) to get all catalogs and aliases.

    Args:
        hostname: Server hostname to query.

    Returns:
        List of registry entries with catalog/alias information.
    """
    server = DerivaServer("https", hostname, credentials=get_credential(hostname))
    registry_catalog = server.connect_ermrest(0)
    pb = registry_catalog.getPathBuilder()
    registry = pb.schemas["ermrest"].tables["registry"]
    return list(registry.entities().fetch())


def register_catalog_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register catalog management tools with the MCP server."""

    @mcp.tool()
    async def connect_catalog(
        hostname: str,
        catalog_id: str,
        domain_schema: str | None = None,
    ) -> str:
        """Connect to an existing DerivaML catalog. Must be called before using other tools.

        On connection, an MCP workflow and execution are automatically created to
        track all operations performed through the MCP server. The workflow type
        "DerivaML MCP" is created if it doesn't exist.

        Args:
            hostname: Server hostname (e.g., "dev.eye-ai.org", "www.atlas-d2k.org").
            catalog_id: Catalog ID number (e.g., "1", "52").
            domain_schema: Schema name for domain tables. Auto-detected if omitted.

        Returns:
            JSON with status, hostname, catalog_id, domain_schema, project_name,
            workflow_rid, execution_rid.

        Example:
            connect_catalog("dev.eye-ai.org", "52") -> connects to eye-ai catalog
        """
        try:
            ml = conn_manager.connect(hostname, catalog_id, domain_schema)
            conn_info = conn_manager.get_active_connection_info()

            result = {
                "status": "connected",
                "hostname": hostname,
                "catalog_id": catalog_id,
                "domain_schema": ml.domain_schema,
                "project_name": ml.project_name,
            }

            # Add workflow/execution info if available
            if conn_info:
                result["workflow_rid"] = conn_info.workflow_rid
                result["execution_rid"] = (
                    conn_info.execution.execution_rid if conn_info.execution else None
                )

            return json.dumps(result)
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
        """Get details about the active catalog: hostname, schemas, project name, and MCP execution."""
        try:
            ml = conn_manager.get_active_or_raise()
            conn_info = conn_manager.get_active_connection_info()

            result = {
                "hostname": ml.host_name,
                "catalog_id": ml.catalog_id,
                "domain_schema": ml.domain_schema,
                "ml_schema": ml.ml_schema,
                "project_name": ml.project_name,
            }

            # Add workflow/execution info if available
            if conn_info:
                result["workflow_rid"] = conn_info.workflow_rid
                result["execution_rid"] = (
                    conn_info.execution.execution_rid if conn_info.execution else None
                )

            return json.dumps(result)
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
        catalog_alias: str | None = None,
    ) -> str:
        """Create a new DerivaML catalog with all ML schema tables.

        Creates a fresh catalog with Dataset, Execution, Workflow, Feature, and
        vocabulary tables. Automatically connects to the new catalog.

        Args:
            hostname: Server hostname (e.g., "localhost", "deriva.example.org").
            project_name: Name for the project, becomes the domain schema name.
            catalog_alias: Optional alias for the catalog. If provided, creates
                an alias that allows accessing the catalog by name instead of
                numeric ID (e.g., /ermrest/catalog/my-project instead of
                /ermrest/catalog/45).

        Returns:
            JSON with status, hostname, catalog_id, catalog_alias (if created),
            domain_schema, project_name.

        Example:
            create_catalog("localhost", "my_ml_project", "my-project")
        """
        try:
            from deriva.core.ermrest_model import Schema
            from deriva_ml.schema import create_ml_catalog

            # Create the catalog with deriva-ml schema (and optional alias)
            catalog = create_ml_catalog(hostname, project_name, catalog_alias=catalog_alias)

            # Create the domain schema (project_name becomes the domain schema)
            model = catalog.getCatalogModel()
            model.create_schema(Schema.define(project_name))

            # Connect to the newly created catalog
            ml = conn_manager.connect(hostname, str(catalog.catalog_id), project_name)

            result = {
                "status": "created",
                "hostname": hostname,
                "catalog_id": str(catalog.catalog_id),
                "domain_schema": ml.domain_schema,
                "project_name": project_name,
            }
            if catalog_alias:
                result["catalog_alias"] = catalog_alias

            return json.dumps(result)
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

    @mcp.tool()
    async def list_catalog_registry(hostname: str) -> str:
        """List all catalogs and aliases available on a Deriva server.

        Queries the ERMrest registry (catalog 0) to retrieve information about
        all catalogs and aliases on the server. This is useful for discovering
        available catalogs before connecting.

        Args:
            hostname: Server hostname (e.g., "www.eye-ai.org", "dev.eye-ai.org").

        Returns:
            JSON with:
            - catalogs: List of actual catalogs with id, name, description, is_persistent
            - aliases: List of aliases with id (alias name), alias_target (catalog id), name

        Example:
            list_catalog_registry("www.eye-ai.org")
            -> {"catalogs": [{"id": "2", "name": "eye-ai", ...}],
                "aliases": [{"id": "eye-ai", "alias_target": "2", ...}]}
        """
        try:
            entries = _get_catalog_registry(hostname)

            catalogs = []
            aliases = []

            for entry in entries:
                # Skip deleted entries
                if entry.get("deleted_on"):
                    continue

                if entry.get("is_catalog"):
                    catalogs.append({
                        "id": entry["id"],
                        "name": entry.get("name"),
                        "description": entry.get("description"),
                        "is_persistent": entry.get("is_persistent"),
                    })
                elif entry.get("alias_target"):
                    aliases.append({
                        "id": entry["id"],
                        "alias_target": entry["alias_target"],
                        "name": entry.get("name"),
                        "description": entry.get("description"),
                    })

            return json.dumps({
                "hostname": hostname,
                "catalogs": catalogs,
                "aliases": aliases,
            })
        except Exception as e:
            logger.error(f"Failed to list catalog registry: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

    @mcp.tool()
    async def create_catalog_alias(
        hostname: str,
        alias_name: str,
        catalog_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """Create an alias for an existing catalog.

        Aliases allow accessing a catalog by a memorable name instead of its
        numeric ID. For example, instead of /ermrest/catalog/21, you can use
        /ermrest/catalog/eye-ai.

        Args:
            hostname: Server hostname (e.g., "www.eye-ai.org").
            alias_name: The alias identifier (e.g., "my-project", "eye-ai").
                Must be unique on the server.
            catalog_id: The numeric ID of the catalog to alias.
            name: Optional display name for the alias.
            description: Optional description of the alias.

        Returns:
            JSON with status and alias details.

        Example:
            create_catalog_alias("localhost", "my-project", "45", "My ML Project")
            -> {"status": "created", "alias": "my-project", "target": "45"}
        """
        try:
            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            server.create_ermrest_alias(
                id=alias_name,
                alias_target=catalog_id,
                name=name,
                description=description,
            )
            return json.dumps({
                "status": "created",
                "hostname": hostname,
                "alias": alias_name,
                "target": catalog_id,
                "name": name,
                "description": description,
            })
        except Exception as e:
            logger.error(f"Failed to create catalog alias: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

    @mcp.tool()
    async def get_catalog_alias(
        hostname: str,
        alias_name: str,
    ) -> str:
        """Get metadata for a catalog alias.

        Retrieves the alias configuration including its target catalog and owner.

        Args:
            hostname: Server hostname (e.g., "www.eye-ai.org").
            alias_name: The alias identifier to look up.

        Returns:
            JSON with alias metadata: id, alias_target, owner.

        Example:
            get_catalog_alias("www.eye-ai.org", "my-project")
            -> {"id": "my-project", "alias_target": "45", "owner": [...]}
        """
        try:
            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            alias = server.connect_ermrest_alias(alias_name)
            metadata = alias.retrieve()
            return json.dumps({
                "hostname": hostname,
                **metadata,
            })
        except Exception as e:
            logger.error(f"Failed to get catalog alias: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

    @mcp.tool()
    async def update_catalog_alias(
        hostname: str,
        alias_name: str,
        alias_target: str | None = None,
        owner: list[str] | None = None,
    ) -> str:
        """Update an existing catalog alias.

        Can change the target catalog the alias points to, or update the owner ACL.

        Args:
            hostname: Server hostname (e.g., "www.eye-ai.org").
            alias_name: The alias identifier to update.
            alias_target: New target catalog ID. If None, target is unchanged.
                Pass empty string "" to unbind the alias from any catalog.
            owner: New owner ACL (list of user/group identifiers).
                If None, owner is unchanged.

        Returns:
            JSON with status and updated alias details.

        Example:
            update_catalog_alias("localhost", "my-project", alias_target="50")
            -> {"status": "updated", "alias": "my-project", "target": "50"}
        """
        try:
            from deriva.core.ermrest_model import NoChange

            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            alias = server.connect_ermrest_alias(alias_name)

            # Build update kwargs - only include changed values
            kwargs: dict[str, Any] = {}
            if alias_target is not None:
                # Empty string means unbind (set to None)
                kwargs["alias_target"] = alias_target if alias_target else None
            if owner is not None:
                kwargs["owner"] = owner

            if kwargs:
                alias.update(**kwargs)

            # Get updated metadata
            metadata = alias.retrieve()
            return json.dumps({
                "status": "updated",
                "hostname": hostname,
                **metadata,
            })
        except Exception as e:
            logger.error(f"Failed to update catalog alias: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

    @mcp.tool()
    async def delete_catalog_alias(
        hostname: str,
        alias_name: str,
    ) -> str:
        """Delete a catalog alias. The target catalog is NOT deleted.

        Args:
            hostname: Server hostname (e.g., "www.eye-ai.org").
            alias_name: The alias identifier to delete.

        Returns:
            JSON with deletion status.

        Example:
            delete_catalog_alias("localhost", "my-project")
            -> {"status": "deleted", "alias": "my-project"}
        """
        try:
            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            alias = server.connect_ermrest_alias(alias_name)
            alias.delete_ermrest_alias(really=True)
            return json.dumps({
                "status": "deleted",
                "hostname": hostname,
                "alias": alias_name,
            })
        except Exception as e:
            logger.error(f"Failed to delete catalog alias: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

    @mcp.tool()
    async def clone_catalog(
        hostname: str,
        source_catalog_id: str,
        copy_data: bool = True,
        copy_annotations: bool = True,
        copy_policy: bool = True,
        truncate_after: bool = True,
        exclude_schemas: list[str] | None = None,
        dest_name: str | None = None,
        dest_description: str | None = None,
    ) -> str:
        """Clone a catalog to create a new copy with the same structure and optionally data.

        Creates a new catalog as a clone of the source. Useful for creating test
        environments, backups, or development copies of production catalogs.

        Args:
            hostname: Server hostname (e.g., "www.eye-ai.org").
            source_catalog_id: ID of the catalog to clone.
            copy_data: If True (default), copy all table contents.
            copy_annotations: If True (default), copy all annotations.
            copy_policy: If True (default), copy ACL (access control) policies.
            truncate_after: If True (default), truncate destination history after cloning.
            exclude_schemas: List of schema names to exclude from cloning.
            dest_name: Optional name for the new catalog.
            dest_description: Optional description for the new catalog.

        Returns:
            JSON with status, source_catalog_id, and new dest_catalog_id.

        Example:
            clone_catalog("www.eye-ai.org", "21", dest_name="My Clone")
            -> {"status": "cloned", "source": "21", "dest_catalog_id": "45"}
        """
        try:
            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            source_catalog = server.connect_ermrest(source_catalog_id)

            # Build destination properties
            dst_properties: dict[str, Any] = {}
            if dest_name:
                dst_properties["name"] = dest_name
            if dest_description:
                dst_properties["description"] = dest_description

            # Clone the catalog
            dest_catalog = source_catalog.clone_catalog(
                dst_catalog=None,  # Create new catalog
                copy_data=copy_data,
                copy_annotations=copy_annotations,
                copy_policy=copy_policy,
                truncate_after=truncate_after,
                exclude_schemas=exclude_schemas,
                dst_properties=dst_properties if dst_properties else None,
            )

            return json.dumps({
                "status": "cloned",
                "hostname": hostname,
                "source_catalog_id": source_catalog_id,
                "dest_catalog_id": str(dest_catalog.catalog_id),
                "copy_data": copy_data,
                "copy_annotations": copy_annotations,
                "copy_policy": copy_policy,
            })
        except Exception as e:
            logger.error(f"Failed to clone catalog: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })
