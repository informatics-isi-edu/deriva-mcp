"""Catalog connection and management tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from deriva.core import DerivaServer, get_credential

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-mcp")


def register_catalog_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register catalog management tools with the MCP server."""

    @mcp.tool()
    async def connect_catalog(
        hostname: str,
        catalog_id: str,
        default_schema: str | None = None,
    ) -> str:
        """Connect to an existing Deriva catalog (DerivaML or plain ERMrest).

        Automatically detects whether the catalog is DerivaML-compliant. For
        DerivaML catalogs, creates an MCP workflow and execution to track all
        operations. For plain ERMrest catalogs, provides basic catalog access.

        **DerivaML Catalogs**: Full access to datasets, features, executions,
        workflows, and ML provenance tracking.

        **Plain ERMrest Catalogs**: Access to schema, data query/insert,
        and display annotations. No dataset/feature/execution support.

        Use `is_derivaml_catalog` to check catalog type before connecting.

        Args:
            hostname: Server hostname (e.g., "dev.eye-ai.org", "www.atlas-d2k.org").
            catalog_id: Catalog ID number (e.g., "1", "52").
            default_schema: Default schema for table creation when multiple domain
                schemas exist. Auto-detected if catalog has exactly one domain schema.

        Returns:
            JSON with status, hostname, catalog_id, catalog_type, and additional
            fields depending on catalog type:
            - DerivaML: domain_schemas, default_schema, project_name, workflow_rid, execution_rid
            - ERMrest: available schemas list

        Example:
            connect_catalog("dev.eye-ai.org", "52") -> connects to eye-ai catalog
            connect_catalog("www.example.org", "1") -> connects to ERMrest catalog
        """
        try:
            conn_info = conn_manager.connect(hostname, catalog_id, default_schema)

            result = {
                "status": "connected",
                "hostname": hostname,
                "catalog_id": catalog_id,
                "catalog_type": conn_info.catalog_type.value,
                "is_derivaml": conn_info.is_derivaml,
            }

            if conn_info.is_derivaml:
                # DerivaML catalog - include ML-specific info
                ml = conn_info.ml_instance
                result["domain_schemas"] = list(ml.domain_schemas)
                result["default_schema"] = ml.default_schema
                result["project_name"] = ml.project_name
                result["workflow_rid"] = conn_info.workflow_rid
                result["execution_rid"] = conn_info.execution.execution_rid if conn_info.execution else None
            else:
                # Plain ERMrest catalog - include basic schema info
                model = conn_info.get_model()
                result["schemas"] = list(model.schemas.keys())
                result["domain_schemas"] = conn_info.domain_schemas
                result["default_schema"] = conn_info.default_schema
                result["note"] = (
                    "This is a plain ERMrest catalog. Dataset, feature, and execution "
                    "tools are not available. Use schema, data, and annotation tools."
                )

            return json.dumps(result)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def disconnect_catalog() -> str:
        """Disconnect from the currently active catalog."""
        if conn_manager.disconnect():
            return json.dumps({"status": "disconnected"})
        return json.dumps({"status": "no_active_connection"})

    @mcp.tool()
    async def set_active_catalog(hostname: str, catalog_id: str) -> str:
        """Switch the active catalog when multiple catalogs are connected.

        Args:
            hostname: Server hostname of the catalog to activate.
            catalog_id: Catalog ID to activate.
        """
        if conn_manager.set_active(hostname, catalog_id):
            return json.dumps(
                {
                    "status": "success",
                    "active_catalog": f"{hostname}:{catalog_id}",
                }
            )
        return json.dumps(
            {
                "status": "error",
                "message": f"No connection found for {hostname}:{catalog_id}",
            }
        )

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

            # Connect to the newly created catalog (project_name is used as default_schema)
            conn_info = conn_manager.connect(hostname, str(catalog.catalog_id), project_name)
            ml = conn_info.ml_instance

            result = {
                "status": "created",
                "hostname": hostname,
                "catalog_id": str(catalog.catalog_id),
                "domain_schemas": list(ml.domain_schemas),
                "default_schema": ml.default_schema,
                "project_name": project_name,
            }
            if catalog_alias:
                result["catalog_alias"] = catalog_alias

            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to create catalog: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def add_ml_schema(
        hostname: str,
        catalog_id: str,
        project_name: str | None = None,
        navbar_brand_text: str = "ML Data Browser",
        head_title: str = "Catalog ML",
    ) -> str:
        """Add the DerivaML schema to an existing ERMrest catalog and initialize it.

        Transforms a plain ERMrest catalog into a DerivaML-compliant catalog by
        adding the 'deriva-ml' schema with all required ML tables (Dataset,
        Execution, Workflow, etc.). The existing schemas and data are preserved.
        Also applies catalog annotations to configure the Chaise web interface.

        Use this when you have an existing catalog with domain data and want to
        add ML workflow tracking capabilities.

        **What gets created:**
        - 'deriva-ml' schema with core ML tables
        - Dataset and Dataset_Version tables for data management
        - Execution and Workflow tables for provenance tracking
        - Execution_Asset and Execution_Metadata for artifact storage
        - Required vocabulary tables (Feature_Name, Asset_Type, Asset_Role, etc.)

        **What gets configured:**
        - Navigation bar with organized menus for ML tables, domain tables, vocabularies, assets
        - Display settings (underscores as spaces, system columns, faceted search)
        - Bulk upload configuration for asset tables
        - Default landing page set to Dataset table

        **Prerequisites:**
        - The catalog must not already have a 'deriva-ml' schema
        - You must have schema creation permissions on the catalog

        Args:
            hostname: Server hostname (e.g., "www.example.org").
            catalog_id: Catalog ID number to add the ML schema to.
            project_name: Optional project name for CURIE prefixes in vocabularies.
                Defaults to "deriva-ml" if not provided.
            navbar_brand_text: Text shown in the navigation bar (default: "ML Data Browser").
            head_title: Browser tab title (default: "Catalog ML").

        Returns:
            JSON with:
            - status: "success" or "error"
            - hostname: Server hostname
            - catalog_id: Catalog ID
            - ml_schema: Name of the created ML schema
            - tables_created: List of tables created
            - annotations_applied: Whether catalog annotations were applied

        Example:
            add_ml_schema("www.example.org", "42", "my_project")
            -> Adds deriva-ml schema to catalog 42 and initializes it
        """
        try:
            from deriva.core import DerivaServer, get_credential
            from deriva_ml import DerivaML
            from deriva_ml.schema import create_ml_schema

            # Connect to the catalog
            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            catalog = server.connect_ermrest(catalog_id)
            model = catalog.getCatalogModel()

            # Check if ML schema already exists
            if "deriva-ml" in model.schemas:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Catalog already has a 'deriva-ml' schema. "
                        "Use is_derivaml_catalog to check catalog type.",
                    }
                )

            # Create the ML schema
            create_ml_schema(catalog, schema_name="deriva-ml", project_name=project_name)

            # Get list of created tables
            model = catalog.getCatalogModel()
            ml_schema_obj = model.schemas.get("deriva-ml")
            tables_created = list(ml_schema_obj.tables.keys()) if ml_schema_obj else []

            # Initialize the catalog by applying annotations
            # Create a DerivaML instance to apply annotations
            ml = DerivaML(hostname=hostname, catalog_id=catalog_id)
            ml.apply_catalog_annotations(
                navbar_brand_text=navbar_brand_text,
                head_title=head_title,
            )

            return json.dumps(
                {
                    "status": "success",
                    "hostname": hostname,
                    "catalog_id": catalog_id,
                    "ml_schema": "deriva-ml",
                    "tables_created": tables_created,
                    "annotations_applied": True,
                    "navbar_brand_text": navbar_brand_text,
                    "head_title": head_title,
                    "message": "DerivaML schema added and initialized successfully. "
                    "Use connect_catalog to connect with full ML capabilities.",
                }
            )
        except Exception as e:
            logger.error(f"Failed to add ML schema: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

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

            return json.dumps(
                {
                    "status": "deleted",
                    "hostname": hostname,
                    "catalog_id": catalog_id,
                }
            )
        except Exception as e:
            logger.error(f"Failed to delete catalog: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

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
            ml = conn_manager.require_derivaml("apply_catalog_annotations")
            ml.apply_catalog_annotations(
                navbar_brand_text=navbar_brand_text,
                head_title=head_title,
            )
            return json.dumps(
                {
                    "status": "success",
                    "navbar_brand_text": navbar_brand_text,
                    "head_title": head_title,
                    "message": "Catalog annotations applied. Navigation bar and display settings configured.",
                }
            )
        except Exception as e:
            logger.error(f"Failed to apply catalog annotations: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

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
            return json.dumps(
                {
                    "status": "created",
                    "hostname": hostname,
                    "alias": alias_name,
                    "target": catalog_id,
                    "name": name,
                    "description": description,
                }
            )
        except Exception as e:
            logger.error(f"Failed to create catalog alias: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

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
            return json.dumps(
                {
                    "status": "updated",
                    "hostname": hostname,
                    **metadata,
                }
            )
        except Exception as e:
            logger.error(f"Failed to update catalog alias: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

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
            return json.dumps(
                {
                    "status": "deleted",
                    "hostname": hostname,
                    "alias": alias_name,
                }
            )
        except Exception as e:
            logger.error(f"Failed to delete catalog alias: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

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

            return json.dumps(
                {
                    "status": "cloned",
                    "hostname": hostname,
                    "source_catalog_id": source_catalog_id,
                    "dest_catalog_id": str(dest_catalog.catalog_id),
                    "copy_data": copy_data,
                    "copy_annotations": copy_annotations,
                    "copy_policy": copy_policy,
                }
            )
        except Exception as e:
            logger.error(f"Failed to clone catalog: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def migrate_catalog(
        source_hostname: str,
        source_catalog_id: str,
        dest_hostname: str,
        dest_catalog_id: str | None = None,
        add_ml_schema: bool = False,
        copy_data: bool = True,
        copy_annotations: bool = True,
        copy_policy: bool = True,
        exclude_schemas: list[str] | None = None,
    ) -> str:
        """Migrate a catalog from one server to another, leaving assets in the source hatrac.

        This tool performs a cross-server catalog migration by:
        1. Creating a backup of the source catalog (schema + data, with asset references)
        2. Restoring the backup to a new catalog on the destination server
        3. Optionally adding the DerivaML schema to the migrated catalog

        **Key Feature**: Asset files (images, model weights, etc.) remain stored in the
        source server's hatrac storage. The migrated catalog contains references to those
        assets via their original URLs. This allows migration without downloading/uploading
        large asset files.

        **Use Cases**:
        - Move a catalog from a development server to production
        - Create a catalog on a new server while keeping assets on the original
        - Add DerivaML capabilities to an existing catalog during migration

        **Prerequisites**:
        - Valid credentials for both source and destination servers
        - Write permissions on the destination server to create catalogs
        - Read permissions on the source catalog

        Args:
            source_hostname: Source server hostname (e.g., "dev.example.org").
            source_catalog_id: ID of the catalog to migrate.
            dest_hostname: Destination server hostname (e.g., "prod.example.org").
            dest_catalog_id: Optional destination catalog ID. If None, creates a new catalog.
            add_ml_schema: If True, add the DerivaML schema after migration.
                Use this when migrating a plain ERMrest catalog that you want to
                convert to a DerivaML catalog.
            copy_data: If True (default), copy all table data.
            copy_annotations: If True (default), copy all annotations.
            copy_policy: If True (default), copy ACL (access control) policies.
            exclude_schemas: List of schema names to exclude from migration.

        Returns:
            JSON with:
            - status: "success" or "error"
            - source_hostname: Source server
            - source_catalog_id: Source catalog ID
            - dest_hostname: Destination server
            - dest_catalog_id: New catalog ID on destination
            - ml_schema_added: Whether DerivaML schema was added
            - message: Summary of migration

        Example:
            migrate_catalog(
                "dev.example.org", "21",
                "prod.example.org",
                add_ml_schema=True
            )
            -> Migrates catalog 21 to prod server and adds DerivaML schema

        Note:
            This operation may take several minutes for large catalogs.
            The backup is created in a temporary directory and cleaned up after migration.
        """
        import tempfile
        from pathlib import Path

        from deriva.transfer.backup import DerivaBackup
        from deriva.transfer.restore import DerivaRestore

        try:
            # Create a temporary directory for the backup
            with tempfile.TemporaryDirectory() as tmpdir:
                backup_dir = Path(tmpdir) / "backup"
                backup_dir.mkdir()

                logger.info(f"Backing up catalog {source_hostname}:{source_catalog_id}")

                # Step 1: Backup the source catalog
                # Use include_assets="references" to store asset URLs without downloading files
                backup_args = {
                    "host": source_hostname,
                    "protocol": "https",
                    "catalog_id": source_catalog_id,
                }
                backup = DerivaBackup(
                    backup_args,
                    output_dir=str(backup_dir),
                    no_data=not copy_data,
                    include_assets="references",  # Store asset URLs, don't download files
                    exclude_data=exclude_schemas if exclude_schemas else [],
                )
                backup.transfer()

                # Find the backup output (it creates a bag in the output dir)
                bag_dirs = list(backup_dir.glob("*"))
                if not bag_dirs:
                    return json.dumps(
                        {
                            "status": "error",
                            "message": "Backup failed: no output created",
                        }
                    )
                bag_path = bag_dirs[0]

                logger.info(f"Restoring to {dest_hostname}")

                # Step 2: Restore to destination server
                restore_args = {
                    "host": dest_hostname,
                    "protocol": "https",
                    "catalog_id": dest_catalog_id,  # None creates new catalog
                }
                restore = DerivaRestore(
                    restore_args,
                    input_path=str(bag_path),
                    no_data=not copy_data,
                    no_annotations=not copy_annotations,
                    no_policy=not copy_policy,
                    no_assets=True,  # Don't try to restore assets (they're references)
                    exclude_schemas=exclude_schemas if exclude_schemas else [],
                )
                restore.restore()

                # Get the destination catalog ID
                dest_id = str(restore.dst_catalog.catalog_id) if restore.dst_catalog else dest_catalog_id

                result = {
                    "status": "success",
                    "source_hostname": source_hostname,
                    "source_catalog_id": source_catalog_id,
                    "dest_hostname": dest_hostname,
                    "dest_catalog_id": dest_id,
                    "copy_data": copy_data,
                    "copy_annotations": copy_annotations,
                    "copy_policy": copy_policy,
                    "ml_schema_added": False,
                }

                # Step 3: Optionally add DerivaML schema
                if add_ml_schema and dest_id:
                    try:
                        from deriva.core import DerivaServer
                        from deriva_ml.schema import create_ml_schema

                        server = DerivaServer(
                            "https", dest_hostname, credentials=get_credential(dest_hostname)
                        )
                        dest_catalog = server.connect_ermrest(dest_id)
                        model = dest_catalog.getCatalogModel()

                        # Only add if not already present
                        if "deriva-ml" not in model.schemas:
                            create_ml_schema(dest_catalog, schema_name="deriva-ml")
                            result["ml_schema_added"] = True
                            result["message"] = (
                                f"Successfully migrated catalog to {dest_hostname}:{dest_id} "
                                "and added DerivaML schema. "
                                "Use connect_catalog to connect with full ML capabilities."
                            )
                        else:
                            result["message"] = (
                                f"Successfully migrated catalog to {dest_hostname}:{dest_id}. "
                                "DerivaML schema was already present."
                            )
                    except Exception as e:
                        result["ml_schema_error"] = str(e)
                        result["message"] = (
                            f"Catalog migrated to {dest_hostname}:{dest_id}, "
                            f"but failed to add ML schema: {e}"
                        )
                else:
                    result["message"] = (
                        f"Successfully migrated catalog to {dest_hostname}:{dest_id}. "
                        "Asset files remain on the source server's hatrac storage."
                    )

                return json.dumps(result)

        except Exception as e:
            logger.error(f"Failed to migrate catalog: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def validate_rids(
        dataset_rids: list[str] | None = None,
        asset_rids: list[str] | None = None,
        dataset_versions: dict[str, str] | None = None,
        workflow_rids: list[str] | None = None,
        execution_rids: list[str] | None = None,
        warn_missing_descriptions: bool = True,
    ) -> str:
        """Validate that RIDs exist in the catalog before running experiments.

        Performs batch validation of RIDs to catch configuration errors early with
        clear error messages. Use this before running experiments to ensure all
        referenced datasets, assets, and other entities actually exist.

        Args:
            dataset_rids: List of dataset RIDs to validate.
            asset_rids: List of asset RIDs to validate (model weights, etc.).
            dataset_versions: Dictionary mapping dataset RID to required version
                string (e.g., {"1-ABC": "0.4.0"}). Validates version exists.
            workflow_rids: List of workflow RIDs to validate.
            execution_rids: List of execution RIDs to validate.
            warn_missing_descriptions: If True (default), include warnings for
                datasets missing descriptions.

        Returns:
            JSON with:
            - is_valid: True if all validations passed
            - errors: List of error messages
            - warnings: List of warning messages
            - validated_rids: Dictionary of validated RID info

        Example:
            validate_rids(
                dataset_rids=["1-ABC", "2-DEF"],
                dataset_versions={"1-ABC": "0.4.0"},
                asset_rids=["3-GHI"]
            )
            -> {
                "is_valid": true,
                "errors": [],
                "warnings": [],
                "validated_rids": {...}
            }
        """
        try:
            from deriva_ml.core.validation import validate_rids as do_validate

            ml = conn_manager.require_derivaml("validate_rids")
            result = do_validate(
                ml,
                dataset_rids=dataset_rids,
                asset_rids=asset_rids,
                dataset_versions=dataset_versions,
                workflow_rids=workflow_rids,
                execution_rids=execution_rids,
                warn_missing_descriptions=warn_missing_descriptions,
            )

            return json.dumps(
                {
                    "is_valid": result.is_valid,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "validated_rids": result.validated_rids,
                    "summary": str(result),  # Include formatted summary
                }
            )
        except Exception as e:
            logger.error(f"Failed to validate RIDs: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def cite(rid: str, current: bool = False) -> str:
        """Generate a citation URL for a catalog entity.

        Creates a permanent, citable URL for any catalog entity (dataset,
        execution, asset, etc.). By default, includes a snapshot timestamp
        for reproducibility. Use current=True for a link to the live data.

        Args:
            rid: RID of the entity to cite (e.g., "1-ABC").
            current: If True, return URL to current state without snapshot.
                If False (default), return permanent URL with snapshot timestamp.

        Returns:
            JSON with:
            - url: The citation URL
            - rid: The entity RID
            - is_snapshot: Whether URL includes snapshot timestamp

        Examples:
            cite("1-ABC")
            -> {"url": "https://host/id/catalog/1-ABC@2024-01-15", "is_snapshot": true}

            cite("1-ABC", current=True)
            -> {"url": "https://host/id/catalog/1-ABC", "is_snapshot": false}
        """
        try:
            ml = conn_manager.require_derivaml("cite")
            url = ml.cite(rid, current=current)

            return json.dumps(
                {
                    "url": url,
                    "rid": rid,
                    "is_snapshot": not current,
                }
            )
        except Exception as e:
            logger.error(f"Failed to generate citation URL: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def is_derivaml_catalog(
        hostname: str,
        catalog_id: str,
    ) -> str:
        """Check if a catalog is a DerivaML catalog.

        Tests whether a catalog has the DerivaML schema and required tables.
        Use this before connecting to understand what features are available.

        A DerivaML catalog has:
        - 'deriva-ml' schema with Dataset, Execution, Workflow, etc. tables
        - Support for datasets, features, executions, and workflow tracking
        - Full ML provenance capabilities

        A plain ERMrest catalog:
        - Supports basic table operations, queries, and annotations
        - Does NOT support datasets, features, or execution tracking
        - Can still use schema, data, and annotation tools

        Args:
            hostname: Server hostname (e.g., "dev.eye-ai.org").
            catalog_id: Catalog ID number (e.g., "1", "52").

        Returns:
            JSON with:
            - is_derivaml: True if catalog has the DerivaML schema
            - ml_schema: Name of the ML schema if found
            - domain_schemas: List of auto-detected domain schema names (if DerivaML)
            - reason: Explanation if not DerivaML
            - missing_tables: List of missing required tables (if partial)
            - available_tools: Summary of available functionality

        Examples:
            is_derivaml_catalog("dev.eye-ai.org", "52")
            -> {"is_derivaml": true, "ml_schema": "deriva-ml", "domain_schemas": ["eye-ai"]}

            is_derivaml_catalog("www.example.org", "1")
            -> {"is_derivaml": false, "reason": "Schema 'deriva-ml' not found"}
        """
        from deriva_ml_mcp.connection import check_is_derivaml_catalog

        try:
            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            catalog = server.connect_ermrest(catalog_id)

            result = check_is_derivaml_catalog(catalog)

            # Add available tools summary
            if result["is_derivaml"]:
                result["available_tools"] = {
                    "datasets": "create_dataset, add_dataset_members, download_dataset, etc.",
                    "features": "create_feature, add_feature_value, etc.",
                    "executions": "create_execution, upload_execution_outputs, etc.",
                    "workflows": "create_workflow, find_workflows, etc.",
                    "vocabularies": "add_term, create_vocabulary, etc.",
                    "schema": "create_table, create_asset_table, etc.",
                    "data": "query_table, insert_records, etc.",
                    "annotations": "set_visible_columns, set_table_display, etc.",
                }
            else:
                result["available_tools"] = {
                    "schema": "create_table (limited), get_schema_description",
                    "data": "query_table, insert_records, get_record, update_record",
                    "annotations": "set_visible_columns, set_table_display, etc.",
                    "note": "Dataset, feature, and execution tools require a DerivaML catalog",
                }

            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to check catalog type: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )
