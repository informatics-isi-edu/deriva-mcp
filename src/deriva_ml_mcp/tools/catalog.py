"""Catalog connection and management tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from deriva.core import DerivaServer, get_credential

from deriva_ml_mcp.tools.background_tasks import _resolve_hostname

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
        domain_schema: str | None = None,
        default_schema: str | None = None,
    ) -> str:
        """Connect to an existing DerivaML catalog. Must be called before using other tools.

        On connection, an MCP workflow and execution are automatically created to
        track all operations performed through the MCP server. The workflow type
        "DerivaML MCP" is created if it doesn't exist.

        Args:
            hostname: Server hostname (e.g., "dev.eye-ai.org", "www.atlas-d2k.org").
            catalog_id: Catalog ID number (e.g., "1", "52").
            domain_schema: Schema name for domain tables. Auto-detected if omitted.
            default_schema: Default schema for table creation and lookups. If omitted
                and there is exactly one domain schema, that schema is used. Required
                when multiple domain schemas exist and you want to avoid specifying
                the schema on every operation.

        Returns:
            JSON with status, hostname, catalog_id, domain_schemas, default_schema,
            project_name, workflow_rid, execution_rid.

        Example:
            connect_catalog("dev.eye-ai.org", "52") -> connects to eye-ai catalog
            connect_catalog("localhost", "10", domain_schema="isa", default_schema="isa")
        """
        try:
            # Resolve hostname for Docker network environments
            resolved_hostname = _resolve_hostname(hostname) or hostname
            # Convert single domain_schema to set for the new API
            domain_schemas = {domain_schema} if domain_schema else None
            ml = conn_manager.connect(resolved_hostname, catalog_id, domain_schemas,
                                      default_schema=default_schema)
            conn_info = conn_manager.get_active_connection_info()

            result = {
                "status": "connected",
                "hostname": hostname,
                "catalog_id": catalog_id,
                "domain_schemas": list(ml.domain_schemas),
                "default_schema": ml.default_schema,
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
    async def set_default_schema(schema_name: str) -> str:
        """Set the default schema for the active catalog connection.

        When a catalog has multiple domain schemas, many operations require
        knowing which schema to use. Setting a default schema avoids having
        to specify it on every call.

        Args:
            schema_name: Name of the domain schema to set as default
                (e.g., "isa", "my_project"). Must be one of the catalog's
                domain schemas.

        Returns:
            JSON with status, default_schema, and domain_schemas.

        Example:
            set_default_schema("isa") -> sets "isa" as the default schema
        """
        try:
            ml = conn_manager.get_active_or_raise()
            if schema_name not in ml.domain_schemas:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"'{schema_name}' is not a domain schema. "
                        f"Available domain schemas: {sorted(ml.domain_schemas)}"
                    ),
                })
            ml.model.default_schema = schema_name
            ml.default_schema = schema_name
            return json.dumps({
                "status": "success",
                "default_schema": schema_name,
                "domain_schemas": sorted(ml.domain_schemas),
            })
        except Exception as e:
            logger.error(f"Failed to set default schema: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

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
            ml = conn_manager.connect(hostname, str(catalog.catalog_id), {project_name})

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
        source_hostname: str,
        source_catalog_id: str,
        root_rid: str,
        dest_hostname: str | None = None,
        alias: str | None = None,
        add_ml_schema: bool = True,
        asset_mode: str = "refs",
        copy_annotations: bool = True,
        copy_policy: bool = True,
        exclude_schemas: list[str] | None = None,
        exclude_objects: list[str] | None = None,
        reinitialize_dataset_versions: bool = True,
        orphan_strategy: str = "fail",
        prune_hidden_fkeys: bool = False,
        truncate_oversized: bool = False,
        include_tables: list[str] | None = None,
        include_associations: bool = True,
        include_vocabularies: bool = True,
    ) -> str:
        """Create an ML workspace by cloning data reachable from a root RID.

        Creates a partial catalog clone containing only data reachable from the
        root RID (e.g., a project, dataset, or experiment). Uses the root table's
        export annotation (if available) to determine which tables and paths to
        follow, then fills in any uncovered tables (vocabularies, associations).

        Uses a three-stage approach:
        1. Create schema WITHOUT foreign keys (only for included tables)
        2. Copy data asynchronously (export paths + fill-in tables)
        3. Apply foreign keys, handling violations based on orphan_strategy

        **Asset handling modes**:
        - "none": Don't copy assets (asset columns will be empty)
        - "refs": Copy asset URLs only, files stay on source server (default)
        - "full": Download and re-upload all assets (fully independent clone)

        **Orphan handling**:
        When source catalog policies hide some data but not references to it,
        cloning can result in dangling foreign keys. The orphan_strategy controls
        how these are handled.

        Args:
            source_hostname: Source server hostname (e.g., "www.facebase.org").
            source_catalog_id: ID of the catalog to clone.
            root_rid: The starting RID from which to trace reachability
                (e.g., a project RID like "3-HXMC").
            dest_hostname: Destination hostname. If None, uses source hostname.
            alias: Optional alias name for the new catalog.
            add_ml_schema: If True, add the DerivaML schema to the clone.
            asset_mode: How to handle assets: "none", "refs" (default), or "full".
            copy_annotations: If True (default), copy all annotations.
            copy_policy: If True (default), copy ACL policies.
            exclude_schemas: List of schema names to exclude from cloning.
            exclude_objects: List of tables ("schema:table" format) to exclude.
            reinitialize_dataset_versions: If True (default), reinitialize dataset versions.
            orphan_strategy: How to handle orphan rows: "fail", "delete", or "nullify".
            prune_hidden_fkeys: If True, skip FKs with hidden reference data.
            truncate_oversized: If True, truncate values exceeding index size limits.
            include_tables: Additional tables to include.
            include_associations: If True, auto-include association tables.
            include_vocabularies: If True, auto-include vocabulary tables.

        Returns:
            JSON with status, source info, destination info, and
            operation details including tables restored and orphan handling stats.

        Examples:
            clone_catalog("www.facebase.org", "1",
                          root_rid="3-HXMC",
                          dest_hostname="localhost",
                          alias="facebase-musmorph",
                          add_ml_schema=True,
                          orphan_strategy="delete")
        """
        try:
            from deriva_ml.catalog import AssetCopyMode, OrphanStrategy, create_ml_workspace

            # Convert string parameters to enums
            asset_mode_enum = AssetCopyMode(asset_mode)
            orphan_strategy_enum = OrphanStrategy(orphan_strategy)

            # Resolve hostnames for Docker network environments
            resolved_dest = _resolve_hostname(dest_hostname)
            resolved_source = _resolve_hostname(source_hostname)
            source_credential = get_credential(source_hostname) if resolved_source != source_hostname else None
            dest_credential = get_credential(dest_hostname) if resolved_dest != dest_hostname else None

            result = create_ml_workspace(
                source_hostname=resolved_source,
                source_catalog_id=source_catalog_id,
                root_rid=root_rid,
                include_tables=include_tables,
                exclude_objects=exclude_objects,
                exclude_schemas=exclude_schemas,
                include_associations=include_associations,
                include_vocabularies=include_vocabularies,
                dest_hostname=resolved_dest,
                alias=alias,
                add_ml_schema=add_ml_schema,
                asset_mode=asset_mode_enum,
                copy_annotations=copy_annotations,
                copy_policy=copy_policy,
                source_credential=source_credential,
                dest_credential=dest_credential,
                orphan_strategy=orphan_strategy_enum,
                prune_hidden_fkeys=prune_hidden_fkeys,
                truncate_oversized=truncate_oversized,
                reinitialize_dataset_versions=reinitialize_dataset_versions,
            )

            # Build response from CloneCatalogResult
            response: dict[str, Any] = {
                "status": "cloned",
                "clone_mode": "partial",
                "source_hostname": source_hostname,
                "source_catalog_id": source_catalog_id,
                "dest_hostname": result.hostname,
                "dest_catalog_id": result.catalog_id,
                "root_rid": root_rid,
                "asset_mode": asset_mode,
            }

            if result.source_snapshot:
                response["source_snapshot"] = result.source_snapshot
            if alias:
                response["alias"] = alias
            if result.datasets_reinitialized:
                response["datasets_reinitialized"] = result.datasets_reinitialized
            if result.ml_schema_added:
                response["ml_schema_added"] = result.ml_schema_added

            # Include stats from report
            if result.report:
                response["orphan_rows_removed"] = result.report.summary.orphan_rows_removed
                response["orphan_rows_nullified"] = result.report.summary.orphan_rows_nullified
                response["fkeys_pruned"] = result.report.summary.fkeys_pruned
                if result.truncated_values:
                    response["truncated_values_count"] = len(result.truncated_values)
                # Include detailed report
                response["report"] = {
                    "summary": {
                        "total_issues": result.report.summary.total_issues,
                        "errors": result.report.summary.errors,
                        "warnings": result.report.summary.warnings,
                        "tables_restored": result.report.summary.tables_restored,
                        "tables_failed": result.report.summary.tables_failed,
                        "tables_skipped": result.report.summary.tables_skipped,
                        "total_rows_restored": result.report.summary.total_rows_restored,
                        "orphan_rows_removed": result.report.summary.orphan_rows_removed,
                        "orphan_rows_nullified": result.report.summary.orphan_rows_nullified,
                        "fkeys_applied": result.report.summary.fkeys_applied,
                        "fkeys_failed": result.report.summary.fkeys_failed,
                        "fkeys_pruned": result.report.summary.fkeys_pruned,
                    },
                    "issues": [
                        {
                            "severity": issue.severity.value,
                            "category": issue.category.value,
                            "message": issue.message,
                            "table": issue.table,
                            "details": issue.details,
                            "action": issue.action,
                            "row_count": issue.row_count,
                        }
                        for issue in result.report.issues
                    ],
                    "tables_restored": result.report.tables_restored,
                    "tables_failed": result.report.tables_failed,
                    "tables_skipped": result.report.tables_skipped,
                    "orphan_details": result.report.orphan_details,
                }
                response["clone_type"] = "cross_server" if dest_hostname and dest_hostname != source_hostname else "same_server"
                response["message"] = (
                    f"Workspace created from {source_hostname}:{source_catalog_id} (root: {root_rid}) "
                    f"to {result.hostname}:{result.catalog_id}"
                )
                response["report_summary"] = result.report.to_text()

            return json.dumps(response)

        except Exception as e:
            logger.error(f"Failed to create workspace: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

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

            ml = conn_manager.get_active_or_raise()
            result = do_validate(
                ml,
                dataset_rids=dataset_rids,
                asset_rids=asset_rids,
                dataset_versions=dataset_versions,
                workflow_rids=workflow_rids,
                execution_rids=execution_rids,
                warn_missing_descriptions=warn_missing_descriptions,
            )

            return json.dumps({
                "is_valid": result.is_valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "validated_rids": result.validated_rids,
                "summary": str(result),  # Include formatted summary
            })
        except Exception as e:
            logger.error(f"Failed to validate RIDs: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

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
            ml = conn_manager.get_active_or_raise()
            url = ml.cite(rid, current=current)

            return json.dumps({
                "url": url,
                "rid": rid,
                "is_snapshot": not current,
            })
        except Exception as e:
            logger.error(f"Failed to generate citation URL: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
            })

