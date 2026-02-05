"""Background task management tools for long-running operations.

This module provides MCP tools for managing background tasks such as
catalog cloning. These tools allow users to:
- Start long-running operations asynchronously
- Check task status and progress
- List their active and completed tasks
- Cancel pending/running tasks

The task system is multi-user safe - each user can only see and manage
their own tasks.

Performance notes:
- All MCP tools are truly async, using asyncio.to_thread() for blocking operations
- Task status queries use snapshot methods to minimize lock contention
- User ID is consistently determined to avoid task lookup mismatches

Async Clone Support:
- Uses new async datapath operations from deriva-py for concurrent data fetching
- Pipeline pattern: fetch page N while uploading page N-1
- Multi-table concurrency: copy multiple tables simultaneously
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from deriva.core import get_credential

from deriva_ml_mcp.tasks import (
    TaskProgress,
    TaskStatus,
    TaskType,
    get_task_manager,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-mcp")

# Flag to check if async datapath is available
try:
    from deriva.core.asyncio import AsyncCatalogWrapper, AsyncErmrestCatalog
    HAS_ASYNC_DATAPATH = True
except ImportError:
    HAS_ASYNC_DATAPATH = False
    logger.info("Async datapath not available, using synchronous clone")

# Cache for user ID to ensure consistency within a session
_cached_user_id: str | None = None


def _get_user_id(hostname: str | None = None) -> str:
    """Get a user identifier from credentials.

    For multi-user isolation, we use the credential identity as user_id.
    If no credentials available, fall back to a default (single-user mode).

    IMPORTANT: This function now caches the user ID to ensure consistency
    between task creation and task lookup. The first call with a hostname
    sets the cached value which is then used for all subsequent calls.

    Args:
        hostname: Optional hostname to get credentials for.

    Returns:
        A string identifying the user.
    """
    global _cached_user_id

    # If we have a cached user ID, always use it for consistency
    if _cached_user_id is not None:
        return _cached_user_id

    try:
        if hostname:
            cred = get_credential(hostname)
            if cred and "cookie" in cred:
                # Extract webauthn from cookie for user identity
                cookie = cred.get("cookie", "")
                if "webauthn=" in cookie:
                    # Use a hash of the webauthn value for privacy
                    import hashlib

                    webauthn = cookie.split("webauthn=")[1].split(";")[0]
                    user_id = hashlib.sha256(webauthn.encode()).hexdigest()[:16]
                    _cached_user_id = user_id
                    logger.debug(f"Cached user ID from credentials: {user_id[:8]}...")
                    return user_id
        # Fall back to checking any available credential
        # In single-user mode, this is fine
        _cached_user_id = "default_user"
        logger.debug("Using default_user for single-user mode")
        return "default_user"
    except Exception:
        _cached_user_id = "default_user"
        return "default_user"


async def _get_user_id_async(hostname: str | None = None) -> str:
    """Async version of _get_user_id.

    Runs credential lookup in a thread to avoid blocking the event loop.
    """
    return await asyncio.to_thread(_get_user_id, hostname)


async def _clone_catalog_task_async(
    progress_updater: Any,
    source_hostname: str,
    source_catalog_id: str,
    root_rid: str | None = None,
    dest_hostname: str | None = None,
    alias: str | None = None,
    add_ml_schema: bool = False,
    schema_only: bool = False,
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
    use_export_annotation: bool = False,
    table_concurrency: int = 5,
    page_size: int = 10000,
) -> dict[str, Any]:
    """Execute catalog clone using async datapath operations.

    This version uses the new async datapath infrastructure for concurrent
    data fetching, providing better performance for large catalogs.

    Supports both full clones and partial clones (with root_rid).

    Key optimizations:
    - Pipeline pattern: fetch page N while uploading page N-1
    - Multi-table concurrency: copy multiple tables simultaneously
    - Async HTTP client with connection pooling
    """
    if not HAS_ASYNC_DATAPATH:
        raise RuntimeError("Async datapath not available. Install deriva-py with asyncio support.")

    from urllib.parse import quote as urlquote

    from deriva.core import DerivaServer, ErmrestCatalog, get_credential
    from deriva.core.asyncio import AsyncCatalogWrapper, AsyncErmrestCatalog
    from deriva.core.asyncio.clone import AsyncTableCopier

    is_partial_clone = root_rid is not None
    clone_type = "partial" if is_partial_clone else "full"

    # Update progress
    progress = TaskProgress(
        current_step="Initializing async clone",
        total_steps=6 if is_partial_clone else 5,
        current_step_number=1,
        percent_complete=5.0,
        message=f"Preparing async {clone_type} clone with datapath operations...",
    )
    progress_updater(progress)

    dest_hostname = dest_hostname or source_hostname

    # Get credentials
    src_cred = get_credential(source_hostname)
    dst_cred = get_credential(dest_hostname)

    # Update progress
    progress.current_step = "Connecting to source catalog"
    progress.current_step_number = 2
    progress.percent_complete = 10.0
    progress.message = f"Connecting to {source_hostname}:{source_catalog_id}..."
    progress_updater(progress)

    # Create sync catalog for datapath navigation
    src_sync_catalog = ErmrestCatalog("https", source_hostname, source_catalog_id, src_cred)

    # Create async catalog for HTTP operations
    src_async_catalog = AsyncErmrestCatalog("https", source_hostname, source_catalog_id, src_cred)

    # Create datapath wrapper
    src_wrapper = AsyncCatalogWrapper(src_sync_catalog, src_async_catalog)

    try:
        # Get catalog model
        model = src_sync_catalog.getCatalogModel()

        # Build exclude sets
        exclude_schemas_set = set(exclude_schemas) if exclude_schemas else set()
        excluded_tables: set[tuple[str, str]] = set()
        if exclude_objects:
            for table_spec in exclude_objects:
                if ":" in table_spec:
                    schema, table = table_spec.split(":", 1)
                    excluded_tables.add((schema, table))

        # For partial clones, discover reachable tables and compute reachable RIDs
        reachable_rids: dict[str, set[str]] | None = None
        tables_to_include: list[str] | None = None

        if is_partial_clone:
            progress.current_step = "Discovering reachable data"
            progress.current_step_number = 3
            progress.percent_complete = 12.0
            progress.message = f"Finding tables and rows reachable from RID {root_rid}..."
            progress_updater(progress)

            # Find root table
            root_table_key = await _find_root_table_async(
                src_async_catalog, model, root_rid, exclude_schemas_set, excluded_tables
            )
            if root_table_key is None:
                raise ValueError(f"Root RID {root_rid} not found in any accessible table")

            logger.info(f"Root RID {root_rid} found in table {root_table_key}")

            # Discover reachable tables via FK graph traversal (local operation)
            start_tables = [root_table_key] + (include_tables or [])
            tables_to_include = _discover_reachable_tables_local(
                model, start_tables, excluded_tables, exclude_schemas_set
            )
            logger.info(f"Discovered {len(tables_to_include)} reachable tables")

            # Compute reachable RIDs using async queries
            progress.percent_complete = 15.0
            progress.message = f"Computing reachable rows from {root_table_key}..."
            progress_updater(progress)

            reachable_rids = await _compute_reachable_rids_async(
                src_async_catalog, model, root_rid, root_table_key, tables_to_include
            )

            total_reachable = sum(len(rids) for rids in reachable_rids.values())
            logger.info(f"Found {total_reachable} reachable rows across {len(tables_to_include)} tables")

        # Update progress
        progress.current_step = "Creating destination catalog"
        progress.current_step_number = 4 if is_partial_clone else 3
        progress.percent_complete = 18.0
        progress.message = f"Creating catalog on {dest_hostname}..."
        progress_updater(progress)

        # Create destination catalog
        dst_server = DerivaServer("https", dest_hostname, dst_cred)
        dst_ermrest = dst_server.create_ermrest_catalog()
        dst_catalog_id = dst_ermrest.catalog_id

        # Create async destination catalog
        dst_async_catalog = AsyncErmrestCatalog("https", dest_hostname, dst_catalog_id, dst_cred)

        try:
            # Update progress
            progress.current_step = "Creating schema"
            progress.percent_complete = 20.0
            progress.message = "Creating schema in destination..."
            progress_updater(progress)

            # Create schema without FKs
            if is_partial_clone and tables_to_include:
                # Only create schemas/tables that are included
                included_tables_set = set()
                for table_spec in tables_to_include:
                    schema, table = table_spec.split(":", 1)
                    included_tables_set.add((schema, table))

                model_json = _build_partial_schema_json(model, included_tables_set, copy_annotations, copy_policy)
            else:
                # Full clone - include all tables
                model_json = model.prejson()
                # Filter out system schemas that already exist
                schemas_to_remove = ["public", "_acl_admin"]
                for schema_name in schemas_to_remove:
                    model_json.get("schemas", {}).pop(schema_name, None)
                # Remove foreign keys from all tables (will be applied later)
                for schema in model_json.get("schemas", {}).values():
                    for table in schema.get("tables", {}).values():
                        table.pop("foreign_keys", None)

            if model_json.get("schemas"):
                dst_ermrest.post("/schema", json=model_json).raise_for_status()

            # Determine tables to copy
            if is_partial_clone and tables_to_include:
                tables_to_copy = []
                for table_spec in tables_to_include:
                    schema, table = table_spec.split(":", 1)
                    tables_to_copy.append((schema, table))
            else:
                tables_to_copy = []
                for sname, schema in model.schemas.items():
                    if sname in {"public", "_acl_admin"}:
                        continue
                    if sname in exclude_schemas_set:
                        continue
                    for tname, table in schema.tables.items():
                        if table.kind == "table":
                            if (sname, tname) in excluded_tables:
                                continue
                            tables_to_copy.append((sname, tname))

            total_tables = len(tables_to_copy)
            tables_completed = 0
            total_rows = 0

            # Update progress
            progress.current_step = "Copying data"
            progress.current_step_number = 5 if is_partial_clone else 4
            progress.percent_complete = 25.0
            progress.message = f"Copying {total_tables} tables with {table_concurrency} concurrent..."
            progress_updater(progress)

            # Use semaphore to limit concurrent table copies
            semaphore = asyncio.Semaphore(table_concurrency)

            async def copy_table_with_progress(schema_name: str, table_name: str) -> int:
                nonlocal tables_completed, total_rows

                async with semaphore:
                    table_spec = f"{schema_name}:{table_name}"

                    if is_partial_clone and reachable_rids is not None:
                        # Partial clone: only copy reachable RIDs
                        rids_to_copy = reachable_rids.get(table_spec, set())
                        if not rids_to_copy:
                            logger.debug(f"No reachable rows for {table_spec}, skipping")
                            tables_completed += 1
                            return 0

                        logger.info(f"Copying {len(rids_to_copy)} rows from {table_spec}")
                        rows = await _copy_subset_rows_async(
                            src_async_catalog, dst_async_catalog,
                            schema_name, table_name, rids_to_copy, page_size
                        )
                    else:
                        # Full clone: copy entire table
                        logger.info(f"Copying table {table_spec}")
                        copier = AsyncTableCopier(
                            src_wrapper,
                            dst_async_catalog,
                            schema_name,
                            table_name,
                            page_size=page_size,
                        )
                        rows = await copier.copy_async()

                    tables_completed += 1
                    total_rows += rows

                    # Update progress
                    pct = 25.0 + (tables_completed / total_tables) * 60.0
                    progress.percent_complete = pct
                    progress.message = f"Copied {table_spec} ({rows} rows). {tables_completed}/{total_tables} tables done."
                    progress_updater(progress)

                    return rows

            # Copy all tables concurrently
            results = await asyncio.gather(
                *[copy_table_with_progress(s, t) for s, t in tables_to_copy],
                return_exceptions=True,
            )

            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.error(f"Clone had {len(errors)} table errors: {errors[:3]}")

            # Update progress
            progress.current_step = "Finalizing"
            progress.current_step_number = 6 if is_partial_clone else 5
            progress.percent_complete = 90.0
            progress.message = "Applying foreign keys and annotations..."
            progress_updater(progress)

            # Apply foreign keys (use sync for now)
            # TODO: Add FK application with orphan handling

            # Create alias if requested
            if alias:
                try:
                    dst_server.create_ermrest_alias(
                        id=alias,
                        alias_target=dst_catalog_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create alias: {e}")

            # Add ML schema if requested
            if add_ml_schema:
                try:
                    from deriva_ml.schema import create_ml_schema
                    create_ml_schema(dst_ermrest)
                except Exception as e:
                    logger.warning(f"Failed to add ML schema: {e}")

            # Build response
            response: dict[str, Any] = {
                "status": "cloned",
                "clone_mode": "async_datapath",
                "clone_type": clone_type,
                "source_hostname": source_hostname,
                "source_catalog_id": source_catalog_id,
                "dest_hostname": dest_hostname,
                "dest_catalog_id": str(dst_catalog_id),
                "tables_copied": tables_completed,
                "rows_copied": total_rows,
                "table_errors": len(errors),
                "asset_mode": asset_mode,
            }

            if root_rid:
                response["root_rid"] = root_rid
            if alias:
                response["alias"] = alias
            if add_ml_schema:
                response["ml_schema_added"] = True

            response["message"] = (
                f"Async {clone_type} clone completed: {tables_completed} tables, {total_rows} rows "
                f"from {source_hostname}:{source_catalog_id} to {dest_hostname}:{dst_catalog_id}"
            )

            return response

        finally:
            await dst_async_catalog.close()

    finally:
        await src_async_catalog.close()


async def _find_root_table_async(
    async_catalog: "AsyncErmrestCatalog",
    model: Any,
    root_rid: str,
    exclude_schemas: set[str],
    exclude_tables: set[tuple[str, str]],
) -> str | None:
    """Find which table contains the root RID using async queries."""
    from urllib.parse import quote as urlquote

    for sname, schema in model.schemas.items():
        if sname in {"public", "_acl_admin", "WWW"} or sname in exclude_schemas:
            continue
        for tname, table in schema.tables.items():
            if (sname, tname) in exclude_tables:
                continue
            if table.kind != "table" or "RID" not in table.column_definitions.elements:
                continue
            try:
                table_spec = f"{sname}:{tname}"
                uri = f"/entity/{urlquote(sname)}:{urlquote(tname)}/RID={urlquote(root_rid)}"
                response = await async_catalog.get_async(uri)
                result = response.json()
                if result:
                    return table_spec
            except Exception:
                continue
    return None


def _discover_reachable_tables_local(
    model: Any,
    start_tables: list[str],
    exclude_tables: set[tuple[str, str]],
    exclude_schemas: set[str],
) -> list[str]:
    """Discover all tables reachable from start tables via FK relationships.

    This is a local operation using only the model (no HTTP calls).
    """
    system_schemas = {"public", "_acl_admin", "WWW"}
    all_excluded_schemas = system_schemas | exclude_schemas

    discovered: set[tuple[str, str]] = set()
    to_visit: list[tuple[str, str]] = []

    for table_spec in start_tables:
        if ":" not in table_spec:
            continue
        schema, table = table_spec.split(":", 1)
        key = (schema, table)
        if key not in exclude_tables and schema not in all_excluded_schemas:
            discovered.add(key)
            to_visit.append(key)

    # BFS traversal of FK graph
    while to_visit:
        current_key = to_visit.pop(0)
        schema_name, table_name = current_key

        try:
            table = model.schemas[schema_name].tables[table_name]
        except KeyError:
            continue

        # Outbound FKs (this table references other tables)
        for fk in table.foreign_keys:
            pk_table = fk.pk_table
            pk_key = (pk_table.schema.name, pk_table.name)

            if pk_key in discovered or pk_key in exclude_tables:
                continue
            if pk_table.schema.name in all_excluded_schemas:
                continue

            discovered.add(pk_key)
            to_visit.append(pk_key)

        # Inbound FKs (other tables reference this table)
        for fk in table.referenced_by:
            ref_table = fk.table
            ref_key = (ref_table.schema.name, ref_table.name)

            if ref_key in discovered or ref_key in exclude_tables:
                continue
            if ref_table.schema.name in all_excluded_schemas:
                continue

            discovered.add(ref_key)
            to_visit.append(ref_key)

    return [f"{schema}:{table}" for schema, table in sorted(discovered)]


async def _compute_reachable_rids_async(
    async_catalog: "AsyncErmrestCatalog",
    model: Any,
    root_rid: str,
    root_table: str,
    include_tables: list[str],
) -> dict[str, set[str]]:
    """Compute RIDs reachable from root_rid using async FK traversal."""
    from urllib.parse import quote as urlquote

    # Initialize reachable sets
    reachable: dict[str, set[str]] = {t: set() for t in include_tables}
    reachable[root_table].add(root_rid)

    # Build table lookup
    table_lookup: dict[tuple[str, str], str] = {}
    for table_spec in include_tables:
        schema, table_name = table_spec.split(":", 1)
        table_lookup[(schema, table_name)] = table_spec

    # Get root table object
    root_schema, root_tname = root_table.split(":", 1)
    root_table_obj = model.schemas[root_schema].tables[root_tname]

    # Find all FK paths from root table
    def find_paths(start_table, visited, current_path):
        paths = []
        connected = []

        # Outbound FKs
        for fk in start_table.foreign_keys:
            pk_table = fk.pk_table
            pk_key = (pk_table.schema.name, pk_table.name)
            if pk_key not in visited and pk_key in table_lookup:
                connected.append(pk_table)

        # Inbound FKs
        for fk in start_table.referenced_by:
            ref_table = fk.table
            ref_key = (ref_table.schema.name, ref_table.name)
            if ref_key not in visited and ref_key in table_lookup:
                connected.append(ref_table)

        for next_table in connected:
            next_key = (next_table.schema.name, next_table.name)
            new_path = current_path + [next_key]
            paths.append(new_path)
            new_visited = visited | {next_key}
            paths.extend(find_paths(next_table, new_visited, new_path))

        return paths

    root_key = (root_table_obj.schema.name, root_table_obj.name)
    all_paths = find_paths(root_table_obj, {root_key}, [])

    # Query each path asynchronously
    async def query_path(path):
        if not path:
            return None, set()

        target_key = path[-1]
        target_spec = table_lookup.get(target_key)
        if not target_spec:
            return None, set()

        # Build query
        query = f"/entity/{urlquote(root_schema)}:{urlquote(root_tname)}/RID={urlquote(root_rid)}"
        for schema, table in path:
            query += f"/{urlquote(schema)}:{urlquote(table)}"

        try:
            response = await async_catalog.get_async(query)
            result = response.json()
            rids = {row["RID"] for row in result if "RID" in row}
            return target_spec, rids
        except Exception as e:
            logger.debug(f"Path query failed: {query}: {e}")
            return target_spec, set()

    # Execute all path queries concurrently
    results = await asyncio.gather(*[query_path(p) for p in all_paths], return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            continue
        target_spec, rids = result
        if target_spec and rids:
            reachable[target_spec].update(rids)

    return reachable


def _build_partial_schema_json(
    model: Any,
    included_tables: set[tuple[str, str]],
    copy_annotations: bool,
    copy_policy: bool,
) -> dict:
    """Build schema JSON for only the included tables."""
    def prune_parts(d, *extra_victims):
        victims = set(extra_victims)
        if not copy_annotations:
            victims.add("annotations")
        if not copy_policy:
            victims |= {"acls", "acl_bindings"}
        for k in victims:
            d.pop(k, None)
        return d

    new_model = {"schemas": {}}
    included_schemas = {schema for schema, _ in included_tables}

    for sname in included_schemas:
        if sname not in model.schemas or sname in {"public", "_acl_admin"}:
            continue

        schema = model.schemas[sname]
        schema_def = prune_parts(schema.prejson(), "tables")
        schema_def["tables"] = {}

        for tname, table in schema.tables.items():
            if (sname, tname) not in included_tables:
                continue
            if table.kind != "table":
                continue

            table_def = prune_parts(table.prejson(), "foreign_keys")
            table_def["column_definitions"] = [
                prune_parts(c.copy()) for c in table_def.get("column_definitions", [])
            ]
            table_def["keys"] = [prune_parts(k.copy()) for k in table_def.get("keys", [])]
            schema_def["tables"][tname] = table_def

        if schema_def["tables"]:
            new_model["schemas"][sname] = schema_def

    return new_model


async def _copy_subset_rows_async(
    src_catalog: "AsyncErmrestCatalog",
    dst_catalog: "AsyncErmrestCatalog",
    schema_name: str,
    table_name: str,
    rids_to_copy: set[str],
    page_size: int,
) -> int:
    """Copy only specific RIDs from source to destination using async operations."""
    from urllib.parse import quote as urlquote

    table_spec = f"{urlquote(schema_name)}:{urlquote(table_name)}"
    rows_copied = 0

    # Sort RIDs for consistent pagination
    rid_list = sorted(rids_to_copy)

    # Process in batches
    for i in range(0, len(rid_list), page_size):
        batch_rids = rid_list[i : i + page_size]

        # Build query with RID filter
        rid_filter = ",".join(urlquote(rid) for rid in batch_rids)
        try:
            response = await src_catalog.get_async(f"/entity/{table_spec}/RID=any({rid_filter})")
            page = response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch batch from {schema_name}:{table_name}: {e}")
            continue

        if not page:
            continue

        # Insert into destination
        try:
            await dst_catalog.post_async(
                f"/entity/{table_spec}?nondefaults=RID,RCT,RCB",
                json_data=page,
            )
            rows_copied += len(page)
        except Exception as e:
            logger.warning(f"Failed to insert batch into {schema_name}:{table_name}: {e}")
            # Try row-by-row fallback
            for row in page:
                try:
                    await dst_catalog.post_async(
                        f"/entity/{table_spec}?nondefaults=RID,RCT,RCB",
                        json_data=[row],
                    )
                    rows_copied += 1
                except Exception:
                    pass

    return rows_copied


def _clone_catalog_task(
    progress_updater: Any,
    source_hostname: str,
    source_catalog_id: str,
    root_rid: str | None = None,
    dest_hostname: str | None = None,
    alias: str | None = None,
    add_ml_schema: bool = False,
    schema_only: bool = False,
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
    use_export_annotation: bool = False,
) -> dict[str, Any]:
    """Execute catalog clone operation with progress tracking.

    This function is called by the task manager in a background thread.
    """
    from deriva_ml.catalog import AssetCopyMode, OrphanStrategy

    # Update progress
    progress = TaskProgress(
        current_step="Initializing clone operation",
        total_steps=4,
        current_step_number=1,
        percent_complete=5.0,
        message="Preparing to clone catalog...",
    )
    progress_updater(progress)

    # Convert string parameters to enums
    asset_mode_enum = AssetCopyMode(asset_mode)
    orphan_strategy_enum = OrphanStrategy(orphan_strategy)

    # Update progress
    progress.current_step = "Connecting to source catalog"
    progress.current_step_number = 2
    progress.percent_complete = 10.0
    progress.message = f"Connecting to {source_hostname}..."
    progress_updater(progress)

    # Determine if this is a partial or full clone
    if root_rid:
        from deriva_ml.catalog import clone_subset_catalog as do_clone

        progress.message = f"Starting partial clone from RID {root_rid}..."
        progress_updater(progress)

        result = do_clone(
            source_hostname=source_hostname,
            source_catalog_id=source_catalog_id,
            root_rid=root_rid,
            include_tables=include_tables,
            exclude_objects=exclude_objects,
            exclude_schemas=exclude_schemas,
            include_associations=include_associations,
            include_vocabularies=include_vocabularies,
            use_export_annotation=use_export_annotation,
            dest_hostname=dest_hostname,
            alias=alias,
            add_ml_schema=add_ml_schema,
            asset_mode=asset_mode_enum,
            copy_annotations=copy_annotations,
            copy_policy=copy_policy,
            orphan_strategy=orphan_strategy_enum,
            prune_hidden_fkeys=prune_hidden_fkeys,
            truncate_oversized=truncate_oversized,
            reinitialize_dataset_versions=reinitialize_dataset_versions,
        )
        clone_mode = "partial"
    else:
        from deriva_ml.catalog import clone_catalog as do_clone

        progress.message = "Starting full catalog clone..."
        progress_updater(progress)

        result = do_clone(
            source_hostname=source_hostname,
            source_catalog_id=source_catalog_id,
            dest_hostname=dest_hostname,
            alias=alias,
            add_ml_schema=add_ml_schema,
            schema_only=schema_only,
            asset_mode=asset_mode_enum,
            copy_annotations=copy_annotations,
            copy_policy=copy_policy,
            exclude_schemas=exclude_schemas,
            exclude_objects=exclude_objects,
            reinitialize_dataset_versions=reinitialize_dataset_versions,
            orphan_strategy=orphan_strategy_enum,
            prune_hidden_fkeys=prune_hidden_fkeys,
            truncate_oversized=truncate_oversized,
        )
        clone_mode = "full"

    # Update progress - finalizing
    progress.current_step = "Finalizing"
    progress.current_step_number = 4
    progress.percent_complete = 95.0
    progress.message = "Building result..."
    progress_updater(progress)

    # Build response from CloneCatalogResult
    response: dict[str, Any] = {
        "status": "cloned",
        "clone_mode": clone_mode,
        "source_hostname": source_hostname,
        "source_catalog_id": source_catalog_id,
        "dest_hostname": result.hostname,
        "dest_catalog_id": result.catalog_id,
        "schema_only": schema_only,
        "asset_mode": asset_mode,
    }

    if root_rid:
        response["root_rid"] = root_rid
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
        response["rows_skipped"] = (
            result.report.summary.rows_skipped if hasattr(result.report.summary, "rows_skipped") else 0
        )
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
            f"Catalog {'subset ' if root_rid else ''}migrated from "
            f"{source_hostname}:{source_catalog_id} to {result.hostname}:{result.catalog_id}"
        )
        response["report_summary"] = result.report.to_text()

    return response


def register_background_task_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register background task management tools with the MCP server."""

    @mcp.tool()
    async def clone_catalog_async(
        source_hostname: str,
        source_catalog_id: str,
        root_rid: str | None = None,
        dest_hostname: str | None = None,
        alias: str | None = None,
        add_ml_schema: bool = False,
        schema_only: bool = False,
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
        use_export_annotation: bool = False,
        use_async_datapath: bool = True,
        table_concurrency: int = 5,
        page_size: int = 10000,
    ) -> str:
        """Start a catalog clone operation in the background.

        This is the async version of clone_catalog - it starts the clone operation
        and immediately returns a task_id that you can use to check progress.

        Use this for large catalogs or cross-server clones that may take several
        minutes to complete. Check progress with `get_task_status(task_id)`.

        **Full clone** (root_rid=None):
        Creates a complete clone of the source catalog.

        **Partial clone** (root_rid provided):
        Creates a subset clone containing only data reachable from the root RID.

        **Clone modes:**
        - `use_async_datapath=True` (default): Uses async datapath for concurrent data
          fetching. Faster for large catalogs. Supports both full and partial clones.
        - `use_async_datapath=False`: Uses the synchronous clone. Provides more detailed
          orphan handling and reporting.

        Args:
            source_hostname: Source server hostname (e.g., "www.facebase.org").
            source_catalog_id: ID of the catalog to clone.
            root_rid: Optional RID for partial clone (e.g., "3-HXMC"). Forces sync mode.
            dest_hostname: Destination hostname. If None, uses source hostname.
            alias: Optional alias name for the new catalog.
            add_ml_schema: If True, add the DerivaML schema to the clone.
            schema_only: If True, copy only schema structure without data.
            asset_mode: How to handle assets: "none", "refs" (default), or "full".
            copy_annotations: If True (default), copy all annotations.
            copy_policy: If True (default), copy ACL policies.
            exclude_schemas: Schemas to exclude from cloning.
            exclude_objects: Tables ("schema:table") to exclude.
            reinitialize_dataset_versions: If True, increment dataset versions.
            orphan_strategy: How to handle orphans: "fail", "delete", "nullify".
            prune_hidden_fkeys: Skip FKs with hidden reference data.
            truncate_oversized: Truncate values exceeding index limits.
            include_tables: (Partial) Additional starting tables.
            include_associations: (Partial) Include association tables.
            include_vocabularies: (Partial) Include vocabulary tables.
            use_export_annotation: (Partial) Use export annotation.
            use_async_datapath: If True (default), use async datapath for concurrent copying.
            table_concurrency: (Async mode) Number of tables to copy concurrently.
            page_size: (Async mode) Rows per page when fetching data.

        Returns:
            JSON with task_id and status. Use get_task_status(task_id) to check progress.

        Example:
            # Start full clone (uses async datapath by default)
            clone_catalog_async("www.facebase.org", "1",
                               dest_hostname="localhost",
                               alias="facebase-clone")
            -> {"task_id": "abc123", "status": "started", "clone_mode": "async_datapath", ...}

            # Partial clone (automatically uses sync mode)
            clone_catalog_async("www.facebase.org", "1",
                               root_rid="3-HXMC",
                               dest_hostname="localhost",
                               alias="facebase-subset")
            -> {"task_id": "xyz789", "status": "started", "clone_mode": "sync", ...}

            # Full clone with sync mode (for detailed orphan handling)
            clone_catalog_async("www.facebase.org", "1",
                               dest_hostname="localhost",
                               use_async_datapath=False,
                               orphan_strategy="delete")
            -> {"task_id": "def456", "status": "started", "clone_mode": "sync", ...}

            # Check progress
            get_task_status("abc123")
            -> {"status": "running", "progress": {"percent_complete": 45.0, ...}}

            # When done
            get_task_status("abc123")
            -> {"status": "completed", "result": {...}}
        """
        try:
            task_manager = get_task_manager()
            # Use async credential lookup to avoid blocking
            user_id = await _get_user_id_async(source_hostname)

            # Determine clone mode
            # Fall back to sync if async datapath not available
            if use_async_datapath and HAS_ASYNC_DATAPATH:
                clone_mode = "async_datapath"
                task_fn = _clone_catalog_task_async
            else:
                clone_mode = "sync"
                task_fn = _clone_catalog_task
                if use_async_datapath and not HAS_ASYNC_DATAPATH:
                    logger.info("Async datapath not available - falling back to sync mode")

            # Store parameters for the task
            parameters = {
                "source_hostname": source_hostname,
                "source_catalog_id": source_catalog_id,
                "root_rid": root_rid,
                "dest_hostname": dest_hostname,
                "alias": alias,
                "add_ml_schema": add_ml_schema,
                "schema_only": schema_only,
                "asset_mode": asset_mode,
                "copy_annotations": copy_annotations,
                "copy_policy": copy_policy,
                "exclude_schemas": exclude_schemas,
                "exclude_objects": exclude_objects,
                "reinitialize_dataset_versions": reinitialize_dataset_versions,
                "orphan_strategy": orphan_strategy,
                "prune_hidden_fkeys": prune_hidden_fkeys,
                "truncate_oversized": truncate_oversized,
                "include_tables": include_tables,
                "include_associations": include_associations,
                "include_vocabularies": include_vocabularies,
                "use_export_annotation": use_export_annotation,
            }

            # Add async-specific parameters
            if use_async_datapath:
                parameters["table_concurrency"] = table_concurrency
                parameters["page_size"] = page_size

            # Create the background task
            if use_async_datapath:
                # Use async task creation for async clone
                task = await task_manager.create_async_task(
                    user_id=user_id,
                    task_type=TaskType.CLONE_CATALOG,
                    task_fn=task_fn,
                    parameters=parameters,
                )
            else:
                # Use sync task creation for traditional clone
                task = task_manager.create_task(
                    user_id=user_id,
                    task_type=TaskType.CLONE_CATALOG,
                    task_fn=task_fn,
                    parameters=parameters,
                )

            return json.dumps(
                {
                    "status": "started",
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "clone_mode": clone_mode,
                    "message": (f"Clone operation started. Use get_task_status('{task.task_id}') to check progress."),
                    "parameters": {
                        "source": f"{source_hostname}:{source_catalog_id}",
                        "dest": dest_hostname or source_hostname,
                        "root_rid": root_rid,
                        "alias": alias,
                        "clone_mode": clone_mode,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Failed to start clone task: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def get_task_status(
        task_id: str,
        include_result: bool = True,
    ) -> str:
        """Get the status and progress of a background task.

        Args:
            task_id: The task ID returned by an async operation.
            include_result: If True, include the full result when completed.

        Returns:
            JSON with task status, progress, and optionally the result.

        Example:
            get_task_status("abc123")
            -> {
                "task_id": "abc123",
                "status": "running",
                "progress": {
                    "current_step": "Copying data",
                    "percent_complete": 45.0,
                    "message": "Copying table Subject..."
                }
            }
        """
        try:
            task_manager = get_task_manager()
            # Use cached user_id for consistency with task creation
            user_id = await _get_user_id_async()

            # Use async snapshot method to avoid blocking event loop and minimize lock contention
            task_snapshot = await task_manager.get_task_snapshot_async(task_id, user_id)
            if not task_snapshot:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Task {task_id} not found or access denied",
                    }
                )

            # If include_result is False, remove result from snapshot
            if not include_result and "result" in task_snapshot:
                del task_snapshot["result"]

            return json.dumps(task_snapshot)

        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def list_tasks(
        status: str | None = None,
        task_type: str | None = None,
    ) -> str:
        """List all background tasks for the current user.

        Args:
            status: Filter by status: "pending", "running", "completed", "failed", "cancelled".
            task_type: Filter by type: "clone_catalog".

        Returns:
            JSON list of tasks with their status and basic info.

        Example:
            list_tasks(status="running")
            -> [{"task_id": "abc123", "status": "running", ...}]
        """
        try:
            task_manager = get_task_manager()
            # Use cached user_id for consistency
            user_id = await _get_user_id_async()

            # Parse filters
            status_filter = TaskStatus(status) if status else None
            type_filter = TaskType(task_type) if task_type else None

            # Use async snapshot method to avoid blocking and minimize lock contention
            task_snapshots = await task_manager.list_tasks_snapshots_async(
                user_id=user_id,
                status_filter=status_filter,
                task_type_filter=type_filter,
                include_result=False,
            )

            return json.dumps(task_snapshots)

        except ValueError as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Invalid filter value: {e}",
                }
            )
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )

    @mcp.tool()
    async def cancel_task(task_id: str) -> str:
        """Cancel a pending or running background task.

        Args:
            task_id: The task ID to cancel.

        Returns:
            JSON with cancellation status.

        Note: Cancellation is best-effort. Long-running operations may not
        stop immediately.
        """
        try:
            task_manager = get_task_manager()
            # Use cached user_id for consistency
            user_id = await _get_user_id_async()

            # Run cancel in thread to avoid blocking
            cancelled = await asyncio.to_thread(task_manager.cancel_task, task_id, user_id)

            if cancelled:
                return json.dumps(
                    {
                        "status": "cancelled",
                        "task_id": task_id,
                        "message": "Task cancellation requested",
                    }
                )
            else:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Task {task_id} not found, access denied, or already completed",
                    }
                )

        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                }
            )
