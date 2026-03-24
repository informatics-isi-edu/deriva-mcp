"""Dataset management tools for DerivaML MCP server.

Datasets are versioned, reproducible collections of data for ML workflows. Key concepts:

**Creating Datasets**:
Datasets must be created through an execution for provenance tracking. Use
create_execution_dataset() from the execution tools to create new datasets.
This ensures all datasets have proper provenance linking to the workflow that
created them.

**Dataset Elements**:
Records from domain tables (e.g., Images, Subjects) that belong to a dataset.
A table must be registered as a "dataset element type" before its records can be
added to datasets. Use list_dataset_element_types() and add_dataset_element_type().

**Dataset Types**:
Controlled vocabulary labels that categorize datasets (e.g., "Training", "Testing",
"Validation", "Complete"). A dataset can have multiple types. Types help organize
datasets by their role in ML workflows.

**Nested Datasets**:
Datasets can contain other datasets as children, creating hierarchies. Common pattern:
a parent "Complete" dataset contains "Training" and "Testing" child datasets that
partition the same underlying data. Children share the parent's elements.

**Dataset Versions** (Semantic Versioning):
Datasets use major.minor.patch versioning for reproducibility:
- **Patch** (0.0.X): Metadata-only changes (descriptions, corrections)
- **Minor** (0.X.0): Added or removed elements, types (auto-incremented)
- **Major** (X.0.0): Schema changes, breaking changes to data structure

Each version captures a catalog snapshot, allowing queries against historical states.
The same version always returns the same data, regardless of later catalog changes.

**Dataset Bags (BDBags)**:
A BDBag is a self-describing, portable archive of a specific dataset version.
Use the DerivaML Python API for downloading and processing bags locally.
See DerivaML documentation for detailed BDBag contents and usage.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-mcp")


def _serialize_dataset(dataset) -> dict:
    """Serialize a Dataset object to a JSON-compatible dictionary."""
    return {
        "dataset_rid": dataset.dataset_rid,
        "description": dataset.description,
        "dataset_types": dataset.dataset_types,
        "current_version": str(dataset.current_version) if dataset.current_version else None,
    }


def register_dataset_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register dataset management tools with the MCP server."""

    @mcp.tool()
    async def create_dataset(
        description: str = "",
        dataset_types: list[str] | None = None,
        version: str | None = None,
    ) -> str:
        """Create a new empty dataset within an execution context.

        The dataset is created through an execution for proper provenance
        tracking. Use add_dataset_members() to populate it after creation.

        Assign Dataset_Type labels to categorize the dataset's role
        (e.g., "Training", "Testing", "Validation").

        Args:
            description: Human-readable description of the dataset's purpose.
            dataset_types: Type labels from Dataset_Type vocabulary (e.g., ["Training", "Image"]).
            version: Initial version string (default: "0.1.0").

        Returns:
            JSON with status, rid, description, dataset_types, version, execution_rid.

        Example:
            create_dataset("Training images for model v2", ["Training"])
        """
        try:
            conn_manager.get_active_or_raise()  # Ensure connection exists

            # Get execution: user-created execution takes priority over MCP connection execution
            execution = None
            conn_info = conn_manager.get_active_connection_info()
            if conn_info and conn_info.active_tool_execution:
                execution = conn_info.active_tool_execution

            # Fallback to MCP connection execution
            if execution is None:
                execution = conn_manager.get_active_execution()

            if execution is None:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution context. Connect to a catalog or use create_execution first.",
                })

            # Create dataset through execution for provenance
            dataset = execution.create_dataset(
                description=description,
                dataset_types=dataset_types or [],
            )

            return json.dumps({
                "status": "created",
                "dataset_rid": dataset.dataset_rid,
                "description": dataset.description,
                "dataset_types": dataset.dataset_types,
                "version": str(dataset.current_version) if dataset.current_version else version or "0.1.0",
                "execution_rid": execution.execution_rid,
            })
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_dataset_spec(dataset_rid: str, version: str | None = None) -> str:
        """Generate a DatasetSpecConfig string for use in Python configuration files.

        Returns the exact Python code to use in hydra-zen config files. This ensures
        the RID and version are correctly formatted and match what's in the catalog.

        **IMPORTANT**: Always prefer specifying explicit versions in configurations.
        Using current_version as a default can lead to unexpected changes in results
        if the dataset is modified after the configuration is written. Pin to a
        specific version for reproducibility.

        Args:
            dataset_rid: The RID of the dataset (e.g., "28CT").
            version: Specific version to use. If not provided, uses the dataset's
                current version (with a warning about reproducibility).

        Returns:
            JSON with the Python code string and metadata including:
            - spec: The DatasetSpecConfig(...) string ready to paste into code
            - rid: The dataset RID
            - version: The version used
            - description: Dataset description for reference
            - warning: Present if using current_version (recommends explicit version)

        Example:
            get_dataset_spec("28CT")
            -> {"spec": "DatasetSpecConfig(rid=\\"28CT\\", version=\\"0.21.0\\")", ...}

            get_dataset_spec("28CT", "0.20.0")
            -> {"spec": "DatasetSpecConfig(rid=\\"28CT\\", version=\\"0.20.0\\")", ...}
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)

            # Use provided version or fall back to current version
            if version:
                use_version = version
                warning = None
            else:
                use_version = str(dataset.current_version) if dataset.current_version else "0.1.0"
                warning = (
                    "Using current_version as default. For reproducibility, consider "
                    "pinning to an explicit version. Dataset contents may change if "
                    "the version is incremented after this configuration is written."
                )

            spec_string = f'DatasetSpecConfig(rid="{dataset_rid}", version="{use_version}")'

            result = {
                "spec": spec_string,
                "dataset_rid": dataset_rid,
                "version": use_version,
                "description": dataset.description,
                "dataset_types": dataset.dataset_types,
            }
            if warning:
                result["warning"] = warning

            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to get dataset spec for {dataset_rid}: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_dataset_members(
        dataset_rid: str,
        member_rids: list[str] | None = None,
        members_by_table: dict[str, list[str]] | None = None,
        description: str = "",
    ) -> str:
        """Add records as dataset elements. Auto-increments minor version.

        Records must be from tables registered as dataset element types.
        Use add_dataset_element_type() to register a table, or list_dataset_element_types()
        to see which tables are already registered.

        Accepts members in two forms:

        **List of RIDs** (member_rids): Each RID is auto-resolved to its table.
        Simpler but slower for large numbers.

        **Dict by table name** (members_by_table): Maps table names to RID lists.
        Faster (skips RID resolution) and lets you add members of different types
        in one call. Recommended when you know the table names.

        Exactly one of member_rids or members_by_table must be provided.

        Args:
            dataset_rid: The RID of the dataset to add members to.
            member_rids: List of RIDs to add (e.g., ["2-ABC", "2-DEF"]).
                Auto-resolves each RID to its table.
            members_by_table: Dict mapping table names to RID lists
                (e.g., {"Subject": ["2-ABC"], "Observation": ["2-DEF", "2-GHI"]}).
                Faster than member_rids for large datasets.
            description: Optional description for the version increment that records
                why these members were added. Stored in the dataset history.

        Returns:
            JSON with status, added_count, dataset_rid.

        Example:
            add_dataset_members("1-ABC", member_rids=["2-DEF", "2-GHI"])
            add_dataset_members("1-ABC", members_by_table={"Subject": ["2-DEF"], "Image": ["2-GHI"]})
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)

            if members_by_table and member_rids:
                return json.dumps({
                    "status": "error",
                    "message": "Provide either member_rids or members_by_table, not both.",
                })
            if members_by_table:
                dataset.add_dataset_members(members=members_by_table, description=description)
                total = sum(len(v) for v in members_by_table.values())
            elif member_rids:
                dataset.add_dataset_members(members=member_rids, description=description)
                total = len(member_rids)
            else:
                return json.dumps({
                    "status": "error",
                    "message": "Provide either member_rids or members_by_table.",
                })

            return json.dumps({
                "status": "success",
                "added_count": total,
                "dataset_rid": dataset_rid,
            })
        except Exception as e:
            logger.error(f"Failed to add dataset members: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def delete_dataset_members(
        dataset_rid: str,
        member_rids: list[str],
    ) -> str:
        """Remove records from a dataset. Auto-increments minor version.

        Removes the specified records from the dataset's membership. The records
        themselves are not deleted from the catalog, only their association with
        this dataset is removed.

        Removing members automatically increments the dataset's minor version for change tracking.

        Args:
            dataset_rid: The RID of the dataset to remove members from.
            member_rids: List of RIDs to remove (e.g., ["2-ABC", "2-DEF", "2-GHI"]).

        Returns:
            JSON with status, removed_count, dataset_rid.

        Example:
            delete_dataset_members("1-ABC", ["2-DEF", "2-GHI"]) -> removes 2 records from dataset
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            dataset.delete_dataset_members(members=member_rids)
            return json.dumps({
                "status": "success",
                "removed_count": len(member_rids),
                "dataset_rid": dataset_rid,
            })
        except Exception as e:
            logger.error(f"Failed to delete dataset members: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def increment_dataset_version(
        dataset_rid: str,
        description: str = "",
        component: str = "minor",
    ) -> str:
        """Manually increment a dataset's semantic version (major.minor.patch).

        **Description Handling (follows generate-descriptions prompt guidelines):**

        1. If user provides description: Use it, potentially improving for clarity
        2. If description is empty: Generate from conversation context:
           - What catalog operations were performed since last version?
           - What was the user's stated goal for these changes?
           - Summarize the changes made (e.g., "Added X, fixed Y, modified Z")

        **Description Generation Guidelines:**
        - Include WHAT changed (added images, fixed labels, new features)
        - Include WHY if known (QA review, batch import, schema update)
        - Include IMPACT if relevant (affects N records, breaking change)
        - Use markdown for complex descriptions (lists, tables)

        Use this tool when:
        - You've modified catalog data and want changes visible in a dataset
        - You need to capture a snapshot of the current catalog state
        - You want to create a reproducible checkpoint before making changes

        Args:
            dataset_rid: The RID of the dataset.
            description: What changed in this version. If empty, LLM should generate
                from context. Good descriptions include:
                - What was added, modified, or fixed
                - Why the change was made (if known)
                - Impact on users of this dataset

                Examples of good descriptions:
                - "Added 500 new labeled training images from batch 3"
                - "Fixed incorrect labels on 23 images identified in QA review"
                - "Schema change: added 'quality_score' column to Image table"
                - "Captured snapshot before label correction workflow"

            component: Which part to increment: "major", "minor", or "patch".
                - major: Breaking changes or schema modifications
                - minor: New data added or non-breaking changes (default)
                - patch: Bug fixes or label corrections

        Returns:
            JSON with status, new_version, previous_version, dataset_rid, description.

        Examples:
            increment_dataset_version("1-ABC", "Added quality labels to all images", "minor")
            increment_dataset_version("1-ABC", "Fixed mislabeled cat images", "patch")
            increment_dataset_version("1-ABC", "Schema change: new metadata columns", "major")
        """
        try:
            from deriva_ml.dataset.aux_classes import VersionPart

            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)

            # Capture previous version before incrementing
            previous_version = dataset.current_version

            component_map = {
                "major": VersionPart.major,
                "minor": VersionPart.minor,
                "patch": VersionPart.patch,
            }
            version_part = component_map.get(component.lower(), VersionPart.minor)

            new_version = dataset.increment_dataset_version(
                component=version_part,
                description=description,
            )
            return json.dumps({
                "status": "success",
                "new_version": str(new_version) if new_version else None,
                "previous_version": str(previous_version) if previous_version else None,
                "dataset_rid": dataset_rid,
                "description": description,
                "component": component,
            })
        except Exception as e:
            logger.error(f"Failed to increment version: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def delete_dataset(dataset_rid: str, recurse: bool = False) -> str:
        """Soft-delete a dataset (marks deleted but preserves data).

        Soft deletion hides the dataset from normal queries but keeps all data intact.
        For nested datasets, use recurse=True to also delete child datasets.

        Args:
            dataset_rid: The RID of the dataset to delete.
            recurse: If True, also delete all nested child datasets.

        Returns:
            JSON with status, dataset_rid, recursive.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            ml.delete_dataset(dataset, recurse=recurse)
            return json.dumps({
                "status": "deleted",
                "dataset_rid": dataset_rid,
                "recursive": recurse,
            })
        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_dataset_description(dataset_rid: str, description: str) -> str:
        """Set or update the description for a dataset.

        Updates the dataset's description in the catalog. Good descriptions help
        users understand the dataset's purpose, contents, and intended use.

        Args:
            dataset_rid: RID of the dataset to update.
            description: New description text.

        Returns:
            JSON with status, dataset_rid, description.

        Example:
            set_dataset_description("1-ABC", "Training images for CIFAR-10 classification")
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)

            # Update the description in the catalog
            pb = ml.pathBuilder()
            dataset_path = pb.schemas[ml.ml_schema].Dataset
            dataset_path.update([{"RID": dataset_rid, "Description": description}])

            # Update the local object
            dataset.description = description

            return json.dumps({
                "status": "updated",
                "dataset_rid": dataset_rid,
                "description": description,
            })
        except Exception as e:
            logger.error(f"Failed to set dataset description: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_dataset_type(dataset_rid: str, dataset_type: str) -> str:
        """Add a type to a dataset.

        Adds a Dataset_Type vocabulary term to this dataset. The type must exist
        in the Dataset_Type vocabulary.

        Args:
            dataset_rid: RID of the dataset.
            dataset_type: Name of the type to add (must exist in Dataset_Type vocabulary).

        Returns:
            JSON with status, dataset_rid, dataset_types list.

        Example:
            add_dataset_type("1-ABC", "Training") -> adds "Training" type to dataset
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            dataset.add_dataset_type(dataset_type)
            return json.dumps({
                "status": "added",
                "dataset_rid": dataset_rid,
                "dataset_types": dataset.dataset_types,
            })
        except Exception as e:
            logger.error(f"Failed to add dataset type: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def remove_dataset_type(dataset_rid: str, dataset_type: str) -> str:
        """Remove a type from a dataset.

        Removes a Dataset_Type vocabulary term from this dataset. The type must exist
        in the Dataset_Type vocabulary.

        Args:
            dataset_rid: RID of the dataset.
            dataset_type: Name of the type to remove.

        Returns:
            JSON with status, dataset_rid, dataset_types list.

        Example:
            remove_dataset_type("1-ABC", "Training") -> removes "Training" type from dataset
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            dataset.remove_dataset_type(dataset_type)
            return json.dumps({
                "status": "removed",
                "dataset_rid": dataset_rid,
                "dataset_types": dataset.dataset_types,
            })
        except Exception as e:
            logger.error(f"Failed to remove dataset type: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_dataset_element_type(table_name: str) -> str:
        """Register a domain table as a dataset element type.

        After registration, records from this table can be added to datasets
        using add_dataset_members(). Creates an association table to link
        records to datasets.

        Args:
            table_name: Name of the domain table to register (e.g., "Subject", "Image").

        Returns:
            JSON with status, table_name, association_table.

        Example:
            add_dataset_element_type("Subject") -> enables Subject records as dataset elements
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.add_dataset_element_type(table_name)
            return json.dumps({
                "status": "success",
                "table_name": table_name,
                "association_table": table.name,
            })
        except Exception as e:
            logger.error(f"Failed to add element type: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_dataset_child(
        parent_rid: str,
        child_rid: str,
    ) -> str:
        """Add a dataset as a nested child of another dataset.

        Creates a parent-child relationship between datasets. Common pattern:
        a "Complete" parent dataset contains "Training" and "Testing" children
        that partition the same data.

        Args:
            parent_rid: RID of the parent dataset.
            child_rid: RID of the child dataset to nest.

        Returns:
            JSON with status, parent_rid, child_rid.

        Example:
            add_dataset_child("1-ABC", "1-DEF") -> nests 1-DEF inside 1-ABC
        """
        try:
            ml = conn_manager.get_active_or_raise()
            parent = ml.lookup_dataset(parent_rid)
            # Add child dataset as a member - Dataset is a dataset element type
            # so adding a dataset RID creates a parent-child relationship
            parent.add_dataset_members(members=[child_rid])
            return json.dumps({
                "status": "added",
                "parent_rid": parent_rid,
                "child_rid": child_rid,
            })
        except Exception as e:
            logger.error(f"Failed to add dataset child: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_dataset_parents(
        dataset_rid: str,
        recurse: bool = False,
        version: str | None = None,
    ) -> str:
        """List all parent datasets that contain this dataset as a child.

        Args:
            dataset_rid: RID of the child dataset.
            recurse: If True, recursively list all ancestors (parents of parents).
            version: Semantic version to query (e.g., "1.0.0"). If not specified,
                uses the current version.

        Returns:
            JSON array of parent datasets with {rid, description, dataset_types, current_version}.

        Example:
            list_dataset_parents("1-ABC") -> direct parents only
            list_dataset_parents("1-ABC", recurse=True) -> all ancestors
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            parents = dataset.list_dataset_parents(recurse=recurse, version=version)
            return json.dumps([_serialize_dataset(p) for p in parents])
        except Exception as e:
            logger.error(f"Failed to list dataset parents: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def estimate_bag_size(
        dataset_rid: str,
        version: str,
        exclude_tables: list[str] | None = None,
    ) -> str:
        """Estimate the size of a dataset bag before downloading.

        Runs the same FK path traversal as a dataset bag download, then queries the
        snapshot catalog for row counts and asset file sizes. Use this to
        preview what a download will contain and how large it will be before
        committing to the full download.

        Args:
            dataset_rid: RID of the dataset to estimate.
            version: Semantic version to estimate (e.g., "1.0.0").
            exclude_tables: Optional list of table names to exclude from FK
                path traversal during bag export.

        Returns:
            JSON with:
                - tables: dict of table name -> {row_count, is_asset, asset_bytes}
                - total_rows: total row count across all tables
                - total_asset_bytes: total asset size in bytes
                - total_asset_size: human-readable size (e.g., "1.2 GB")
        """
        try:
            from deriva_ml.dataset.aux_classes import DatasetSpec

            ml = conn_manager.get_active_or_raise()
            spec = DatasetSpec(
                rid=dataset_rid,
                version=version,
                exclude_tables=set(exclude_tables) if exclude_tables else None,
            )
            estimate = ml.estimate_bag_size(spec)
            return json.dumps({"status": "success"} | estimate)
        except Exception as e:
            logger.error(f"Failed to estimate bag size: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def bag_info(
        dataset_rid: str,
        version: str,
        exclude_tables: list[str] | None = None,
    ) -> str:
        """Get comprehensive info about a dataset bag: size, contents, and cache status.

        Combines the size estimate (row counts, asset sizes per table) with
        local cache status. Use this to decide whether to cache a bag
        before running an experiment.

        Cache status values:
        - "not_cached": No local copy exists
        - "cached_metadata_only": Table data downloaded, assets not fetched
        - "cached_materialized": Fully downloaded and validated
        - "cached_incomplete": Was cached but some assets are missing

        Args:
            dataset_rid: RID of the dataset to inspect.
            version: Semantic version to inspect (e.g., "1.0.0").
            exclude_tables: Optional list of table names to exclude from FK
                path traversal.

        Returns:
            JSON with size info (tables, total_rows, total_asset_bytes,
            total_asset_size) plus cache_status and cache_path.
        """
        try:
            from deriva_ml.dataset.aux_classes import DatasetSpec

            ml = conn_manager.get_active_or_raise()
            spec = DatasetSpec(
                rid=dataset_rid,
                version=version,
                exclude_tables=set(exclude_tables) if exclude_tables else None,
            )
            info = ml.bag_info(spec)
            return json.dumps({"status": "success"} | info)
        except Exception as e:
            logger.error(f"Failed to get bag info: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def cache_dataset(
        dataset_rid: str | None = None,
        asset_rid: str | None = None,
        version: str | None = None,
        materialize: bool = True,
        exclude_tables: list[str] | None = None,
    ) -> str:
        """Download a dataset bag or asset into the local cache without creating an execution.

        Use this to warm the cache before running experiments. No execution or
        provenance records are created — this is purely a local download
        operation. After caching, subsequent download_dataset or
        download_execution_dataset calls will use the cached copy.

        Provide either dataset_rid (for bags) or asset_rid (for individual
        assets), not both.

        Args:
            dataset_rid: RID of a dataset to cache (mutually exclusive with asset_rid).
            asset_rid: RID of an asset to cache (mutually exclusive with dataset_rid).
            version: Dataset version to cache (required when using dataset_rid).
            materialize: If True (default), download all asset files in the bag.
                If False, download only table metadata (faster, smaller).
                Ignored for asset cache.
            exclude_tables: Optional list of table names to exclude from FK
                path traversal during bag export. Only applies to dataset cache.

        Returns:
            JSON with cache results. For datasets: bag_info including
            cache_status and size. For assets: file path and metadata.
        """
        try:
            ml = conn_manager.get_active_or_raise()

            if dataset_rid and asset_rid:
                return json.dumps({
                    "status": "error",
                    "message": "Provide either dataset_rid or asset_rid, not both.",
                })

            if dataset_rid:
                if not version:
                    dataset = ml.lookup_dataset(dataset_rid)
                    version = str(dataset.current_version)

                from deriva_ml.dataset.aux_classes import DatasetSpec
                spec = DatasetSpec(
                    rid=dataset_rid,
                    version=version,
                    materialize=materialize,
                    exclude_tables=set(exclude_tables) if exclude_tables else None,
                )
                result = ml.cache_dataset(spec, materialize=materialize)
                return json.dumps({"status": "success", "type": "dataset"} | result)

            elif asset_rid:
                # Download asset to cache without provenance
                from pathlib import Path
                cache_dir = ml.cache_dir / "assets"
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Resolve the asset to get its metadata
                asset_table = ml.resolve_rid(asset_rid).table
                asset_record = ml.retrieve_rid(asset_rid)
                asset_url = asset_record.get("URL")
                filename = asset_record.get("Filename", "unknown")
                md5 = asset_record.get("MD5")

                # Check cache first
                if md5:
                    cache_key = f"{asset_rid}_{md5}"
                    cached_file = cache_dir / cache_key / filename
                    if cached_file.exists():
                        return json.dumps({
                            "status": "success",
                            "type": "asset",
                            "cache_status": "cached",
                            "file_path": str(cached_file),
                            "asset_table": asset_table.name if hasattr(asset_table, "name") else str(asset_table),
                            "filename": filename,
                        })

                # Download the asset
                asset_dest = cache_dir / f"{asset_rid}_{md5 or 'nomd5'}"
                asset_dest.mkdir(parents=True, exist_ok=True)
                dest_file = asset_dest / filename

                from deriva.core import get_credential
                hostname = ml.catalog.deriva_server.server
                credentials = get_credential(hostname)
                from bdbag.fetch.fetcher import fetch_single_file
                fetch_single_file(asset_url, output_path=dest_file)

                return json.dumps({
                    "status": "success",
                    "type": "asset",
                    "cache_status": "cached",
                    "file_path": str(dest_file),
                    "asset_table": asset_table.name if hasattr(asset_table, "name") else str(asset_table),
                    "filename": filename,
                })

            else:
                return json.dumps({
                    "status": "error",
                    "message": "Provide either dataset_rid or asset_rid.",
                })
        except Exception as e:
            logger.error(f"Failed to cache dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def preview_denormalized_dataset(
        dataset_rid: str,
        include_tables: list[str],
        version: str | None = None,
        limit: int = 25,
    ) -> str:
        """Preview a denormalized (wide table) view of dataset tables.

        Joins related dataset tables into a single wide table and returns a sample
        of rows. Useful for understanding data shape, column names, and relationships
        before building ML pipelines.

        **This is a preview only** — results are not cached or stored. Use this to
        understand the data shape and decide what you need, then use the DerivaML
        Python API to access the full dataset for building subsets or ML pipelines.

        Tables are joined based on their foreign key relationships. Column names are
        prefixed with the source table name using dots (e.g., "Image.Filename",
        "Subject.RID", "Diagnosis.Label").

        Args:
            dataset_rid: RID of the dataset to preview.
            include_tables: List of table names to include in the join.
                Tables are joined based on their foreign key relationships.
                Order doesn't matter - the join order is determined automatically.
            version: Semantic version to query (e.g., "1.0.0"). If not specified,
                uses the current version.
            limit: Maximum rows to return (default: 25, max: 100).

        Returns:
            JSON with columns list and rows array.

        Example:
            preview_denormalized_dataset("1-ABC", ["Image", "Diagnosis"])
            -> {"columns": ["Image.RID", "Image.Filename", "Diagnosis.Label"], "rows": [...]}

            # Include subject info for analysis by demographics
            preview_denormalized_dataset("1-ABC", ["Subject", "Image", "Diagnosis"])
            -> {"columns": ["Subject.Age", "Subject.Gender", "Image.RID", ...], "rows": [...]}
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)

            # Hard cap at 100 rows
            limit = min(limit, 100)

            # Get denormalized data as dict
            rows = []
            for i, row in enumerate(dataset.denormalize_as_dict(include_tables, version=version)):
                if i >= limit:
                    break
                rows.append(dict(row))

            # Get column names from first row
            columns = list(rows[0].keys()) if rows else []

            return json.dumps({
                "dataset_rid": dataset_rid,
                "include_tables": include_tables,
                "columns": columns,
                "rows": rows,
                "count": len(rows),
                "limit": limit,
            })
        except Exception as e:
            logger.error(f"Failed to preview denormalized dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})


    # ========================================================================
    # Dataset Type convenience tools
    # ========================================================================

    @mcp.tool()
    async def create_dataset_type_term(
        type_name: str,
        description: str,
        synonyms: list[str] | None = None,
    ) -> str:
        """Create a new dataset type term in the Dataset_Type vocabulary.

        This creates a new vocabulary term that can then be assigned to datasets
        using add_dataset_type(). Dataset types help categorize datasets by their
        role in ML workflows.

        Common types include "Training", "Testing", "Validation", "Complete".

        Args:
            type_name: Name for the dataset type (must be unique).
            description: What this type of dataset is used for.
            synonyms: Alternative names that can match this type (e.g., ["train"] for "Training").

        Returns:
            JSON with status, name, description, synonyms, rid.

        Example:
            create_dataset_type_term("Validation", "Held-out data for hyperparameter tuning", ["val", "valid"])
        """
        try:
            ml = conn_manager.get_active_or_raise()
            term = ml.add_term(
                table="Dataset_Type",
                term_name=type_name,
                description=description,
                synonyms=synonyms or [],
                exists_ok=False,
            )
            return json.dumps({
                "status": "created",
                "name": term.name,
                "description": term.description,
                "synonyms": term.synonyms or [],
                "rid": term.rid,
            })
        except Exception as e:
            if "already exists" in str(e).lower():
                try:
                    existing = ml.lookup_term("Dataset_Type", type_name)
                    return json.dumps({
                        "status": "exists",
                        "name": existing.name,
                        "description": existing.description,
                        "synonyms": existing.synonyms or [],
                        "rid": existing.rid,
                    })
                except Exception:
                    pass
            logger.error(f"Failed to create dataset type term: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def delete_dataset_type_term(type_name: str) -> str:
        """Delete a dataset type term from the Dataset_Type vocabulary.

        WARNING: Only delete types that are not referenced by any datasets.
        If datasets use this type, the delete will fail with a foreign key error.
        Use remove_dataset_type() first to remove the type from all datasets.

        Args:
            type_name: Name of the dataset type to delete.

        Returns:
            JSON with status and deleted type name.

        Example:
            delete_dataset_type_term("Obsolete") -> {"status": "deleted", "name": "Obsolete"}
        """
        try:
            ml = conn_manager.get_active_or_raise()
            ml.delete_term("Dataset_Type", type_name)
            return json.dumps({
                "status": "deleted",
                "name": type_name,
            })
        except Exception as e:
            logger.error(f"Failed to delete dataset type term: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def split_dataset(
        source_dataset_rid: str,
        test_size: float = 0.2,
        train_size: float | None = None,
        val_size: float | None = None,
        seed: int = 42,
        shuffle: bool = True,
        stratify_by_column: str | None = None,
        stratify_missing: str = "error",
        element_table: str | None = None,
        include_tables: list[str] | None = None,
        training_types: list[str] | None = None,
        testing_types: list[str] | None = None,
        validation_types: list[str] | None = None,
        split_description: str = "",
        dry_run: bool = False,
    ) -> str:
        """Split a dataset into training, testing, and optionally validation subsets.

        Creates a new dataset hierarchy with full provenance tracking:
        - Split (parent, type: "Split")
          - Training (child, type: "Training" + training_types)
          - Validation (child, type: "Validation" + validation_types)  # if val_size
          - Testing (child, type: "Testing" + testing_types)

        The API follows scikit-learn's train_test_split conventions for
        test_size, train_size, val_size, shuffle, and seed parameters.

        **Splitting strategies:**

        - **Random** (default): Shuffles members and splits at the boundary.
          No denormalization needed. Fast for any dataset size.
        - **Stratified**: Maintains class distribution across splits.
          Requires stratify_by_column and include_tables. Uses scikit-learn
          internally.

        **Column naming for stratification:**

        When using stratify_by_column, the column name must match the
        denormalized DataFrame format: ``{TableName}_{ColumnName}``.
        For example, to stratify by the Image_Class column from the
        Image_Classification feature table, use
        ``Image_Classification_Image_Class``.

        Derive the column name from the table schema (via the
        ``deriva://catalog/schema`` or ``deriva://catalog/features`` resource)
        rather than calling denormalize_dataset().

        Args:
            source_dataset_rid: RID of the source dataset to split.
            test_size: Test set size as a fraction (0-1) or absolute count.
                Default: 0.2 (20% of data).
            train_size: Train set size as a fraction (0-1) or absolute count.
                Default: None (complement of test_size and val_size).
            val_size: Validation set size as a fraction (0-1) or absolute
                count. Default: None (no validation split, two-way only).
                When provided, creates a three-way train/val/test split.
            seed: Random seed for reproducibility. Default: 42.
            shuffle: Whether to shuffle before splitting. Default: True.
            stratify_by_column: Column name in the denormalized DataFrame
                for stratified splitting. Maintains class distribution
                across all partitions. Requires include_tables.
                Example: "Image_Classification_Image_Class".
            stratify_missing: Policy for null values in the stratify column.
                "error" (default): raise if any nulls exist, reporting count
                and percentage. "drop": exclude rows with null values from
                the split. "include": treat nulls as a separate class.
                Only used when stratify_by_column is set.
            element_table: Element table to split (e.g., "Image"). If not
                specified, auto-detected from the dataset's members.
            include_tables: Tables to include when denormalizing. Required
                when using stratify_by_column.
                Example: ["Image", "Image_Classification"].
            training_types: Additional dataset types for the training set
                beyond "Training". Example: ["Labeled"].
            testing_types: Additional dataset types for the testing set
                beyond "Testing". Example: ["Labeled"].
            validation_types: Additional dataset types for the validation
                set beyond "Validation". Example: ["Labeled"].
                Ignored when val_size is None.
            split_description: Description for the parent Split dataset.
            dry_run: If True, return what would happen without modifying
                the catalog. Useful for previewing split sizes.

        Returns:
            JSON with split results including:
            - split: RID, version, and count of the parent Split dataset
            - training: RID, version, and count of the Training dataset
            - validation: RID, version, and count of the Validation dataset (if val_size)
            - testing: RID, version, and count of the Testing dataset
            - source: RID of the source dataset

        Example:
            # Random 80/20 split
            split_dataset("28D0", test_size=0.2, seed=42)

            # Three-way train/val/test split
            split_dataset("28D0", test_size=0.2, val_size=0.1, seed=42)

            # Stratified split maintaining class balance
            split_dataset("28D0", test_size=0.2,
                         stratify_by_column="Image_Classification_Image_Class",
                         include_tables=["Image", "Image_Classification"])

            # Fixed-count split with labeled types
            split_dataset("28D0", train_size=400, test_size=100,
                         training_types=["Labeled"], testing_types=["Labeled"])

            # Dry run to preview
            split_dataset("28D0", test_size=0.2, dry_run=True)
        """
        try:
            from deriva_ml.dataset.split import split_dataset as _split_dataset

            ml = conn_manager.get_active_or_raise()

            result = _split_dataset(
                ml=ml,
                source_dataset_rid=source_dataset_rid,
                test_size=test_size,
                train_size=train_size,
                val_size=val_size,
                seed=seed,
                shuffle=shuffle,
                stratify_by_column=stratify_by_column,
                stratify_missing=stratify_missing,
                element_table=element_table,
                include_tables=include_tables,
                training_types=training_types,
                testing_types=testing_types,
                validation_types=validation_types,
                split_description=split_description,
                dry_run=dry_run,
            )

            return json.dumps({"status": "success", **result.model_dump()})
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})
