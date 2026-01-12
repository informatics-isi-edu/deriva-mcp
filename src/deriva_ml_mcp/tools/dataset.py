"""Dataset management tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-ml-mcp")


def register_dataset_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register dataset management tools with the MCP server."""

    @mcp.tool()
    async def list_datasets(include_deleted: bool = False) -> str:
        """List all datasets in the catalog with their RIDs, types, and versions.

        Args:
            include_deleted: Set True to include soft-deleted datasets.

        Returns:
            JSON array of {rid, description, dataset_types, current_version}.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            datasets = ml.find_datasets(deleted=include_deleted)
            result = []
            for ds in datasets:
                result.append({
                    "rid": ds.dataset_rid,
                    "description": ds.description,
                    "dataset_types": ds.dataset_types,
                    "current_version": str(ds.current_version) if ds.current_version else None,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_dataset(dataset_rid: str) -> str:
        """Get full details about a dataset including its children and parents.

        Args:
            dataset_rid: The RID of the dataset (e.g., "1-ABC").

        Returns:
            JSON with rid, description, dataset_types, current_version, children, parents.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            return json.dumps({
                "rid": dataset.dataset_rid,
                "description": dataset.description,
                "dataset_types": dataset.dataset_types,
                "current_version": str(dataset.current_version) if dataset.current_version else None,
                "children": dataset.list_dataset_children(),
                "parents": dataset.list_dataset_parents(),
            })
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_rid}: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def create_dataset(
        description: str = "",
        dataset_types: list[str] | None = None,
        version: str | None = None,
    ) -> str:
        """Create a new empty dataset. Use add_dataset_members to populate it.

        Args:
            description: Human-readable description of the dataset's purpose.
            dataset_types: Type labels from Dataset_Type vocabulary (e.g., ["Training", "Image"]).
            version: Initial version string (default: "0.1.0").

        Returns:
            JSON with status, rid, description, dataset_types, version.

        Example:
            create_dataset("Training images for model v2", ["Training", "Image"])
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.create_dataset(
                description=description,
                dataset_types=dataset_types or [],
                version=version,
            )
            version_str = str(dataset.current_version) if dataset.current_version else None
            return json.dumps({
                "status": "created",
                "rid": dataset.dataset_rid,
                "description": dataset.description,
                "dataset_types": dataset.dataset_types,
                "version": version_str,
            })
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_dataset_members(dataset_rid: str, version: str | None = None) -> str:
        """List all members (records) in a dataset, grouped by table.

        Args:
            dataset_rid: The RID of the dataset.
            version: Specific version to query (default: current version).

        Returns:
            JSON object mapping table names to arrays of {RID} objects.

        Example:
            list_dataset_members("1-ABC") -> {"Image": [{"RID": "2-DEF"}, ...], "Subject": [...]}
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            if version:
                dataset = dataset.set_version(version)
            members = dataset.list_dataset_members()
            result = {}
            for table_name, items in members.items():
                result[table_name] = [{"RID": m["RID"]} for m in items]
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list dataset members: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_dataset_members(
        dataset_rid: str,
        member_rids: list[str],
    ) -> str:
        """Add records to a dataset by their RIDs. Auto-increments minor version.

        Args:
            dataset_rid: The RID of the dataset to add members to.
            member_rids: List of RIDs to add (e.g., ["2-ABC", "2-DEF", "2-GHI"]).

        Returns:
            JSON with status, added_count, dataset_rid.

        Example:
            add_dataset_members("1-ABC", ["2-DEF", "2-GHI"]) -> adds 2 members
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            dataset.add_dataset_members(members=member_rids)
            return json.dumps({
                "status": "success",
                "added_count": len(member_rids),
                "dataset_rid": dataset_rid,
            })
        except Exception as e:
            logger.error(f"Failed to add dataset members: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_dataset_version_history(dataset_rid: str) -> str:
        """Get all versions of a dataset with their descriptions and timestamps.

        Args:
            dataset_rid: The RID of the dataset.

        Returns:
            JSON array of {version, description, snapshot} entries.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            history = dataset.dataset_history()
            result = []
            for entry in history:
                result.append({
                    "version": str(entry.version) if entry.version else None,
                    "description": entry.description,
                    "snapshot": entry.snapshot,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to get dataset history: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def increment_dataset_version(
        dataset_rid: str,
        component: str = "minor",
        description: str = "",
    ) -> str:
        """Manually increment a dataset's semantic version (major.minor.patch).

        Args:
            dataset_rid: The RID of the dataset.
            component: Which part to increment: "major", "minor", or "patch".
            description: Description of what changed in this version.

        Returns:
            JSON with status, new_version, dataset_rid.

        Examples:
            increment_dataset_version("1-ABC", "major", "Schema change")  -> 2.0.0
            increment_dataset_version("1-ABC", "patch", "Fixed labels")   -> 1.0.1
        """
        try:
            from deriva_ml.dataset.aux_classes import VersionPart

            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)

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
                "dataset_rid": dataset_rid,
            })
        except Exception as e:
            logger.error(f"Failed to increment version: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def delete_dataset(dataset_rid: str, recurse: bool = False) -> str:
        """Soft-delete a dataset (marks deleted but keeps data).

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
    async def list_dataset_element_types() -> str:
        """List which table types can be added as dataset members.

        Use add_dataset_element_type to enable additional tables.

        Returns:
            JSON array of table names that can be dataset members.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            element_types = ml.list_dataset_element_types()
            result = [t.name for t in element_types]
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list element types: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_dataset_element_type(table_name: str) -> str:
        """Enable a table to be included as dataset members.

        Args:
            table_name: Name of the table to enable (e.g., "Subject", "Image").

        Returns:
            JSON with status, table_name, association_table.

        Example:
            add_dataset_element_type("Subject") -> enables Subject records in datasets
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
