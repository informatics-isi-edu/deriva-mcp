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
        """List all datasets in the catalog.

        Returns a list of all datasets, including their RIDs, descriptions,
        types, and current versions.

        Args:
            include_deleted: If True, include deleted datasets in the list.

        Returns:
            JSON array of dataset information.
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
                    "current_version": ds.current_version,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_dataset(dataset_rid: str) -> str:
        """Get detailed information about a specific dataset.

        Retrieves full details about a dataset including its members,
        version history, and nested datasets.

        Args:
            dataset_rid: Resource Identifier of the dataset.

        Returns:
            JSON object with dataset details.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            return json.dumps({
                "rid": dataset.dataset_rid,
                "description": dataset.description,
                "dataset_types": dataset.dataset_types,
                "current_version": dataset.current_version,
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
        """Create a new dataset in the catalog.

        Creates an empty dataset that can then have members added to it.

        Args:
            description: Description of the dataset's purpose.
            dataset_types: List of dataset type terms from the Dataset_Type vocabulary.
            version: Optional initial version (defaults to '0.1.0').

        Returns:
            JSON object with the created dataset's RID and details.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.create_dataset(
                description=description,
                dataset_types=dataset_types or [],
                version=version,
            )
            return json.dumps({
                "status": "created",
                "rid": dataset.dataset_rid,
                "description": dataset.description,
                "dataset_types": dataset.dataset_types,
                "version": dataset.current_version,
            })
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_dataset_members(dataset_rid: str, version: str | None = None) -> str:
        """List all members of a dataset.

        Returns the objects (entities) that are part of this dataset,
        organized by table type.

        Args:
            dataset_rid: Resource Identifier of the dataset.
            version: Optional version to list members for (defaults to current).

        Returns:
            JSON object mapping table names to arrays of member RIDs.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            if version:
                dataset = dataset.set_version(version)
            members = dataset.list_dataset_members()
            # Convert to JSON-serializable format
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
        """Add members to a dataset.

        Adds the specified objects to the dataset by their RIDs.

        Args:
            dataset_rid: Resource Identifier of the dataset.
            member_rids: List of RIDs to add to the dataset.

        Returns:
            Status message with count of added members.
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
        """Get the version history of a dataset.

        Returns the complete version history including timestamps
        and descriptions for each version.

        Args:
            dataset_rid: Resource Identifier of the dataset.

        Returns:
            JSON array of version history entries.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            history = dataset.dataset_history()
            result = []
            for entry in history:
                result.append({
                    "version": entry.version,
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
        """Increment the version of a dataset.

        Updates the dataset version using semantic versioning.

        Args:
            dataset_rid: Resource Identifier of the dataset.
            component: Version component to increment: 'major', 'minor', or 'patch'.
            description: Description of what changed in this version.

        Returns:
            JSON object with the new version number.
        """
        try:
            from deriva_ml.dataset.aux_classes import VersionPart

            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)

            # Map string to enum
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
                "new_version": new_version,
                "dataset_rid": dataset_rid,
            })
        except Exception as e:
            logger.error(f"Failed to increment version: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def delete_dataset(dataset_rid: str, recurse: bool = False) -> str:
        """Delete a dataset from the catalog.

        Soft-deletes the dataset (marks as deleted but retains data).

        Args:
            dataset_rid: Resource Identifier of the dataset to delete.
            recurse: If True, also delete nested child datasets.

        Returns:
            Status message indicating success or failure.
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
        """List types of entities that can be added to datasets.

        Returns the table types that are configured to be included
        as elements in datasets.

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
        """Add a table type as a valid dataset element.

        Configures the specified table to be includable as an
        element in datasets.

        Args:
            table_name: Name of the table to enable as a dataset element type.

        Returns:
            Status message with the association table name.
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
