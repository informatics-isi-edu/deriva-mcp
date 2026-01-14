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
- **Minor** (0.X.0): Added or removed elements (auto-incremented by add_dataset_members)
- **Major** (X.0.0): Schema changes, breaking changes to data structure

Each version captures a catalog snapshot, allowing queries against historical states.

**Dataset Bags (BDBags)**:
Exported, portable packages containing dataset elements and metadata. Bags can be:
- Downloaded locally for offline ML training
- Shared via MINID (Minimal Viable Identifier) for reproducibility
- Materialized to fetch all referenced asset files
"""

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
        """List all datasets in the catalog with their types and current versions.

        Returns datasets with their Dataset_Type labels (e.g., "Training", "Testing")
        and current semantic version. Use get_dataset() for full details including
        nested dataset relationships.

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
        """Get full details about a dataset including nested dataset relationships.

        Returns dataset metadata plus its position in the dataset hierarchy:
        - children: Nested datasets contained within this dataset
        - parents: Datasets that contain this dataset as a nested child

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
    async def list_dataset_members(dataset_rid: str, version: str | None = None) -> str:
        """List all dataset elements (records) grouped by table type.

        Dataset elements are records from domain tables that have been added to this
        dataset. Results are grouped by table name. Specify a version to query
        historical states for reproducibility.

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
        """Add records as dataset elements. Auto-increments minor version.

        Records must be from tables registered as dataset element types.
        Use add_dataset_element_type() to register a table, or list_dataset_element_types()
        to see which tables are already registered.

        Adding members automatically increments the dataset's minor version for change tracking.

        This tool accepts a list of RIDs and automatically resolves each RID to
        determine which table it belongs to. For better performance with large
        numbers of members, use the Python API directly with a dict mapping
        table names to RID lists (skips RID resolution).

        Args:
            dataset_rid: The RID of the dataset to add members to.
            member_rids: List of RIDs to add (e.g., ["2-ABC", "2-DEF", "2-GHI"]).

        Returns:
            JSON with status, added_count, dataset_rid.

        Example:
            add_dataset_members("1-ABC", ["2-DEF", "2-GHI"]) -> adds 2 Image records
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
        """Get all versions of a dataset for reproducibility tracking.

        Returns the complete version history with semantic versions (major.minor.patch),
        descriptions of changes, and catalog snapshots. Each snapshot allows querying
        the exact state of data at that version.

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
    async def list_dataset_element_types() -> str:
        """List which tables are registered as dataset element types.

        Only records from registered element types can be added to datasets.
        Use add_dataset_element_type() to register additional domain tables.

        Returns:
            JSON array of table names that can contain dataset elements.
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
    async def list_dataset_children(dataset_rid: str) -> str:
        """List all nested child datasets.

        Args:
            dataset_rid: RID of the parent dataset.

        Returns:
            JSON array of child dataset RIDs.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            children = dataset.list_dataset_children()
            return json.dumps({
                "dataset_rid": dataset_rid,
                "children": children,
            })
        except Exception as e:
            logger.error(f"Failed to list dataset children: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_dataset_parents(dataset_rid: str) -> str:
        """List all parent datasets that contain this dataset as a child.

        Args:
            dataset_rid: RID of the child dataset.

        Returns:
            JSON array of parent dataset RIDs.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            dataset = ml.lookup_dataset(dataset_rid)
            parents = dataset.list_dataset_parents()
            return json.dumps({
                "dataset_rid": dataset_rid,
                "parents": parents,
            })
        except Exception as e:
            logger.error(f"Failed to list dataset parents: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def download_dataset(
        dataset_rid: str,
        version: str | None = None,
        materialize: bool = True,
    ) -> str:
        """Download a dataset as a BDBag for local processing.

        Downloads the dataset to a local directory. Use this for standalone
        processing outside an execution context. For tracked ML workflows,
        use download_execution_dataset instead.

        Args:
            dataset_rid: RID of the dataset to download.
            version: Specific version (default: current).
            materialize: Fetch all referenced asset files (default: True).

        Returns:
            JSON with dataset_rid, version, path (local directory), bag_id.
        """
        try:
            from deriva_ml.dataset.aux_classes import DatasetSpec

            ml = conn_manager.get_active_or_raise()
            spec = DatasetSpec(rid=dataset_rid, version=version, materialize=materialize)
            bag = ml.download_dataset_bag(spec)

            return json.dumps({
                "status": "downloaded",
                "dataset_rid": bag.dataset_rid,
                "version": str(bag.version) if bag.version else None,
                "path": str(bag.path),
                "bag_id": id(bag),  # For referencing in subsequent calls
            })
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def denormalize_dataset(
        dataset_rid: str,
        include_tables: list[str],
        version: str | None = None,
        limit: int = 1000,
    ) -> str:
        """Denormalize dataset tables into a flat structure for ML.

        Joins related tables together to produce a "wide" view of the data,
        with columns from multiple tables combined. Column names are prefixed
        with the source table name (e.g., "Image.Filename", "Subject.RID").

        This is useful for:
        - Creating training data with all features in one table
        - Joining images with their labels/diagnoses
        - Combining subject metadata with associated records

        Args:
            dataset_rid: RID of the dataset to denormalize.
            include_tables: List of table names to include in the join.
            version: Specific version (default: current).
            limit: Maximum rows to return (default: 1000).

        Returns:
            JSON with columns list and rows array.

        Example:
            denormalize_dataset("1-ABC", ["Image", "Diagnosis"])
            -> {"columns": ["Image.RID", "Image.Filename", "Diagnosis.Name"], "rows": [...]}
        """
        try:
            from deriva_ml.dataset.aux_classes import DatasetSpec

            ml = conn_manager.get_active_or_raise()
            spec = DatasetSpec(rid=dataset_rid, version=version, materialize=False)
            bag = ml.download_dataset_bag(spec)

            # Get denormalized data as dict
            rows = []
            for i, row in enumerate(bag.denormalize_as_dict(include_tables)):
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
            logger.error(f"Failed to denormalize dataset: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_dataset_table(
        dataset_rid: str,
        table_name: str,
        version: str | None = None,
        limit: int = 1000,
    ) -> str:
        """Get all records from a specific table in a dataset.

        Returns records from the specified table that are part of this dataset.
        Useful for accessing raw table data before denormalization.

        Args:
            dataset_rid: RID of the dataset.
            table_name: Name of the table to retrieve.
            version: Specific version (default: current).
            limit: Maximum records to return (default: 1000).

        Returns:
            JSON with table name and records array.

        Example:
            get_dataset_table("1-ABC", "Image") -> all images in dataset
        """
        try:
            from deriva_ml.dataset.aux_classes import DatasetSpec

            ml = conn_manager.get_active_or_raise()
            spec = DatasetSpec(rid=dataset_rid, version=version, materialize=False)
            bag = ml.download_dataset_bag(spec)

            # Get table data
            rows = []
            for i, row in enumerate(bag.get_table_as_dict(table_name)):
                if i >= limit:
                    break
                rows.append(row)

            return json.dumps({
                "dataset_rid": dataset_rid,
                "table": table_name,
                "records": rows,
                "count": len(rows),
                "limit": limit,
            })
        except Exception as e:
            logger.error(f"Failed to get dataset table: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    # ========================================================================
    # Dataset Type convenience tools
    # ========================================================================

    @mcp.tool()
    async def list_dataset_types() -> str:
        """List all available dataset types from the Dataset_Type vocabulary.

        Dataset types categorize datasets by their role in ML workflows
        (e.g., "Training", "Testing", "Validation", "Complete").

        Returns:
            JSON array of {name, description, synonyms, rid} for each type.

        Example:
            list_dataset_types() -> [
                {"name": "Training", "description": "Data for model training", ...},
                {"name": "Testing", "description": "Held-out test data", ...}
            ]
        """
        try:
            ml = conn_manager.get_active_or_raise()
            terms = ml.list_vocabulary_terms("Dataset_Type")
            result = []
            for term in terms:
                result.append({
                    "name": term.name,
                    "description": term.description,
                    "synonyms": term.synonyms or [],
                    "rid": term.rid,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list dataset types: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_dataset_type(
        type_name: str,
        description: str,
        synonyms: list[str] | None = None,
    ) -> str:
        """Add a new dataset type to the Dataset_Type vocabulary.

        Dataset types help categorize datasets by their role in ML workflows.
        Common types include "Training", "Testing", "Validation", "Complete".

        Args:
            type_name: Name for the dataset type (must be unique).
            description: What this type of dataset is used for.
            synonyms: Alternative names that can match this type (e.g., ["train"] for "Training").

        Returns:
            JSON with status, name, description, synonyms, rid.

        Example:
            add_dataset_type("Validation", "Held-out data for hyperparameter tuning", ["val", "valid"])
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
            logger.error(f"Failed to add dataset type: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def delete_dataset_type(type_name: str) -> str:
        """Delete a dataset type from the Dataset_Type vocabulary.

        WARNING: Only delete types that are not referenced by any datasets.
        If datasets use this type, the delete will fail with a foreign key error.

        Args:
            type_name: Name of the dataset type to delete.

        Returns:
            JSON with status and deleted type name.

        Example:
            delete_dataset_type("Obsolete") -> {"status": "deleted", "name": "Obsolete"}
        """
        try:
            ml = conn_manager.get_active_or_raise()
            # Look up the term to get its RID
            term = ml.lookup_term("Dataset_Type", type_name)

            # Delete using direct ERMrest delete
            vocab_table = ml.model.schemas[ml.ml_schema].tables["Dataset_Type"]
            vocab_table.delete_rows(ml.catalog, [{"RID": term.rid}])

            return json.dumps({
                "status": "deleted",
                "name": type_name,
                "rid": term.rid,
            })
        except Exception as e:
            error_msg = str(e)
            if "foreign key" in error_msg.lower() or "constraint" in error_msg.lower():
                return json.dumps({
                    "status": "error",
                    "message": f"Cannot delete '{type_name}': it is referenced by existing datasets. "
                               "Remove the type from all datasets before deleting.",
                })
            logger.error(f"Failed to delete dataset type: {e}")
            return json.dumps({"status": "error", "message": error_msg})
