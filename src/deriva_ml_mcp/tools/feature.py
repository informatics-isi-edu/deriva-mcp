"""Feature management tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-ml-mcp")


def register_feature_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register feature management tools with the MCP server."""

    @mcp.tool()
    async def list_features(table_name: str) -> str:
        """List all features defined for a table.

        Features are named attributes attached to records (e.g., labels, scores, tags).

        Args:
            table_name: Name of the table to list features for (e.g., "Image", "Subject").

        Returns:
            JSON array of {feature_name, target_table, feature_table}.

        Example:
            list_features("Image") -> [{"feature_name": "Quality", ...}, ...]
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)
            features = ml.model.find_features(table)
            result = []
            for f in features:
                result.append({
                    "feature_name": f.feature_name,
                    "target_table": f.target_table.name,
                    "feature_table": f.feature_table.name,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list features: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def lookup_feature(table_name: str, feature_name: str) -> str:
        """Get details about a specific feature including its columns.

        Args:
            table_name: Name of the table the feature is attached to.
            feature_name: Name of the feature.

        Returns:
            JSON with feature_name, target_table, feature_table, columns.

        Example:
            lookup_feature("Image", "Quality") -> shows Quality feature structure
        """
        try:
            ml = conn_manager.get_active_or_raise()
            feature = ml.lookup_feature(table_name, feature_name)
            return json.dumps({
                "feature_name": feature.feature_name,
                "target_table": feature.target_table.name,
                "feature_table": feature.feature_table.name,
                "columns": [c.name for c in feature.feature_table.columns],
            })
        except Exception as e:
            logger.error(f"Failed to lookup feature: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_feature_values(table_name: str, feature_name: str) -> str:
        """Get all values/instances of a feature.

        Args:
            table_name: Name of the table the feature is attached to.
            feature_name: Name of the feature.

        Returns:
            JSON array of feature value records.

        Example:
            list_feature_values("Image", "Quality") -> [{"Image": "1-ABC", "Quality": "Good"}, ...]
        """
        try:
            ml = conn_manager.get_active_or_raise()
            values = ml.list_feature_values(table_name, feature_name)
            result = [dict(v) for v in values]
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list feature values: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def create_feature(
        table_name: str,
        feature_name: str,
        comment: str = "",
        terms: list[str] | None = None,
        assets: list[str] | None = None,
    ) -> str:
        """Create a new feature definition for a table.

        Features can reference vocabulary terms (categorical) and/or asset tables.

        Args:
            table_name: Table to attach the feature to (e.g., "Image").
            feature_name: Unique name for the feature (e.g., "Quality", "Label").
            comment: Description of what this feature represents.
            terms: Vocabulary table names whose terms can be values (e.g., ["Quality_Level"]).
            assets: Asset table names that can be referenced.

        Returns:
            JSON with status, feature_name, target_table.

        Example:
            create_feature("Image", "Quality", "Image quality rating", terms=["Quality_Level"])
        """
        try:
            ml = conn_manager.get_active_or_raise()
            ml.create_feature(
                target_table=table_name,
                feature_name=feature_name,
                terms=terms or [],
                assets=assets or [],
                comment=comment,
            )
            return json.dumps({
                "status": "created",
                "feature_name": feature_name,
                "target_table": table_name,
            })
        except Exception as e:
            logger.error(f"Failed to create feature: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def delete_feature(table_name: str, feature_name: str) -> str:
        """Delete a feature definition and all its values. Cannot be undone.

        Args:
            table_name: Table the feature is attached to.
            feature_name: Name of the feature to delete.

        Returns:
            JSON with status, feature_name, table_name.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            success = ml.delete_feature(table_name, feature_name)
            if success:
                return json.dumps({
                    "status": "deleted",
                    "feature_name": feature_name,
                    "table_name": table_name,
                })
            return json.dumps({
                "status": "not_found",
                "message": f"Feature '{feature_name}' not found in table '{table_name}'",
            })
        except Exception as e:
            logger.error(f"Failed to delete feature: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_feature_names() -> str:
        """List all defined feature names across the catalog.

        Returns:
            JSON array of {name, description} for each feature name.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            terms = ml.list_vocabulary_terms("Feature_Name")
            result = []
            for term in terms:
                result.append({
                    "name": term.name,
                    "description": term.description,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list feature names: {e}")
            return json.dumps({"status": "error", "message": str(e)})
