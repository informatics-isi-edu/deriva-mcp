"""Feature management tools for DerivaML MCP server.

Features are a core concept in DerivaML for ML data engineering. A feature associates
metadata with domain objects (e.g., Images, Subjects) to support ML workflows:

- **Labels/Annotations**: Categorical values from controlled vocabularies (e.g., "diagnosis", "quality_score")
- **Computed Values**: Numeric or text values produced by ML models or processing pipelines
- **Related Assets**: Links to derived assets (e.g., segmentation masks, embeddings)

Key properties of features:
1. **Provenance Tracking**: Every feature value records which Execution produced it
2. **Vocabulary-Controlled**: Categorical features use controlled vocabulary terms for consistency
3. **Multi-valued**: An object can have multiple values for the same feature (e.g., multiple labels)
4. **Versioned**: Feature values are included in dataset versions for reproducibility

Common use cases:
- Ground truth labels for training data
- Model predictions for inference results
- Quality scores from data validation
- Derived measurements from analysis pipelines
"""

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

        Features associate metadata (labels, scores, assets) with domain objects for ML workflows.
        Each feature tracks provenance via the Execution that produced its values.

        Args:
            table_name: Name of the table to list features for (e.g., "Image", "Subject").

        Returns:
            JSON array of {feature_name, target_table, feature_table}.

        Example:
            list_features("Image") -> [{"feature_name": "Diagnosis", ...}, {"feature_name": "Quality", ...}]
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
        """Get details about a specific feature including its structure.

        Returns the feature's underlying table structure, which includes columns for:
        - The target object reference (e.g., Image RID)
        - The Execution that produced this value (provenance)
        - The feature value(s) - vocabulary terms, numeric values, or asset references

        Args:
            table_name: Name of the table the feature is attached to.
            feature_name: Name of the feature.

        Returns:
            JSON with feature_name, target_table, feature_table, columns.

        Example:
            lookup_feature("Image", "Diagnosis") -> shows Diagnosis feature structure
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
        """Get all feature values across objects, including provenance.

        Returns every instance where this feature has been assigned to an object.
        Each record includes the target object RID, the feature value(s), and
        the Execution RID that produced it (for provenance tracking).

        Args:
            table_name: Name of the table the feature is attached to.
            feature_name: Name of the feature.

        Returns:
            JSON array of feature value records with target RID, value, and Execution.

        Example:
            list_feature_values("Image", "Diagnosis") -> [{"Image": "1-ABC", "Diagnosis": "Normal", "Execution": "2-XYZ"}, ...]
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
        """Create a new feature definition to associate metadata with domain objects.

        Features enable ML data engineering by linking labels, scores, or derived assets
        to domain objects. The feature definition specifies what types of values are valid.

        Feature types:
        - **Term-based**: Values come from controlled vocabularies (e.g., diagnosis labels)
        - **Asset-based**: Values reference asset files (e.g., segmentation masks)
        - **Mixed**: Can reference both terms and assets

        The feature automatically tracks which Execution produced each value for provenance.

        Args:
            table_name: Table to attach the feature to (e.g., "Image", "Subject").
            feature_name: Unique name for the feature (e.g., "Diagnosis", "Quality_Score").
            comment: Description of what this feature represents.
            terms: Vocabulary table names whose terms can be values (e.g., ["Diagnosis_Type"]).
            assets: Asset table names that can be referenced (e.g., ["Segmentation_Mask"]).

        Returns:
            JSON with status, feature_name, target_table.

        Example:
            create_feature("Image", "Diagnosis", "Clinical diagnosis label", terms=["Diagnosis_Type"])
            create_feature("Image", "Segmentation", "Derived segmentation mask", assets=["Segmentation_Mask"])
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

        WARNING: This permanently removes the feature table and all associated values.
        All provenance information for this feature will be lost.

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
        """List all registered feature names from the Feature_Name vocabulary.

        Feature names are controlled vocabulary terms that identify features across tables.
        The same feature name can be used on multiple tables (e.g., "Quality" on both
        Image and Subject tables).

        Returns:
            JSON array of {name, description} for each registered feature name.
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
