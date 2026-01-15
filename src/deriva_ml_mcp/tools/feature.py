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
            features = ml.find_features(table_name)
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

        **Understanding feature fields:**
        When adding feature values with `add_feature_value`, you need to provide values
        for the feature's fields. This tool shows you:
        - `target_column`: The column name for the target record RID (required)
        - `term_columns`: Columns that accept vocabulary term names
        - `asset_columns`: Columns that accept asset RIDs
        - `value_columns`: Columns that accept direct values (numbers, strings)

        For example, if a "Diagnosis" feature has term_columns=["Diagnosis_Type"],
        you would call add_feature_value with values={"Diagnosis_Type": "Normal"}.

        Args:
            table_name: Name of the table the feature is attached to.
            feature_name: Name of the feature.

        Returns:
            JSON with feature structure including:
            - feature_name: Name of the feature
            - target_table: Table the feature is attached to
            - target_column: Column name for the target record RID
            - term_columns: Columns accepting vocabulary terms (with their vocabulary table)
            - asset_columns: Columns accepting asset RIDs (with their asset table)
            - value_columns: Columns accepting direct values (with their type)
            - required_fields: Fields that must be provided (non-nullable)

        Example:
            lookup_feature("Image", "Diagnosis") -> shows Diagnosis feature structure
        """
        try:
            ml = conn_manager.get_active_or_raise()
            feature = ml.lookup_feature(table_name, feature_name)

            # Get column details for better documentation
            term_cols = {}
            for col in feature.term_columns:
                # Find the FK to get the vocabulary table name
                for fk in feature.feature_table.foreign_keys:
                    if col in fk.foreign_key_columns:
                        term_cols[col.name] = {
                            "vocabulary_table": fk.pk_table.name,
                            "required": not col.nullok,
                        }
                        break

            asset_cols = {}
            for col in feature.asset_columns:
                for fk in feature.feature_table.foreign_keys:
                    if col in fk.foreign_key_columns:
                        asset_cols[col.name] = {
                            "asset_table": fk.pk_table.name,
                            "required": not col.nullok,
                        }
                        break

            value_cols = {}
            for col in feature.value_columns:
                value_cols[col.name] = {
                    "type": col.type.typename,
                    "required": not col.nullok,
                }

            # Determine required fields
            required_fields = [feature.target_table.name]  # Target is always required
            required_fields.extend([c for c, info in term_cols.items() if info["required"]])
            required_fields.extend([c for c, info in asset_cols.items() if info["required"]])
            required_fields.extend([c for c, info in value_cols.items() if info["required"]])

            return json.dumps({
                "feature_name": feature.feature_name,
                "target_table": feature.target_table.name,
                "target_column": feature.target_table.name,
                "feature_table": feature.feature_table.name,
                "term_columns": term_cols,
                "asset_columns": asset_cols,
                "value_columns": value_cols,
                "required_fields": required_fields,
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

        **What this creates**:
        1. A new association table in the domain schema to store feature values
        2. A dynamically generated Pydantic model class for creating validated feature instances

        The Pydantic model class (accessible via `feature_record_class()` in Python) provides
        type-safe construction of feature records with automatic validation against the
        feature's definition.

        **Feature types**:
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

    @mcp.tool()
    async def add_feature_value(
        table_name: str,
        feature_name: str,
        target_rid: str,
        value: str,
        execution_rid: str | None = None,
    ) -> str:
        """Add a feature value to a domain object.

        Associates a feature value (term, asset, or other) with a target record.
        If an execution is active, it will be used for provenance. Otherwise,
        provide an execution_rid explicitly.

        **For simple features** (single term or asset column):
        Use this tool with a single `value` string.

        **For complex features** (multiple columns):
        Use `add_feature_value_record` instead, which accepts a dictionary of
        field values. Use `lookup_feature` first to see what fields are available.

        Args:
            table_name: Table the target record belongs to (e.g., "Image").
            feature_name: Name of the feature (e.g., "Diagnosis").
            target_rid: RID of the target record to annotate.
            value: The feature value - either a term name or asset RID.
            execution_rid: Execution RID for provenance (uses active if not provided).

        Returns:
            JSON with status, target_rid, feature_name, value.

        Example:
            add_feature_value("Image", "Diagnosis", "1-ABC", "Normal")
        """
        try:
            from deriva_ml_mcp.tools.execution import _active_executions

            ml = conn_manager.get_active_or_raise()

            # Get execution RID from active execution or parameter
            exe_rid = execution_rid
            if not exe_rid:
                key = f"{ml.host_name}:{ml.catalog_id}"
                if key in _active_executions:
                    exe_rid = _active_executions[key].execution_rid

            if not exe_rid:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution. Provide execution_rid or create_execution first.",
                })

            # Look up the feature to get its structure
            feature = ml.lookup_feature(table_name, feature_name)

            # Determine which column gets the value
            # Priority: term columns, then asset columns, then value columns
            value_column = None
            if feature.term_columns:
                value_column = next(iter(feature.term_columns)).name
            elif feature.asset_columns:
                value_column = next(iter(feature.asset_columns)).name
            elif feature.value_columns:
                value_column = next(iter(feature.value_columns)).name

            if not value_column:
                return json.dumps({
                    "status": "error",
                    "message": f"Feature '{feature_name}' has no value columns. Use add_feature_value_record for complex features.",
                })

            # Build the record dict
            record_dict = {
                feature.target_table.name: target_rid,
                "Feature_Name": feature_name,
                "Execution": exe_rid,
                value_column: value,
            }

            # Insert the record
            pb = ml.pathBuilder()
            path = pb.schemas[feature.feature_table.schema.name].tables[feature.feature_table.name]
            result = list(path.insert([record_dict]))

            return json.dumps({
                "status": "added",
                "target_rid": target_rid,
                "feature_name": feature_name,
                value_column: value,
                "execution_rid": exe_rid,
                "rid": result[0].get("RID") if result else None,
            })
        except Exception as e:
            logger.error(f"Failed to add feature value: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_feature_value_record(
        table_name: str,
        feature_name: str,
        target_rid: str,
        values: dict[str, str | int | float],
        execution_rid: str | None = None,
    ) -> str:
        """Add a feature value with multiple fields to a domain object.

        For features with multiple columns (e.g., a diagnosis with confidence score),
        use this tool to provide values for each field. Use `lookup_feature` first
        to see the available fields and their types.

        **Feature columns are dynamically generated** based on the feature definition:
        - `term_columns`: Accept vocabulary term names (strings)
        - `asset_columns`: Accept asset RIDs (strings)
        - `value_columns`: Accept direct values (strings, numbers)

        The feature's dynamically generated Pydantic class (accessible via
        `feature_record_class()` in Python) validates all values automatically.

        Args:
            table_name: Table the target record belongs to (e.g., "Image").
            feature_name: Name of the feature (e.g., "Diagnosis").
            target_rid: RID of the target record to annotate.
            values: Dictionary mapping column names to values. Use `lookup_feature`
                to see available columns and their types.
            execution_rid: Execution RID for provenance (uses active if not provided).

        Returns:
            JSON with status, target_rid, feature_name, values, rid.

        Example:
            # First check the feature structure:
            lookup_feature("Image", "Diagnosis")
            # -> {"term_columns": {"Diagnosis_Type": {...}}, "value_columns": {"confidence": {"type": "float4"}}}

            # Then add a value with all fields:
            add_feature_value_record(
                "Image", "Diagnosis", "1-ABC",
                {"Diagnosis_Type": "Normal", "confidence": 0.95}
            )
        """
        try:
            from deriva_ml_mcp.tools.execution import _active_executions

            ml = conn_manager.get_active_or_raise()

            # Get execution RID from active execution or parameter
            exe_rid = execution_rid
            if not exe_rid:
                key = f"{ml.host_name}:{ml.catalog_id}"
                if key in _active_executions:
                    exe_rid = _active_executions[key].execution_rid

            if not exe_rid:
                return json.dumps({
                    "status": "error",
                    "message": "No active execution. Provide execution_rid or create_execution first.",
                })

            # Look up the feature
            feature = ml.lookup_feature(table_name, feature_name)

            # Build the record dict with required fields
            record_dict = {
                feature.target_table.name: target_rid,
                "Feature_Name": feature_name,
                "Execution": exe_rid,
            }

            # Add user-provided values
            record_dict.update(values)

            # Insert the record
            pb = ml.pathBuilder()
            path = pb.schemas[feature.feature_table.schema.name].tables[feature.feature_table.name]
            result = list(path.insert([record_dict]))

            return json.dumps({
                "status": "added",
                "target_rid": target_rid,
                "feature_name": feature_name,
                "values": values,
                "execution_rid": exe_rid,
                "rid": result[0].get("RID") if result else None,
            })
        except Exception as e:
            logger.error(f"Failed to add feature value record: {e}")
            return json.dumps({"status": "error", "message": str(e)})
