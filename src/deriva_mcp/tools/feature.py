"""Feature management tools for DerivaML MCP server.

Features are a core concept in DerivaML for ML data engineering. A feature associates
metadata with domain objects (e.g., Images, Subjects) to support ML workflows:

- **Categorical values**: From controlled vocabularies (e.g., "diagnosis", "quality_score")
- **Computed values**: Numeric or text values produced by ML models or processing pipelines
- **Related assets**: Links to derived assets (e.g., segmentation masks, embeddings)

Key properties of features:
1. **Provenance Tracking**: Every feature value records which Execution produced it
2. **Vocabulary-Controlled**: Categorical features use controlled vocabulary terms for consistency
3. **Multi-valued**: An object can have multiple values for the same feature (e.g., from different annotators)
4. **Versioned**: Feature values are included in dataset versions for reproducibility

Common use cases:
- Ground truth feature values for training data
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

    from deriva_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-mcp")


def register_feature_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register feature management tools with the MCP server."""

    @mcp.tool()
    async def create_feature(
        table_name: str,
        feature_name: str,
        comment: str = "",
        terms: list[str] | None = None,
        assets: list[str] | None = None,
        metadata: list[str | dict] | None = None,
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
            metadata: Additional columns or table references to include in the feature.
                Each item can be:
                - A string: Treated as a table name (adds a foreign key reference)
                - A dict: Column definition with at minimum "name" and "type" keys.
                  The "type" value should be a dict like {"typename": "float4"}.
                  Valid type names: text, int2, int4, int8, float4, float8, boolean,
                  date, timestamp, timestamptz, json, jsonb.
                  Optional keys: "nullok" (bool), "default", "comment".

        Returns:
            JSON with status, feature_name, target_table.

        Examples:
            # Simple term-based feature
            create_feature("Image", "Diagnosis", "Clinical diagnosis label", terms=["Diagnosis_Type"])

            # Feature with a confidence score column
            create_feature("Image", "Diagnosis", "Diagnosis with confidence",
                terms=["Diagnosis_Type"],
                metadata=[{"name": "confidence", "type": {"typename": "float4"}}])

            # Feature referencing another table
            create_feature("Image", "Review", "Review annotations",
                terms=["Review_Status"],
                metadata=["Reviewer"])
        """
        try:
            ml = conn_manager.get_active_or_raise()

            # Layer 3: Check for semantic near-duplicates
            from deriva_mcp.rag.helpers import rag_suggest_entity, DUPLICATE_RELEVANCE_THRESHOLD
            conn_info = conn_manager.get_active_connection_info()
            similar = rag_suggest_entity(feature_name, conn_info, limit=3)
            dup_warnings = [
                s for s in similar
                if s["relevance"] > DUPLICATE_RELEVANCE_THRESHOLD
                and s["name"].lower() != feature_name.lower()
            ]

            ml.create_feature(
                target_table=table_name,
                feature_name=feature_name,
                terms=terms or [],
                assets=assets or [],
                metadata=metadata or [],
                comment=comment,
            )
            from deriva_mcp.rag.helpers import trigger_schema_reindex
            trigger_schema_reindex(conn_manager.get_active_connection_info())
            result = {
                "status": "created",
                "feature_name": feature_name,
                "target_table": table_name,
            }
            if dup_warnings:
                result["similar_existing"] = dup_warnings
                result["warning"] = (
                    f"Created '{feature_name}', but similar entities exist: "
                    f"{', '.join(w['name'] for w in dup_warnings)}. "
                    f"Verify this isn't a duplicate."
                )
            return json.dumps(result)
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
                from deriva_mcp.rag.helpers import trigger_schema_reindex
                trigger_schema_reindex(conn_manager.get_active_connection_info())
                return json.dumps(
                    {
                        "status": "deleted",
                        "feature_name": feature_name,
                        "table_name": table_name,
                    }
                )
            return json.dumps(
                {
                    "status": "not_found",
                    "message": f"Feature '{feature_name}' not found in table '{table_name}'",
                }
            )
        except Exception as e:
            logger.error(f"Failed to delete feature: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_feature_value(
        table_name: str,
        feature_name: str,
        entries: list[dict[str, str]],
        execution_rid: str | None = None,
    ) -> str:
        """Add feature values to one or more domain objects.

        Associates feature values (terms, assets, or other) with target records.
        Accepts a list of entries, each mapping a target RID to a value. All entries
        are inserted in a single batch for efficiency.

        If an execution is active, it will be used for provenance. Otherwise,
        provide an execution_rid explicitly.

        **For simple features** (single term or asset column):
        Use this tool — each entry needs only `target_rid` and `value`.

        **For complex features** (multiple columns per record):
        Use `add_feature_value_record` instead, which accepts arbitrary field dicts.

        Args:
            table_name: Table the target records belong to (e.g., "Image").
            feature_name: Name of the feature (e.g., "Diagnosis").
            entries: List of dicts, each with:
                - target_rid (str): RID of the target record to annotate.
                - value (str): The feature value — a term name or asset RID.
            execution_rid: Execution RID for provenance (uses active if not provided).

        Returns:
            JSON with status, feature_name, count, execution_rid, rids.

        Examples:
            # Single value
            add_feature_value("Image", "Diagnosis",
                [{"target_rid": "1-ABC", "value": "Normal"}])

            # Batch values
            add_feature_value("Image", "Diagnosis", [
                {"target_rid": "1-ABC", "value": "Normal"},
                {"target_rid": "1-DEF", "value": "Abnormal"},
                {"target_rid": "1-GHI", "value": "Normal"},
            ])
        """
        try:
            ml = conn_manager.get_active_or_raise()

            # Get the active Execution object
            # Priority: explicit execution_rid > user-created execution > MCP connection execution
            execution = None
            exe_rid = execution_rid

            if not exe_rid:
                conn_info = conn_manager.get_active_connection_info()
                if conn_info and conn_info.active_tool_execution:
                    execution = conn_info.active_tool_execution
                    exe_rid = execution.execution_rid

            if not exe_rid:
                mcp_execution = conn_manager.get_active_execution()
                if mcp_execution:
                    execution = mcp_execution
                    exe_rid = mcp_execution.execution_rid

            if not exe_rid:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "No active execution. Connect to a catalog or use create_execution first.",
                    }
                )

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
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Feature '{feature_name}' has no value columns. Use add_feature_value_record for complex features.",
                    }
                )

            # Build typed FeatureRecord instances
            RecordClass = feature.feature_record_class()
            records = []
            for entry in entries:
                records.append(RecordClass(**{
                    feature.target_table.name: entry["target_rid"],
                    value_column: entry["value"],
                }))

            # Stage feature values via the execution's deferred write mechanism
            if execution is not None:
                execution.add_features(records)
            else:
                # Explicit execution_rid without an Execution object — use catalog API
                for r in records:
                    r.Execution = exe_rid
                ml.add_features(records)

            return json.dumps(
                {
                    "status": "added",
                    "feature_name": feature_name,
                    "count": len(records),
                    "execution_rid": exe_rid,
                }
            )
        except Exception as e:
            logger.error(f"Failed to add feature values: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_feature_value_record(
        table_name: str,
        feature_name: str,
        entries: list[dict],
        execution_rid: str | None = None,
    ) -> str:
        """Add feature values with multiple fields to one or more domain objects.

        For features with multiple columns (e.g., a diagnosis with confidence score),
        use this tool to provide values for each field. Accepts a list of entries
        for batch insertion. Use `lookup_feature` first to see available fields.

        **Feature columns are dynamically generated** based on the feature definition:
        - `term_columns`: Accept vocabulary term names (strings)
        - `asset_columns`: Accept asset RIDs (strings)
        - `value_columns`: Accept direct values (strings, numbers)

        Args:
            table_name: Table the target records belong to (e.g., "Image").
            feature_name: Name of the feature (e.g., "Diagnosis").
            entries: List of dicts, each with:
                - target_rid (str, required): RID of the target record.
                - Plus any feature column names mapped to their values.
                  Use `lookup_feature` to see available columns and types.
            execution_rid: Execution RID for provenance (uses active if not provided).

        Returns:
            JSON with status, feature_name, count, execution_rid, rids.

        Example:
            # First check the feature structure:
            lookup_feature("Image", "Diagnosis")
            # -> {"term_columns": {"Diagnosis_Type": {...}}, "value_columns": {"confidence": {...}}}

            # Then add values (single or batch):
            add_feature_value_record("Image", "Diagnosis", [
                {"target_rid": "1-ABC", "Diagnosis_Type": "Normal", "confidence": 0.95},
                {"target_rid": "1-DEF", "Diagnosis_Type": "Abnormal", "confidence": 0.87},
            ])
        """
        try:
            ml = conn_manager.get_active_or_raise()

            # Get the active Execution object
            # Priority: explicit execution_rid > user-created execution > MCP connection execution
            execution = None
            exe_rid = execution_rid

            if not exe_rid:
                conn_info = conn_manager.get_active_connection_info()
                if conn_info and conn_info.active_tool_execution:
                    execution = conn_info.active_tool_execution
                    exe_rid = execution.execution_rid

            if not exe_rid:
                mcp_execution = conn_manager.get_active_execution()
                if mcp_execution:
                    execution = mcp_execution
                    exe_rid = mcp_execution.execution_rid

            if not exe_rid:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "No active execution. Connect to a catalog or use create_execution first.",
                    }
                )

            # Look up the feature
            feature = ml.lookup_feature(table_name, feature_name)

            # Build typed FeatureRecord instances
            RecordClass = feature.feature_record_class()
            records = []
            for entry in entries:
                record_kwargs = {
                    feature.target_table.name: entry["target_rid"],
                }
                # Add all user-provided values (everything except target_rid)
                for k, v in entry.items():
                    if k != "target_rid":
                        record_kwargs[k] = v
                records.append(RecordClass(**record_kwargs))

            # Stage feature values via the execution's deferred write mechanism
            if execution is not None:
                execution.add_features(records)
            else:
                # Explicit execution_rid without an Execution object — use catalog API
                for r in records:
                    r.Execution = exe_rid
                ml.add_features(records)

            return json.dumps(
                {
                    "status": "added",
                    "feature_name": feature_name,
                    "count": len(records),
                    "execution_rid": exe_rid,
                }
            )
        except Exception as e:
            logger.error(f"Failed to add feature value records: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def fetch_table_features(
        table_name: str,
        feature_name: str | None = None,
        selector: str | None = None,
        workflow: str | None = None,
        execution: str | None = None,
    ) -> str:
        """Fetch all feature values for a table, grouped by feature name.

        Returns a dictionary mapping feature names to lists of feature value records.
        Useful for retrieving all annotations on a table at once — for example, getting
        all classification labels and quality scores for images in a single call.

        **Resolving multiple values per object:**

        When the same object has multiple values for a feature (e.g., labels from
        different annotators or model runs), use ``selector``, ``workflow``, or
        ``execution`` to pick one value per object:

        - **selector="newest"**: Picks the value with the most recent creation time
          (RCT). Good for getting the latest annotation regardless of source.
        - **workflow**: Filters to values produced by a specific workflow, then picks
          the newest. Pass a Workflow RID (e.g., "2-ABC1") or a Workflow_Type name
          (e.g., "Training"). Auto-detected.
        - **execution**: Filters to values produced by a specific execution RID,
          then picks the newest. Use this when multiple executions of the same
          workflow have produced values and you want a specific run's results.

        Only one of ``selector``, ``workflow``, or ``execution`` may be specified.

        Args:
            table_name: Table to fetch features for (e.g., "Image", "Subject").
            feature_name: If provided, only fetch this specific feature.
                If not provided, fetches all features on the table.
            selector: Built-in selector name. Currently supported: "newest".
                Picks one value per target object using the most recent RCT.
            workflow: Workflow RID or Workflow_Type name. Filters values to those
                produced by executions of the matching workflow, then picks the
                newest per target object.
            execution: Execution RID. Filters values to those produced by a
                specific execution, then picks the newest per target object.

        Returns:
            JSON dict mapping feature names to lists of feature value records.
            Each record includes target RID, feature values, Execution RID,
            RCT (creation time), and Feature_Name.

        Examples:
            # Get all features for Image table
            fetch_table_features("Image")

            # Get just Classification, deduplicated to newest per image
            fetch_table_features("Image", feature_name="Classification", selector="newest")

            # Get values from Training workflow only
            fetch_table_features("Image", feature_name="Classification", workflow="Training")

            # Get values from a specific execution
            fetch_table_features("Image", feature_name="FooBar", execution="3WY2")
        """
        try:
            ml = conn_manager.get_active_or_raise()

            # Validate mutual exclusivity
            specified = [x for x in [selector, workflow, execution] if x is not None]
            if len(specified) > 1:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Only one of 'selector', 'workflow', or 'execution' may be specified.",
                    }
                )

            # Resolve selector callable
            from deriva_ml.feature import FeatureRecord

            selector_fn = None
            if selector == "newest":
                selector_fn = FeatureRecord.select_newest
            elif selector is not None:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Unknown selector '{selector}'. Supported: 'newest'.",
                    }
                )

            if execution:
                selector_fn = FeatureRecord.select_by_execution(execution)
                features = ml.fetch_table_features(
                    table_name,
                    feature_name=feature_name,
                    selector=selector_fn,
                )
                result = {fname: [r.model_dump(mode="json") for r in records] for fname, records in features.items()}
            elif workflow:
                # Fetch without selector, then apply select_by_workflow per group
                features = ml.fetch_table_features(table_name, feature_name=feature_name)
                result = {}
                for fname, records in features.items():
                    if not records:
                        result[fname] = []
                        continue
                    # Group by target column and apply select_by_workflow
                    feat = ml.lookup_feature(table_name, fname)
                    target_col = feat.target_table.name
                    from collections import defaultdict

                    grouped: dict[str, list] = defaultdict(list)
                    for rec in records:
                        target_rid = getattr(rec, target_col, None)
                        if target_rid is not None:
                            grouped[target_rid].append(rec)
                    selected = []
                    for group in grouped.values():
                        try:
                            selected.append(ml.select_by_workflow(group, workflow))
                        except Exception:
                            pass  # Skip targets with no matching workflow records
                    result[fname] = [r.model_dump(mode="json") for r in selected]
            else:
                features = ml.fetch_table_features(
                    table_name,
                    feature_name=feature_name,
                    selector=selector_fn,
                )
                result = {fname: [r.model_dump(mode="json") for r in records] for fname, records in features.items()}

            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Failed to fetch table features: {e}")
            error_msg = str(e)
            result = {"status": "error", "message": error_msg}

            # Layer 2: Suggest alternatives on entity-not-found errors
            from deriva_mcp.rag.helpers import _is_not_found_error, rag_suggest_entity
            if _is_not_found_error(error_msg):
                conn_info = conn_manager.get_active_connection_info()
                suggestions = rag_suggest_entity(table_name, conn_info)
                if suggestions:
                    result["suggestions"] = suggestions
                    result["hint"] = f"Did you mean: {suggestions[0]['name']}?"

            return json.dumps(result)
