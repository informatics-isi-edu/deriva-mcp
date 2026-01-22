"""MCP Resources for DerivaML.

This module provides resource registration functions that expose
DerivaML information as MCP resources for LLM applications.

Resources provide read-only access to:
- Server version information
- Configuration templates for hydra-zen (static)
- Catalog information (dynamic, queries actual catalog)
- Documentation from GitHub repositories (dynamic, fetched with caching)
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from deriva_ml_mcp import __version__
from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.github_docs import fetch_doc


def register_resources(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register all DerivaML resources with the MCP server."""

    # =========================================================================
    # Server Information
    # =========================================================================

    @mcp.resource(
        "deriva-ml://server/version",
        name="Server Version",
        description="DerivaML MCP server version information",
        mime_type="application/json",
    )
    def get_server_version() -> str:
        """Return the DerivaML MCP server version."""
        return json.dumps({
            "name": "deriva-ml-mcp",
            "version": __version__,
        }, indent=2)

    # =========================================================================
    # Static Resources - Configuration Templates
    # =========================================================================

    @mcp.resource(
        "deriva-ml://config/deriva-ml-template",
        name="DerivaML Config Template",
        description="Hydra-zen configuration template for DerivaML connection",
        mime_type="text/x-python",
    )
    def get_deriva_ml_config_template() -> str:
        """Return a hydra-zen configuration template for DerivaML."""
        return '''"""DerivaML Configuration with hydra-zen.

This template shows how to configure DerivaML for different environments.
"""
from hydra_zen import builds, store
from deriva_ml import DerivaMLConfig

# Create a structured config using hydra-zen
DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

# Store configurations for different environments
deriva_store = store(group="deriva_ml")

# Development configuration
deriva_store(DerivaMLConf(
    hostname="localhost",
    catalog_id=1,
    use_minid=False,
), name="dev")

# Production configuration
deriva_store(DerivaMLConf(
    hostname="your-server.org",
    catalog_id="your-catalog",
    use_minid=True,
), name="prod")
'''

    @mcp.resource(
        "deriva-ml://config/dataset-spec-template",
        name="Dataset Spec Config Template",
        description="Hydra-zen configuration template for dataset specifications",
        mime_type="text/x-python",
    )
    def get_dataset_spec_template() -> str:
        """Return a hydra-zen configuration template for dataset specs."""
        return '''"""Dataset Specification Configuration with hydra-zen.

This template shows how to configure dataset collections for experiments.
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

# Define dataset collections
training_datasets = [
    DatasetSpecConfig(rid="XXXX", version="1.0.0", materialize=True),
    DatasetSpecConfig(rid="YYYY", version="1.0.0", materialize=True),
]

validation_datasets = [
    DatasetSpecConfig(rid="ZZZZ", version="1.0.0", materialize=False),
]

# Store them in hydra-zen store
datasets_store = store(group="datasets")
datasets_store(training_datasets, name="training")
datasets_store(validation_datasets, name="validation")
'''

    @mcp.resource(
        "deriva-ml://config/execution-template",
        name="Execution Config Template",
        description="Hydra-zen configuration template for ML executions",
        mime_type="text/x-python",
    )
    def get_execution_config_template() -> str:
        """Return a hydra-zen configuration template for executions."""
        return '''"""Execution Configuration with hydra-zen.

This template shows how to configure ML executions with datasets and assets.
"""
from hydra_zen import builds, instantiate
from deriva_ml.execution import ExecutionConfiguration, AssetRIDConfig
from deriva_ml.dataset import DatasetSpecConfig

# Build execution config
ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)

# Define input assets
assets = [
    AssetRIDConfig(rid="MODL", description="Pretrained model weights"),
    AssetRIDConfig(rid="CNFG", description="Model configuration"),
]

# Configure execution with datasets and assets
exec_conf = ExecConf(
    description="ML training run",
    datasets=[
        DatasetSpecConfig(rid="DATA", version="1.0.0", materialize=True),
    ],
    assets=[a.rid for a in assets],
)

# Instantiate to get ExecutionConfiguration
exec_config = instantiate(exec_conf)
'''

    @mcp.resource(
        "deriva-ml://config/model-template",
        name="Model Config Template",
        description="Hydra-zen configuration template for ML models with zen_partial",
        mime_type="text/x-python",
    )
    def get_model_config_template() -> str:
        """Return a hydra-zen configuration template for ML models."""
        return '''"""Model Configuration with hydra-zen using zen_partial.

This template shows how to configure ML models that integrate with DerivaML.
The key is using zen_partial=True to create partially configured functions
that receive the execution context at runtime.
"""
from hydra_zen import builds, store
from deriva_ml.execution import Execution
from deriva_ml import DerivaML

# Define your model function
def train_model(
    learning_rate: float,
    epochs: int,
    batch_size: int,
    ml_instance: DerivaML,
    execution: Execution | None = None,
) -> None:
    """Train model using DerivaML execution context."""
    # Access datasets and assets through execution
    for dataset in execution.datasets:
        bag = execution.download_dataset_bag(dataset)
        # Process data...

    # Your training code here
    print(f"Training with lr={learning_rate}, epochs={epochs}")

    # Register output files
    model_path = execution.asset_file_path("Model", "trained_model.pt")
    # Save model to model_path...

# Build config with zen_partial=True
# This creates a callable waiting for ml_instance and execution
ModelConfig = builds(
    train_model,
    learning_rate=1e-3,
    epochs=10,
    batch_size=32,
    populate_full_signature=True,
    zen_partial=True,  # Key: creates partial function
)

# Register configurations
model_store = store(group="model_config")
model_store(ModelConfig, name="default")
model_store(ModelConfig, name="fast", epochs=5, learning_rate=1e-2)
model_store(ModelConfig, name="long", epochs=100, learning_rate=1e-4)
'''

    @mcp.resource(
        "deriva-ml://config/experiment-template",
        name="Experiment Config Template",
        description="Hydra-zen configuration template for experiment presets",
        mime_type="text/x-python",
    )
    def get_experiment_config_template() -> str:
        """Return a hydra-zen configuration template for experiments."""
        return '''"""Experiment Configuration with hydra-zen.

This template shows how to define experiment presets that combine
model configs, datasets, and assets into reproducible configurations.

Experiments use Hydra's defaults list to override specific config groups
and inherit from a base configuration.

Usage:
    # Run a single experiment
    uv run deriva-ml-run +experiment=my_quick_experiment

    # Override experiment settings
    uv run deriva-ml-run +experiment=my_quick_experiment datasets=different_dataset
"""
from hydra_zen import make_config, store

from configs.base import MyBaseConfig  # Your base config (a builds() of run_model)

# Use _global_ package to allow overrides at the root level
experiment_store = store(group="experiment", package="_global_")

# Quick test experiment - minimal training for validation
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "quick"},      # Use quick model config
            {"override /datasets": "small_split"},    # Use small dataset
        ],
        description="Quick training: 3 epochs for fast validation",
        bases=(MyBaseConfig,),  # Inherit from base config
    ),
    name="my_quick_experiment",
)

# Extended training experiment - full training run
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "extended"},   # Use extended model config
            {"override /datasets": "full_split"},     # Use full dataset
        ],
        description="Extended training: 50 epochs with regularization",
        bases=(MyBaseConfig,),
    ),
    name="my_extended_experiment",
)

# Test-only experiment - evaluation without training
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "test_only"},  # Model config with test_only=True
            {"override /datasets": "testing"},        # Test dataset only
            {"override /assets": "pretrained_weights"},  # Load pretrained weights
        ],
        description="Evaluation only: load weights and evaluate on test set",
        bases=(MyBaseConfig,),
    ),
    name="my_test_only_experiment",
)

# Key patterns:
# 1. Use package="_global_" in experiment_store for root-level overrides
# 2. Use "override /group": "name" syntax in hydra_defaults
# 3. Inherit from base config using bases=(BaseConfig,)
# 4. Add description field for documentation
# 5. Experiments are selected with +experiment=name (note the + prefix)
'''

    @mcp.resource(
        "deriva-ml://config/multirun-template",
        name="Multirun Config Template",
        description="Hydra-zen configuration template for multirun sweeps",
        mime_type="text/x-python",
    )
    def get_multirun_config_template() -> str:
        """Return a hydra-zen configuration template for multiruns."""
        return '''"""Multirun Configuration for experiment sweeps.

This template shows how to define named multirun configurations that bundle
Hydra overrides with rich markdown descriptions. Multiruns allow running
multiple experiment variations in a single command.

Usage:
    # Run a defined multirun (no --multirun flag needed)
    uv run deriva-ml-run +multirun=my_lr_sweep

    # Override parameters from the multirun config
    uv run deriva-ml-run +multirun=my_lr_sweep model_config.epochs=5

    # Show available multiruns
    uv run deriva-ml-run --info

Benefits:
    - Explicit declaration of sweep experiments
    - Rich markdown descriptions for parent executions
    - Reproducible sweeps documented in code
    - Same Hydra override syntax as command line
    - No need to remember --multirun flag
"""
from deriva_ml.execution import multirun_config

# =============================================================================
# Experiment Comparisons
# =============================================================================
# Compare different experiment configurations side by side

multirun_config(
    "quick_vs_extended",
    overrides=[
        "+experiment=my_quick,my_extended",  # Run both experiments
    ],
    description="""## Quick vs Extended Training Comparison

**Objective:** Compare fast validation training against full extended training.

### Configurations

| Experiment | Epochs | Model Size | Dataset |
|------------|--------|------------|---------|
| my_quick | 3 | Small | small_split |
| my_extended | 50 | Large | full_split |

### Expected Outcomes

- Quick: Fast baseline, lower accuracy
- Extended: Best accuracy, longer training time
""",
)

# =============================================================================
# Hyperparameter Sweeps
# =============================================================================
# Sweep over parameter ranges to find optimal values.
# Build on existing experiments with parameter overrides.

multirun_config(
    "my_lr_sweep",
    overrides=[
        "+experiment=my_quick",                           # Base experiment
        "model_config.epochs=10",                         # Override epochs
        "model_config.learning_rate=0.0001,0.001,0.01,0.1",  # Sweep values
    ],
    description="""## Learning Rate Sweep

**Objective:** Find optimal learning rate for the model.

### Parameter Grid

| Parameter | Values |
|-----------|--------|
| Learning Rate | 0.0001, 0.001, 0.01, 0.1 |

**Total runs:** 4

### Analysis

Compare final test accuracy and training loss curves across runs.
""",
)

# =============================================================================
# Grid Searches
# =============================================================================
# Sweep multiple parameters simultaneously (creates N*M runs)

multirun_config(
    "my_lr_batch_grid",
    overrides=[
        "+experiment=my_quick",
        "model_config.epochs=10",
        "model_config.learning_rate=0.001,0.01",    # 2 values
        "model_config.batch_size=64,128",           # 2 values
    ],
    description="""## Learning Rate and Batch Size Grid Search

**Objective:** Find optimal combination of learning rate and batch size.

### Parameter Grid

| Parameter | Values |
|-----------|--------|
| Learning Rate | 0.001, 0.01 |
| Batch Size | 64, 128 |

**Total runs:** 4 (2 x 2 grid)

### Expected Outcomes

- Smaller batch sizes may need lower learning rates
- Larger batch sizes can often tolerate higher learning rates
- Look for the combination with best test accuracy and stable training
""",
)

# Key patterns:
# 1. Use multirun_config() from deriva_ml.execution
# 2. Build sweeps on existing experiments with "+experiment=name"
# 3. Override specific parameters with "group.param=value1,value2,..."
# 4. Use markdown descriptions for documentation in parent execution
# 5. Comma-separated values create multiple runs
# 6. Multiple parameters with commas create grid search (N*M runs)
'''

    # =========================================================================
    # Dynamic Resources - Catalog Information
    # =========================================================================

    @mcp.resource(
        "deriva-ml://catalog/schema",
        name="Catalog Schema",
        description="Current catalog schema structure in JSON format",
        mime_type="application/json",
    )
    def get_catalog_schema() -> str:
        """Return the current catalog schema as JSON."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            schema_info = {
                "hostname": ml.host_name,
                "catalog_id": str(ml.catalog_id),
                "domain_schema": ml.domain_schema,
                "ml_schema": ml.ml_schema,
                "tables": [],
            }

            # Get domain schema tables
            domain = ml.model.schemas.get(ml.domain_schema)
            if domain:
                for table in domain.tables.values():
                    table_info = {
                        "name": table.name,
                        "columns": [
                            {"name": col.name, "type": str(col.type)}
                            for col in table.columns
                        ],
                        "is_vocabulary": hasattr(table, "is_vocabulary") and table.is_vocabulary,
                    }
                    schema_info["tables"].append(table_info)

            return json.dumps(schema_info, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/vocabularies",
        name="Catalog Vocabularies",
        description="All vocabulary tables and their terms",
        mime_type="application/json",
    )
    def get_catalog_vocabularies() -> str:
        """Return all vocabulary tables and terms as JSON."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            vocabularies = {}
            # Iterate through schemas to find vocabulary tables
            for schema_name in [ml.ml_schema, ml.domain_schema]:
                schema = ml.model.schemas.get(schema_name)
                if schema:
                    for table in schema.tables.values():
                        if ml.model.is_vocabulary(table):
                            terms = ml.list_vocabulary_terms(table.name)
                            vocabularies[table.name] = [
                                {"name": t.name, "description": t.description}
                                for t in terms
                            ]
            return json.dumps(vocabularies, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/datasets",
        name="Catalog Datasets",
        description="All datasets in the current catalog",
        mime_type="application/json",
    )
    def get_catalog_datasets() -> str:
        """Return all datasets as JSON."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            datasets = []
            for ds in ml.find_datasets():
                datasets.append({
                    "rid": ds.dataset_rid,
                    "description": ds.description,
                    "types": ds.dataset_types,
                    "version": str(ds.current_version),
                })
            return json.dumps(datasets, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/dataset-element-types",
        name="Dataset Element Types",
        description="Tables that can contain dataset elements",
        mime_type="application/json",
    )
    def get_dataset_element_types() -> str:
        """Return all dataset element type tables."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            tables = list(ml.list_dataset_element_types())
            return json.dumps([
                {"name": t.name, "schema": t.schema.name, "comment": t.comment or ""}
                for t in tables
            ], indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/workflows",
        name="Catalog Workflows",
        description="All registered workflows in the catalog",
        mime_type="application/json",
    )
    def get_catalog_workflows() -> str:
        """Return all workflows as JSON."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            workflows = []
            for wf in ml.find_workflows():
                workflows.append({
                    "rid": wf.rid,
                    "name": wf.name,
                    "url": wf.url,
                    "workflow_type": wf.workflow_type,
                    "description": wf.description,
                })
            return json.dumps(workflows, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/workflow-types",
        name="Workflow Types",
        description="Available workflow type vocabulary terms",
        mime_type="application/json",
    )
    def get_workflow_types() -> str:
        """Return all workflow type terms."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            terms = ml.list_vocabulary_terms("Workflow_Type")
            return json.dumps([
                {"name": t.name, "description": t.description, "rid": t.rid}
                for t in terms
            ], indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/features",
        name="Catalog Features",
        description="All feature names defined in the catalog",
        mime_type="application/json",
    )
    def get_catalog_features() -> str:
        """Return all feature names as JSON."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            terms = ml.list_vocabulary_terms("Feature_Name")
            features = [
                {"name": t.name, "description": t.description}
                for t in terms
            ]
            return json.dumps(features, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # =========================================================================
    # Template Resources - Parameterized by dataset/table
    # =========================================================================

    @mcp.resource(
        "deriva-ml://dataset/{dataset_rid}",
        name="Dataset Details",
        description="Detailed information about a specific dataset including nested relationships",
        mime_type="application/json",
    )
    def get_dataset_details(dataset_rid: str) -> str:
        """Return detailed information about a dataset including children and parents."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            ds = ml.lookup_dataset(dataset_rid)
            members = ds.list_dataset_members()
            history = ds.dataset_history()
            children = ds.list_dataset_children()
            parents = ds.list_dataset_parents()

            return json.dumps({
                "rid": ds.dataset_rid,
                "description": ds.description,
                "types": ds.dataset_types,
                "current_version": str(ds.current_version),
                "member_counts": {k: len(v) for k, v in members.items()},
                "children": [
                    {
                        "rid": c.dataset_rid,
                        "description": c.description,
                        "types": c.dataset_types,
                    }
                    for c in children
                ],
                "parents": [
                    {
                        "rid": p.dataset_rid,
                        "description": p.description,
                        "types": p.dataset_types,
                    }
                    for p in parents
                ],
                "version_history": [
                    {
                        "version": str(h.version) if h.version else None,
                        "description": h.description,
                        "snapshot": h.snapshot,
                    }
                    for h in history
                ],
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://dataset/{dataset_rid}/members",
        name="Dataset Members",
        description="All elements (records) belonging to a specific dataset, grouped by table",
        mime_type="application/json",
    )
    def get_dataset_members(dataset_rid: str) -> str:
        """Return all dataset members grouped by table."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            ds = ml.lookup_dataset(dataset_rid)
            members = ds.list_dataset_members()

            return json.dumps({
                "dataset_rid": ds.dataset_rid,
                "description": ds.description,
                "current_version": str(ds.current_version),
                "members": {
                    table_name: [{"RID": m["RID"]} for m in items]
                    for table_name, items in members.items()
                },
                "member_counts": {k: len(v) for k, v in members.items()},
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://dataset/{dataset_rid}/versions",
        name="Dataset Version History",
        description="Complete version history for a dataset with semantic versions and snapshots",
        mime_type="application/json",
    )
    def get_dataset_versions(dataset_rid: str) -> str:
        """Return complete version history for a dataset."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            ds = ml.lookup_dataset(dataset_rid)
            history = ds.dataset_history()

            return json.dumps({
                "dataset_rid": ds.dataset_rid,
                "description": ds.description,
                "current_version": str(ds.current_version),
                "versions": [
                    {
                        "version": str(h.version) if h.version else None,
                        "description": h.description,
                        "snapshot": h.snapshot,
                    }
                    for h in history
                ],
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://table/{table_name}/features",
        name="Table Features",
        description="Features defined for a specific table",
        mime_type="application/json",
    )
    def get_table_features(table_name: str) -> str:
        """Return features for a specific table."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            features = list(ml.find_features(table_name))
            return json.dumps([
                {
                    "name": f.feature_name,
                    "target_table": f.target_table.name,
                    "feature_table": f.feature_table.name,
                    "asset_columns": [c.name for c in f.asset_columns],
                    "term_columns": [c.name for c in f.term_columns],
                    "value_columns": [c.name for c in f.value_columns],
                }
                for f in features
            ], indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://feature/{table_name}/{feature_name}",
        name="Feature Details",
        description="Detailed information about a specific feature including column types and requirements",
        mime_type="application/json",
    )
    def get_feature_details(table_name: str, feature_name: str) -> str:
        """Return detailed information about a feature.

        Includes column details needed for add_feature_value:
        - term_columns: columns accepting vocabulary terms (with vocab table name)
        - asset_columns: columns accepting asset RIDs (with asset table name)
        - value_columns: columns accepting direct values (with type)
        - required_fields: fields that must be provided (non-nullable)
        """
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            feature = ml.lookup_feature(table_name, feature_name)

            # Get detailed column info with FK references and requirements
            term_cols = {}
            for col in feature.term_columns:
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
                "optional": feature.optional,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://feature/{table_name}/{feature_name}/values",
        name="Feature Values",
        description="All feature values for a specific feature, with provenance",
        mime_type="application/json",
    )
    def get_feature_values(table_name: str, feature_name: str) -> str:
        """Return all values for a feature across objects.

        Each record includes the target object RID, the feature value(s), and
        the Execution RID that produced it (for provenance tracking).
        """
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            values = ml.list_feature_values(table_name, feature_name)
            result = [dict(v) for v in values]
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://vocabulary/{vocab_name}",
        name="Vocabulary Terms",
        description="Terms in a specific vocabulary table",
        mime_type="application/json",
    )
    def get_vocabulary_terms(vocab_name: str) -> str:
        """Return terms for a specific vocabulary."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            terms = ml.list_vocabulary_terms(vocab_name)
            return json.dumps([
                {
                    "name": t.name,
                    "description": t.description,
                    "synonyms": getattr(t, "synonyms", []),
                }
                for t in terms
            ], indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # =========================================================================
    # Documentation Resources - Fetched from GitHub
    # =========================================================================

    # DerivaML User Guide
    @mcp.resource(
        "deriva-ml://docs/overview",
        name="DerivaML Overview",
        description="Overview of DerivaML concepts and architecture",
        mime_type="text/markdown",
    )
    def get_overview_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/overview.md")

    @mcp.resource(
        "deriva-ml://docs/datasets",
        name="Datasets Guide",
        description="Guide to creating and managing datasets in DerivaML",
        mime_type="text/markdown",
    )
    def get_datasets_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/datasets.md")

    @mcp.resource(
        "deriva-ml://docs/features",
        name="Features Guide",
        description="Guide to defining and using features in DerivaML",
        mime_type="text/markdown",
    )
    def get_features_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/features.md")

    @mcp.resource(
        "deriva-ml://docs/execution-configuration",
        name="Execution Configuration Guide",
        description="Guide to configuring ML executions with datasets and assets",
        mime_type="text/markdown",
    )
    def get_execution_config_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/execution-configuration.md")

    @mcp.resource(
        "deriva-ml://docs/hydra-zen",
        name="Hydra-zen Configuration Guide",
        description="Guide to using hydra-zen for configuration management",
        mime_type="text/markdown",
    )
    def get_hydra_zen_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/hydra-zen-configuration.md")

    @mcp.resource(
        "deriva-ml://docs/file-assets",
        name="File Assets Guide",
        description="Guide to managing file assets in DerivaML",
        mime_type="text/markdown",
    )
    def get_file_assets_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/file-assets.md")

    @mcp.resource(
        "deriva-ml://docs/notebooks",
        name="Notebooks Guide",
        description="Guide to using Jupyter notebooks with DerivaML",
        mime_type="text/markdown",
    )
    def get_notebooks_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/notebooks.md")

    @mcp.resource(
        "deriva-ml://docs/annotations",
        name="Catalog Annotations Guide",
        description="Guide to configuring Chaise display using annotation builders",
        mime_type="text/markdown",
    )
    def get_annotations_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/annotations.md")

    @mcp.resource(
        "deriva-ml://docs/identifiers",
        name="Identifiers Guide",
        description="Guide to RIDs, MINIDs and other identifiers in DerivaML",
        mime_type="text/markdown",
    )
    def get_identifiers_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/identifiers.md")

    @mcp.resource(
        "deriva-ml://docs/install",
        name="Installation Guide",
        description="Installation instructions for DerivaML",
        mime_type="text/markdown",
    )
    def get_install_doc() -> str:
        return fetch_doc("deriva-ml", "docs/user-guide/install.md")

    # ERMrest Documentation
    @mcp.resource(
        "deriva-ml://docs/ermrest/data-api",
        name="ERMrest Data API",
        description="ERMrest REST API for data operations",
        mime_type="text/markdown",
    )
    def get_ermrest_data_doc() -> str:
        return fetch_doc("ermrest", "docs/api-doc/data/rest.md")

    @mcp.resource(
        "deriva-ml://docs/ermrest/naming",
        name="ERMrest Naming Conventions",
        description="ERMrest URL naming conventions for entities and attributes",
        mime_type="text/markdown",
    )
    def get_ermrest_naming_doc() -> str:
        return fetch_doc("ermrest", "docs/api-doc/data/naming.md")

    @mcp.resource(
        "deriva-ml://docs/ermrest/catalog",
        name="ERMrest Catalog API",
        description="ERMrest REST API for catalog operations",
        mime_type="text/markdown",
    )
    def get_ermrest_catalog_doc() -> str:
        return fetch_doc("ermrest", "docs/api-doc/rest-catalog.md")

    # Chaise Documentation
    @mcp.resource(
        "deriva-ml://docs/chaise/config",
        name="Chaise Configuration",
        description="Configuration options for the Chaise web UI",
        mime_type="text/markdown",
    )
    def get_chaise_config_doc() -> str:
        return fetch_doc("chaise", "docs/user-docs/chaise-config.md")

    @mcp.resource(
        "deriva-ml://docs/chaise/query-parameters",
        name="Chaise Query Parameters",
        description="URL query parameters for Chaise pages",
        mime_type="text/markdown",
    )
    def get_chaise_query_params_doc() -> str:
        return fetch_doc("chaise", "docs/user-docs/query-parameters.md")

    # deriva-py Documentation
    @mcp.resource(
        "deriva-ml://docs/deriva-py/install",
        name="Deriva-py Installation",
        description="Installation guide for the deriva-py Python SDK",
        mime_type="text/markdown",
    )
    def get_derivapy_install_doc() -> str:
        return fetch_doc("deriva-py", "docs/install.md")

    @mcp.resource(
        "deriva-ml://docs/deriva-py/tutorial",
        name="Deriva-py Tutorial",
        description="Project tutorial for deriva-py",
        mime_type="text/markdown",
    )
    def get_derivapy_tutorial_doc() -> str:
        return fetch_doc("deriva-py", "docs/project-tutorial.md")

    # =========================================================================
    # Dynamic Resources - Assets
    # =========================================================================

    @mcp.resource(
        "deriva-ml://catalog/asset-tables",
        name="Asset Tables",
        description="List of all asset tables in the catalog",
        mime_type="application/json",
    )
    def get_asset_tables() -> str:
        """Return list of all asset tables."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            tables = ml.list_asset_tables()
            return json.dumps([
                {"name": t.name, "schema": t.schema.name, "comment": t.comment or ""}
                for t in tables
            ], indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/assets",
        name="Catalog Assets",
        description="Summary of all asset tables and their contents",
        mime_type="application/json",
    )
    def get_catalog_assets() -> str:
        """Return summary of all asset tables."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            asset_tables = ml.model.find_assets()
            result = {}
            for table in asset_tables:
                assets = ml.list_assets(table.name)
                result[table.name] = {
                    "count": len(assets),
                    "columns": [c.name for c in table.columns if c.name not in ["RID", "RCT", "RMT", "RCB", "RMB"]],
                }
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://asset/{asset_rid}",
        name="Asset Details",
        description="Detailed information about a specific asset including provenance",
        mime_type="application/json",
    )
    def get_asset_details(asset_rid: str) -> str:
        """Return detailed information about an asset."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            asset = ml.lookup_asset(asset_rid)
            executions = ml.list_asset_executions(asset_rid)

            # Convert ExecutionRecord objects to dicts for JSON serialization
            execution_list = []
            for exe in executions:
                execution_list.append({
                    "execution_rid": exe.execution_rid,
                    "workflow_rid": exe.workflow_rid,
                    "status": exe.status.value if hasattr(exe.status, 'value') else str(exe.status),
                    "description": exe.description,
                })

            return json.dumps({
                "rid": asset.asset_rid,
                "table": asset.asset_table,
                "filename": asset.filename,
                "url": asset.url,
                "length": asset.length,
                "md5": asset.md5,
                "description": asset.description,
                "types": asset.asset_types,
                "executions": execution_list,
                "chaise_url": asset.get_chaise_url(),
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/executions",
        name="Catalog Executions",
        description="Recent executions in the catalog",
        mime_type="application/json",
    )
    def get_catalog_executions() -> str:
        """Return recent executions."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            pb = ml.pathBuilder()
            exec_path = pb.schemas[ml.ml_schema].Execution
            executions = list(exec_path.entities().fetch(limit=50))

            result = []
            for exe in executions:
                result.append({
                    "rid": exe.get("RID"),
                    "workflow": exe.get("Workflow"),
                    "status": exe.get("Status"),
                    "description": exe.get("Description", ""),
                    "created": exe.get("RCT"),
                })
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://execution/{execution_rid}",
        name="Execution Details",
        description="Detailed information about a specific execution",
        mime_type="application/json",
    )
    def get_execution_details(execution_rid: str) -> str:
        """Return detailed information about an execution."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            exe = ml.lookup_execution(execution_rid)

            # Get nested executions
            nested = exe.list_nested_executions()
            parents = exe.list_parent_executions()

            return json.dumps({
                "rid": exe.execution_rid,
                "workflow_rid": exe.workflow_rid,
                "status": exe.status.value if hasattr(exe.status, 'value') else str(exe.status),
                "description": exe.configuration.description if exe.configuration else "",
                "nested_executions": [n["Nested_Execution"] for n in nested],
                "parent_executions": [p["Execution"] for p in parents],
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # =========================================================================
    # Catalog Metadata Resources (converted from tools)
    # =========================================================================

    @mcp.resource(
        "deriva-ml://catalog/info",
        name="Catalog Info",
        description="Details about the active catalog: hostname, schemas, project name",
        mime_type="application/json",
    )
    def get_catalog_info() -> str:
        """Return details about the active catalog."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            conn_info = conn_manager.get_active_connection_info()
            result = {
                "hostname": ml.host_name,
                "catalog_id": str(ml.catalog_id),
                "domain_schema": ml.domain_schema,
                "ml_schema": ml.ml_schema,
                "project_name": ml.project_name,
            }
            if conn_info:
                result["workflow_rid"] = conn_info.workflow_rid
                result["execution_rid"] = (
                    conn_info.execution.execution_rid if conn_info.execution else None
                )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/users",
        name="Catalog Users",
        description="All users who have access to the active catalog",
        mime_type="application/json",
    )
    def get_catalog_users() -> str:
        """Return list of users with catalog access."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            users = ml.user_list()
            return json.dumps(users, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/connections",
        name="Active Connections",
        description="All open catalog connections and which is active",
        mime_type="application/json",
    )
    def get_connections() -> str:
        """Return list of all open connections."""
        connections = conn_manager.list_connections()
        return json.dumps(connections, indent=2)

    @mcp.resource(
        "deriva-ml://chaise-url/{table_or_rid}",
        name="Chaise URL",
        description="Web UI URL for viewing a table or specific record in Chaise",
        mime_type="application/json",
    )
    def get_chaise_url(table_or_rid: str) -> str:
        """Return the Chaise web interface URL for a table or RID."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            # First try as a table name
            try:
                url = ml.chaise_url(table_or_rid)
                return json.dumps({"url": url, "table_or_rid": table_or_rid})
            except Exception:
                # If not a table name, try as a RID
                result = ml.resolve_rid(table_or_rid)
                schema_name = result.table.schema.name
                table_name = result.table.name
                base_url = f"https://{ml.host_name}/chaise/record/#{ml.catalog_id}"
                url = f"{base_url}/{schema_name}:{table_name}/RID={result.rid}"
                return json.dumps({"url": url, "table_or_rid": table_or_rid})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://rid/{rid}",
        name="RID Resolution",
        description="Find which table a RID belongs to and get its Chaise URL",
        mime_type="application/json",
    )
    def resolve_rid(rid: str) -> str:
        """Resolve a RID to its table and Chaise URL."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            result = ml.resolve_rid(rid)
            schema_name = result.table.schema.name
            table_name = result.table.name
            base_url = f"https://{ml.host_name}/chaise/record/#{ml.catalog_id}"
            url = f"{base_url}/{schema_name}:{table_name}/RID={result.rid}"
            return json.dumps({
                "rid": rid,
                "schema": schema_name,
                "table": table_name,
                "url": url,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://cite/{rid}",
        name="Citation URL",
        description="Generate a permanent citation URL for a RID with optional snapshot",
        mime_type="application/json",
    )
    def get_citation_url(rid: str) -> str:
        """Generate permanent and current citation URLs for a RID."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            # Get both permanent (with snapshot) and current URLs
            permanent_url = ml.cite(rid, current=False)
            current_url = ml.cite(rid, current=True)
            return json.dumps({
                "rid": rid,
                "permanent_url": permanent_url,
                "current_url": current_url,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://registry/{hostname}",
        name="Catalog Registry",
        description="All catalogs and aliases available on a Deriva server",
        mime_type="application/json",
    )
    def get_catalog_registry(hostname: str) -> str:
        """List all catalogs and aliases on a server."""
        try:
            from deriva.core import DerivaServer, get_credential

            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            registry_catalog = server.connect_ermrest(0)
            pb = registry_catalog.getPathBuilder()
            registry = pb.schemas["ermrest"].tables["registry"]
            entries = list(registry.entities().fetch())

            catalogs = []
            aliases = []

            for entry in entries:
                if entry.get("deleted_on"):
                    continue
                if entry.get("is_catalog"):
                    catalogs.append({
                        "id": entry["id"],
                        "name": entry.get("name"),
                        "description": entry.get("description"),
                        "is_persistent": entry.get("is_persistent"),
                    })
                elif entry.get("alias_target"):
                    aliases.append({
                        "id": entry["id"],
                        "alias_target": entry["alias_target"],
                        "name": entry.get("name"),
                        "description": entry.get("description"),
                    })

            return json.dumps({
                "hostname": hostname,
                "catalogs": catalogs,
                "aliases": aliases,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://alias/{hostname}/{alias_name}",
        name="Catalog Alias",
        description="Metadata for a catalog alias including target and owner",
        mime_type="application/json",
    )
    def get_catalog_alias(hostname: str, alias_name: str) -> str:
        """Get metadata for a catalog alias."""
        try:
            from deriva.core import DerivaServer, get_credential

            server = DerivaServer("https", hostname, credentials=get_credential(hostname))
            alias = server.connect_ermrest_alias(alias_name)
            metadata = alias.retrieve()
            return json.dumps({
                "hostname": hostname,
                **metadata,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # =========================================================================
    # Execution Resources (converted from tools)
    # =========================================================================

    @mcp.resource(
        "deriva-ml://execution/{execution_rid}/inputs",
        name="Execution Inputs",
        description="Input datasets and assets for an execution",
        mime_type="application/json",
    )
    def get_execution_inputs(execution_rid: str) -> str:
        """Return input datasets and assets for an execution."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            execution = ml.lookup_execution(execution_rid)

            # Get input datasets
            input_datasets = []
            for ds in execution.list_input_datasets():
                input_datasets.append({
                    "rid": ds.dataset_rid,
                    "description": ds.description,
                    "types": ds.dataset_types,
                    "version": str(ds.current_version) if ds.current_version else None,
                })

            # Get input assets
            input_assets = []
            for asset in execution.list_input_assets():
                input_assets.append({
                    "rid": asset.rid,
                    "table": asset.table_name,
                    "filename": asset.filename,
                })

            return json.dumps({
                "execution_rid": execution_rid,
                "input_datasets": input_datasets,
                "input_assets": input_assets,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://experiment/{execution_rid}",
        name="Experiment Details",
        description="Experiment analysis for an execution with Hydra configuration",
        mime_type="application/json",
    )
    def get_experiment_details(execution_rid: str) -> str:
        """Return experiment summary for an execution."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            exp = ml.lookup_experiment(execution_rid)
            return json.dumps(exp.summary(), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://catalog/experiments",
        name="Catalog Experiments",
        description="All experiments (executions with Hydra config) in the catalog",
        mime_type="application/json",
    )
    def get_catalog_experiments() -> str:
        """Return all experiments in the catalog."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            experiments = list(ml.find_experiments())
            result = []
            for exp in experiments[:50]:  # Limit to 50
                result.append(exp.summary())

            return json.dumps({
                "count": len(result),
                "experiments": result,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://storage/summary",
        name="Storage Summary",
        description="Local storage usage summary for DerivaML",
        mime_type="application/json",
    )
    def get_storage_summary() -> str:
        """Return storage usage summary."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            summary = ml.get_storage_summary()
            return json.dumps(summary, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://storage/cache",
        name="Cache Size",
        description="Dataset cache directory size and statistics",
        mime_type="application/json",
    )
    def get_cache_stats() -> str:
        """Return cache size statistics."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            stats = ml.get_cache_size()
            return json.dumps(stats, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "deriva-ml://storage/execution-dirs",
        name="Execution Directories",
        description="List of execution working directories with sizes",
        mime_type="application/json",
    )
    def get_execution_dirs() -> str:
        """Return list of execution directories."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            dirs = ml.list_execution_dirs()

            # Convert datetime to ISO format
            for d in dirs:
                if 'modified' in d:
                    d['modified'] = d['modified'].isoformat()

            return json.dumps({
                "count": len(dirs),
                "execution_dirs": dirs,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
