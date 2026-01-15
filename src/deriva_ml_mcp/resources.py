"""MCP Resources for DerivaML.

This module provides resource registration functions that expose
DerivaML information as MCP resources for LLM applications.

Resources provide read-only access to:
- Schema documentation and structure
- Configuration templates for hydra-zen
- Vocabulary definitions
- Dataset metadata
- Annotation documentation and context reference
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from deriva_ml_mcp.connection import ConnectionManager


def register_resources(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register all DerivaML resources with the MCP server."""

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
        description="Detailed information about a specific dataset",
        mime_type="application/json",
    )
    def get_dataset_details(dataset_rid: str) -> str:
        """Return detailed information about a dataset."""
        ml = conn_manager.get_active_connection()
        if ml is None:
            return json.dumps({"error": "No active catalog connection"})

        try:
            ds = ml.lookup_dataset(dataset_rid)
            members = ds.list_dataset_members()
            history = ds.dataset_history()

            return json.dumps({
                "rid": ds.dataset_rid,
                "description": ds.description,
                "types": ds.dataset_types,
                "current_version": str(ds.current_version),
                "member_counts": {k: len(v) for k, v in members.items()},
                "version_history": [
                    {
                        "version": str(h.dataset_version),
                        "snapshot": h.snapshot,
                        "minid": h.minid,
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
    # Annotation Documentation Resources
    # =========================================================================

    @mcp.resource(
        "deriva-ml://docs/annotations",
        name="Annotation Reference",
        description="Complete reference documentation for Deriva annotations",
        mime_type="application/json",
    )
    def get_annotation_reference() -> str:
        """Return comprehensive annotation documentation."""
        return json.dumps({
            "title": "Deriva Annotation Reference",
            "description": "Complete guide to Deriva catalog annotations for UI customization",
            "annotations": {
                "display": {
                    "tag_uri": "tag:isrd.isi.edu,2015:display",
                    "applies_to": ["catalog", "schema", "table", "column", "foreign_key"],
                    "description": "Controls display name and basic presentation options",
                    "schema": {
                        "name": {"type": "string", "description": "Display name (mutually exclusive with markdown_name)"},
                        "markdown_name": {"type": "string", "description": "Markdown-formatted display name"},
                        "name_style": {
                            "type": "object",
                            "properties": {
                                "underline_space": {"type": "boolean", "description": "Replace underscores with spaces"},
                                "title_case": {"type": "boolean", "description": "Apply title case formatting"},
                                "markdown": {"type": "boolean", "description": "Interpret name as markdown"}
                            }
                        },
                        "comment": {"type": "string", "description": "Tooltip/description text"},
                        "show_null": {
                            "type": "object",
                            "description": "Per-context null value display (true=show 'No value', false=hide, string=custom text)",
                            "pattern_properties": {"context": "boolean | string"}
                        },
                        "show_foreign_key_link": {
                            "type": "object",
                            "description": "Per-context control of FK link display",
                            "pattern_properties": {"context": "boolean"}
                        }
                    },
                    "examples": [
                        {"name": "Images", "_comment": "Simple display name"},
                        {"name_style": {"underline_space": True}, "_comment": "Auto-format names"},
                        {"name": "My Table", "comment": "Tooltip shown on hover"}
                    ]
                },
                "visible_columns": {
                    "tag_uri": "tag:isrd.isi.edu,2016:visible-columns",
                    "applies_to": ["table"],
                    "description": "Controls which columns appear in each UI context and their order",
                    "contexts": {
                        "*": "Default for all unspecified contexts",
                        "compact": "Main list/table view (record lists, search results)",
                        "compact/brief": "Abbreviated display (tooltips, inline previews)",
                        "compact/brief/inline": "Minimal inline display (FK cell values)",
                        "compact/select": "Selection modal (FK picker dialogs)",
                        "detailed": "Full single-record view",
                        "entry": "Data entry forms (both create and edit)",
                        "entry/create": "Create new record form only",
                        "entry/edit": "Edit existing record form only",
                        "export": "Data export (CSV/JSON)",
                        "filter": "Faceted search sidebar (special format)"
                    },
                    "column_formats": {
                        "string": {
                            "description": "Simple column name",
                            "example": "\"Filename\""
                        },
                        "array": {
                            "description": "Foreign key reference [schema, constraint_name]",
                            "example": "[\"domain\", \"Image_Subject_fkey\"]"
                        },
                        "object": {
                            "description": "Pseudo-column with source path and display options",
                            "properties": {
                                "source": "Column name or path array with outbound/inbound FK traversal",
                                "sourcekey": "Reference to predefined source in source-definitions",
                                "entity": "boolean - Show as entity (row link) vs scalar value",
                                "aggregate": "Aggregation function: min, max, cnt, cnt_d, array, array_d",
                                "self_link": "boolean - Link to current row",
                                "markdown_name": "Custom column header display",
                                "comment": "Column tooltip text",
                                "display": {
                                    "markdown_pattern": "Template using {{{column}}} syntax",
                                    "template_engine": "handlebars or mustache",
                                    "show_foreign_key_link": "boolean",
                                    "array_ux_mode": "raw, csv, olist, or ulist"
                                }
                            }
                        }
                    },
                    "filter_format": {
                        "description": "Special format for filter context (faceted search)",
                        "structure": {
                            "and": [
                                {
                                    "source": "column_name or path",
                                    "markdown_name": "Filter label",
                                    "open": "boolean - Expand by default",
                                    "ux_mode": "choices, ranges, or check_presence",
                                    "bar_plot": "boolean - Show distribution chart",
                                    "hide_null_choice": "boolean",
                                    "hide_not_null_choice": "boolean",
                                    "choices": "Array of preset values",
                                    "ranges": "Array of {min, max} range objects"
                                }
                            ]
                        }
                    },
                    "examples": [
                        {
                            "_comment": "Basic column list per context",
                            "compact": ["Name", "Status", "RCT"],
                            "detailed": ["Name", "Status", "Description", "Notes", "RCT", "RMT"]
                        },
                        {
                            "_comment": "Include FK and hide in entry",
                            "compact": ["Name", ["domain", "Parent_fkey"]],
                            "entry": ["Name", "Description"]
                        },
                        {
                            "_comment": "Pseudo-column traversing FK",
                            "compact": [
                                "Name",
                                {
                                    "source": [{"outbound": ["domain", "Image_Subject_fkey"]}, "Name"],
                                    "markdown_name": "Subject Name"
                                }
                            ]
                        }
                    ]
                },
                "visible_foreign_keys": {
                    "tag_uri": "tag:isrd.isi.edu,2016:visible-foreign-keys",
                    "applies_to": ["table"],
                    "description": "Controls which related tables (inbound FKs) appear in detailed view",
                    "contexts": {
                        "*": "Default for all contexts",
                        "detailed": "Full record view - related tables section"
                    },
                    "important_note": "Only INBOUND foreign keys are valid - other tables that reference this table",
                    "foreign_key_formats": {
                        "array": {
                            "description": "Inbound FK reference [schema, constraint_name]",
                            "example": "[\"domain\", \"Image_Subject_fkey\"]"
                        },
                        "object": {
                            "description": "Pseudo-column for complex relationships",
                            "properties": {
                                "source": "Path starting with inbound FK",
                                "sourcekey": "Reference to source-definitions",
                                "markdown_name": "Section header",
                                "comment": "Section tooltip",
                                "display": "Display options object"
                            }
                        }
                    },
                    "examples": [
                        {
                            "_comment": "Show specific related tables",
                            "detailed": [
                                ["domain", "Image_Subject_fkey"],
                                ["domain", "Diagnosis_Subject_fkey"]
                            ]
                        },
                        {
                            "_comment": "Hide all related tables",
                            "detailed": []
                        }
                    ]
                },
                "table_display": {
                    "tag_uri": "tag:isrd.isi.edu,2016:table-display",
                    "applies_to": ["table"],
                    "description": "Controls table-level display like row naming, page size, and sort order",
                    "contexts": {
                        "*": "Default for all contexts",
                        "row_name": "Special context for row identifier display",
                        "compact": "List view options",
                        "detailed": "Record view options"
                    },
                    "options": {
                        "row_markdown_pattern": {"type": "string", "description": "Template for row display using {{{column}}}"},
                        "template_engine": {"type": "string", "enum": ["handlebars", "mustache"]},
                        "row_order": {"type": "array", "description": "Default sort order - column names or {column, descending} objects"},
                        "page_size": {"type": "number", "description": "Rows per page"},
                        "hide_column_headers": {"type": "boolean", "description": "Hide column headers (detailed only)"},
                        "collapse_toc_panel": {"type": "boolean", "description": "Collapse table of contents (detailed only)"},
                        "page_markdown_pattern": {"type": "string", "description": "Template for entire page"},
                        "separator_markdown": {"type": "string", "description": "Separator between rows"},
                        "prefix_markdown": {"type": "string", "description": "Content before rows"},
                        "suffix_markdown": {"type": "string", "description": "Content after rows"}
                    },
                    "examples": [
                        {
                            "_comment": "Set row name pattern",
                            "row_name": {"row_markdown_pattern": "{{{Name}}} ({{{Species}}})"}
                        },
                        {
                            "_comment": "Set sort order and page size",
                            "compact": {
                                "row_order": [{"column": "RCT", "descending": True}],
                                "page_size": 50
                            }
                        }
                    ]
                },
                "column_display": {
                    "tag_uri": "tag:isrd.isi.edu,2016:column-display",
                    "applies_to": ["column"],
                    "description": "Controls how column values are rendered",
                    "contexts": {
                        "*": "Default for all contexts",
                        "compact": "List view rendering",
                        "detailed": "Record view rendering",
                        "entry": "Entry form rendering"
                    },
                    "options": {
                        "pre_format": {
                            "type": "object",
                            "properties": {
                                "format": "printf-style format string (e.g., '%.2f')",
                                "bool_true_value": "Text for boolean true",
                                "bool_false_value": "Text for boolean false"
                            }
                        },
                        "markdown_pattern": {"type": "string", "description": "Template using {{{_value}}} or {{{column}}}"},
                        "template_engine": {"type": "string", "enum": ["handlebars", "mustache"]},
                        "column_order": {"description": "Sort config or false to disable sorting"}
                    },
                    "template_variables": {
                        "{{{_value}}}": "The column's own value",
                        "{{{value}}}": "Alias for _value",
                        "{{{_row.column_name}}}": "Another column's value from same row",
                        "{{{$fkeys.schema.fkey.values.col}}}": "Value from related table via FK"
                    },
                    "examples": [
                        {
                            "_comment": "Format number with 2 decimals",
                            "*": {"pre_format": {"format": "%.2f"}}
                        },
                        {
                            "_comment": "Display boolean as Yes/No",
                            "*": {"pre_format": {"bool_true_value": "Yes", "bool_false_value": "No"}}
                        },
                        {
                            "_comment": "Clickable image thumbnail",
                            "detailed": {"markdown_pattern": "[![Image]({{{_value}}})]({{{_value}}})"}
                        }
                    ]
                }
            },
            "tools_reference": {
                "read_tools": [
                    {"name": "get_table_annotations", "description": "Get all annotations for a table"},
                    {"name": "get_column_annotations", "description": "Get annotations for a column"},
                    {"name": "get_annotation_contexts", "description": "Get context documentation"},
                    {"name": "list_foreign_keys", "description": "List FKs to find constraint names"}
                ],
                "write_tools": [
                    {"name": "set_display_annotation", "description": "Set display annotation on table/column"},
                    {"name": "set_visible_columns", "description": "Set visible-columns annotation"},
                    {"name": "set_visible_foreign_keys", "description": "Set visible-foreign-keys annotation"},
                    {"name": "set_table_display", "description": "Set table-display annotation"},
                    {"name": "set_column_display", "description": "Set column-display annotation"}
                ],
                "convenience_tools": [
                    {"name": "add_visible_column", "description": "Add column to visible list"},
                    {"name": "remove_visible_column", "description": "Remove column from visible list"},
                    {"name": "reorder_visible_columns", "description": "Reorder visible columns"},
                    {"name": "add_visible_foreign_key", "description": "Add FK to visible list"},
                    {"name": "remove_visible_foreign_key", "description": "Remove FK from visible list"},
                    {"name": "reorder_visible_foreign_keys", "description": "Reorder visible FKs"}
                ],
                "apply_tool": {"name": "apply_annotations", "description": "Commit staged changes to catalog"}
            },
            "workflow_example": [
                "1. get_table_annotations('Image')  # See current state",
                "2. list_foreign_keys('Image')  # Find FK constraint names",
                "3. add_visible_column('Image', 'compact', 'Description')  # Add column",
                "4. reorder_visible_columns('Image', 'compact', [1, 0, 2, 3])  # Reorder",
                "5. set_table_display('Image', {'row_name': {'row_markdown_pattern': '{{{Filename}}}'}})  # Set row name",
                "6. apply_annotations()  # Commit all changes"
            ]
        }, indent=2)

    @mcp.resource(
        "deriva-ml://docs/annotation-contexts",
        name="Annotation Contexts Quick Reference",
        description="Quick reference for annotation contexts and when they are used",
        mime_type="application/json",
    )
    def get_annotation_contexts_reference() -> str:
        """Return a quick reference for annotation contexts."""
        return json.dumps({
            "title": "Annotation Contexts Quick Reference",
            "visible_columns": {
                "contexts": [
                    {"name": "*", "when_used": "Fallback for unspecified contexts"},
                    {"name": "compact", "when_used": "Main record list, search results, related tables"},
                    {"name": "compact/brief", "when_used": "Tooltips, hover previews"},
                    {"name": "compact/brief/inline", "when_used": "FK cell values in tables"},
                    {"name": "compact/select", "when_used": "FK picker/selection dialogs"},
                    {"name": "detailed", "when_used": "Single record page"},
                    {"name": "entry", "when_used": "Create and edit forms"},
                    {"name": "entry/create", "when_used": "New record form only"},
                    {"name": "entry/edit", "when_used": "Edit form only"},
                    {"name": "export", "when_used": "CSV/JSON export"},
                    {"name": "filter", "when_used": "Faceted search (special format)"}
                ],
                "fallback_chain": "specific → parent → * → system default"
            },
            "visible_foreign_keys": {
                "contexts": [
                    {"name": "*", "when_used": "Default for all views"},
                    {"name": "detailed", "when_used": "Related tables on record page"}
                ],
                "note": "Only inbound FKs (tables referencing this table) are valid"
            },
            "table_display": {
                "contexts": [
                    {"name": "*", "when_used": "Default options"},
                    {"name": "row_name", "when_used": "Row identifier in links/titles"},
                    {"name": "compact", "when_used": "List view settings"},
                    {"name": "detailed", "when_used": "Record view settings"}
                ]
            },
            "column_display": {
                "contexts": [
                    {"name": "*", "when_used": "Default rendering"},
                    {"name": "compact", "when_used": "Table cell rendering"},
                    {"name": "detailed", "when_used": "Record field rendering"},
                    {"name": "entry", "when_used": "Form field rendering"}
                ]
            }
        }, indent=2)

    @mcp.resource(
        "deriva-ml://docs/handlebars-templates",
        name="Handlebars Template Guide",
        description="Complete guide to Handlebars templates for Deriva annotations",
        mime_type="application/json",
    )
    def get_handlebars_guide() -> str:
        """Return comprehensive Handlebars template documentation from ermrestjs."""
        return json.dumps({
            "title": "Handlebars Template Guide for Deriva",
            "description": "Templates are used in row_markdown_pattern, markdown_pattern, and other display options",
            "source": "https://github.com/informatics-isi-edu/ermrestjs/blob/master/docs/user-docs/handlebars.md",
            "handlebars_vs_mustache": {
                "description": "Handlebars is almost similar to Mustache with additional benefits like helpers",
                "null_checking": {
                    "mustache": "{{#name}}Hello {{{name}}}{{/name}}{{^name}}No name available{{/name}}",
                    "handlebars": "{{#if name}}Hello {{{name}}}{{else}}No name available{{/if}}",
                    "note": "Handlebars if doesn't change context; use #with for context-changing null checks"
                },
                "encode_syntax": {
                    "mustache": "{{#encode}}{{{col}}}{{/encode}}",
                    "handlebars": "{{#encode col}}{{/encode}}"
                },
                "attribute_syntax": {
                    "issue": "Handlebars doesn't recognize {{{{}}}} for markdown attributes",
                    "solution": "Add space: { {{{btn-class}}} } instead of {{{{{btn_class}}}}}"
                }
            },
            "basic_syntax": {
                "variable_output": {
                    "raw": "{{{column_name}}} - Triple braces for raw output (no HTML escaping)",
                    "escaped": "{{column_name}} - Double braces for HTML-escaped output",
                    "recommendation": "Use triple braces {{{...}}} for most Deriva use cases"
                },
                "nested_paths": {
                    "example": "{{author.name}} - Access nested properties",
                    "parent_context": "../ - Access parent context in nested blocks"
                },
                "raw_values": {
                    "syntax": "{{{_COLUMN_NAME}}} - Prepend underscore for raw ERMrest values",
                    "jsonb": "{{{_col.name}}} - Access fields in jsonb columns via raw value"
                },
                "array_access": {
                    "by_index": "{{{arr.0.value}}} - Access array element by index (0-based)"
                },
                "escaping": {
                    "syntax": "\\\\{{{escaped}}} - Prefix with \\\\ to output literal handlebars"
                },
                "special_characters": {
                    "syntax": "{{[str with a space]}} - Square brackets for keys with spaces/special chars",
                    "nested": "{{{values.[power (uW)]}}}"
                },
                "subexpressions": {
                    "syntax": "{{#escape (encode arg1) arg2}}{{/escape}}",
                    "description": "Nest helpers using parentheses"
                }
            },
            "predefined_variables": {
                "$moment": {
                    "description": "Datetime object with current app load time",
                    "properties": {
                        "date": "Day of month (e.g., 19)",
                        "day": "Day of week (e.g., 4)",
                        "month": "Month number (e.g., 10)",
                        "year": "Year (e.g., 2017)",
                        "dateString": "e.g., Thu Oct 19 2017",
                        "hours": "24-hour format",
                        "minutes": "Minutes",
                        "seconds": "Seconds",
                        "milliseconds": "Milliseconds",
                        "ISOString": "ISO 8601 format",
                        "UTCString": "UTC format",
                        "LocaleString": "Locale-specific format"
                    },
                    "examples": [
                        "{{formatDatetime $moment.ISOString 'YYYY/M/D'}}",
                        "{{{$moment.month}}}/{{{$moment.date}}}/{{{$moment.year}}}"
                    ]
                },
                "$catalog": {
                    "description": "Catalog information object",
                    "properties": {
                        "id": "Catalog identifier without version",
                        "snapshot": "Catalog ID with version if present",
                        "version": "Version string if present"
                    }
                },
                "$dcctx": {
                    "description": "Current page/context IDs for generating links with ppid/pcid",
                    "properties": {
                        "pid": "Page ID",
                        "cid": "Context ID (app name)"
                    }
                },
                "$location": {
                    "description": "Current document location from URL",
                    "properties": {
                        "origin": "URL origin",
                        "host": "Hostname with port",
                        "hostname": "Hostname only",
                        "chaise_path": "Path to Chaise install (default: /chaise/)"
                    }
                },
                "$session": {
                    "description": "Current user session from webauthn",
                    "properties": {
                        "attributes": "Array of groups/identities with id, display_name, type, webpage",
                        "client.display_name": "User display name",
                        "client.email": "User email",
                        "client.full_name": "User full name",
                        "client.id": "User ID",
                        "client.identities": "User identities array",
                        "client.extensions": "Additional permissions (e.g., ras_dbgap_permissions)"
                    }
                },
                "$fkeys": {
                    "description": "Access outbound foreign key data",
                    "syntax": "$fkey_schema_constraint (preferred) or $fkeys.schema.constraint",
                    "attributes": {
                        "values": "Object with column values from related table",
                        "values.col1": "Formatted value, values._col1 for unformatted",
                        "rowName": "Row name of foreign key record",
                        "uri.detailed": "URI to FK in record app"
                    },
                    "example": "{{#with $fkey_schema_constraint}}[{{rowName}}]({{{uri.detailed}}}){{/with}}",
                    "limitation": "Only accesses tables one level away; use for column-display annotation"
                }
            },
            "helpers": {
                "printf": {
                    "description": "Format values using PreFormat syntax",
                    "syntax": "{{printf value format}}",
                    "examples": [
                        "{{printf 3.1415 '%.1f'}} → 3.1",
                        "{{printf 43 '%4d'}} → '  43'"
                    ]
                },
                "formatDatetime": {
                    "description": "Format date/timestamp/timestamptz values",
                    "syntax": "{{formatDatetime value format}}",
                    "examples": [
                        "{{formatDatetime '30-08-2018' 'YYYY'}} → 2018",
                        "{{formatDatetime '30-08-2018' 'YYYY-MM-DD'}} → 2018-08-30",
                        "{{formatDatetime '2018-09-25T00:12:34.00-07:00' 'MM/DD/YYYY HH:mm A'}} → 09/25/2018 03:12 AM"
                    ],
                    "note": "Previously called formatDate (still works but deprecated)"
                },
                "datetimeToSnapshot": {
                    "description": "Encode datetime to snapshot ID for catalog versioning",
                    "syntax": "{{datetimeToSnapshot value}}",
                    "example": "{{datetimeToSnapshot '2025-07-26T19:20:30Z'}} → 33N-PFKM-4DR0"
                },
                "snapshotToDatetime": {
                    "description": "Decode snapshot ID to datetime",
                    "syntax": "{{snapshotToDatetime value}} or {{snapshotToDatetime value format}}",
                    "examples": [
                        "{{snapshotToDatetime '33N-PFKM-4DR0'}} → 2025-07-26T19:20:30.000000+00:00",
                        "{{snapshotToDatetime '33N-PFKM-4DR0' 'YYYY-MM-DD'}} → 2025-07-26"
                    ]
                },
                "humanizeBytes": {
                    "description": "Convert byte count to human-readable format",
                    "syntax": "{{humanizeBytes value}} with optional named arguments",
                    "arguments": {
                        "mode": "'si' (default) or 'binary' (MiB instead of MB)",
                        "precision": "Number of digits (min 3 for si, 4 for binary)",
                        "tooltip": "true to include tooltip with exact bytes"
                    },
                    "examples": [
                        "{{humanizeBytes 41235532}} → 41.2 MB",
                        "{{humanizeBytes 41235532 precision=4}} → 41.23 MB",
                        "{{humanizeBytes 41235532 mode='binary'}} → 39.32 MiB",
                        "{{humanizeBytes 41235532 mode='binary' tooltip=true}} → span with tooltip"
                    ]
                },
                "stringLength": {
                    "description": "Get length of a string",
                    "syntax": "{{stringLength value}}",
                    "example": "{{stringLength '123123'}} → 6"
                },
                "add": {
                    "description": "Add two numbers",
                    "syntax": "{{add value1 value2}}",
                    "note": "Converts strings to numbers to avoid concatenation"
                },
                "subtract": {
                    "description": "Subtract value2 from value1",
                    "syntax": "{{subtract value1 value2}}"
                }
            },
            "block_helpers": {
                "if": {
                    "description": "Conditionally render content",
                    "syntax": "{{#if value}}...{{/if}} or {{#if value}}...{{else}}...{{/if}}",
                    "falsy_values": "false, undefined, null, '', 0, []",
                    "else_if": "{{#if val1}}...{{else if val2}}...{{else}}...{{/if}}",
                    "note": "Does NOT change context - use #with for context change"
                },
                "unless": {
                    "description": "Inverse of if - renders when falsy",
                    "syntax": "{{#unless value}}...{{/unless}}"
                },
                "each": {
                    "description": "Iterate over arrays or objects",
                    "syntax": "{{#each array}}...{{/each}}",
                    "variables": {
                        "this": "Current item",
                        "@index": "Current loop index (0-based)",
                        "@key": "Current key (for objects)",
                        "@first": "Boolean - is first item",
                        "@last": "Boolean - is last item"
                    },
                    "parent_access": "{{../array.length}} - Access parent context",
                    "block_params": "{{#each array as |value key|}}...{{/each}}",
                    "else": "{{#each items}}...{{else}}No items{{/each}}"
                },
                "with": {
                    "description": "Shift context to a value (also does truthy check)",
                    "syntax": "{{#with value}}...{{/with}}",
                    "current_value": "{{#with column}}{{{.}}}{{/with}} - Use . for current value",
                    "parent_access": "{{#with author}}{{{../title}}}{{/with}}",
                    "block_params": "{{#with author as |myAuthor|}}...{{/with}}",
                    "else": "{{#with author}}{{name}}{{else}}No author{{/with}}"
                },
                "lookup": {
                    "description": "Dynamic parameter resolution",
                    "syntax": "{{lookup map key}}",
                    "example": "{{lookup $session.client.extensions.ras_dbgap_phs_ids dbgap_study_id}}",
                    "note": "Returns null if value is null, undefined if key not present"
                },
                "encode": {
                    "description": "URL encode strings",
                    "syntax": "{{#encode value}}{{/encode}} or {{#encode key '=' value}}{{/encode}}",
                    "example": "age={{#encode ageVar}}{{/encode}} → age%3D10"
                },
                "escape": {
                    "description": "Escape special characters (hyphens, brackets, etc.)",
                    "syntax": "{{#escape value}}{{/escape}}",
                    "example": "{{#escape '**value ] ! special'}}{{/escape}} → \\*\\*value \\] \\! special"
                },
                "encodeFacet": {
                    "description": "Compress JSON for facet URLs",
                    "string_syntax": "{{#encodeFacet}}{JSON string}{{/encodeFacet}}",
                    "object_syntax": "{{encodeFacet jsonb_column}}",
                    "example": "[link](example.com/recordset/#1/S:T/*::facets::{{encodeFacet facet_obj}})"
                },
                "jsonStringify": {
                    "description": "Convert JSON object to string (like JSON.stringify)",
                    "syntax": "{{#jsonStringify column}}{{/jsonStringify}} or {{jsonStringify col}}"
                },
                "regexFindFirst": {
                    "description": "Return first matching substring",
                    "syntax": "{{#regexFindFirst string pattern}}{{this}}{{/regexFindFirst}}",
                    "flags": "{{#regexFindFirst str pattern flags='i'}}...{{/regexFindFirst}}",
                    "example": "{{#regexFindFirst '/var/www/index.html' '[^\\/]+$'}}{{this}}{{/regexFindFirst}} → index.html"
                },
                "regexFindAll": {
                    "description": "Return all matching substrings as array",
                    "syntax": "{{#each (regexFindAll string pattern)}}{{this}}{{/each}}"
                },
                "replace": {
                    "description": "Replace matches with substring (like String.replace)",
                    "syntax": "{{#replace pattern replacement}}string{{/replace}}",
                    "flags": "{{#replace pattern replacement flags=''}}...{{/replace}} - empty flags for first match only",
                    "example": "{{#replace '_' ' '}}table_name{{/replace}} → table name"
                },
                "toTitleCase": {
                    "description": "Capitalize first letter of each word",
                    "syntax": "{{#toTitleCase}}string{{/toTitleCase}}",
                    "example": "{{#toTitleCase}}hello world{{/toTitleCase}} → Hello World"
                }
            },
            "boolean_helpers": {
                "comparison": {
                    "eq": "{{#if (eq var1 var2)}}...{{/if}} - Equality",
                    "ne": "{{#if (ne var1 var2)}}...{{/if}} - Inequality",
                    "lt": "{{#if (lt var1 var2)}}...{{/if}} - Less than",
                    "gt": "{{#if (gt var1 var2)}}...{{/if}} - Greater than",
                    "lte": "{{#if (lte var1 var2)}}...{{/if}} - Less than or equal",
                    "gte": "{{#if (gte var1 var2)}}...{{/if}} - Greater than or equal"
                },
                "regexMatch": {
                    "description": "Check if value matches regex",
                    "syntax": "{{#if (regexMatch value pattern)}}...{{/if}}",
                    "flags": "{{#if (regexMatch value 'text' flags='i')}}...{{/if}}"
                },
                "isUserInAcl": {
                    "description": "Check if user is in ACL group(s)",
                    "syntax": "{{#if (isUserInAcl 'https://group-id')}}...{{/if}}",
                    "multiple": "{{#if (isUserInAcl 'id1' 'id2')}}...{{/if}}",
                    "array": "{{#if (isUserInAcl groupArray)}}...{{/if}}"
                },
                "logical": {
                    "and": "{{#if (and var1 var2)}}...{{/if}}",
                    "or": "{{#if (or var1 var2)}}...{{/if}}",
                    "not": "{{#if (not var1)}}...{{/if}}",
                    "nested": "{{#if (or (eq a 1) (and (gt b 5) (lt b 10)))}}...{{/if}}",
                    "multiple": "{{#if (or cond1 cond2 cond3)}}...{{/if}}"
                }
            },
            "automatic_null_detection": {
                "description": "If any column in markdown_pattern is null/empty, falls back to show_null",
                "limitations": [
                    "Disabled when block syntax ({{#) is used anywhere in template",
                    "Column names with # character may break detection"
                ],
                "workaround": "Use {{#with [# Column]}}{{{.}}}{{/with}} for special column names"
            },
            "common_patterns": {
                "row_name": [
                    {"description": "Simple", "pattern": "{{{Name}}}"},
                    {"description": "Composite", "pattern": "{{{Last_Name}}}, {{{First_Name}}}"},
                    {"description": "With FK", "pattern": "{{{Filename}}} ({{{$fkey_domain_fk.rowName}}})"},
                    {"description": "Conditional", "pattern": "{{{Name}}}{{#if Nickname}} ({{{Nickname}}}){{/if}}"}
                ],
                "column_display": [
                    {"description": "URL as link", "pattern": "[Download]({{{_value}}})"},
                    {"description": "Email link", "pattern": "[{{{_value}}}](mailto:{{{_value}}})"},
                    {"description": "Image thumbnail", "pattern": "[![]({{{_value}}}?h=80)]({{{_value}}})"},
                    {"description": "Conditional", "pattern": "{{#if _value}}✓ Active{{else}}✗ Inactive{{/if}}"}
                ],
                "foreign_key_link": {
                    "pattern": "{{#with $fkey_schema_constraint}}[{{rowName}}]({{{uri.detailed}}}){{/with}}"
                },
                "array_with_separator": {
                    "pattern": "{{#each items}}{{{this}}}{{#unless @last}}, {{/unless}}{{/each}}"
                },
                "facet_url": {
                    "pattern": "[View](url/*::facets::{{encodeFacet facet_obj}})"
                }
            },
            "debugging_tips": [
                "Use {{jsonStringify variable}} to inspect complex objects",
                "Start simple and add complexity incrementally",
                "Check browser console for template rendering errors",
                "Triple braces {{{...}}} prevent HTML escaping issues",
                "Use {{#if true}}...{{/if}} to disable automatic null detection without side effects"
            ]
        }, indent=2)

    # =========================================================================
    # JSON Schema Resources - Official Deriva Annotation Schemas
    # =========================================================================

    @mcp.resource(
        "deriva-ml://schemas/display",
        name="Display Annotation Schema",
        description="Official JSON Schema for the display annotation (tag:isrd.isi.edu,2015:display)",
        mime_type="application/json",
    )
    def get_display_schema() -> str:
        """Return the official JSON schema for the display annotation."""
        return json.dumps({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "http://deriva.isi.edu/schemas/display.schema.json",
            "title": "tag:isrd.isi.edu,2015:display",
            "description": "Schema document for the 'display' annotation. Controls display name and presentation options for tables, columns, and foreign keys.",
            "type": "object",
            "properties": {
                "comment": {
                    "type": "string",
                    "description": "Tooltip or description text shown on hover"
                },
                "name": {
                    "type": "string",
                    "description": "Plain text display name (mutually exclusive with markdown_name)"
                },
                "markdown_name": {
                    "type": "string",
                    "description": "Markdown-formatted display name (mutually exclusive with name)"
                },
                "name_style": {
                    "type": "object",
                    "description": "Automatic name formatting options",
                    "properties": {
                        "underline_space": {
                            "type": "boolean",
                            "description": "Replace underscores with spaces in names"
                        },
                        "title_case": {
                            "type": "boolean",
                            "description": "Apply title case formatting to names"
                        },
                        "markdown": {
                            "type": "boolean",
                            "description": "Interpret name as markdown"
                        }
                    }
                },
                "show_null": {
                    "type": "object",
                    "description": "How to display null values per context. Values can be: true (show 'No value'), false (hide), or a quoted string for custom text",
                    "patternProperties": {
                        "^[*]$|^compact([/]select|[/]brief([/]inline)?)?$|^detailed$": {
                            "oneOf": [
                                {"type": "boolean"},
                                {"type": "string", "pattern": "^[\"].*[\"]$", "description": "Quoted custom text for null display"}
                            ]
                        }
                    },
                    "additionalProperties": False
                },
                "show_foreign_key_link": {
                    "type": "object",
                    "description": "Whether to show foreign key values as clickable links per context",
                    "patternProperties": {
                        "^[*]$|^compact([/]select|[/]brief([/]inline)?)?$|^detailed$": {"type": "boolean"}
                    },
                    "additionalProperties": False
                }
            },
            "anyOf": [
                {"required": ["name"], "not": {"required": ["markdown_name"]}},
                {"required": ["markdown_name"], "not": {"required": ["name"]}},
                {"not": {"required": ["name", "markdown_name"]}}
            ],
            "examples": [
                {"name": "Images", "_comment": "Simple display name"},
                {"markdown_name": "**Images**", "_comment": "Markdown formatted name"},
                {"name_style": {"underline_space": True, "title_case": True}, "_comment": "Auto-format names"},
                {"name": "My Table", "comment": "Tooltip shown on hover"},
                {"show_null": {"*": True, "compact": False}, "_comment": "Show null in most views, hide in compact"}
            ]
        }, indent=2)

    @mcp.resource(
        "deriva-ml://schemas/visible-columns",
        name="Visible Columns Annotation Schema",
        description="Official JSON Schema for the visible-columns annotation (tag:isrd.isi.edu,2016:visible-columns)",
        mime_type="application/json",
    )
    def get_visible_columns_schema() -> str:
        """Return the official JSON schema for the visible-columns annotation."""
        return json.dumps({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "http://deriva.isi.edu/schemas/visible_columns.schema.json",
            "title": "tag:isrd.isi.edu,2016:visible-columns",
            "description": "Schema document for the 'visible-columns' annotation. Controls which columns appear in each UI context and their order.",
            "definitions": {
                "column-name": {
                    "type": "string",
                    "description": "A column name from the annotated table"
                },
                "constraint-name": {
                    "type": "array",
                    "description": "Foreign key reference as [schema_name, constraint_name]",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "pseudo-column": {
                    "type": "object",
                    "description": "A computed/virtual column with custom source path and display",
                    "properties": {
                        "source": {
                            "description": "Column name or path through foreign keys",
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {
                                        "anyOf": [
                                            {"type": "string", "description": "Column name (must be last item)"},
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "inbound": {"$ref": "#/definitions/constraint-name"},
                                                    "outbound": {"$ref": "#/definitions/constraint-name"}
                                                },
                                                "minProperties": 1,
                                                "maxProperties": 1
                                            }
                                        ]
                                    }
                                }
                            ]
                        },
                        "sourcekey": {
                            "type": "string",
                            "description": "Reference to a predefined source in source-definitions annotation"
                        },
                        "entity": {
                            "type": "boolean",
                            "description": "Show as entity (row link) vs scalar value. Default: true for FK paths"
                        },
                        "aggregate": {
                            "type": "string",
                            "enum": ["min", "max", "cnt", "cnt_d", "array", "array_d"],
                            "description": "Aggregation function for multi-valued results"
                        },
                        "self_link": {
                            "type": "boolean",
                            "description": "Make the column value a link to the current row"
                        },
                        "markdown_name": {
                            "type": "string",
                            "description": "Custom column header display"
                        },
                        "comment": {
                            "anyOf": [{"type": "string"}, {"type": "boolean", "const": False}],
                            "description": "Column tooltip text, or false to suppress"
                        },
                        "display": {
                            "type": "object",
                            "properties": {
                                "markdown_pattern": {"type": "string", "description": "Template using {{{column}}} syntax"},
                                "template_engine": {"type": "string", "enum": ["handlebars", "mustache"]},
                                "show_foreign_key_link": {"type": "boolean"},
                                "array_ux_mode": {"type": "string", "enum": ["raw", "csv", "olist", "ulist"]}
                            }
                        }
                    }
                },
                "facet-entry": {
                    "type": "object",
                    "description": "Faceted search filter configuration",
                    "properties": {
                        "source": {"description": "Column or path to filter on"},
                        "sourcekey": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {"type": ["string", "number", "boolean", "null"]},
                            "description": "Preset filter choices"
                        },
                        "ranges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "min": {"type": "number"},
                                    "max": {"type": "number"},
                                    "min_exclusive": {"type": "boolean"},
                                    "max_exclusive": {"type": "boolean"}
                                }
                            },
                            "description": "Preset filter ranges"
                        },
                        "not_null": {"type": "boolean", "const": True},
                        "entity": {"type": "boolean"},
                        "markdown_name": {"type": "string", "description": "Filter label"},
                        "comment": {"type": "string"},
                        "open": {"type": "boolean", "description": "Expand facet by default"},
                        "bar_plot": {"type": "boolean", "description": "Show distribution chart"},
                        "ux_mode": {
                            "type": "string",
                            "enum": ["choices", "ranges", "check_presence"],
                            "description": "Filter interaction mode"
                        },
                        "hide_null_choice": {"type": "boolean"},
                        "hide_not_null_choice": {"type": "boolean"},
                        "n_bins": {"type": "number", "minimum": 1, "description": "Number of histogram bins"}
                    }
                }
            },
            "type": "object",
            "patternProperties": {
                "^[*]$|^compact([/]select|[/]brief([/]inline)?)?$|^detailed$|^entry([/]edit|[/]create)?$|^export$": {
                    "oneOf": [
                        {"type": "string", "description": "Reference to another context"},
                        {
                            "type": "array",
                            "description": "List of columns/foreign keys/pseudo-columns",
                            "items": {
                                "anyOf": [
                                    {"$ref": "#/definitions/column-name"},
                                    {"$ref": "#/definitions/constraint-name"},
                                    {"$ref": "#/definitions/pseudo-column"}
                                ]
                            }
                        }
                    ]
                }
            },
            "properties": {
                "filter": {
                    "type": "object",
                    "description": "Faceted search configuration (special format)",
                    "properties": {
                        "and": {
                            "type": "array",
                            "items": {"$ref": "#/definitions/facet-entry"}
                        }
                    }
                }
            },
            "additionalProperties": False,
            "context_reference": {
                "*": "Default for all unspecified contexts",
                "compact": "Main list/table view (record lists, search results)",
                "compact/brief": "Abbreviated display (tooltips, inline previews)",
                "compact/brief/inline": "Minimal inline display (FK cell values)",
                "compact/select": "Selection modal (FK picker dialogs)",
                "detailed": "Full single-record view",
                "entry": "Data entry forms (both create and edit)",
                "entry/create": "Create new record form only",
                "entry/edit": "Edit existing record form only",
                "export": "Data export (CSV/JSON)",
                "filter": "Faceted search sidebar (special format with 'and' array)"
            },
            "examples": [
                {
                    "_comment": "Basic column lists per context",
                    "compact": ["RID", "Name", "Status"],
                    "detailed": ["RID", "Name", "Status", "Description", "Notes", "RCT", "RMT"]
                },
                {
                    "_comment": "Include foreign key reference",
                    "compact": ["Name", ["domain", "Image_Subject_fkey"]]
                },
                {
                    "_comment": "Pseudo-column traversing foreign key",
                    "detailed": [
                        "Name",
                        {
                            "source": [{"outbound": ["domain", "Image_Subject_fkey"]}, "Name"],
                            "markdown_name": "Subject Name"
                        }
                    ]
                },
                {
                    "_comment": "Faceted search configuration",
                    "filter": {
                        "and": [
                            {"source": "Species", "open": True, "bar_plot": True},
                            {"source": "Quality", "ux_mode": "choices"}
                        ]
                    }
                }
            ]
        }, indent=2)

    @mcp.resource(
        "deriva-ml://schemas/visible-foreign-keys",
        name="Visible Foreign Keys Annotation Schema",
        description="Official JSON Schema for the visible-foreign-keys annotation (tag:isrd.isi.edu,2016:visible-foreign-keys)",
        mime_type="application/json",
    )
    def get_visible_foreign_keys_schema() -> str:
        """Return the official JSON schema for the visible-foreign-keys annotation."""
        return json.dumps({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "http://deriva.isi.edu/schemas/visible_foreign_keys.schema.json",
            "title": "tag:isrd.isi.edu,2016:visible-foreign-keys",
            "description": "Schema document for the 'visible-foreign-keys' annotation. Controls which related tables (via inbound FKs) appear in detailed view.",
            "definitions": {
                "constraint-name": {
                    "type": "array",
                    "description": "Foreign key reference as [schema_name, constraint_name]",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "pseudo-column": {
                    "type": "object",
                    "description": "A computed relationship with custom source path",
                    "properties": {
                        "source": {
                            "type": "array",
                            "description": "Path starting with inbound FK",
                            "items": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "inbound": {"$ref": "#/definitions/constraint-name"},
                                            "outbound": {"$ref": "#/definitions/constraint-name"}
                                        }
                                    }
                                ]
                            }
                        },
                        "sourcekey": {"type": "string"},
                        "markdown_name": {"type": "string", "description": "Section header"},
                        "comment": {"type": "string", "description": "Section tooltip"},
                        "display": {
                            "type": "object",
                            "properties": {
                                "markdown_pattern": {"type": "string"},
                                "template_engine": {"type": "string", "enum": ["handlebars", "mustache"]}
                            }
                        }
                    }
                }
            },
            "type": "object",
            "properties": {
                "*": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"$ref": "#/definitions/constraint-name"},
                                    {"$ref": "#/definitions/pseudo-column"}
                                ]
                            }
                        }
                    ],
                    "description": "Default for all contexts"
                },
                "detailed": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"$ref": "#/definitions/constraint-name"},
                                    {"$ref": "#/definitions/pseudo-column"}
                                ]
                            }
                        }
                    ],
                    "description": "Related tables in detailed view"
                }
            },
            "additionalProperties": False,
            "important_note": "Only INBOUND foreign keys are valid - these are foreign keys from OTHER tables that reference THIS table. Use list_foreign_keys() tool to see available inbound FKs.",
            "examples": [
                {
                    "_comment": "Show specific related tables in order",
                    "detailed": [
                        ["domain", "Image_Subject_fkey"],
                        ["domain", "Diagnosis_Subject_fkey"]
                    ]
                },
                {
                    "_comment": "Hide all related tables",
                    "detailed": []
                },
                {
                    "_comment": "Pseudo-column for custom relationship display",
                    "detailed": [
                        {
                            "source": [{"inbound": ["domain", "Image_Subject_fkey"]}],
                            "markdown_name": "Subject Images"
                        }
                    ]
                }
            ]
        }, indent=2)

    @mcp.resource(
        "deriva-ml://schemas/table-display",
        name="Table Display Annotation Schema",
        description="Official JSON Schema for the table-display annotation (tag:isrd.isi.edu,2016:table-display)",
        mime_type="application/json",
    )
    def get_table_display_schema() -> str:
        """Return the official JSON schema for the table-display annotation."""
        return json.dumps({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "http://deriva.isi.edu/schemas/table_display.schema.json",
            "title": "tag:isrd.isi.edu,2016:table-display",
            "description": "Schema document for the 'table-display' annotation. Controls table-level display like row naming, page size, and sort order.",
            "definitions": {
                "template-engine": {
                    "type": "string",
                    "enum": ["handlebars", "mustache"],
                    "description": "Template engine to use for pattern rendering"
                },
                "sort-key": {
                    "oneOf": [
                        {"type": "string", "description": "Column name for ascending sort"},
                        {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "description": "Column name"},
                                "descending": {"type": "boolean", "default": False}
                            },
                            "required": ["column"]
                        }
                    ]
                },
                "row-order": {
                    "type": "array",
                    "description": "Default sort order as array of sort keys",
                    "minItems": 1,
                    "items": {"$ref": "#/definitions/sort-key"}
                },
                "table-display-options": {
                    "type": "object",
                    "properties": {
                        "row_order": {
                            "$ref": "#/definitions/row-order",
                            "description": "Default sort order for rows"
                        },
                        "page_size": {
                            "type": "number",
                            "description": "Number of rows per page"
                        },
                        "collapse_toc_panel": {
                            "type": "boolean",
                            "description": "Collapse table of contents panel (detailed view only)"
                        },
                        "hide_column_headers": {
                            "type": "boolean",
                            "description": "Hide column headers (detailed view only)"
                        },
                        "page_markdown_pattern": {
                            "type": "string",
                            "description": "Template for entire page layout"
                        },
                        "row_markdown_pattern": {
                            "type": "string",
                            "description": "Template for row display using {{{column}}} syntax"
                        },
                        "separator_markdown": {
                            "type": "string",
                            "description": "Markdown separator between rows"
                        },
                        "prefix_markdown": {
                            "type": "string",
                            "description": "Markdown content before rows"
                        },
                        "suffix_markdown": {
                            "type": "string",
                            "description": "Markdown content after rows"
                        },
                        "template_engine": {
                            "$ref": "#/definitions/template-engine"
                        }
                    }
                }
            },
            "type": "object",
            "patternProperties": {
                "^[*]$|^detailed$|^row_name.*$|^compact.*$": {
                    "oneOf": [
                        {"type": "string", "description": "Reference to another context"},
                        {"$ref": "#/definitions/table-display-options"},
                        {"type": "null"}
                    ]
                }
            },
            "additionalProperties": False,
            "context_reference": {
                "*": "Default options for all contexts",
                "row_name": "Special context for row identifier display (used in links, breadcrumbs, titles)",
                "compact": "List view settings",
                "detailed": "Record view settings"
            },
            "examples": [
                {
                    "_comment": "Set row name pattern for display in links and titles",
                    "row_name": {
                        "row_markdown_pattern": "{{{Name}}} ({{{Species}}})"
                    }
                },
                {
                    "_comment": "Set default sort order and page size for list view",
                    "compact": {
                        "row_order": [{"column": "RCT", "descending": True}],
                        "page_size": 50
                    }
                },
                {
                    "_comment": "Configure detailed view appearance",
                    "detailed": {
                        "hide_column_headers": True,
                        "collapse_toc_panel": True
                    }
                },
                {
                    "_comment": "Row name with foreign key value",
                    "row_name": {
                        "row_markdown_pattern": "{{{Filename}}} - {{{$fkeys.domain.Image_Subject_fkey.rowName}}}"
                    }
                }
            ]
        }, indent=2)

    @mcp.resource(
        "deriva-ml://schemas/column-display",
        name="Column Display Annotation Schema",
        description="Official JSON Schema for the column-display annotation (tag:isrd.isi.edu,2016:column-display)",
        mime_type="application/json",
    )
    def get_column_display_schema() -> str:
        """Return the official JSON schema for the column-display annotation."""
        return json.dumps({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "http://deriva.isi.edu/schemas/column_display.schema.json",
            "title": "tag:isrd.isi.edu,2016:column-display",
            "description": "Schema document for the 'column-display' annotation. Controls how column values are rendered.",
            "definitions": {
                "template-engine": {
                    "type": "string",
                    "enum": ["handlebars", "mustache"]
                },
                "column-order": {
                    "oneOf": [
                        {"type": "boolean", "const": False, "description": "Disable sorting on this column"},
                        {
                            "type": "array",
                            "description": "Custom sort order",
                            "items": {
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "column": {"type": "string"},
                                            "descending": {"type": "boolean"}
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
                "column-display-options": {
                    "type": "object",
                    "properties": {
                        "pre_format": {
                            "type": "object",
                            "description": "Pre-processing before display",
                            "properties": {
                                "format": {
                                    "type": "string",
                                    "description": "printf-style format string (e.g., '%.2f' for 2 decimal places)"
                                },
                                "bool_true_value": {
                                    "type": "string",
                                    "description": "Text to display for boolean true"
                                },
                                "bool_false_value": {
                                    "type": "string",
                                    "description": "Text to display for boolean false"
                                }
                            }
                        },
                        "markdown_pattern": {
                            "type": "string",
                            "description": "Template using {{{_value}}} or {{{column_name}}} substitution"
                        },
                        "template_engine": {
                            "$ref": "#/definitions/template-engine"
                        },
                        "column_order": {
                            "$ref": "#/definitions/column-order"
                        }
                    },
                    "dependencies": {
                        "template_engine": ["markdown_pattern"]
                    }
                }
            },
            "type": "object",
            "patternProperties": {
                "^[*]$|^detailed$|^compact.*$|^entry.*$": {
                    "oneOf": [
                        {"type": "string", "description": "Reference to another context"},
                        {"$ref": "#/definitions/column-display-options"}
                    ]
                }
            },
            "additionalProperties": False,
            "template_variables": {
                "{{{_value}}}": "The column's own value",
                "{{{value}}}": "Alias for _value",
                "{{{_row.column_name}}}": "Another column's value from the same row",
                "{{{$fkeys.schema.fkey.values.col}}}": "Value from related table via foreign key"
            },
            "context_reference": {
                "*": "Default rendering for all contexts",
                "compact": "Table cell rendering in list views",
                "detailed": "Field rendering on record page",
                "entry": "Form field rendering (usually not customized)"
            },
            "examples": [
                {
                    "_comment": "Format number with 2 decimal places",
                    "*": {"pre_format": {"format": "%.2f"}}
                },
                {
                    "_comment": "Display boolean as Yes/No",
                    "*": {
                        "pre_format": {
                            "bool_true_value": "Yes",
                            "bool_false_value": "No"
                        }
                    }
                },
                {
                    "_comment": "Clickable image thumbnail",
                    "detailed": {
                        "markdown_pattern": "[![Image]({{{_value}}}?h=100)]({{{_value}}})"
                    }
                },
                {
                    "_comment": "URL as clickable link",
                    "*": {
                        "markdown_pattern": "[{{{_value}}}]({{{_value}}})"
                    }
                },
                {
                    "_comment": "Disable sorting on this column",
                    "*": {"column_order": False}
                },
                {
                    "_comment": "Custom sort using another column",
                    "*": {
                        "column_order": [{"column": "sort_order", "descending": False}]
                    }
                }
            ]
        }, indent=2)

    # =========================================================================
    # Dataset Operations Documentation
    # =========================================================================

    @mcp.resource(
        "deriva-ml://docs/dataset-denormalization",
        name="Dataset Denormalization Guide",
        description="Guide to denormalizing datasets for ML workflows",
        mime_type="application/json",
    )
    def get_denormalization_guide() -> str:
        """Return comprehensive dataset denormalization documentation."""
        return json.dumps({
            "title": "Dataset Denormalization Guide",
            "description": "Denormalization joins related tables into a flat structure suitable for ML training",
            "overview": {
                "what_it_does": "Joins tables from a dataset into a single 'wide' table with columns from multiple source tables",
                "why_use_it": [
                    "ML frameworks expect flat tabular data",
                    "Combines features from related tables (e.g., Image + Subject + Diagnosis)",
                    "Creates training-ready datasets without manual joins"
                ],
                "column_naming": "Columns are prefixed with table name: 'Image.Filename', 'Subject.Name', 'Diagnosis.Type'"
            },
            "tool_reference": {
                "name": "denormalize_dataset",
                "parameters": {
                    "dataset_rid": "RID of the dataset to denormalize",
                    "include_tables": "List of table names to include in the join",
                    "version": "Optional: specific dataset version (default: current)",
                    "limit": "Maximum rows to return (default: 1000)"
                },
                "returns": {
                    "columns": "List of column names (prefixed with table name)",
                    "rows": "Array of row objects with column values",
                    "count": "Number of rows returned",
                    "limit": "Limit that was applied"
                }
            },
            "how_tables_are_joined": {
                "description": "Tables are joined based on foreign key relationships in the schema",
                "join_types": "Uses inner joins following FK paths between included tables",
                "important_note": "Only tables that are related (directly or through other tables) will produce rows"
            },
            "examples": [
                {
                    "description": "Basic denormalization of Images with Subject info",
                    "call": "denormalize_dataset('1-ABC', ['Image', 'Subject'])",
                    "result_columns": ["Image.RID", "Image.Filename", "Image.URL", "Subject.RID", "Subject.Name", "Subject.Species"],
                    "use_case": "Get images with their associated subject metadata"
                },
                {
                    "description": "Three-table join for ML training data",
                    "call": "denormalize_dataset('1-ABC', ['Image', 'Subject', 'Diagnosis'])",
                    "result_columns": ["Image.RID", "Image.Filename", "Subject.Name", "Diagnosis.Type", "Diagnosis.Confidence"],
                    "use_case": "Create labeled training dataset with image paths and diagnosis labels"
                },
                {
                    "description": "Get specific version of dataset",
                    "call": "denormalize_dataset('1-ABC', ['Image', 'Subject'], version='1.0.0')",
                    "use_case": "Reproducible ML training with pinned dataset version"
                }
            ],
            "common_patterns": {
                "image_classification": {
                    "tables": ["Image", "Label"],
                    "description": "Image paths with classification labels",
                    "example_columns": ["Image.URL", "Image.Filename", "Label.Class", "Label.Confidence"]
                },
                "patient_data": {
                    "tables": ["Image", "Subject", "Diagnosis"],
                    "description": "Medical images with patient info and diagnoses",
                    "example_columns": ["Image.URL", "Subject.Age", "Subject.Sex", "Diagnosis.Disease", "Diagnosis.Stage"]
                },
                "multi_modal": {
                    "tables": ["Image", "Scan", "Subject"],
                    "description": "Multiple imaging modalities per subject",
                    "example_columns": ["Image.URL", "Scan.Modality", "Scan.Date", "Subject.ID"]
                }
            },
            "workflow_example": [
                "1. list_datasets() - Find your dataset",
                "2. list_dataset_members(dataset_rid) - See which tables have data",
                "3. get_table_schema(table_name) - Understand column structure",
                "4. denormalize_dataset(dataset_rid, ['Table1', 'Table2']) - Get flat data",
                "5. Use rows for ML training (e.g., create DataFrame, build data loader)"
            ],
            "related_tools": {
                "list_dataset_members": "See which tables are in a dataset",
                "get_dataset_table": "Get raw data from a single table (no joins)",
                "get_table_schema": "View table columns and types",
                "download_dataset": "Download entire dataset as BDBag for offline use"
            },
            "tips": [
                "Start with list_dataset_members() to see available tables",
                "Include only tables you need - more tables = slower queries",
                "Use limit parameter for testing before fetching all data",
                "For production ML, use download_dataset() and process locally",
                "Column names are prefixed, so 'RID' becomes 'Image.RID' vs 'Subject.RID'"
            ],
            "limitations": [
                "Tables must be related through foreign keys to produce join results",
                "Very large datasets should use download_dataset() for local processing",
                "Default limit is 1000 rows - increase if needed",
                "Returns dict rows - convert to DataFrame/tensor for ML use"
            ]
        }, indent=2)

    @mcp.resource(
        "deriva-ml://schemas/source-definitions",
        name="Source Definitions Annotation Schema",
        description="Official JSON Schema for the source-definitions annotation (tag:isrd.isi.edu,2019:source-definitions)",
        mime_type="application/json",
    )
    def get_source_definitions_schema() -> str:
        """Return the official JSON schema for the source-definitions annotation."""
        return json.dumps({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "http://deriva.isi.edu/schemas/source_definitions.schema.json",
            "title": "tag:isrd.isi.edu,2019:source-definitions",
            "description": "Schema document for the 'source-definitions' annotation. Defines reusable source paths and column/fkey subsets for use in other annotations.",
            "definitions": {
                "column-name": {
                    "type": "string",
                    "description": "A column name from the table"
                },
                "constraint-name": {
                    "type": "array",
                    "description": "Foreign key reference as [schema_name, constraint_name]",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "foreign-key-path": {
                    "type": "object",
                    "description": "A step in a source path traversing a foreign key",
                    "properties": {
                        "inbound": {"$ref": "#/definitions/constraint-name", "description": "Traverse inbound FK (from other table to this)"},
                        "outbound": {"$ref": "#/definitions/constraint-name", "description": "Traverse outbound FK (from this table to other)"}
                    },
                    "minProperties": 1,
                    "maxProperties": 1
                },
                "source-entry": {
                    "description": "A source path specification",
                    "oneOf": [
                        {"$ref": "#/definitions/column-name"},
                        {
                            "type": "array",
                            "description": "Path: [fkey-step, ..., fkey-step, column-name]",
                            "items": {
                                "anyOf": [
                                    {"type": "string", "description": "Column name (must be last)"},
                                    {"$ref": "#/definitions/foreign-key-path"}
                                ]
                            },
                            "minItems": 1
                        }
                    ]
                },
                "pseudo-column": {
                    "type": "object",
                    "description": "A named source definition",
                    "properties": {
                        "source": {"$ref": "#/definitions/source-entry"},
                        "entity": {"type": "boolean"},
                        "aggregate": {"type": "string", "enum": ["min", "max", "cnt", "cnt_d", "array", "array_d"]},
                        "self_link": {"type": "boolean"},
                        "markdown_name": {"type": "string"},
                        "comment": {"anyOf": [{"type": "string"}, {"type": "boolean", "const": False}]},
                        "display": {
                            "type": "object",
                            "properties": {
                                "column_order": {},
                                "markdown_pattern": {"type": "string"},
                                "template_engine": {"type": "string", "enum": ["handlebars", "mustache"]},
                                "wait_for": {"type": "array", "items": {"type": "string"}},
                                "show_foreign_key_link": {"type": "boolean"},
                                "array_ux_mode": {"type": "string", "enum": ["raw", "csv", "olist", "ulist"]}
                            }
                        },
                        "array_options": {
                            "type": "object",
                            "properties": {
                                "order": {},
                                "max_length": {"type": "number", "minimum": 1}
                            }
                        }
                    }
                },
                "search-column": {
                    "type": "object",
                    "properties": {
                        "source": {"$ref": "#/definitions/column-name"},
                        "markdown_name": {"type": "string"}
                    },
                    "required": ["source"]
                }
            },
            "type": "object",
            "properties": {
                "columns": {
                    "oneOf": [
                        {"type": "boolean", "const": True, "description": "Include all columns"},
                        {
                            "type": "array",
                            "description": "Subset of column names to include",
                            "items": {"$ref": "#/definitions/column-name"}
                        }
                    ],
                    "description": "Columns available for template rendering"
                },
                "fkeys": {
                    "oneOf": [
                        {"type": "boolean", "const": True, "description": "Include all outbound FKs"},
                        {
                            "type": "array",
                            "description": "Subset of outbound foreign keys",
                            "items": {"$ref": "#/definitions/constraint-name"}
                        }
                    ],
                    "description": "Foreign keys available for template rendering ($fkeys.schema.constraint)"
                },
                "sources": {
                    "type": "object",
                    "description": "Named source definitions that can be referenced by 'sourcekey' in other annotations",
                    "properties": {
                        "search-box": {
                            "type": "object",
                            "description": "Columns to include in search",
                            "properties": {
                                "or": {
                                    "type": "array",
                                    "items": {"$ref": "#/definitions/search-column"},
                                    "minItems": 1
                                }
                            },
                            "required": ["or"]
                        }
                    },
                    "patternProperties": {
                        "^[^$].*": {"$ref": "#/definitions/pseudo-column"}
                    }
                }
            },
            "examples": [
                {
                    "_comment": "Expose specific columns and FKs for templates",
                    "columns": ["RID", "Name", "Description"],
                    "fkeys": [["domain", "Image_Subject_fkey"]]
                },
                {
                    "_comment": "Define reusable source paths",
                    "sources": {
                        "subject_name": {
                            "source": [{"outbound": ["domain", "Image_Subject_fkey"]}, "Name"],
                            "markdown_name": "Subject"
                        },
                        "image_count": {
                            "source": [{"inbound": ["domain", "Image_Subject_fkey"]}],
                            "aggregate": "cnt",
                            "markdown_name": "# Images"
                        }
                    }
                },
                {
                    "_comment": "Configure search box columns",
                    "sources": {
                        "search-box": {
                            "or": [
                                {"source": "Name"},
                                {"source": "Description"},
                                {"source": "Notes"}
                            ]
                        }
                    }
                }
            ]
        }, indent=2)

    # =========================================================================
    # Development Tools Documentation Resources
    # =========================================================================

    @mcp.resource(
        "deriva-ml://docs/bump-version",
        name="Bump Version Documentation",
        description="Documentation for semantic versioning with bump-version",
        mime_type="text/markdown",
    )
    def get_bump_version_docs() -> str:
        """Return documentation for the bump-version tool."""
        return """# Bump Version Tool

A command-line tool for managing semantic version tags in a git repository.

## Semantic Versioning

This tool follows semantic versioning (semver) conventions:

| Bump Type | Description | Example |
|-----------|-------------|---------|
| **major** | Incompatible API changes | 1.0.0 → 2.0.0 |
| **minor** | New backward-compatible functionality | 1.0.0 → 1.1.0 |
| **patch** | Backward-compatible bug fixes | 1.0.0 → 1.0.1 |

## How It Works

1. If no semver tag exists, creates an initial tag (default: v0.1.0)
2. If a tag exists, uses bump-my-version to increment the specified component
3. Pushes the new tag and any commits to the remote repository

## Dynamic Versioning with setuptools_scm

DerivaML uses **setuptools_scm** to derive package versions dynamically from git tags:

- **At a tag**: Version is clean (e.g., `1.2.3`)
- **After a tag**: Version includes distance and commit hash (e.g., `1.2.3.post2+g1234abc`)
- **Dirty working tree**: Adds `.dirty` suffix

## Usage

### MCP Tool
```python
# Bump patch version (default): v1.0.0 -> v1.0.1
bump_version("patch")

# Bump minor version: v1.0.0 -> v1.1.0
bump_version("minor")

# Bump major version: v1.0.0 -> v2.0.0
bump_version("major")

# Check current version
get_current_version()
```

### Command Line
```bash
# Using uv
uv run bump-version patch
uv run bump-version minor
uv run bump-version major

# Check current version
uv run python -m setuptools_scm
```

## Configuration

The tool requires:
- **git**: Version control system
- **uv**: Python package manager
- **bump-my-version**: Configured in pyproject.toml

### pyproject.toml Configuration

```toml
[project]
dynamic = ["version"]  # Version is not hardcoded

[build-system]
requires = ["setuptools>=80", "setuptools_scm[toml]>=8", "wheel"]

[tool.setuptools_scm]
version_scheme = "post-release"  # Use .postN for commits after a tag

[tool.bumpversion]
current_version = "0.1.0"
commit = true
tag = true
```

## Best Practices

1. **Commit all changes** before bumping version
2. **Use appropriate bump type**:
   - `patch` for bug fixes
   - `minor` for new features
   - `major` for breaking changes
3. **Tag before running experiments** for reproducibility
4. **Push tags** to remote for team visibility

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| START | 0.1.0 | Initial version if no tag exists |
| PREFIX | v | Tag prefix |
"""

    @mcp.resource(
        "deriva-ml://docs/install-kernel",
        name="Install Kernel Documentation",
        description="Documentation for Jupyter kernel installation",
        mime_type="text/markdown",
    )
    def get_install_kernel_docs() -> str:
        """Return documentation for the install-kernel tool."""
        return """# Install Jupyter Kernel Tool

A utility for installing a Jupyter kernel that points to the current Python virtual environment.

## Why Install a Kernel?

When working with Jupyter notebooks, the kernel determines which Python environment executes
the code. By default, Jupyter may not see packages installed in your virtual environment.
Installing a kernel creates a link so Jupyter can find and use your DerivaML environment.

## How It Works

1. Detects the current virtual environment name from `pyvenv.cfg`
2. Normalizes the name to be Jupyter-compatible (lowercase, alphanumeric)
3. Registers the kernel with Jupyter using ipykernel's install mechanism
4. The kernel appears in Jupyter's kernel selector with a friendly display name

## Usage

### MCP Tool
```python
# Install kernel for current venv (auto-detects name)
install_jupyter_kernel()

# Custom kernel name and display name
install_jupyter_kernel("my-kernel", "My Custom Kernel")

# List all installed kernels
list_jupyter_kernels()
```

### Command Line
```bash
# Install kernel
uv run deriva-ml-install-kernel

# Or run as a module
uv run python -m deriva_ml.install_kernel
```

## Example Workflow

Setting up a new DerivaML project with Jupyter support:

```bash
# Create and activate virtual environment
uv venv --prompt my-ml-project
source .venv/bin/activate

# Install DerivaML
uv pip install deriva-ml

# Install Jupyter kernel
uv run deriva-ml-install-kernel
# Output: Installed Jupyter kernel 'my-ml-project' with display name 'Python (my-ml-project)'

# Start Jupyter and select the new kernel
jupyter lab
```

## Kernel Location

Kernels are installed to the user's Jupyter data directory:

| Platform | Location |
|----------|----------|
| Linux/macOS | `~/.local/share/jupyter/kernels/` |
| Windows | `%APPDATA%\\jupyter\\kernels\\` |

Each kernel is a directory containing a `kernel.json` file that specifies
the Python executable path and display name.

## Requirements

- Must be run from within a virtual environment
- ipykernel must be installed in the environment

```bash
uv add ipykernel
```

## Troubleshooting

### Kernel not showing in Jupyter
- Verify kernel was installed: `list_jupyter_kernels()`
- Check kernel location exists
- Restart Jupyter server

### Wrong Python version
- Ensure you activated the correct virtual environment
- Check `kernel.json` points to correct Python executable
"""

    @mcp.resource(
        "deriva-ml://docs/run-notebook",
        name="Run Notebook Documentation",
        description="Documentation for running Jupyter notebooks with DerivaML tracking",
        mime_type="text/markdown",
    )
    def get_run_notebook_docs() -> str:
        """Return documentation for the run-notebook tool."""
        return """# Run Notebook Tool

A command-line tool for executing Jupyter notebooks with DerivaML execution tracking.

## Overview

This tool runs notebooks using papermill while automatically tracking the execution
in a Deriva catalog. It handles:

- Parameter injection into notebooks from command-line arguments or config files
- Automatic kernel detection for the current virtual environment
- Execution tracking with workflow provenance
- Conversion of executed notebooks to Markdown format
- Upload of notebook outputs as execution assets

## Usage

### MCP Tool
```python
# Run notebook with parameters
run_notebook(
    "notebooks/train_model.ipynb",
    hostname="deriva.example.org",
    catalog_id="42",
    parameters={"learning_rate": 0.001, "epochs": 100},
    kernel="my-ml-project"
)

# Inspect notebook parameters first
inspect_notebook("notebooks/train_model.ipynb")
```

### Command Line
```bash
# Basic usage
uv run deriva-ml-run-notebook notebook.ipynb --host example.org --catalog 1

# With parameters
uv run deriva-ml-run-notebook notebook.ipynb \\
    --host deriva.example.org \\
    --catalog 42 \\
    -p learning_rate 0.001 \\
    -p epochs 100 \\
    --kernel my_ml_env

# Parameters from file
uv run deriva-ml-run-notebook notebook.ipynb --file parameters.yaml

# Inspect available parameters
uv run deriva-ml-run-notebook notebook.ipynb --inspect
```

## Environment Variables

The tool sets these environment variables for the notebook:

| Variable | Description |
|----------|-------------|
| `DERIVA_ML_WORKFLOW_URL` | URL to the notebook source (e.g., GitHub URL) |
| `DERIVA_ML_WORKFLOW_CHECKSUM` | MD5 checksum of the notebook file |
| `DERIVA_ML_NOTEBOOK_PATH` | Local filesystem path to the notebook |
| `DERIVA_ML_SAVE_EXECUTION_RID` | Path where notebook should save execution info |

## Notebook Requirements

The notebook being executed should:

1. Use DerivaML's execution context to record its workflow
2. Save execution metadata to the path in `DERIVA_ML_SAVE_EXECUTION_RID`
3. Have a parameter cell for papermill to inject values

### Example Notebook Structure

```python
# Parameters cell (tagged with "parameters" in notebook metadata)
host = "localhost"
catalog = "1"
learning_rate = 0.001
epochs = 10

# DerivaML setup
from deriva_ml import DerivaML
import os
import json

ml = DerivaML(hostname=host, catalog_id=catalog)

# Create execution (save metadata for the runner)
execution = ml.create_execution(...)

# Save execution info for the runner
rid_path = os.environ.get("DERIVA_ML_SAVE_EXECUTION_RID")
if rid_path:
    with open(rid_path, "w") as f:
        json.dump({
            "execution_rid": execution.rid,
            "hostname": host,
            "catalog_id": catalog,
            "workflow_rid": execution.workflow_rid,
        }, f)

# ... rest of notebook code ...
```

## Output Assets

After execution, the tool uploads:
- The executed notebook (`.ipynb`) with all outputs
- A Markdown conversion (`.md`) for easy viewing

Both are registered as `Execution_Asset` records with type `notebook_output`.

## Best Practices

1. **Configure nbstripout** to keep notebooks clean in version control
2. **Use parameter cells** for values that should be injectable
3. **Pin dataset versions** in parameters for reproducibility
4. **Commit code before running** for proper provenance tracking

## Requirements

- papermill: For notebook execution
- nbformat: For notebook parsing
- nbconvert: For Markdown conversion
- deriva_ml: For catalog integration
- ipykernel: For kernel detection

```bash
uv add papermill nbformat nbconvert
```
"""

    # =========================================================================
    # ERMrest Documentation Resources
    # =========================================================================

    @mcp.resource(
        "deriva-ml://docs/ermrest-data-paths",
        name="ERMrest Data Path Reference",
        description="Reference documentation for ERMrest data path syntax and query building",
        mime_type="text/markdown",
    )
    def get_ermrest_data_paths_docs() -> str:
        """Return documentation for ERMrest data path syntax."""
        return """# ERMrest Data Path Reference

ERMrest uses URL-based "data paths" to query and manipulate data in Deriva catalogs.
This reference covers the syntax and patterns for building effective queries.

## Overview

Data paths are URL segments that specify:
- Which tables to query
- How to join related tables
- Which filters to apply
- Which columns to return

## Basic Syntax

### Entity Paths (Full Records)
```
/entity/schema:table
```
Returns all columns from all rows in the table.

### Attribute Paths (Column Projection)
```
/attribute/schema:table/column1,column2,column3
```
Returns only specified columns.

### Filtered Queries
```
/entity/schema:table/column=value
/entity/schema:table/column::gt::10
/entity/schema:table/column::null::
```

## Filter Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equals | `Name=John` |
| `::gt::` | Greater than | `Age::gt::18` |
| `::lt::` | Less than | `Score::lt::100` |
| `::geq::` | Greater or equal | `Count::geq::5` |
| `::leq::` | Less or equal | `Price::leq::50` |
| `::null::` | Is null | `Email::null::` |
| `::regexp::` | Regular expression | `Name::regexp::^J.*` |
| `::ciregexp::` | Case-insensitive regex | `Name::ciregexp::john` |
| `::ts::` | Text search | `Description::ts::machine learning` |

## Joining Tables (Links)

Use `/` to traverse foreign key relationships:

```
/entity/schema:Image/Subject/Species=Human
```

This:
1. Starts at the Image table
2. Follows the foreign key to Subject
3. Filters where Species equals "Human"

## Table Aliases

Use `alias:=` to reference the same table multiple times or for clarity:

```
/entity/I:=schema:Image/$I/Subject/I:Name,I:RID
```

- `I:=` creates an alias "I" for the Image table
- `$I` returns to the aliased table
- `I:Name` references the Name column from alias I

## Aggregates

### Count
```
/aggregate/schema:table/cnt:=cnt(*)
```

### Count Distinct
```
/aggregate/schema:table/unique_values:=cnt_d(column)
```

### Min/Max
```
/aggregate/schema:table/min_val:=min(column),max_val:=max(column)
```

### Array Aggregates
```
/aggregate/schema:table/all_names:=array(Name)
```

## Grouped Aggregates

```
/attributegroup/schema:table/grouping_column/cnt:=cnt(*)
```

Group by multiple columns:
```
/attributegroup/schema:table/col1,col2/cnt:=cnt(*),avg_val:=avg(value)
```

## Sorting and Pagination

### Sort Results
```
/entity/schema:table@sort(column)
/entity/schema:table@sort(column::desc::)
```

### Limit Results
```
/entity/schema:table?limit=100
```

### Pagination
```
/entity/schema:table@after(last_rid)?limit=100
```

## Common Patterns

### Get Images with Subject Information
```
/entity/domain:Image/Subject/RID,Filename,Subject:Name,Subject:Age
```

### Count Images per Subject
```
/attributegroup/domain:Image/Subject/image_count:=cnt(*)
```

### Filter by Related Table
```
/entity/domain:Image/Subject/Species=Human/$Image
```
The `$Image` returns to the Image table after filtering through Subject.

### Multiple Conditions (AND)
```
/entity/domain:Image/Width::gt::100/Height::gt::100
```

### OR Conditions
Use `;` for OR within same column:
```
/entity/domain:Image/Format=PNG;Format=JPEG
```

## Snapshot Queries

Query historical data at a specific point in time:
```
/ermrest/catalog/{id}@{snapshot}/entity/...
```

The snapshot is a timestamp or snapshot ID from the catalog's version history.

## Python API

The `deriva.core.datapath` module provides a Pythonic interface:

```python
from deriva.core import ErmrestCatalog

catalog = ErmrestCatalog('https', 'example.org', '1')
pb = catalog.getPathBuilder()

# Build query
path = pb.schemas['domain'].tables['Image']
path = path.filter(path.Width > 100)
path = path.link(pb.schemas['domain'].tables['Subject'])
path = path.filter(path.Subject.Species == 'Human')

# Execute
results = path.entities()
```

## Tips

1. **Use aliases** when joining the same table multiple times
2. **Project columns** to reduce data transfer
3. **Add filters early** to reduce intermediate result sizes
4. **Use snapshots** for reproducible queries
5. **Paginate large results** using `@after()` and `limit`

## Related Tools

- `query_table()` - Simple table queries with filters
- `get_table_schema()` - View table structure
- `denormalize_dataset()` - Join dataset tables automatically
"""

    @mcp.resource(
        "deriva-ml://docs/ermrest-model-management",
        name="ERMrest Model Management Reference",
        description="Reference for schema and table management in Deriva catalogs",
        mime_type="text/markdown",
    )
    def get_ermrest_model_docs() -> str:
        """Return documentation for ERMrest model management."""
        return """# ERMrest Model Management Reference

ERMrest provides a programmatic interface for managing catalog schemas, tables,
columns, and constraints. This reference covers common model management operations.

## Overview

The model hierarchy in Deriva:
- **Catalog**: Top-level container
- **Schema**: Namespace for related tables
- **Table**: Data entity with columns and constraints
- **Column**: Data attribute with type and constraints
- **Key**: Uniqueness constraint
- **Foreign Key**: Referential integrity constraint

## Built-in Column Types

| Type | Description | Example Use |
|------|-------------|-------------|
| `text` | Variable-length string | Names, descriptions |
| `int4` | 32-bit integer | Counts, small IDs |
| `int8` | 64-bit integer | Large numbers |
| `float4` | 32-bit floating point | Measurements |
| `float8` | 64-bit floating point | Precise values |
| `boolean` | True/false | Flags |
| `date` | Calendar date | Birth dates |
| `timestamptz` | Timestamp with timezone | Event times |
| `jsonb` | Binary JSON | Structured metadata |

## System Columns

Every table automatically includes:

| Column | Type | Description |
|--------|------|-------------|
| `RID` | `ermrest_rid` | Unique record identifier |
| `RCT` | `ermrest_rct` | Record creation timestamp |
| `RMT` | `ermrest_rmt` | Record modification timestamp |
| `RCB` | `ermrest_rcb` | Record created by (user) |
| `RMB` | `ermrest_rmb` | Record modified by (user) |

## Creating Tables

### Basic Table
```python
from deriva.core import ErmrestCatalog
import deriva.core.ermrest_model as em
from deriva.core.ermrest_model import builtin_types as typ

catalog = ErmrestCatalog('https', 'example.org', '1')
model = catalog.getCatalogModel()

# Define columns
column_defs = [
    em.Column.define("Name", typ.text, nullok=False),
    em.Column.define("Description", typ.text),
    em.Column.define("Count", typ.int4, default=0),
]

# Define table
table_def = em.Table.define(
    "MyTable",
    column_defs,
    comment="My new data table",
)

# Create in schema
schema = model.schemas['my_schema']
new_table = schema.create_table(table_def)
```

### Vocabulary Table
```python
# Convenience method for controlled vocabularies
vocab_table = schema.create_table(
    em.Table.define_vocabulary(
        "My_Vocabulary",
        "MYPROJECT:{RID}",
        "https://example.org/id/{RID}"
    )
)
```

Vocabulary tables automatically include:
- `Name`: Term name (unique)
- `Description`: Term description
- `Synonyms`: Alternative names (array)
- `ID`: CURIE identifier
- `URI`: Full URI

## Creating Columns

```python
table = model.table('schema_name', 'table_name')

column_def = em.Column.define(
    "NewColumn",
    typ.text,
    nullok=True,
    comment="Description of the column",
)

new_column = table.create_column(column_def)
```

## Creating Keys

```python
# Unique constraint on single column
key_def = em.Key.define(
    ["Email"],  # Column(s)
    constraint_names=[["schema", "Table_Email_key"]],
    comment="Email must be unique",
)
table.create_key(key_def)

# Compound unique constraint
key_def = em.Key.define(
    ["FirstName", "LastName"],
    constraint_names=[["schema", "Table_Name_key"]],
)
```

## Creating Foreign Keys

```python
fkey_def = em.ForeignKey.define(
    ["Subject"],  # Local column(s)
    "domain",     # Referenced schema
    "Subject",    # Referenced table
    ["RID"],      # Referenced column(s)
    on_update='CASCADE',
    on_delete='SET NULL',
    constraint_names=[["schema", "Image_Subject_fkey"]],
)
table.create_fkey(fkey_def)
```

### Foreign Key Actions

| Action | On Delete | On Update |
|--------|-----------|-----------|
| `NO ACTION` | Block if references exist | Block |
| `CASCADE` | Delete referencing rows | Update references |
| `SET NULL` | Set to NULL | Set to NULL |

## Altering Tables

```python
table = model.table('schema', 'old_name')

# Rename table
table.alter(table_name='new_name')

# Move to different schema
table.alter(schema_name='other_schema')

# Both at once
table.alter(schema_name='other_schema', table_name='new_name')
```

## Altering Columns

```python
column = table.column_definitions['old_name']

# Rename
column.alter(name='new_name')

# Change type (if compatible)
column.alter(type=typ.int8)

# Change nullability
column.alter(nullok=False)

# Change default
column.alter(default='unknown')
```

## Dropping Elements

```python
# Drop column
table.column_definitions['column_name'].drop()

# Drop key
table.keys[(schema, 'key_name')].drop()

# Drop foreign key
table.foreign_keys[(schema, 'fkey_name')].drop()

# Drop table
table.drop()

# Drop schema (must be empty)
schema.drop()
```

## Annotations

Annotations control UI display and behavior:

```python
# Set table annotation
table.annotations[em.tag.display] = {"name": "Friendly Name"}

# Set column annotation
column = table.column_definitions['col']
column.annotations[em.tag.display] = {"name": "Column Label"}

# Apply changes
model.apply()
```

### Common Annotation Tags

| Tag | Purpose |
|-----|---------|
| `tag:isrd.isi.edu,2016:visible-columns` | Control column visibility |
| `tag:isrd.isi.edu,2016:visible-foreign-keys` | Control FK visibility |
| `tag:isrd.isi.edu,2016:table-display` | Table display settings |
| `tag:misd.isi.edu,2015:display` | Display name/format |

## DerivaML Convenience Methods

DerivaML provides higher-level methods:

```python
from deriva_ml import DerivaML

ml = DerivaML(hostname='example.org', catalog_id='1')

# Create table
ml.create_table("MyTable", [
    {"name": "Name", "type": "text"},
    {"name": "Value", "type": "float8"},
])

# Create vocabulary
ml.create_vocabulary("My_Type", "Types for my data")

# Create asset table (with URL, checksum tracking)
ml.create_asset_table("MyAsset", [
    {"name": "Width", "type": "int4"},
])
```

## Best Practices

1. **Use vocabularies** for controlled values
2. **Define foreign keys** to maintain referential integrity
3. **Add comments** to document schema elements
4. **Use annotations** to improve UI experience
5. **Test in development** before modifying production schemas

## Related Tools

- `create_table()` - Create domain tables
- `create_vocabulary()` - Create vocabulary tables
- `create_asset_table()` - Create asset tables
- `get_table_schema()` - View existing structure
- `apply_annotations()` - Apply UI customizations
"""

    @mcp.resource(
        "deriva-ml://docs/chaise-annotations",
        name="Chaise Annotation Reference",
        description="Reference for UI customization annotations in Deriva/Chaise",
        mime_type="text/markdown",
    )
    def get_chaise_annotations_docs() -> str:
        """Return documentation for Chaise UI annotations."""
        return """# Chaise Annotation Reference

Annotations customize how data is displayed in Chaise, Deriva's web interface.
They don't affect the data itself, only its presentation.

## Annotation Overview

| Annotation | Applies To | Purpose |
|------------|-----------|---------|
| `display` | All elements | Names, tooltips, formatting |
| `visible-columns` | Tables | Column order and visibility |
| `visible-foreign-keys` | Tables | Related table visibility |
| `table-display` | Tables | Row ordering, markdown patterns |
| `column-display` | Columns | Column-specific display |
| `key-display` | Keys | Key presentation |
| `foreign-key` | Foreign Keys | FK names and filtering |
| `asset` | Columns | File/asset handling |
| `export` | Tables | Export templates |
| `citation` | Tables | Citation formatting |

## Annotation Tag Format

Annotation keys follow URI format:
- `tag:misd.isi.edu,2015:display`
- `tag:isrd.isi.edu,2016:visible-columns`

The date indicates when the annotation was defined.

## Display Contexts

Many annotations are context-sensitive:

| Context | Where Used |
|---------|-----------|
| `*` | Default for all contexts |
| `compact` | Table listings, search results |
| `detailed` | Single record view |
| `entry` | Create/edit forms |
| `entry/create` | Create form only |
| `entry/edit` | Edit form only |
| `filter` | Facet panel |
| `compact/brief` | Inline references |
| `compact/select` | Foreign key selection |

## Common Annotations

### Display (`tag:misd.isi.edu,2015:display`)

Controls names and formatting:

```json
{
  "name": "Friendly Name",
  "markdown_name": "**Bold Name**",
  "comment": "Tooltip text",
  "name_style": {
    "underline_space": true,
    "title_case": true
  },
  "show_null": {
    "*": true,
    "detailed": false
  }
}
```

**Key options:**
- `name`: Override element name
- `markdown_name`: Markdown-formatted name
- `comment`: Tooltip/description
- `name_style.underline_space`: Convert `_` to spaces
- `name_style.title_case`: Capitalize words
- `show_null`: How to display NULL values

### Visible Columns (`tag:isrd.isi.edu,2016:visible-columns`)

Controls which columns appear and in what order:

```json
{
  "compact": ["RID", "Name", "Description"],
  "detailed": ["RID", "Name", "Description", "Created"],
  "entry": ["Name", "Description"],
  "filter": {
    "and": [
      {"source": "Name"},
      {"source": "Status", "choices": ["Active"]}
    ]
  }
}
```

**Column directives can be:**
- Column name: `"Name"`
- Foreign key: `["Schema", "FK_Name"]`
- Path with source: `{"source": [{"outbound": ["S", "FK"]}, "column"]}`

### Visible Foreign Keys (`tag:isrd.isi.edu,2016:visible-foreign-keys`)

Controls related tables shown on record page:

```json
{
  "detailed": [
    ["Schema", "Image_Subject_fkey"],
    ["Schema", "Diagnosis_Subject_fkey"]
  ]
}
```

Only inbound foreign keys (tables referencing this one) apply.

### Table Display (`tag:isrd.isi.edu,2016:table-display`)

Controls table-level presentation:

```json
{
  "row_order": [
    {"column": "RCT", "descending": true}
  ],
  "page_size": 25,
  "row_markdown_pattern": "{{{Name}}} ({{{RID}}})"
}
```

**Key options:**
- `row_order`: Default sort order
- `page_size`: Results per page
- `row_markdown_pattern`: Custom row display

### Column Display (`tag:isrd.isi.edu,2016:column-display`)

Column-specific display options:

```json
{
  "*": {
    "markdown_pattern": "[{{{_self}}}](https://example.com/{{{_self}}})"
  },
  "compact": {
    "markdown_pattern": "{{{_self}}}"
  }
}
```

Use `{{{_self}}}` for raw value, `{{{$self}}}` for formatted value.

### Foreign Key (`tag:isrd.isi.edu,2016:foreign-key`)

Customize foreign key display:

```json
{
  "from_name": "Images",
  "to_name": "Subject",
  "to_comment": "The subject this image belongs to",
  "domain_filter": {
    "ermrest_path_pattern": "Status=Active"
  }
}
```

### Asset (`tag:isrd.isi.edu,2017:asset`)

Mark a column as containing file assets:

```json
{
  "url_pattern": "/hatrac/data/{{{MD5}}}/{{{Filename}}}",
  "filename_column": "Filename",
  "byte_count_column": "Length",
  "md5_column": "MD5",
  "browser_upload": true
}
```

### Export (`tag:isrd.isi.edu,2019:export`)

Define export templates:

```json
{
  "templates": [
    {
      "displayname": "BDBag",
      "type": "BAG",
      "outputs": [
        {
          "source": {"api": "entity"},
          "destination": {"name": "data", "type": "csv"}
        }
      ]
    }
  ]
}
```

## Pattern Expansion

Many annotations support Handlebars/Mustache templates:

```
{{{column_name}}}           - Column value
{{{$fkey_schema_fkey_name}}} - Foreign key value
{{{_date}}}                 - Current date
{{{_timestamp}}}            - Current timestamp
{{{#if column}}}...{{{/if}}} - Conditional
{{{#each array}}}...{{{/each}}} - Loop
```

## Applying Annotations

### Via DerivaML MCP

```
apply_catalog_annotations("My Catalog", "ML Browser")
```

This sets up navigation, display settings, and bulk upload.

### Via deriva-workbench

The DERIVA Workbench provides a graphical editor for annotations:
1. Connect to catalog
2. Browse schema tree
3. Double-click annotation to edit
4. Click Update to save

### Via Python API

```python
from deriva.core import ErmrestCatalog
import deriva.core.ermrest_model as em

catalog = ErmrestCatalog('https', 'example.org', '1')
model = catalog.getCatalogModel()

table = model.table('schema', 'table')
table.annotations[em.tag.display] = {"name": "Friendly Name"}
model.apply()
```

## Tips

1. **Use `*` context** as default, override specific contexts
2. **Test in Chaise** after applying annotations
3. **Use workbench** for complex visual editing
4. **Export before changes** using dump/restore for backup
5. **Check inheritance** - annotations cascade from catalog → schema → table

## Related Resources

- `deriva-ml://docs/ermrest-model-management` - Schema management
- Deriva Workbench - GUI annotation editor
- Chaise documentation: https://docs.derivacloud.org/
"""

    # =========================================================================
    # DerivaML Conceptual Documentation Resources
    # =========================================================================

    @mcp.resource(
        "deriva-ml://docs/data-model",
        name="DerivaML Data Model",
        description="Conceptual overview of DerivaML entities and their relationships",
        mime_type="application/json",
    )
    def get_data_model_docs() -> str:
        """Return conceptual documentation for DerivaML data model."""
        return json.dumps({
            "title": "DerivaML Data Model",
            "description": "Understanding the core entities and their relationships in DerivaML",
            "core_concepts": {
                "dataset": {
                    "definition": "A versioned collection of records for reproducible ML experiments",
                    "key_properties": [
                        "Semantic versioning (major.minor.patch)",
                        "Contains records from domain tables (e.g., Image, Subject)",
                        "Can be nested (parent/child relationships for train/test splits)",
                        "Immutable snapshots via catalog versioning"
                    ],
                    "created_by": "Execution (datasets are always created within an execution context)",
                    "relationships": {
                        "contains": "Records from registered dataset element types",
                        "created_by": "Execution that produced this dataset",
                        "nested_in": "Optional parent dataset (for train/test partitioning)"
                    }
                },
                "execution": {
                    "definition": "A single run of an ML workflow with full provenance tracking",
                    "key_properties": [
                        "Links to input datasets and assets",
                        "Tracks output datasets, assets, and feature values",
                        "Records timing (start/stop) and status",
                        "Provides working directory for file staging"
                    ],
                    "lifecycle": [
                        "1. create_execution() - Initialize with workflow and inputs",
                        "2. start_execution() - Begin timing",
                        "3. [Perform ML operations]",
                        "4. asset_file_path() - Register output files",
                        "5. stop_execution() - End timing",
                        "6. upload_execution_outputs() - REQUIRED: Upload files to catalog"
                    ],
                    "relationships": {
                        "runs": "Workflow definition",
                        "consumes": "Input datasets and assets",
                        "produces": "Output datasets, assets, and feature values"
                    }
                },
                "workflow": {
                    "definition": "A registered computational process or analysis pipeline",
                    "key_properties": [
                        "Has a unique URL (typically git repository)",
                        "Has a checksum for version tracking",
                        "Categorized by workflow_type vocabulary"
                    ],
                    "purpose": "Enables reproducibility by tracking which code produced which results"
                },
                "feature": {
                    "definition": "Metadata, labels, or annotations associated with domain objects",
                    "key_properties": [
                        "Attached to a target table (e.g., 'Diagnosis' feature on 'Image')",
                        "Values come from controlled vocabularies or assets",
                        "Every value records which execution produced it (provenance)"
                    ],
                    "types": {
                        "term_based": "Values from vocabulary tables (e.g., diagnosis labels)",
                        "asset_based": "Values reference asset files (e.g., segmentation masks)",
                        "mixed": "Can have multiple columns of different types"
                    },
                    "relationships": {
                        "targets": "Domain table records (via foreign key)",
                        "produced_by": "Execution that created this value"
                    }
                },
                "vocabulary": {
                    "definition": "A controlled set of terms (like an enum) for standardized values",
                    "key_properties": [
                        "Terms have names, descriptions, and optional synonyms",
                        "Used by features, dataset types, workflow types, asset types"
                    ],
                    "examples": [
                        "Dataset_Type: Training, Testing, Validation",
                        "Workflow_Type: Training, Inference, Data Preparation",
                        "Asset_Type: Model_File, Checkpoint, Segmentation_Mask"
                    ]
                },
                "asset": {
                    "definition": "A file with URL, checksum, and provenance metadata",
                    "key_properties": [
                        "Automatic URL and MD5 checksum tracking",
                        "Categorized by Asset_Type vocabulary",
                        "Can be inputs or outputs of executions"
                    ],
                    "tables": {
                        "Execution_Asset": "General execution outputs",
                        "Execution_Metadata": "Configuration and log files",
                        "Custom asset tables": "Domain-specific files (e.g., Image, Model)"
                    }
                }
            },
            "relationship_diagram": {
                "description": "How entities connect",
                "flows": [
                    "Workflow → defines → Execution",
                    "Execution → consumes → Dataset (input)",
                    "Execution → produces → Dataset (output)",
                    "Execution → produces → Asset (output)",
                    "Execution → produces → Feature Value",
                    "Dataset → contains → Domain Records (Image, Subject, etc.)",
                    "Feature → targets → Domain Table",
                    "Feature Value → references → Vocabulary Term or Asset"
                ]
            },
            "provenance_chain": {
                "description": "Complete traceability from results back to source",
                "example": [
                    "Feature value 'Diagnosis=Normal' on Image 'IMG-001'",
                    "← produced by Execution 'EXE-123'",
                    "← which ran Workflow 'Classification Pipeline v1.2'",
                    "← using Dataset 'Training Set v2.0.0'",
                    "← containing 1000 images from domain table"
                ]
            }
        }, indent=2)

    @mcp.resource(
        "deriva-ml://docs/python-api",
        name="DerivaML Python API Reference",
        description="Python API patterns for writing deriva-ml programs",
        mime_type="application/json",
    )
    def get_python_api_docs() -> str:
        """Return Python API reference documentation."""
        return json.dumps({
            "title": "DerivaML Python API Reference",
            "description": "Patterns and examples for writing Python programs using deriva-ml",
            "connection": {
                "basic": {
                    "code": "from deriva_ml import DerivaML\\nml = DerivaML(hostname='example.org', catalog_id='1')",
                    "note": "Creates connection with automatic credential lookup from ~/.deriva"
                },
                "with_working_dir": {
                    "code": "ml = DerivaML(hostname='example.org', catalog_id='1', working_dir='/tmp/ml_work')",
                    "note": "Specify directory for dataset downloads and execution outputs"
                }
            },
            "execution_configuration": {
                "class": "ExecutionConfiguration",
                "import": "from deriva_ml.execution import ExecutionConfiguration",
                "required_fields": {
                    "workflow": "Workflow object (from ml.create_workflow())",
                    "description": "Human-readable description of this execution"
                },
                "optional_fields": {
                    "datasets": "List[DatasetSpec] - Input datasets to consume",
                    "assets": "List[str] - Input asset RIDs"
                },
                "example": """from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.dataset.aux_classes import DatasetSpec

config = ExecutionConfiguration(
    workflow=workflow,
    description="Training run with augmented data",
    datasets=[
        DatasetSpec(rid="1-ABC", version="2.0.0", materialize=True),
    ],
)"""
            },
            "execution_context_manager": {
                "description": "Recommended pattern for running executions",
                "pattern": """# Create execution
execution = ml.create_execution(config)

# Use context manager for automatic timing
with execution.execute() as exe:
    # Download input datasets
    for ds in exe.datasets:
        bag_path = exe.download_dataset_bag(ds)
        # Process data from bag_path...

    # Register output files
    model_path = exe.asset_file_path("Model", "trained_model.pt")
    # Save model to model_path...

    # Create output dataset
    output_ds = exe.create_dataset(
        dataset_types=["Training"],
        description="Processed training data"
    )

# IMPORTANT: Upload outputs AFTER context manager exits
execution.upload_execution_outputs()""",
                "timing_notes": [
                    "Context manager automatically calls execution_start() on enter",
                    "Context manager automatically calls execution_stop() on exit",
                    "upload_execution_outputs() must be called AFTER exiting context"
                ]
            },
            "workflow_creation": {
                "description": "Register a workflow for provenance tracking",
                "pattern": """# Ensure workflow type exists
ml.add_term('Workflow_Type', 'Training', description='Model training workflow')

# Create workflow object
workflow = ml.create_workflow(
    name="ResNet50 Training",
    workflow_type="Training",
    description="Train ResNet50 classifier on image data"
)

# Register in catalog (returns RID, or existing RID if same URL/checksum)
workflow_rid = ml.add_workflow(workflow)""",
                "notes": [
                    "Workflow URL and checksum are auto-detected from git repository",
                    "Same workflow code will return same RID (idempotent)"
                ]
            },
            "dataset_creation": {
                "description": "Create datasets within execution context",
                "pattern": """with execution.execute() as exe:
    # Create a new dataset
    dataset = exe.create_dataset(
        dataset_types=["Training", "Augmented"],
        description="Augmented training images"
    )

    # Add members from domain tables
    # First ensure table is registered as element type
    ml.add_dataset_element_type("Image")

    # Add records by RID
    image_rids = ["2-ABC", "2-DEF", "2-GHI"]
    dataset.add_elements({"Image": image_rids})""",
                "version_notes": [
                    "Adding elements auto-increments minor version",
                    "Manual version bump: dataset.increment_version('major', 'Schema change')"
                ]
            },
            "feature_creation": {
                "description": "Define and populate features for ML labeling",
                "pattern": """# Create vocabulary for feature values
ml.create_vocabulary("Diagnosis_Type", comment="Medical diagnosis categories")
ml.add_term("Diagnosis_Type", "Normal", description="No abnormality")
ml.add_term("Diagnosis_Type", "Abnormal", description="Abnormality detected")

# Create feature definition
ml.create_feature(
    table_name="Image",
    feature_name="Diagnosis",
    comment="Clinical diagnosis label",
    terms=["Diagnosis_Type"]  # Values from this vocabulary
)

# Add feature values (within execution context)
with execution.execute() as exe:
    # Get the dynamically generated feature record class
    DiagnosisFeature = ml.feature_record_class("Image", "Diagnosis")

    # Create feature records
    features = [
        DiagnosisFeature(Image="2-ABC", Diagnosis_Type="Normal"),
        DiagnosisFeature(Image="2-DEF", Diagnosis_Type="Abnormal"),
    ]

    # Add to catalog with provenance
    exe.add_features(features)""",
                "notes": [
                    "feature_record_class() returns a Pydantic model with validation",
                    "Feature values automatically link to the execution for provenance"
                ]
            },
            "asset_handling": {
                "description": "Register and upload output files",
                "pattern": """with execution.execute() as exe:
    # Get path to save file (creates staging directory)
    model_path = exe.asset_file_path(
        asset_name="Execution_Asset",  # Target asset table
        file_name="model.pt",
        asset_types=["Model_File"]  # Asset_Type vocabulary terms
    )

    # Save your file to this path
    torch.save(model.state_dict(), model_path)

    # For existing files, copy/link them
    config_path = exe.asset_file_path(
        "Execution_Metadata",
        "/path/to/config.json",
        copy_file=True  # Copy instead of symlink
    )

# Upload all registered files
uploaded = execution.upload_execution_outputs()
# Returns: {"Execution_Asset": [AssetInfo(rid=..., url=...)], ...}"""
            },
            "querying_data": {
                "description": "Query records from domain tables",
                "pattern": """# Find datasets
datasets = list(ml.find_datasets())
for ds in datasets:
    print(f"{ds.dataset_rid}: {ds.description}")

# Query domain table
pb = ml.pathBuilder()
images = list(pb.schemas['domain'].Image.entities().fetch())

# Get dataset contents
members = ml.list_dataset_members(dataset_rid)
# Returns: {"Image": [{"RID": "2-ABC"}, ...], "Subject": [...]}"""
            },
            "denormalization": {
                "description": "Join tables for ML training data",
                "pattern": """# Get flat/wide view of dataset
flat_data = ml.denormalize_dataset(
    dataset_rid="1-ABC",
    include_tables=["Image", "Subject", "Diagnosis"]
)
# Returns: {
#   "columns": ["Image.RID", "Image.Filename", "Subject.Name", "Diagnosis.Label"],
#   "rows": [[...], [...], ...]
# }

# Convert to pandas DataFrame
import pandas as pd
df = pd.DataFrame(flat_data["rows"], columns=flat_data["columns"])"""
            }
        }, indent=2)

    @mcp.resource(
        "deriva-ml://docs/tool-sequences",
        name="MCP Tool Sequences",
        description="Correct ordering of MCP tool calls for common workflows",
        mime_type="application/json",
    )
    def get_tool_sequences_docs() -> str:
        """Return documentation for proper MCP tool call sequences."""
        return json.dumps({
            "title": "MCP Tool Call Sequences",
            "description": "Proper ordering of tool calls for common DerivaML workflows",
            "important_notes": [
                "MCP tools do NOT use Python context managers - you must manually sequence calls",
                "upload_execution_outputs() is REQUIRED after any execution - files are not saved without it",
                "Most operations require an active catalog connection first"
            ],
            "sequences": {
                "connect_and_explore": {
                    "description": "Connect to catalog and explore its contents",
                    "steps": [
                        {"tool": "connect_catalog", "params": {"hostname": "example.org", "catalog_id": "1"}, "note": "Establishes connection and creates MCP tracking execution"},
                        {"tool": "get_catalog_info", "params": {}, "note": "View catalog details including schema names"},
                        {"tool": "list_tables", "params": {}, "note": "See all domain tables"},
                        {"tool": "list_vocabularies", "params": {}, "note": "See all vocabulary tables"},
                        {"tool": "list_datasets", "params": {}, "note": "See all datasets"}
                    ]
                },
                "create_dataset_with_members": {
                    "description": "Create a new dataset and populate it with records",
                    "prerequisites": ["Active catalog connection"],
                    "steps": [
                        {"tool": "create_execution", "params": {"workflow_name": "Data Curation", "workflow_type": "Data Preparation", "description": "Creating training dataset"}, "note": "Start execution for provenance"},
                        {"tool": "start_execution", "params": {}, "note": "Begin timing"},
                        {"tool": "create_execution_dataset", "params": {"description": "Training images", "dataset_types": ["Training"]}, "note": "Create empty dataset"},
                        {"tool": "add_dataset_element_type", "params": {"table_name": "Image"}, "note": "Register Image as valid element type (if not already)"},
                        {"tool": "add_dataset_members", "params": {"dataset_rid": "<from step 3>", "member_rids": ["2-ABC", "2-DEF"]}, "note": "Add records to dataset"},
                        {"tool": "stop_execution", "params": {}, "note": "End timing"},
                        {"tool": "upload_execution_outputs", "params": {}, "note": "REQUIRED: Finalize execution"}
                    ]
                },
                "add_feature_values": {
                    "description": "Add labels/annotations to domain objects",
                    "prerequisites": ["Active catalog connection", "Feature definition exists", "Vocabulary terms exist"],
                    "steps": [
                        {"tool": "create_execution", "params": {"workflow_name": "Labeling", "workflow_type": "Annotation", "description": "Adding diagnosis labels"}, "note": "Start execution"},
                        {"tool": "start_execution", "params": {}, "note": "Begin timing"},
                        {"tool": "lookup_feature", "params": {"table_name": "Image", "feature_name": "Diagnosis"}, "note": "Get feature structure to know what fields to provide"},
                        {"tool": "add_feature_value", "params": {"table_name": "Image", "feature_name": "Diagnosis", "target_rid": "2-ABC", "value": "Normal"}, "note": "Add value for one record"},
                        {"tool": "stop_execution", "params": {}, "note": "End timing"},
                        {"tool": "upload_execution_outputs", "params": {}, "note": "REQUIRED: Finalize"}
                    ],
                    "alternative": "Use add_feature_value_record for features with multiple fields"
                },
                "create_feature_definition": {
                    "description": "Define a new feature for labeling domain objects",
                    "prerequisites": ["Active catalog connection"],
                    "steps": [
                        {"tool": "create_vocabulary", "params": {"vocabulary_name": "Diagnosis_Type", "comment": "Diagnosis categories"}, "note": "Create vocabulary for values (if needed)"},
                        {"tool": "add_term", "params": {"vocabulary_name": "Diagnosis_Type", "term_name": "Normal", "description": "No abnormality"}, "note": "Add vocabulary terms"},
                        {"tool": "add_term", "params": {"vocabulary_name": "Diagnosis_Type", "term_name": "Abnormal", "description": "Abnormality detected"}, "note": "Add more terms"},
                        {"tool": "create_feature", "params": {"table_name": "Image", "feature_name": "Diagnosis", "comment": "Clinical diagnosis", "terms": ["Diagnosis_Type"]}, "note": "Create feature definition"}
                    ]
                },
                "run_ml_execution": {
                    "description": "Run an ML workflow with input datasets and output assets",
                    "prerequisites": ["Active catalog connection", "Input datasets exist"],
                    "steps": [
                        {"tool": "create_execution", "params": {"workflow_name": "Model Training", "workflow_type": "Training", "description": "Train classifier", "dataset_rids": ["1-ABC"]}, "note": "Create with input datasets"},
                        {"tool": "start_execution", "params": {}, "note": "Begin timing"},
                        {"tool": "download_execution_dataset", "params": {"dataset_rid": "1-ABC"}, "note": "Download input data"},
                        {"tool": "get_execution_working_dir", "params": {}, "note": "Get path for outputs"},
                        {"tool": "asset_file_path", "params": {"asset_name": "Execution_Asset", "file_name": "model.pt", "asset_types": ["Model_File"]}, "note": "Register output file"},
                        {"tool": "stop_execution", "params": {}, "note": "End timing"},
                        {"tool": "upload_execution_outputs", "params": {}, "note": "REQUIRED: Upload files"}
                    ],
                    "note": "Between steps 5 and 6, you would actually save your model file to the path returned by asset_file_path"
                },
                "create_schema_elements": {
                    "description": "Create tables, vocabularies, and features for a new domain",
                    "prerequisites": ["Active catalog connection (to a new or existing catalog)"],
                    "steps": [
                        {"tool": "create_vocabulary", "params": {"vocabulary_name": "Species", "comment": "Animal species"}, "note": "Create vocabulary"},
                        {"tool": "add_term", "params": {"vocabulary_name": "Species", "term_name": "Human", "description": "Homo sapiens"}, "note": "Populate vocabulary"},
                        {"tool": "create_table", "params": {"table_name": "Subject", "columns": [{"name": "Name", "type": "text"}, {"name": "Age", "type": "int4"}], "comment": "Study subjects"}, "note": "Create domain table"},
                        {"tool": "create_asset_table", "params": {"asset_name": "Image", "columns": [{"name": "Width", "type": "int4"}], "referenced_tables": ["Subject"], "comment": "Medical images"}, "note": "Create asset table with FK to Subject"},
                        {"tool": "add_dataset_element_type", "params": {"table_name": "Image"}, "note": "Allow Images in datasets"},
                        {"tool": "create_feature", "params": {"table_name": "Image", "feature_name": "Quality", "terms": ["Quality_Level"]}, "note": "Create feature for labeling"}
                    ]
                }
            },
            "common_mistakes": [
                {
                    "mistake": "Forgetting to call upload_execution_outputs()",
                    "consequence": "All registered assets are lost, execution appears incomplete",
                    "fix": "Always call upload_execution_outputs() after stop_execution()"
                },
                {
                    "mistake": "Adding feature values without an execution context",
                    "consequence": "Error: 'No active execution'",
                    "fix": "Create an execution first, or the MCP connection execution will be used"
                },
                {
                    "mistake": "Creating dataset outside execution context",
                    "consequence": "Error or missing provenance",
                    "fix": "Use create_execution_dataset() within an execution"
                },
                {
                    "mistake": "Adding members to dataset before registering element type",
                    "consequence": "Error: table not registered as dataset element type",
                    "fix": "Call add_dataset_element_type() first"
                }
            ]
        }, indent=2)

    @mcp.resource(
        "deriva-ml://docs/dataset-versioning",
        name="Dataset Versioning Policy",
        description="Semantic versioning policy for DerivaML datasets",
        mime_type="application/json",
    )
    def get_dataset_versioning_docs() -> str:
        """Return documentation for dataset semantic versioning."""
        return json.dumps({
            "title": "Dataset Semantic Versioning Policy",
            "description": "How DerivaML uses semantic versioning (major.minor.patch) for datasets",
            "format": {
                "pattern": "MAJOR.MINOR.PATCH",
                "example": "2.1.3",
                "initial": "0.1.0 (default for new datasets)"
            },
            "version_components": {
                "patch": {
                    "incremented_when": "Metadata-only changes",
                    "examples": [
                        "Updating dataset description",
                        "Changing dataset type labels",
                        "Fixing typos in metadata"
                    ],
                    "data_compatibility": "Fully compatible - same records, same structure"
                },
                "minor": {
                    "incremented_when": "Element changes (additions/removals)",
                    "examples": [
                        "Adding new records to dataset",
                        "Removing records from dataset",
                        "Changing which records are included"
                    ],
                    "data_compatibility": "Forward compatible - may have more/fewer records",
                    "auto_increment": "Adding members via add_dataset_members() auto-increments minor"
                },
                "major": {
                    "incremented_when": "Schema or structural changes",
                    "examples": [
                        "Adding new element types (e.g., adding Subject records to Image-only dataset)",
                        "Changing the dataset's fundamental structure",
                        "Breaking changes to how data should be interpreted"
                    ],
                    "data_compatibility": "Not compatible - consumers may need code changes"
                }
            },
            "automatic_versioning": {
                "description": "Some operations auto-increment versions",
                "rules": [
                    "add_dataset_members() → increments MINOR",
                    "Metadata changes via update → increments PATCH (when tracked)",
                    "Manual increment via increment_dataset_version() for MAJOR changes"
                ]
            },
            "version_history": {
                "description": "Every version is preserved with a catalog snapshot",
                "access": "Use get_dataset_version_history() to see all versions",
                "querying": "Use version parameter in list_dataset_members() to query historical state",
                "immutability": "Previous versions are immutable - changes create new versions"
            },
            "best_practices": [
                {
                    "practice": "Start at 0.1.0 for development",
                    "reason": "Indicates pre-release/experimental status"
                },
                {
                    "practice": "Bump to 1.0.0 when dataset is production-ready",
                    "reason": "Signals stability to consumers"
                },
                {
                    "practice": "Document version changes in description",
                    "reason": "Helps track what changed between versions"
                },
                {
                    "practice": "Use nested datasets for train/test splits",
                    "reason": "Parent dataset versioning encompasses all children"
                }
            ],
            "example_history": [
                {"version": "0.1.0", "description": "Initial dataset creation"},
                {"version": "0.2.0", "description": "Added 500 more images"},
                {"version": "0.3.0", "description": "Removed 50 low-quality images"},
                {"version": "0.3.1", "description": "Updated description"},
                {"version": "1.0.0", "description": "Production release - schema finalized"},
                {"version": "1.1.0", "description": "Added 200 new validation images"},
                {"version": "2.0.0", "description": "Added Subject records - major structural change"}
            ],
            "mcp_tools": {
                "view_history": "get_dataset_version_history(dataset_rid)",
                "manual_bump": "increment_dataset_version(dataset_rid, component='minor', description='...')",
                "query_version": "list_dataset_members(dataset_rid, version='1.0.0')"
            }
        }, indent=2)
