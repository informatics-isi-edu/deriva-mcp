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
            for wf in ml.list_workflows():
                workflows.append({
                    "rid": wf.get("RID"),
                    "name": wf.get("Name"),
                    "url": wf.get("URL"),
                    "workflow_type": wf.get("Workflow_Type"),
                    "description": wf.get("Description"),
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
        """Return comprehensive Handlebars template documentation."""
        return json.dumps({
            "title": "Handlebars Template Guide for Deriva",
            "description": "Templates are used in row_markdown_pattern, markdown_pattern, and other display options",
            "template_engines": {
                "handlebars": {
                    "description": "Default template engine with more features",
                    "triple_braces": "{{{value}}} - Raw output (no HTML escaping)",
                    "double_braces": "{{value}} - HTML-escaped output"
                },
                "mustache": {
                    "description": "Simpler template engine, subset of Handlebars",
                    "note": "Use template_engine: 'mustache' to select"
                }
            },
            "basic_syntax": {
                "variable_output": {
                    "raw": "{{{column_name}}}",
                    "escaped": "{{column_name}}",
                    "recommendation": "Use triple braces {{{...}}} for most Deriva use cases"
                },
                "accessing_values": {
                    "current_column": "{{{_value}}} or {{{value}}}",
                    "other_column": "{{{column_name}}}",
                    "nested_property": "{{{object.property}}}",
                    "array_element": "{{{array.0}}}"
                }
            },
            "deriva_specific_variables": {
                "row_context": {
                    "_value": "Current column/field value",
                    "_row": "Object with all columns in current row",
                    "_row.RID": "RID of current row",
                    "_row.column_name": "Any column value from current row"
                },
                "foreign_key_values": {
                    "$fkeys": "Object containing FK-related data",
                    "$fkeys.schema.constraint.values": "Values from related table via FK",
                    "$fkeys.schema.constraint.values.column_name": "Specific column from related row",
                    "$fkeys.schema.constraint.rowName": "Row name of related record"
                },
                "system_values": {
                    "$moment": "Moment.js for date formatting",
                    "$catalog": "Catalog information object",
                    "$catalog.id": "Catalog ID",
                    "$catalog.snapshot": "Current snapshot ID"
                },
                "url_encoding": {
                    "$uri_path": "URI path component (for paths)",
                    "$uri_component": "URI component (for query params)"
                }
            },
            "handlebars_helpers": {
                "conditionals": {
                    "#if": {
                        "syntax": "{{#if value}}...{{/if}}",
                        "example": "{{#if Description}}{{{Description}}}{{else}}No description{{/if}}",
                        "note": "Tests for truthiness (false, null, undefined, '', 0, [] are falsy)"
                    },
                    "#unless": {
                        "syntax": "{{#unless value}}...{{/unless}}",
                        "example": "{{#unless Active}}(Inactive){{/unless}}"
                    },
                    "if_else": {
                        "syntax": "{{#if cond}}...{{else}}...{{/if}}",
                        "example": "{{#if URL}}[Link]({{{URL}}}){{else}}No link{{/if}}"
                    }
                },
                "iteration": {
                    "#each": {
                        "syntax": "{{#each array}}...{{/each}}",
                        "example": "{{#each Tags}}{{{this}}}{{#unless @last}}, {{/unless}}{{/each}}",
                        "variables": {
                            "this": "Current item",
                            "@index": "Current index (0-based)",
                            "@first": "Boolean - is first item",
                            "@last": "Boolean - is last item",
                            "@key": "Current key (for objects)"
                        }
                    }
                },
                "comparison": {
                    "#ifCond": {
                        "syntax": "{{#ifCond val1 op val2}}...{{/ifCond}}",
                        "operators": ["==", "!=", "<", ">", "<=", ">=", "===", "!==", "&&", "||"],
                        "example": "{{#ifCond Status '==' 'Active'}}✓{{/ifCond}}"
                    }
                },
                "string_helpers": {
                    "escape": "{{escape value}} - HTML escape",
                    "toJSON": "{{toJSON object}} - Convert to JSON string",
                    "encodeFacet": "{{encodeFacet object}} - Encode for facet URL"
                },
                "math_helpers": {
                    "math": {
                        "syntax": "{{math val1 op val2}}",
                        "operators": ["+", "-", "*", "/", "%"],
                        "example": "{{math Price '*' Quantity}}"
                    }
                },
                "date_helpers": {
                    "formatDate": {
                        "syntax": "{{formatDate date format}}",
                        "formats": ["YYYY-MM-DD", "MM/DD/YYYY", "MMMM D, YYYY", "relative"],
                        "example": "{{formatDate RCT 'MMMM D, YYYY'}}"
                    },
                    "humanizeBytes": {
                        "syntax": "{{humanizeBytes bytes}}",
                        "example": "{{humanizeBytes Length}} → '1.5 MB'"
                    }
                }
            },
            "common_patterns": {
                "row_name_patterns": [
                    {
                        "description": "Simple column value",
                        "pattern": "{{{Name}}}"
                    },
                    {
                        "description": "Multiple columns",
                        "pattern": "{{{First_Name}}} {{{Last_Name}}}"
                    },
                    {
                        "description": "With conditional",
                        "pattern": "{{{Name}}}{{#if Nickname}} ({{{Nickname}}}){{/if}}"
                    },
                    {
                        "description": "With FK value",
                        "pattern": "{{{Filename}}} - {{{$fkeys.domain.Image_Subject_fkey.rowName}}}"
                    }
                ],
                "column_display_patterns": [
                    {
                        "description": "Link from URL column",
                        "pattern": "[{{{_value}}}]({{{_value}}})"
                    },
                    {
                        "description": "Image thumbnail",
                        "pattern": "[![Thumbnail]({{{_value}}}?h=100)]({{{_value}}})"
                    },
                    {
                        "description": "Email link",
                        "pattern": "[{{{_value}}}](mailto:{{{_value}}})"
                    },
                    {
                        "description": "Badge/status",
                        "pattern": "**{{{_value}}}**"
                    },
                    {
                        "description": "Conditional formatting",
                        "pattern": "{{#ifCond _value '>' 0}}+{{{_value}}}{{else}}{{{_value}}}{{/ifCond}}"
                    }
                ],
                "pseudo_column_patterns": [
                    {
                        "description": "Count related items",
                        "pattern": "{{{$self.values.length}}} items"
                    },
                    {
                        "description": "List with separator",
                        "pattern": "{{#each $self.values}}{{{this.Name}}}{{#unless @last}}, {{/unless}}{{/each}}"
                    }
                ]
            },
            "debugging_tips": [
                "Use {{toJSON variable}} to inspect complex objects",
                "Start simple and add complexity incrementally",
                "Check browser console for template rendering errors",
                "Test with sample data using preview_handlebars_template tool",
                "Triple braces {{{...}}} prevent HTML escaping issues"
            ],
            "examples": {
                "row_name": {
                    "simple": {
                        "pattern": "{{{Name}}}",
                        "output": "John Smith"
                    },
                    "composite": {
                        "pattern": "{{{Last_Name}}}, {{{First_Name}}}",
                        "output": "Smith, John"
                    },
                    "with_fk": {
                        "pattern": "{{{Filename}}} ({{{$fkeys.domain.Image_Subject_fkey.rowName}}})",
                        "output": "scan001.jpg (Patient A)"
                    }
                },
                "column_display": {
                    "url_as_link": {
                        "pattern": "[Download]({{{_value}}})",
                        "input": "https://example.com/file.pdf",
                        "output": "[Download](https://example.com/file.pdf)"
                    },
                    "image_thumbnail": {
                        "pattern": "[![]({{{_value}}}?h=80)]({{{_value}}})",
                        "note": "Creates clickable thumbnail"
                    },
                    "conditional_status": {
                        "pattern": "{{#if _value}}✓ Active{{else}}✗ Inactive{{/if}}",
                        "input_true": True,
                        "output_true": "✓ Active"
                    }
                }
            },
            "tools_for_templates": {
                "preview_handlebars_template": "Test a template with sample data",
                "get_table_sample_data": "Get sample row data for template testing",
                "get_handlebars_template_variables": "List available variables for a table"
            }
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
