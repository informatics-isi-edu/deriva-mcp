"""MCP Tools for DerivaML.

This module provides tool registration functions that expose
DerivaML operations as MCP tools.
"""

from deriva_ml_mcp.tools.annotation import register_annotation_tools
from deriva_ml_mcp.tools.auth import register_auth_tools
from deriva_ml_mcp.tools.catalog import register_catalog_tools
from deriva_ml_mcp.tools.data import register_data_tools
from deriva_ml_mcp.tools.dataset import register_dataset_tools
from deriva_ml_mcp.tools.devtools import register_devtools
from deriva_ml_mcp.tools.execution import register_execution_tools, register_storage_tools
from deriva_ml_mcp.tools.feature import register_feature_tools
from deriva_ml_mcp.tools.schema import register_schema_tools
from deriva_ml_mcp.tools.vocabulary import register_vocabulary_tools
from deriva_ml_mcp.tools.workflow import register_workflow_tools

__all__ = [
    "register_annotation_tools",
    "register_auth_tools",
    "register_catalog_tools",
    "register_dataset_tools",
    "register_vocabulary_tools",
    "register_workflow_tools",
    "register_feature_tools",
    "register_schema_tools",
    "register_execution_tools",
    "register_storage_tools",
    "register_data_tools",
    "register_devtools",
]
