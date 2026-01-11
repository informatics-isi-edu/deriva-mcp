"""MCP Tools for DerivaML.

This module provides tool registration functions that expose
DerivaML operations as MCP tools.
"""

from deriva_ml_mcp.tools.catalog import register_catalog_tools
from deriva_ml_mcp.tools.dataset import register_dataset_tools
from deriva_ml_mcp.tools.vocabulary import register_vocabulary_tools
from deriva_ml_mcp.tools.workflow import register_workflow_tools
from deriva_ml_mcp.tools.feature import register_feature_tools

__all__ = [
    "register_catalog_tools",
    "register_dataset_tools",
    "register_vocabulary_tools",
    "register_workflow_tools",
    "register_feature_tools",
]
