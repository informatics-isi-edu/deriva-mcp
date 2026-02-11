"""Pytest configuration and shared fixtures for deriva-mcp tests.

This module provides fixtures for both unit tests (mocked DerivaML) and
integration tests (live catalog) of the MCP server tools and resources.

Unit Test Fixtures:
    - mock_ml: Mocked DerivaML instance
    - mock_conn_manager: ConnectionManager that returns mock_ml
    - disconnected_conn_manager: ConnectionManager with no active connection
    - *_tools: Per-module captured tool functions
    - *_tools_disconnected: Per-module tools with no connection

Integration Test Fixtures (require DERIVA_HOST):
    - catalog_host: Test server hostname
    - catalog_manager: CatalogManager instance for test catalog lifecycle
    - mcp_connection_manager: Pre-connected ConnectionManager

Configuration:
    DERIVA_HOST: Environment variable for test server (default: localhost)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from deriva_ml_mcp.connection import ConnectionManager

if TYPE_CHECKING:
    from deriva_ml import DerivaML
    from deriva_ml.demo_catalog import DatasetDescription


# =============================================================================
# Tool / Resource Capture Utilities
# =============================================================================


def _create_tool_capture():
    """Create a mock MCP and a dict to capture registered tool functions.

    Returns:
        (mcp_mock, captured_tools_dict)
    """
    mcp = MagicMock()
    tools: dict[str, Any] = {}

    def capture_tool(**kwargs):
        def decorator(func):
            tools[func.__name__] = func
            return func
        return decorator

    mcp.tool = capture_tool
    return mcp, tools


def _create_resource_capture():
    """Create a mock MCP and a dict to capture registered resource functions.

    Returns:
        (mcp_mock, captured_resources_dict)
    """
    mcp = MagicMock()
    resources: dict[str, Any] = {}

    def capture_resource(uri, **kwargs):
        def decorator(func):
            resources[uri] = func
            return func
        return decorator

    mcp.resource = capture_resource
    return mcp, resources


# =============================================================================
# Helper Functions
# =============================================================================


def parse_json_result(result: str) -> dict[str, Any]:
    """Parse a JSON string result from an MCP tool."""
    return json.loads(result)


def assert_success(result: str) -> dict[str, Any]:
    """Assert that a tool result indicates success and return parsed JSON."""
    data = parse_json_result(result)
    assert data.get("status") != "error", f"Tool returned error: {data.get('message')}"
    return data


def assert_error(result: str, expected_message: str | None = None) -> dict[str, Any]:
    """Assert that a tool result indicates an error."""
    data = parse_json_result(result)
    assert "error" in str(data.get("status", "")).lower() or "error" in str(data.get("message", "")).lower() or data.get("status") == "error", \
        f"Expected error but got: {data}"
    if expected_message:
        assert expected_message.lower() in str(data.get("message", "")).lower(), \
            f"Expected '{expected_message}' in error message: {data.get('message')}"
    return data


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_ml():
    """Create a comprehensive mock DerivaML instance."""
    ml = MagicMock()
    ml.host_name = "test.example.org"
    ml.catalog_id = "1"
    ml.domain_schemas = {"test_schema"}
    ml.default_schema = "test_schema"
    ml.domain_schema = "test_schema"
    ml.ml_schema = "deriva-ml"
    ml.project_name = "test_project"
    ml.credential = None
    return ml


@pytest.fixture
def mock_conn_manager(mock_ml):
    """Create a ConnectionManager that returns mock_ml as active connection."""
    from deriva_ml import DerivaMLException

    conn_manager = MagicMock(spec=ConnectionManager)
    conn_manager.get_active.return_value = mock_ml
    conn_manager.get_active_or_raise.return_value = mock_ml

    mock_conn_info = MagicMock()
    mock_conn_info.execution = MagicMock()
    mock_conn_info.execution.execution_rid = "EXE-TEST"
    mock_conn_info.workflow_rid = "WF-TEST"
    mock_conn_info.user_id = "test_user"
    mock_conn_info.active_tool_execution = None
    conn_manager.get_active_connection_info.return_value = mock_conn_info
    conn_manager.get_active_connection_info_or_raise.return_value = mock_conn_info
    conn_manager.get_active_execution.return_value = mock_conn_info.execution
    conn_manager.get_active_execution_or_raise.return_value = mock_conn_info.execution

    return conn_manager


@pytest.fixture
def disconnected_conn_manager():
    """Create a ConnectionManager with no active connection."""
    from deriva_ml import DerivaMLException

    conn_manager = MagicMock(spec=ConnectionManager)
    conn_manager.get_active.return_value = None
    conn_manager.get_active_or_raise.side_effect = DerivaMLException(
        "No active catalog connection. Use 'connect' tool to connect to a catalog first."
    )
    conn_manager.get_active_connection_info.return_value = None
    conn_manager.get_active_connection_info_or_raise.side_effect = DerivaMLException(
        "No active catalog connection."
    )
    conn_manager.get_active_execution.return_value = None
    conn_manager.get_active_execution_or_raise.side_effect = DerivaMLException(
        "No active execution context."
    )
    return conn_manager


# =============================================================================
# Per-Module Tool Fixtures
# =============================================================================


@pytest.fixture
def vocab_tools(mock_conn_manager):
    """Capture vocabulary tools with a connected mock."""
    from deriva_ml_mcp.tools.vocabulary import register_vocabulary_tools
    mcp, tools = _create_tool_capture()
    register_vocabulary_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def vocab_tools_disconnected(disconnected_conn_manager):
    """Capture vocabulary tools with no connection."""
    from deriva_ml_mcp.tools.vocabulary import register_vocabulary_tools
    mcp, tools = _create_tool_capture()
    register_vocabulary_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def data_tools(mock_conn_manager):
    """Capture data tools with a connected mock."""
    from deriva_ml_mcp.tools.data import register_data_tools
    mcp, tools = _create_tool_capture()
    register_data_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def data_tools_disconnected(disconnected_conn_manager):
    """Capture data tools with no connection."""
    from deriva_ml_mcp.tools.data import register_data_tools
    mcp, tools = _create_tool_capture()
    register_data_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def workflow_tools(mock_conn_manager):
    """Capture workflow tools with a connected mock."""
    from deriva_ml_mcp.tools.workflow import register_workflow_tools
    mcp, tools = _create_tool_capture()
    register_workflow_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def workflow_tools_disconnected(disconnected_conn_manager):
    """Capture workflow tools with no connection."""
    from deriva_ml_mcp.tools.workflow import register_workflow_tools
    mcp, tools = _create_tool_capture()
    register_workflow_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def feature_tools(mock_conn_manager):
    """Capture feature tools with a connected mock."""
    from deriva_ml_mcp.tools.feature import register_feature_tools
    mcp, tools = _create_tool_capture()
    register_feature_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def feature_tools_disconnected(disconnected_conn_manager):
    """Capture feature tools with no connection."""
    from deriva_ml_mcp.tools.feature import register_feature_tools
    mcp, tools = _create_tool_capture()
    register_feature_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def catalog_tools(mock_conn_manager):
    """Capture catalog tools with a connected mock."""
    from deriva_ml_mcp.tools.catalog import register_catalog_tools
    mcp, tools = _create_tool_capture()
    register_catalog_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def catalog_tools_disconnected(disconnected_conn_manager):
    """Capture catalog tools with no connection."""
    from deriva_ml_mcp.tools.catalog import register_catalog_tools
    mcp, tools = _create_tool_capture()
    register_catalog_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def dataset_tools(mock_conn_manager):
    """Capture dataset tools with a connected mock."""
    from deriva_ml_mcp.tools.dataset import register_dataset_tools
    mcp, tools = _create_tool_capture()
    register_dataset_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def dataset_tools_disconnected(disconnected_conn_manager):
    """Capture dataset tools with no connection."""
    from deriva_ml_mcp.tools.dataset import register_dataset_tools
    mcp, tools = _create_tool_capture()
    register_dataset_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def schema_tools(mock_conn_manager):
    """Capture schema tools with a connected mock."""
    from deriva_ml_mcp.tools.schema import register_schema_tools
    mcp, tools = _create_tool_capture()
    register_schema_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def schema_tools_disconnected(disconnected_conn_manager):
    """Capture schema tools with no connection."""
    from deriva_ml_mcp.tools.schema import register_schema_tools
    mcp, tools = _create_tool_capture()
    register_schema_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def execution_tools(mock_conn_manager):
    """Capture execution tools with a connected mock."""
    from deriva_ml_mcp.tools.execution import register_execution_tools
    mcp, tools = _create_tool_capture()
    register_execution_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def execution_tools_disconnected(disconnected_conn_manager):
    """Capture execution tools with no connection."""
    from deriva_ml_mcp.tools.execution import register_execution_tools
    mcp, tools = _create_tool_capture()
    register_execution_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def storage_tools(mock_conn_manager):
    """Capture storage tools with a connected mock."""
    from deriva_ml_mcp.tools.execution import register_storage_tools
    mcp, tools = _create_tool_capture()
    register_storage_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def annotation_tools(mock_conn_manager):
    """Capture annotation tools with a connected mock."""
    from deriva_ml_mcp.tools.annotation import register_annotation_tools
    mcp, tools = _create_tool_capture()
    register_annotation_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def annotation_tools_disconnected(disconnected_conn_manager):
    """Capture annotation tools with no connection."""
    from deriva_ml_mcp.tools.annotation import register_annotation_tools
    mcp, tools = _create_tool_capture()
    register_annotation_tools(mcp, disconnected_conn_manager)
    return tools


@pytest.fixture
def bg_task_tools(mock_conn_manager):
    """Capture background task tools with a connected mock."""
    from deriva_ml_mcp.tools.background_tasks import register_background_task_tools
    mcp, tools = _create_tool_capture()
    register_background_task_tools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def devtools(mock_conn_manager):
    """Capture devtools with a connected mock."""
    from deriva_ml_mcp.tools.devtools import register_devtools
    mcp, tools = _create_tool_capture()
    register_devtools(mcp, mock_conn_manager)
    return tools


@pytest.fixture
def devtools_disconnected(disconnected_conn_manager):
    """Capture devtools with no connection."""
    from deriva_ml_mcp.tools.devtools import register_devtools
    mcp, tools = _create_tool_capture()
    register_devtools(mcp, disconnected_conn_manager)
    return tools


# =============================================================================
# Resource Fixtures
# =============================================================================


@pytest.fixture
def captured_resources(mock_conn_manager):
    """Capture all resources with a connected mock."""
    from deriva_ml_mcp.resources import register_resources
    mcp, resources = _create_resource_capture()
    register_resources(mcp, mock_conn_manager)
    return resources


@pytest.fixture
def captured_resources_disconnected(disconnected_conn_manager):
    """Capture all resources with no connection."""
    from deriva_ml_mcp.resources import register_resources
    mcp, resources = _create_resource_capture()
    register_resources(mcp, disconnected_conn_manager)
    return resources


# =============================================================================
# Integration Test Fixtures
# =============================================================================


class CatalogState:
    """Enum-like class for catalog population states."""
    EMPTY = 0
    POPULATED = 1
    WITH_FEATURES = 2
    WITH_DATASETS = 3


class CatalogManager:
    """Manages a test catalog lifecycle efficiently.

    Creates the catalog once per session and resets data between tests.
    """

    def __init__(self, hostname: str):
        self.hostname = hostname
        self.domain_schema = "test-schema"
        self.project_name = "ml-test"
        self.catalog = None
        self.catalog_id = None
        self.state = CatalogState.EMPTY
        self._dataset_description: DatasetDescription | None = None
        self._create_catalog()

    def _create_catalog(self) -> None:
        """Create the ML catalog and domain schema."""
        from deriva_ml.demo_catalog import create_domain_schema
        from deriva_ml.schema import create_ml_catalog

        self.catalog = create_ml_catalog(self.hostname, project_name=self.project_name)
        self.catalog_id = self.catalog.catalog_id
        create_domain_schema(self.catalog, self.domain_schema)
        self.state = CatalogState.EMPTY

    def destroy(self) -> None:
        """Destroy the catalog and clean up resources."""
        if self.catalog:
            self.catalog.delete_ermrest_catalog(really=True)
            self.catalog = None
            self.catalog_id = None

    def reset(self) -> None:
        """Reset catalog to empty state (schema only, no data)."""
        if self.state == CatalogState.EMPTY:
            return

        pb = self.catalog.getPathBuilder()
        ml_path = pb.schemas["deriva-ml"]
        domain_path = pb.schemas[self.domain_schema]

        ml_tables = [
            "Dataset_Execution", "Dataset_Version", "Dataset_Dataset",
            "Dataset", "Workflow_Execution", "Execution", "Workflow",
        ]
        for t in ml_tables:
            self._delete_table_data(ml_path, t)

        domain_assoc_tables = [
            "Dataset_Subject", "Dataset_Image", "Image_Subject",
        ]
        for t in domain_assoc_tables:
            self._delete_table_data(domain_path, t)

        feature_tables = [
            "Execution_Image_BoundingBox", "Execution_Image_Quality",
            "Execution_Subject_Health",
        ]
        for t in feature_tables:
            self._delete_table_data(domain_path, t)

        for t in ["Image", "Subject"]:
            self._delete_table_data(domain_path, t)

        for t in ["SubjectHealth", "ImageQuality"]:
            self._delete_table_data(domain_path, t)

        self.state = CatalogState.EMPTY
        self._dataset_description = None

    def _delete_table_data(self, schema_path, table_name: str) -> None:
        """Delete all data from a table, ignoring missing tables."""
        try:
            schema_path.tables[table_name].path.delete()
        except Exception:
            pass

    def get_ml_instance(self, working_dir: Path | str) -> DerivaML:
        from deriva_ml import DerivaML
        return DerivaML(
            self.hostname,
            self.catalog_id,
            domain_schema=self.domain_schema,
            working_dir=working_dir,
            use_minid=False,
        )

    def ensure_populated(self, working_dir: Path | str) -> DerivaML:
        """Ensure catalog has basic data (subjects, images)."""
        from deriva_ml.demo_catalog import populate_demo_catalog
        from deriva_ml.execution import ExecutionConfiguration
        from deriva_ml.core.definitions import MLVocab

        ml = self.get_ml_instance(working_dir)
        if self.state >= CatalogState.POPULATED:
            return ml

        self._add_workflow_type(ml)
        workflow = ml.create_workflow(name="Test Population", workflow_type="Test Workflow")
        execution = ml.create_execution(workflow=workflow, configuration=ExecutionConfiguration())
        with execution.execute() as exe:
            populate_demo_catalog(exe)

        self.state = CatalogState.POPULATED
        return ml

    def ensure_features(self, working_dir: Path | str) -> DerivaML:
        """Ensure catalog has features defined."""
        from deriva_ml.demo_catalog import create_demo_features
        from deriva_ml.execution import ExecutionConfiguration

        ml = self.ensure_populated(working_dir)
        if self.state >= CatalogState.WITH_FEATURES:
            return ml

        workflow = ml.create_workflow(name="Feature Creation", workflow_type="Test Workflow")
        execution = ml.create_execution(workflow=workflow, configuration=ExecutionConfiguration())
        with execution.execute() as exe:
            create_demo_features(exe)

        self.state = CatalogState.WITH_FEATURES
        return ml

    def ensure_datasets(self, working_dir: Path | str) -> tuple[DerivaML, DatasetDescription]:
        """Ensure catalog has dataset hierarchy."""
        from deriva_ml.demo_catalog import create_demo_datasets
        from deriva_ml.execution import ExecutionConfiguration

        ml = self.ensure_features(working_dir)
        if self.state == CatalogState.WITH_DATASETS and self._dataset_description:
            return ml, self._dataset_description

        workflow = ml.create_workflow(name="Dataset Creation", workflow_type="Test Workflow")
        execution = ml.create_execution(workflow=workflow, configuration=ExecutionConfiguration())
        with execution.execute() as exe:
            self._dataset_description = create_demo_datasets(exe)

        self.state = CatalogState.WITH_DATASETS
        return ml, self._dataset_description

    def _add_workflow_type(self, ml: DerivaML) -> None:
        from deriva_ml.core.definitions import MLVocab
        try:
            ml.lookup_term(MLVocab.workflow_type, "Test Workflow")
        except Exception:
            ml.add_term(
                MLVocab.workflow_type,
                "Test Workflow",
                description="Workflow type for testing",
            )


@pytest.fixture(scope="session")
def catalog_host() -> str:
    """Get the test host from environment or use default."""
    return os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture(scope="session")
def catalog_manager(catalog_host: str) -> CatalogManager:
    """Create a session-scoped catalog manager (integration tests only)."""
    manager = CatalogManager(catalog_host)
    yield manager
    manager.destroy()


@pytest.fixture(scope="function")
def test_ml(catalog_manager: CatalogManager, tmp_path: Path) -> DerivaML:
    """Create a clean DerivaML instance for testing."""
    catalog_manager.reset()
    ml = catalog_manager.get_ml_instance(tmp_path)
    yield ml


@pytest.fixture(scope="function")
def populated_catalog(catalog_manager: CatalogManager, tmp_path: Path) -> DerivaML:
    """Create a DerivaML instance with populated data."""
    catalog_manager.reset()
    ml = catalog_manager.ensure_populated(tmp_path)
    yield ml


@pytest.fixture(scope="function")
def mcp_connection_manager(catalog_manager: CatalogManager) -> ConnectionManager:
    """Create a ConnectionManager pre-connected to the test catalog."""
    catalog_manager.reset()
    conn_manager = ConnectionManager()
    conn_manager.connect(
        hostname=catalog_manager.hostname,
        catalog_id=str(catalog_manager.catalog_id),
        domain_schema=catalog_manager.domain_schema,
    )
    yield conn_manager
    conn_manager.disconnect()
