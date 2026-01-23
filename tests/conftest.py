"""Pytest configuration and shared fixtures for deriva-ml-mcp tests.

This module provides fixtures for integration testing of the MCP server tools.
It reuses the CatalogManager pattern from deriva-ml for efficient test catalog management.

Session-Scoped (created once per test session):
    - catalog_host: Test server hostname
    - catalog_manager: CatalogManager instance that owns the test catalog
    - mcp_server: FastMCP server instance with all tools registered

Function-Scoped (reset per test):
    - test_ml: Clean DerivaML instance with empty catalog
    - connection_manager: Fresh ConnectionManager for each test
    - populated_catalog: DerivaML instance with subjects/images/features/datasets

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

from deriva_ml import DerivaML
from deriva_ml.demo_catalog import (
    DatasetDescription,
    create_demo_datasets,
    create_demo_features,
    create_domain_schema,
    populate_demo_catalog,
)
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.schema import create_ml_catalog
from deriva_ml.core.definitions import MLVocab

from deriva_ml_mcp.connection import ConnectionManager

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


# =============================================================================
# Catalog State Management (adapted from deriva-ml)
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
        self.default_schema = "test-schema"  # The default schema for table creation
        self.domain_schemas = ["test-schema"]  # List of all domain schemas
        self.project_name = "ml-test"
        self.catalog = None
        self.catalog_id = None
        self.state = CatalogState.EMPTY
        self._dataset_description: DatasetDescription | None = None
        self._create_catalog()

    def _create_catalog(self) -> None:
        """Create the ML catalog and domain schema."""
        print(f"\n🚀 Creating test catalog on {self.hostname}")
        self.catalog = create_ml_catalog(self.hostname, project_name=self.project_name)
        self.catalog_id = self.catalog.catalog_id
        create_domain_schema(self.catalog, self.default_schema)
        self.state = CatalogState.EMPTY
        print(f"   Created catalog {self.catalog_id}")

    def destroy(self) -> None:
        """Destroy the catalog and clean up resources."""
        if self.catalog:
            print(f"\n🗑️ Destroying test catalog {self.catalog_id}")
            self.catalog.delete_ermrest_catalog(really=True)
            self.catalog = None
            self.catalog_id = None

    def reset(self) -> None:
        """Reset catalog to empty state (schema only, no data)."""
        if self.state == CatalogState.EMPTY:
            return

        print("   Resetting catalog to empty state...")
        pb = self.catalog.getPathBuilder()
        ml_path = pb.schemas["deriva-ml"]
        domain_path = pb.schemas[self.default_schema]

        # Clear ML schema tables in dependency order
        ml_tables = [
            "Dataset_Execution",
            "Dataset_Version",
            "Dataset_Dataset",
            "Dataset",
            "Workflow_Execution",
            "Execution",
            "Workflow",
        ]
        for t in ml_tables:
            self._delete_table_data(ml_path, t)

        # Clear domain schema association tables
        domain_assoc_tables = [
            "Dataset_Subject",
            "Dataset_Image",
            "Image_Subject",
        ]
        for t in domain_assoc_tables:
            self._delete_table_data(domain_path, t)

        # Clear feature execution tables
        feature_tables = [
            "Execution_Image_BoundingBox",
            "Execution_Image_Quality",
            "Execution_Subject_Health",
        ]
        for t in feature_tables:
            self._delete_table_data(domain_path, t)

        # Clear data tables (Image before Subject due to FK)
        for t in ["Image", "Subject"]:
            self._delete_table_data(domain_path, t)

        # Clear custom vocabularies
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
        """Get a DerivaML instance for this catalog."""
        return DerivaML(
            self.hostname,
            self.catalog_id,
            default_schema=self.default_schema,
            working_dir=working_dir,
            use_minid=False,
        )

    def ensure_populated(self, working_dir: Path | str) -> DerivaML:
        """Ensure catalog has basic data (subjects, images)."""
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
        """Add the test workflow type if not already present."""
        try:
            ml.lookup_term(MLVocab.workflow_type, "Test Workflow")
        except Exception:
            ml.add_term(
                MLVocab.workflow_type,
                "Test Workflow",
                description="Workflow type for testing",
            )


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def catalog_host() -> str:
    """Get the test host from the environment or use default."""
    return os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture(scope="session")
def catalog_manager(catalog_host: str) -> CatalogManager:
    """Create a session-scoped catalog manager."""
    manager = CatalogManager(catalog_host)
    yield manager
    manager.destroy()


# =============================================================================
# Function-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def test_ml(catalog_manager: CatalogManager, tmp_path: Path) -> DerivaML:
    """Create a clean DerivaML instance for testing."""
    catalog_manager.reset()
    ml = catalog_manager.get_ml_instance(tmp_path)
    yield ml
    catalog_manager.state = CatalogState.POPULATED


@pytest.fixture(scope="function")
def connection_manager() -> ConnectionManager:
    """Create a fresh ConnectionManager for each test."""
    return ConnectionManager()


@pytest.fixture(scope="function")
def populated_catalog(catalog_manager: CatalogManager, tmp_path: Path) -> DerivaML:
    """Create a DerivaML instance with populated data."""
    catalog_manager.reset()
    ml = catalog_manager.ensure_populated(tmp_path)
    yield ml


@pytest.fixture(scope="function")
def catalog_with_features(catalog_manager: CatalogManager, tmp_path: Path) -> DerivaML:
    """Create a DerivaML instance with features defined."""
    catalog_manager.reset()
    ml = catalog_manager.ensure_features(tmp_path)
    yield ml


@pytest.fixture(scope="function")
def catalog_with_datasets(
    catalog_manager: CatalogManager, tmp_path: Path
) -> tuple[DerivaML, DatasetDescription]:
    """Create a DerivaML instance with full dataset hierarchy."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path)
    yield ml, dataset_desc


# =============================================================================
# MCP Server Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def mcp_connection_manager(
    catalog_manager: CatalogManager, tmp_path: Path
) -> ConnectionManager:
    """Create a ConnectionManager pre-connected to the test catalog."""
    catalog_manager.reset()
    conn_manager = ConnectionManager()
    # Connect to the test catalog
    conn_manager.connect(
        hostname=catalog_manager.hostname,
        catalog_id=str(catalog_manager.catalog_id),
        default_schema=catalog_manager.default_schema,
    )
    yield conn_manager
    # Disconnect after test
    conn_manager.disconnect()


@pytest.fixture(scope="function")
def populated_connection_manager(
    catalog_manager: CatalogManager, tmp_path: Path
) -> ConnectionManager:
    """Create a ConnectionManager connected to a populated catalog."""
    catalog_manager.reset()
    catalog_manager.ensure_populated(tmp_path)
    conn_manager = ConnectionManager()
    conn_manager.connect(
        hostname=catalog_manager.hostname,
        catalog_id=str(catalog_manager.catalog_id),
        default_schema=catalog_manager.default_schema,
    )
    yield conn_manager
    conn_manager.disconnect()


@pytest.fixture(scope="function")
def dataset_connection_manager(
    catalog_manager: CatalogManager, tmp_path: Path
) -> tuple[ConnectionManager, DatasetDescription]:
    """Create a ConnectionManager connected to a catalog with datasets."""
    catalog_manager.reset()
    _, dataset_desc = catalog_manager.ensure_datasets(tmp_path)
    conn_manager = ConnectionManager()
    conn_manager.connect(
        hostname=catalog_manager.hostname,
        catalog_id=str(catalog_manager.catalog_id),
        default_schema=catalog_manager.default_schema,
    )
    yield conn_manager, dataset_desc
    conn_manager.disconnect()


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
    assert data.get("status") == "error", f"Expected error but got: {data}"
    if expected_message:
        assert expected_message in data.get("message", ""), \
            f"Expected '{expected_message}' in error message: {data.get('message')}"
    return data
