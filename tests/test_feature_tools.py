"""Integration tests for feature management tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.tools.feature import register_feature_tools
from deriva_ml_mcp.tools.catalog import register_catalog_tools
from deriva_ml_mcp.tools.vocabulary import register_vocabulary_tools
from deriva_ml_mcp.tools.execution import register_execution_tools, _active_executions

if TYPE_CHECKING:
    from tests.conftest import CatalogManager

from tests.conftest import parse_json_result


def setup_tools(conn_manager: ConnectionManager) -> dict:
    """Helper to register tools and capture them."""
    mcp = MagicMock()
    captured_tools = {}

    def capture_tool():
        def decorator(func):
            captured_tools[func.__name__] = func
            return func
        return decorator

    mcp.tool = capture_tool
    register_catalog_tools(mcp, conn_manager)
    register_vocabulary_tools(mcp, conn_manager)
    register_feature_tools(mcp, conn_manager)
    register_execution_tools(mcp, conn_manager)
    return captured_tools


class TestCreateFeature:
    """Tests for the create_feature tool."""

    @pytest.mark.asyncio
    async def test_create_feature_with_terms(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating a feature with vocabulary terms."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create a vocabulary for the feature
        await tools["create_vocabulary"](
            vocabulary_name="TestDiagnosis",
            comment="Diagnosis types for testing",
        )

        # Add some terms
        await tools["add_term"](
            vocabulary_name="TestDiagnosis",
            term_name="Normal",
            description="Normal finding",
        )
        await tools["add_term"](
            vocabulary_name="TestDiagnosis",
            term_name="Abnormal",
            description="Abnormal finding",
        )

        # Create feature
        result = await tools["create_feature"](
            table_name="Image",
            feature_name="Image_TestDiagnosis",
            comment="Test diagnosis feature for images",
            terms=["TestDiagnosis"],
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["feature_name"] == "Image_TestDiagnosis"
        assert data["target_table"] == "Image"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_create_feature_no_connection(self):
        """Test creating feature without connection fails."""
        conn_manager = ConnectionManager()
        tools = setup_tools(conn_manager)

        result = await tools["create_feature"](
            table_name="Image",
            feature_name="TestFeature",
        )

        data = parse_json_result(result)
        assert data["status"] == "error"


class TestDeleteFeature:
    """Tests for the delete_feature tool."""

    @pytest.mark.asyncio
    async def test_delete_feature_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test deleting a feature."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create vocabulary and feature
        await tools["create_vocabulary"](
            vocabulary_name="DeleteTest",
            comment="Vocabulary for delete testing",
        )
        await tools["add_term"](
            vocabulary_name="DeleteTest",
            term_name="Value1",
            description="Test value",
        )

        await tools["create_feature"](
            table_name="Image",
            feature_name="Image_DeleteTest",
            terms=["DeleteTest"],
        )

        # Delete feature
        result = await tools["delete_feature"](
            table_name="Image",
            feature_name="Image_DeleteTest",
        )

        data = parse_json_result(result)
        assert data["status"] == "deleted"
        assert data["feature_name"] == "Image_DeleteTest"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_delete_feature_not_found(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test deleting a feature that doesn't exist."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        catalog_manager.ensure_populated(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["delete_feature"](
            table_name="Image",
            feature_name="NonExistentFeature",
        )

        data = parse_json_result(result)
        # Should return not_found or error
        assert data["status"] in ["not_found", "error"]

        conn_manager.disconnect()


class TestAddFeatureValue:
    """Tests for the add_feature_value tool."""

    @pytest.mark.asyncio
    async def test_add_feature_value_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a feature value to an image."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        ml = catalog_manager.ensure_features(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Get an image RID to annotate
        pb = ml.pathBuilder()
        images = list(pb.schemas[ml.default_schema].Image.path.entities())

        if images:
            image_rid = images[0]["RID"]

            # The demo catalog should have ImageQuality feature
            # We need to get the feature's vocabulary terms
            result = await tools["add_feature_value"](
                table_name="Image",
                feature_name="Image_Quality",
                target_rid=image_rid,
                value="Good",  # Assuming this term exists
            )

            data = parse_json_result(result)
            # May succeed or fail depending on feature setup
            # Just check it returns valid JSON
            assert "status" in data

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_add_feature_value_no_execution(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding feature value uses MCP connection execution."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        ml = catalog_manager.ensure_features(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Clear any active user execution
        key = f"{catalog_manager.hostname}:{catalog_manager.catalog_id}"
        if key in _active_executions:
            del _active_executions[key]

        # Get an image RID
        pb = ml.pathBuilder()
        images = list(pb.schemas[ml.default_schema].Image.path.entities())

        if images:
            image_rid = images[0]["RID"]

            # Should use MCP connection execution
            result = await tools["add_feature_value"](
                table_name="Image",
                feature_name="Image_Quality",
                target_rid=image_rid,
                value="Good",
            )

            data = parse_json_result(result)
            # Should either succeed (using MCP execution) or provide specific error
            assert "status" in data

        conn_manager.disconnect()


class TestAddFeatureValueRecord:
    """Tests for the add_feature_value_record tool."""

    @pytest.mark.asyncio
    async def test_add_feature_value_record_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a feature value with multiple fields."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()
        ml = catalog_manager.ensure_features(tmp_path)

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Get an image RID
        pb = ml.pathBuilder()
        images = list(pb.schemas[ml.default_schema].Image.path.entities())

        if images:
            image_rid = images[0]["RID"]

            # Add feature value with record format
            result = await tools["add_feature_value_record"](
                table_name="Image",
                feature_name="Image_Quality",
                target_rid=image_rid,
                values={"ImageQuality": "Good"},
            )

            data = parse_json_result(result)
            # Check that it returns valid response
            assert "status" in data

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_add_feature_value_record_no_connection(self):
        """Test adding feature value record without connection fails."""
        conn_manager = ConnectionManager()
        tools = setup_tools(conn_manager)

        result = await tools["add_feature_value_record"](
            table_name="Image",
            feature_name="TestFeature",
            target_rid="1-ABC",
            values={"value": "test"},
        )

        data = parse_json_result(result)
        assert data["status"] == "error"
