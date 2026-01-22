"""Integration tests for vocabulary management tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from deriva_ml_mcp.connection import ConnectionManager
from deriva_ml_mcp.tools.vocabulary import register_vocabulary_tools
from deriva_ml_mcp.tools.catalog import register_catalog_tools

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
    return captured_tools


class TestCreateVocabulary:
    """Tests for the create_vocabulary tool."""

    @pytest.mark.asyncio
    async def test_create_vocabulary_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test creating a new vocabulary table."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["create_vocabulary"](
            vocabulary_name="TestQuality",
            comment="Quality levels for testing",
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["name"] == "TestQuality"
        assert data["comment"] == "Quality levels for testing"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_create_vocabulary_no_connection(self):
        """Test creating vocabulary without connection fails."""
        conn_manager = ConnectionManager()
        tools = setup_tools(conn_manager)

        result = await tools["create_vocabulary"](
            vocabulary_name="TestVocab",
        )

        data = parse_json_result(result)
        assert data["status"] == "error"


class TestAddTerm:
    """Tests for the add_term tool."""

    @pytest.mark.asyncio
    async def test_add_term_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a term to a vocabulary."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create vocabulary first
        await tools["create_vocabulary"](
            vocabulary_name="TestColors",
            comment="Color vocabulary for testing",
        )

        # Add term
        result = await tools["add_term"](
            vocabulary_name="TestColors",
            term_name="Red",
            description="The color red",
            synonyms=["crimson", "scarlet"],
        )

        data = parse_json_result(result)
        assert data["status"] == "created"
        assert data["name"] == "Red"
        assert data["description"] == "The color red"
        assert "rid" in data

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_add_term_exists(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a term that already exists."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create vocabulary
        await tools["create_vocabulary"](
            vocabulary_name="TestSizes",
            comment="Size vocabulary",
        )

        # Add term
        await tools["add_term"](
            vocabulary_name="TestSizes",
            term_name="Large",
            description="Large size",
        )

        # Try to add same term again
        result = await tools["add_term"](
            vocabulary_name="TestSizes",
            term_name="Large",
            description="Large size again",
        )

        data = parse_json_result(result)
        assert data["status"] == "exists"
        assert data["name"] == "Large"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_add_term_to_builtin_vocab(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a term to a built-in vocabulary like Dataset_Type."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        result = await tools["add_term"](
            vocabulary_name="Dataset_Type",
            term_name="CustomType",
            description="A custom dataset type",
        )

        data = parse_json_result(result)
        assert data["status"] in ["created", "exists"]

        conn_manager.disconnect()


class TestAddSynonym:
    """Tests for the add_synonym tool."""

    @pytest.mark.asyncio
    async def test_add_synonym_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test adding a synonym to a term."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create vocabulary and term
        await tools["create_vocabulary"](
            vocabulary_name="TestAnimals",
            comment="Animal vocabulary",
        )
        await tools["add_term"](
            vocabulary_name="TestAnimals",
            term_name="Dog",
            description="A domestic canine",
        )

        # Add synonym
        result = await tools["add_synonym"](
            vocabulary_name="TestAnimals",
            term_name="Dog",
            synonym="canine",
        )

        data = parse_json_result(result)
        assert data["status"] == "added"
        assert "canine" in data["synonyms"]

        conn_manager.disconnect()


class TestRemoveSynonym:
    """Tests for the remove_synonym tool."""

    @pytest.mark.asyncio
    async def test_remove_synonym_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test removing a synonym from a term."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create vocabulary and term with synonyms
        await tools["create_vocabulary"](
            vocabulary_name="TestPlants",
            comment="Plant vocabulary",
        )
        await tools["add_term"](
            vocabulary_name="TestPlants",
            term_name="Tree",
            description="A woody plant",
            synonyms=["arbor", "timber"],
        )

        # Remove synonym
        result = await tools["remove_synonym"](
            vocabulary_name="TestPlants",
            term_name="Tree",
            synonym="arbor",
        )

        data = parse_json_result(result)
        assert data["status"] == "removed"
        assert "arbor" not in data["synonyms"]

        conn_manager.disconnect()


class TestUpdateTermDescription:
    """Tests for the update_term_description tool."""

    @pytest.mark.asyncio
    async def test_update_description_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test updating a term's description."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create vocabulary and term
        await tools["create_vocabulary"](
            vocabulary_name="TestShapes",
            comment="Shape vocabulary",
        )
        await tools["add_term"](
            vocabulary_name="TestShapes",
            term_name="Circle",
            description="A round shape",
        )

        # Update description
        new_description = "A perfectly round shape with no corners"
        result = await tools["update_term_description"](
            vocabulary_name="TestShapes",
            term_name="Circle",
            description=new_description,
        )

        data = parse_json_result(result)
        assert data["status"] == "updated"
        assert data["description"] == new_description

        conn_manager.disconnect()


class TestDeleteTerm:
    """Tests for the delete_term tool."""

    @pytest.mark.asyncio
    async def test_delete_term_success(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test deleting a term from a vocabulary."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create vocabulary and term
        await tools["create_vocabulary"](
            vocabulary_name="TestFruits",
            comment="Fruit vocabulary",
        )
        await tools["add_term"](
            vocabulary_name="TestFruits",
            term_name="Apple",
            description="A popular fruit",
        )

        # Delete term
        result = await tools["delete_term"](
            vocabulary_name="TestFruits",
            term_name="Apple",
        )

        data = parse_json_result(result)
        assert data["status"] == "deleted"
        assert data["name"] == "Apple"

        conn_manager.disconnect()

    @pytest.mark.asyncio
    async def test_delete_term_nonexistent(
        self,
        catalog_manager: "CatalogManager",
        tmp_path,
    ):
        """Test deleting a term that doesn't exist."""
        conn_manager = ConnectionManager()
        catalog_manager.reset()

        tools = setup_tools(conn_manager)

        await tools["connect_catalog"](
            hostname=catalog_manager.hostname,
            catalog_id=str(catalog_manager.catalog_id),
        )

        # Create vocabulary without any terms
        await tools["create_vocabulary"](
            vocabulary_name="TestEmpty",
            comment="Empty vocabulary",
        )

        # Try to delete non-existent term
        result = await tools["delete_term"](
            vocabulary_name="TestEmpty",
            term_name="NonExistent",
        )

        data = parse_json_result(result)
        assert data["status"] == "error"

        conn_manager.disconnect()
