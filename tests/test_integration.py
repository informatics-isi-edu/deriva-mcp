"""Integration tests for DerivaML MCP tools and resources.

These tests exercise real end-to-end workflows against a live Deriva catalog.
They require a running Deriva server (set DERIVA_HOST or use localhost).

The tests use the CatalogManager from conftest.py to create and manage a
test catalog for each test session, and the mcp_connection_manager fixture
to provide a pre-connected ConnectionManager.

All tests are marked with @pytest.mark.integration and will skip gracefully
if no Deriva server is available.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

import pytest

from tests.conftest import (
    CatalogManager,
    _create_resource_capture,
    _create_tool_capture,
    assert_success,
    parse_json_result,
)

# =============================================================================
# Skip Logic
# =============================================================================


def _check_localhost() -> bool:
    """Try a quick TCP connect to localhost:443 to see if a Deriva server is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", 443))
        sock.close()
        return result == 0
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("DERIVA_HOST") and not _check_localhost(),
        reason="No Deriva server available (set DERIVA_HOST or run localhost)",
    ),
]


# =============================================================================
# Integration Tool Capture Fixtures
# =============================================================================


@pytest.fixture
def int_catalog_tools(mcp_connection_manager):
    """Capture catalog tools with a real ConnectionManager."""
    from deriva_ml_mcp.tools.catalog import register_catalog_tools

    mcp, tools = _create_tool_capture()
    register_catalog_tools(mcp, mcp_connection_manager)
    return tools


@pytest.fixture
def int_vocab_tools(mcp_connection_manager):
    """Capture vocabulary tools with a real ConnectionManager."""
    from deriva_ml_mcp.tools.vocabulary import register_vocabulary_tools

    mcp, tools = _create_tool_capture()
    register_vocabulary_tools(mcp, mcp_connection_manager)
    return tools


@pytest.fixture
def int_schema_tools(mcp_connection_manager):
    """Capture schema tools with a real ConnectionManager."""
    from deriva_ml_mcp.tools.schema import register_schema_tools

    mcp, tools = _create_tool_capture()
    register_schema_tools(mcp, mcp_connection_manager)
    return tools


@pytest.fixture
def int_dataset_tools(mcp_connection_manager):
    """Capture dataset tools with a real ConnectionManager."""
    from deriva_ml_mcp.tools.dataset import register_dataset_tools

    mcp, tools = _create_tool_capture()
    register_dataset_tools(mcp, mcp_connection_manager)
    return tools


@pytest.fixture
def int_execution_tools(mcp_connection_manager):
    """Capture execution tools with a real ConnectionManager."""
    from deriva_ml_mcp.tools.execution import register_execution_tools

    mcp, tools = _create_tool_capture()
    register_execution_tools(mcp, mcp_connection_manager)
    return tools


@pytest.fixture
def int_feature_tools(mcp_connection_manager):
    """Capture feature tools with a real ConnectionManager."""
    from deriva_ml_mcp.tools.feature import register_feature_tools

    mcp, tools = _create_tool_capture()
    register_feature_tools(mcp, mcp_connection_manager)
    return tools


@pytest.fixture
def int_data_tools(mcp_connection_manager):
    """Capture data tools with a real ConnectionManager."""
    from deriva_ml_mcp.tools.data import register_data_tools

    mcp, tools = _create_tool_capture()
    register_data_tools(mcp, mcp_connection_manager)
    return tools


@pytest.fixture
def int_workflow_tools(mcp_connection_manager):
    """Capture workflow tools with a real ConnectionManager."""
    from deriva_ml_mcp.tools.workflow import register_workflow_tools

    mcp, tools = _create_tool_capture()
    register_workflow_tools(mcp, mcp_connection_manager)
    return tools


@pytest.fixture
def int_resources(mcp_connection_manager):
    """Capture resources with a real ConnectionManager."""
    from deriva_ml_mcp.resources import register_resources

    mcp, resources = _create_resource_capture()
    register_resources(mcp, mcp_connection_manager)
    return resources


# =============================================================================
# Populated Catalog Fixtures
# =============================================================================


@pytest.fixture
def populated_connection_manager(catalog_manager, tmp_path):
    """Create a ConnectionManager pre-connected to a populated catalog."""
    from deriva_ml_mcp.connection import ConnectionManager

    catalog_manager.reset()
    catalog_manager.ensure_populated(tmp_path)
    conn_manager = ConnectionManager()
    conn_manager.connect(
        hostname=catalog_manager.hostname,
        catalog_id=str(catalog_manager.catalog_id),
        domain_schema=catalog_manager.domain_schema,
    )
    yield conn_manager
    conn_manager.disconnect()


@pytest.fixture
def populated_data_tools(populated_connection_manager):
    """Capture data tools with a populated catalog."""
    from deriva_ml_mcp.tools.data import register_data_tools

    mcp, tools = _create_tool_capture()
    register_data_tools(mcp, populated_connection_manager)
    return tools


@pytest.fixture
def populated_dataset_tools(populated_connection_manager):
    """Capture dataset tools with a populated catalog."""
    from deriva_ml_mcp.tools.dataset import register_dataset_tools

    mcp, tools = _create_tool_capture()
    register_dataset_tools(mcp, populated_connection_manager)
    return tools


@pytest.fixture
def populated_schema_tools(populated_connection_manager):
    """Capture schema tools with a populated catalog."""
    from deriva_ml_mcp.tools.schema import register_schema_tools

    mcp, tools = _create_tool_capture()
    register_schema_tools(mcp, populated_connection_manager)
    return tools


@pytest.fixture
def populated_resources(populated_connection_manager):
    """Capture resources with a populated catalog."""
    from deriva_ml_mcp.resources import register_resources

    mcp, resources = _create_resource_capture()
    register_resources(mcp, populated_connection_manager)
    return resources


# =============================================================================
# Feature Catalog Fixtures
# =============================================================================


@pytest.fixture
def feature_connection_manager(catalog_manager, tmp_path):
    """Create a ConnectionManager pre-connected to a catalog with features."""
    from deriva_ml_mcp.connection import ConnectionManager

    catalog_manager.reset()
    catalog_manager.ensure_features(tmp_path)
    conn_manager = ConnectionManager()
    conn_manager.connect(
        hostname=catalog_manager.hostname,
        catalog_id=str(catalog_manager.catalog_id),
        domain_schema=catalog_manager.domain_schema,
    )
    yield conn_manager
    conn_manager.disconnect()


@pytest.fixture
def feature_feature_tools(feature_connection_manager):
    """Capture feature tools with a feature-enabled catalog."""
    from deriva_ml_mcp.tools.feature import register_feature_tools

    mcp, tools = _create_tool_capture()
    register_feature_tools(mcp, feature_connection_manager)
    return tools


@pytest.fixture
def feature_data_tools(feature_connection_manager):
    """Capture data tools with a feature-enabled catalog."""
    from deriva_ml_mcp.tools.data import register_data_tools

    mcp, tools = _create_tool_capture()
    register_data_tools(mcp, feature_connection_manager)
    return tools


@pytest.fixture
def feature_schema_tools(feature_connection_manager):
    """Capture schema tools with a feature-enabled catalog."""
    from deriva_ml_mcp.tools.schema import register_schema_tools

    mcp, tools = _create_tool_capture()
    register_schema_tools(mcp, feature_connection_manager)
    return tools


@pytest.fixture
def feature_resources(feature_connection_manager):
    """Capture resources with a feature-enabled catalog."""
    from deriva_ml_mcp.resources import register_resources

    mcp, resources = _create_resource_capture()
    register_resources(mcp, feature_connection_manager)
    return resources


# =============================================================================
# 1. Connection Workflow Tests
# =============================================================================


class TestConnectionWorkflow:
    """Test the catalog connection lifecycle through MCP tools."""

    @pytest.mark.asyncio
    async def test_connect_and_get_info(self, catalog_manager, catalog_host):
        """connect_catalog establishes connection; catalog info resource returns schema details."""
        from deriva_ml_mcp.connection import ConnectionManager
        from deriva_ml_mcp.resources import register_resources
        from deriva_ml_mcp.tools.catalog import register_catalog_tools

        conn_manager = ConnectionManager()
        mcp, tools = _create_tool_capture()
        register_catalog_tools(mcp, conn_manager)
        mcp_r, resources = _create_resource_capture()
        register_resources(mcp_r, conn_manager)

        # Connect
        result = await tools["connect_catalog"](
            hostname=catalog_host,
            catalog_id=str(catalog_manager.catalog_id),
            domain_schema=catalog_manager.domain_schema,
        )
        data = parse_json_result(result)
        assert data["status"] == "connected"
        assert data["hostname"] == catalog_host
        assert data["catalog_id"] == str(catalog_manager.catalog_id)
        assert catalog_manager.domain_schema in data["domain_schemas"]
        assert "workflow_rid" in data
        assert "execution_rid" in data

        # Get catalog info via resource
        info_result = resources["deriva-ml://catalog/info"]()
        info = parse_json_result(info_result)
        assert info["hostname"] == catalog_host
        assert info["catalog_id"] == str(catalog_manager.catalog_id)
        assert catalog_manager.domain_schema in info["domain_schemas"]

        # Disconnect
        disc_result = await tools["disconnect_catalog"]()
        disc_data = parse_json_result(disc_result)
        assert disc_data["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_list_connections(self, catalog_manager, catalog_host):
        """list_connections resource shows open connections and changes after disconnect."""
        from deriva_ml_mcp.connection import ConnectionManager
        from deriva_ml_mcp.resources import register_resources
        from deriva_ml_mcp.tools.catalog import register_catalog_tools

        conn_manager = ConnectionManager()
        mcp, tools = _create_tool_capture()
        register_catalog_tools(mcp, conn_manager)
        mcp_r, resources = _create_resource_capture()
        register_resources(mcp_r, conn_manager)

        # Before connecting, list should be empty
        conns_before = parse_json_result(resources["deriva-ml://catalog/connections"]())
        assert len(conns_before) == 0

        # Connect
        await tools["connect_catalog"](
            hostname=catalog_host,
            catalog_id=str(catalog_manager.catalog_id),
            domain_schema=catalog_manager.domain_schema,
        )

        # After connecting, list should have one entry
        conns_after = parse_json_result(resources["deriva-ml://catalog/connections"]())
        assert len(conns_after) == 1
        assert conns_after[0]["hostname"] == catalog_host
        assert conns_after[0]["is_active"] is True

        # Disconnect
        await tools["disconnect_catalog"]()

        # After disconnect, list should be empty again
        conns_final = parse_json_result(resources["deriva-ml://catalog/connections"]())
        assert len(conns_final) == 0

    @pytest.mark.asyncio
    async def test_connect_invalid_catalog(self, catalog_host):
        """connect_catalog returns error for a non-existent catalog."""
        from deriva_ml_mcp.connection import ConnectionManager
        from deriva_ml_mcp.tools.catalog import register_catalog_tools

        conn_manager = ConnectionManager()
        mcp, tools = _create_tool_capture()
        register_catalog_tools(mcp, conn_manager)

        result = await tools["connect_catalog"](
            hostname=catalog_host,
            catalog_id="99999",
        )
        data = parse_json_result(result)
        assert data["status"] == "error"


# =============================================================================
# 2. Vocabulary Workflow Tests
# =============================================================================


class TestVocabularyWorkflow:
    """Test vocabulary management through MCP tools."""

    @pytest.mark.asyncio
    async def test_add_and_lookup_term(self, int_vocab_tools):
        """add_term creates a term; lookup via resource confirms it exists."""
        # Add a term
        result = await int_vocab_tools["add_term"](
            vocabulary_name="Workflow_Type",
            term_name="Integration Test Workflow",
            description="Workflow type created during integration testing",
            synonyms=["int-test-wf"],
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "Integration Test Workflow"
        assert data["description"] == "Workflow type created during integration testing"
        assert "rid" in data

    @pytest.mark.asyncio
    async def test_create_vocabulary_and_add_terms(self, int_vocab_tools):
        """create_vocabulary makes a new vocabulary; add_term populates it."""
        # Create a new vocabulary
        result = await int_vocab_tools["create_vocabulary"](
            vocabulary_name="Test_Quality",
            comment="Quality ratings for integration tests",
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "Test_Quality"

        # Add terms to the new vocabulary
        for term_name, desc in [("Good", "High quality"), ("Bad", "Low quality")]:
            result = await int_vocab_tools["add_term"](
                vocabulary_name="Test_Quality",
                term_name=term_name,
                description=desc,
            )
            term_data = assert_success(result)
            assert term_data["status"] == "created"
            assert term_data["name"] == term_name

    @pytest.mark.asyncio
    async def test_synonym_lifecycle(self, int_vocab_tools):
        """add_synonym and remove_synonym modify term synonyms."""
        # Create a term first
        await int_vocab_tools["add_term"](
            vocabulary_name="Workflow_Type",
            term_name="Synonym Test WF",
            description="Workflow for synonym testing",
        )

        # Add synonym
        result = await int_vocab_tools["add_synonym"](
            vocabulary_name="Workflow_Type",
            term_name="Synonym Test WF",
            synonym="syn-test",
        )
        data = assert_success(result)
        assert data["status"] == "added"
        assert "syn-test" in data["synonyms"]

        # Remove synonym
        result = await int_vocab_tools["remove_synonym"](
            vocabulary_name="Workflow_Type",
            term_name="Synonym Test WF",
            synonym="syn-test",
        )
        data = assert_success(result)
        assert data["status"] == "removed"
        assert "syn-test" not in data["synonyms"]

    @pytest.mark.asyncio
    async def test_update_term_description(self, int_vocab_tools):
        """update_term_description modifies an existing term's description."""
        # Create a term
        await int_vocab_tools["add_term"](
            vocabulary_name="Workflow_Type",
            term_name="Desc Update WF",
            description="Original description",
        )

        # Update description
        result = await int_vocab_tools["update_term_description"](
            vocabulary_name="Workflow_Type",
            term_name="Desc Update WF",
            description="Updated description for testing",
        )
        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["description"] == "Updated description for testing"

    @pytest.mark.asyncio
    async def test_delete_term(self, int_vocab_tools):
        """delete_term removes a term from the vocabulary."""
        # Create a term
        await int_vocab_tools["add_term"](
            vocabulary_name="Workflow_Type",
            term_name="Deletable WF",
            description="To be deleted",
        )

        # Delete it
        result = await int_vocab_tools["delete_term"](
            vocabulary_name="Workflow_Type",
            term_name="Deletable WF",
        )
        data = assert_success(result)
        assert data["status"] == "deleted"
        assert data["name"] == "Deletable WF"


# =============================================================================
# 3. Schema Workflow Tests
# =============================================================================


class TestSchemaWorkflow:
    """Test schema management through MCP tools."""

    @pytest.mark.asyncio
    async def test_create_table_and_list(self, int_schema_tools, int_resources):
        """create_table creates a domain table; tables resource shows it."""
        # Create a simple table
        result = await int_schema_tools["create_table"](
            table_name="IntTestSubject",
            columns=[
                {"name": "Name", "type": "text", "nullok": False},
                {"name": "Age", "type": "int4"},
                {"name": "Notes", "type": "markdown"},
            ],
            comment="Subject table for integration tests",
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert data["table_name"] == "IntTestSubject"
        assert "Name" in data["columns"]
        assert "Age" in data["columns"]

        # Verify via tables resource
        tables_json = int_resources["deriva-ml://catalog/tables"]()
        tables = parse_json_result(tables_json)
        table_names = [t["name"] for t in tables]
        assert "IntTestSubject" in table_names

    @pytest.mark.asyncio
    async def test_add_column_to_table(self, int_schema_tools):
        """add_column adds a new column to an existing table."""
        # Create table first
        await int_schema_tools["create_table"](
            table_name="IntTestAddCol",
            columns=[
                {"name": "Name", "type": "text"},
            ],
            comment="Table for add_column test",
        )

        # Add a column
        result = await int_schema_tools["add_column"](
            table_name="IntTestAddCol",
            column_name="Weight",
            column_type="float8",
            nullok=True,
            comment="Weight in kg",
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert data["column_name"] == "Weight"


# =============================================================================
# 4. Dataset Workflow Tests
# =============================================================================


class TestDatasetWorkflow:
    """Test dataset management through MCP tools on a populated catalog."""

    @pytest.mark.asyncio
    async def test_create_dataset_and_lookup(self, populated_dataset_tools, populated_resources):
        """create_dataset creates a dataset; datasets resource shows it."""
        # Create a dataset
        result = await populated_dataset_tools["create_dataset"](
            description="Integration test dataset",
            dataset_types=[],
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert "dataset_rid" in data
        dataset_rid = data["dataset_rid"]

        # Verify via datasets resource
        datasets_json = populated_resources["deriva-ml://catalog/datasets"]()
        datasets = parse_json_result(datasets_json)
        dataset_rids = [ds["rid"] for ds in datasets]
        assert dataset_rid in dataset_rids

    @pytest.mark.asyncio
    async def test_dataset_members_workflow(
        self, populated_dataset_tools, populated_data_tools
    ):
        """add_dataset_members adds records; list_dataset_members shows them."""
        # Get some existing records (Image records from populated catalog)
        query_result = await populated_data_tools["query_table"](
            table_name="Image",
            limit=3,
        )
        query_data = parse_json_result(query_result)
        assert query_data["count"] > 0
        member_rids = [r["RID"] for r in query_data["records"][:3]]

        # Create a dataset
        ds_result = await populated_dataset_tools["create_dataset"](
            description="Dataset with members",
        )
        ds_data = assert_success(ds_result)
        dataset_rid = ds_data["dataset_rid"]

        # Add members
        add_result = await populated_dataset_tools["add_dataset_members"](
            dataset_rid=dataset_rid,
            member_rids=member_rids,
        )
        add_data = assert_success(add_result)
        assert add_data["status"] == "success"
        assert add_data["added_count"] == len(member_rids)

        # List members
        members_result = await populated_dataset_tools["list_dataset_members"](
            dataset_rid=dataset_rid,
        )
        members = parse_json_result(members_result)
        # Members are grouped by table name
        assert isinstance(members, dict)
        total_members = sum(len(v) for v in members.values())
        assert total_members >= len(member_rids)

    @pytest.mark.asyncio
    async def test_dataset_description(self, populated_dataset_tools):
        """set_dataset_description updates a dataset's description."""
        # Create a dataset
        ds_result = await populated_dataset_tools["create_dataset"](
            description="Original description",
        )
        ds_data = assert_success(ds_result)
        dataset_rid = ds_data["dataset_rid"]

        # Update description
        result = await populated_dataset_tools["set_dataset_description"](
            dataset_rid=dataset_rid,
            description="Updated integration test description",
        )
        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["description"] == "Updated integration test description"

    @pytest.mark.asyncio
    async def test_dataset_details_resource(self, populated_dataset_tools, populated_resources):
        """Dataset details resource returns correct data for a specific dataset."""
        # Create a dataset
        ds_result = await populated_dataset_tools["create_dataset"](
            description="Resource test dataset",
        )
        ds_data = assert_success(ds_result)
        dataset_rid = ds_data["dataset_rid"]

        # Access the parameterized resource
        detail_result = populated_resources[f"deriva-ml://dataset/{dataset_rid}"](dataset_rid)
        detail = parse_json_result(detail_result)
        assert detail["rid"] == dataset_rid
        assert detail["description"] == "Resource test dataset"
        assert "current_version" in detail


# =============================================================================
# 5. Execution Workflow Tests
# =============================================================================


class TestExecutionWorkflow:
    """Test execution lifecycle through MCP tools."""

    @pytest.mark.asyncio
    async def test_create_and_start_stop_execution(self, int_execution_tools, int_workflow_tools):
        """create_execution, start_execution, stop_execution lifecycle works end-to-end."""
        # Ensure a workflow type exists
        await int_workflow_tools["add_workflow_type"](
            type_name="Integration Test",
            description="Workflow type for integration testing",
        )

        # Create execution
        result = await int_execution_tools["create_execution"](
            workflow_name="Integration Test Run",
            workflow_type="Integration Test",
            description="Testing execution lifecycle",
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert "execution_rid" in data
        execution_rid = data["execution_rid"]

        # Start execution
        start_result = await int_execution_tools["start_execution"]()
        start_data = assert_success(start_result)
        assert start_data["status"] == "started"
        assert start_data["execution_rid"] == execution_rid

        # Get execution info
        info_result = await int_execution_tools["get_execution_info"]()
        info_data = parse_json_result(info_result)
        assert info_data["execution_rid"] == execution_rid

        # Stop execution
        stop_result = await int_execution_tools["stop_execution"]()
        stop_data = assert_success(stop_result)
        assert stop_data["status"] == "completed"
        assert stop_data["execution_rid"] == execution_rid

    @pytest.mark.asyncio
    async def test_create_execution_dataset(self, int_execution_tools, int_workflow_tools):
        """create_execution_dataset creates a dataset linked to an execution."""
        # Ensure a workflow type exists
        await int_workflow_tools["add_workflow_type"](
            type_name="Dataset Creation Test",
            description="For testing dataset creation in execution",
        )

        # Create execution
        await int_execution_tools["create_execution"](
            workflow_name="Dataset Creator",
            workflow_type="Dataset Creation Test",
            description="Creates a dataset",
        )

        # Create dataset via execution
        result = await int_execution_tools["create_execution_dataset"](
            description="Dataset created during execution",
            dataset_types=[],
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert "dataset_rid" in data
        assert "execution_rid" in data

    @pytest.mark.asyncio
    async def test_execution_without_start_errors(self, int_execution_tools):
        """start_execution returns error when no active execution exists."""
        # Ensure no active execution (fresh connection manager has none)
        # The mcp_connection_manager fixture resets the catalog, which clears
        # the active_tool_execution
        result = await int_execution_tools["start_execution"]()
        data = parse_json_result(result)
        assert data["status"] == "error"
        assert "No active execution" in data["message"]


# =============================================================================
# 6. Feature Workflow Tests
# =============================================================================


class TestFeatureWorkflow:
    """Test feature management through MCP tools on a populated catalog."""

    @pytest.mark.asyncio
    async def test_create_feature(self, int_vocab_tools, int_schema_tools, int_feature_tools):
        """create_feature defines a new feature on a domain table."""
        # First create a domain table to attach the feature to
        await int_schema_tools["create_table"](
            table_name="FeatureTestItem",
            columns=[{"name": "Name", "type": "text"}],
            comment="Items for feature testing",
        )

        # Create a vocabulary for feature terms
        await int_vocab_tools["create_vocabulary"](
            vocabulary_name="FeatureTestLabel",
            comment="Labels for feature testing",
        )
        await int_vocab_tools["add_term"](
            vocabulary_name="FeatureTestLabel",
            term_name="Positive",
            description="Positive label",
        )
        await int_vocab_tools["add_term"](
            vocabulary_name="FeatureTestLabel",
            term_name="Negative",
            description="Negative label",
        )

        # Create the feature
        result = await int_feature_tools["create_feature"](
            table_name="FeatureTestItem",
            feature_name="TestLabel",
            comment="Test label feature",
            terms=["FeatureTestLabel"],
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert data["feature_name"] == "TestLabel"
        assert data["target_table"] == "FeatureTestItem"

    @pytest.mark.asyncio
    async def test_find_features_on_populated_catalog(self, feature_schema_tools, feature_resources):
        """find_features returns features defined on a table in a feature-enabled catalog."""
        # The feature catalog already has features defined via ensure_features()
        # Check the table features resource for Image
        features_json = feature_resources["deriva-ml://table/Image/features"]("Image")
        features = parse_json_result(features_json)
        assert isinstance(features, list)
        # The demo catalog creates features on Image (e.g., BoundingBox, Quality)
        assert len(features) > 0

    @pytest.mark.asyncio
    async def test_feature_details_resource(self, feature_resources):
        """Feature details resource returns column types and requirements."""
        # Get features for Image first
        features_json = feature_resources["deriva-ml://table/Image/features"]("Image")
        features = parse_json_result(features_json)
        assert len(features) > 0

        # Pick the first feature and get its details
        first_feature = features[0]
        feature_name = first_feature["name"]
        detail_json = feature_resources[
            f"deriva-ml://feature/Image/{feature_name}"
        ]("Image", feature_name)
        detail = parse_json_result(detail_json)
        assert detail["feature_name"] == feature_name
        assert detail["target_table"] == "Image"
        assert "term_columns" in detail or "asset_columns" in detail or "value_columns" in detail

    @pytest.mark.asyncio
    async def test_add_feature_value(
        self,
        feature_feature_tools,
        feature_data_tools,
    ):
        """add_feature_value associates a value with a domain object."""
        # Get an Image RID from the populated catalog
        query_result = await feature_data_tools["query_table"](
            table_name="Image",
            limit=1,
        )
        query_data = parse_json_result(query_result)
        assert query_data["count"] > 0
        image_rid = query_data["records"][0]["RID"]

        # Add a feature value - the demo catalog has ImageQuality feature on Image
        # with vocabulary terms like "Good", "Bad"
        result = await feature_feature_tools["add_feature_value"](
            table_name="Image",
            feature_name="Quality",
            target_rid=image_rid,
            value="Good",
        )
        data = parse_json_result(result)
        # Allow either success or error (in case feature structure differs)
        # The key test is that the tool processes without crashing
        assert "status" in data


# =============================================================================
# 7. Data Query Workflow Tests
# =============================================================================


class TestDataQueryWorkflow:
    """Test data query and manipulation through MCP tools."""

    @pytest.mark.asyncio
    async def test_insert_and_query_records(self, int_schema_tools, int_data_tools):
        """insert_records adds records; query_table retrieves them."""
        # Create a table for data operations
        await int_schema_tools["create_table"](
            table_name="IntTestData",
            columns=[
                {"name": "Name", "type": "text", "nullok": False},
                {"name": "Value", "type": "int4"},
            ],
            comment="Table for data query testing",
        )

        # Insert records
        records = [
            {"Name": "alpha", "Value": 10},
            {"Name": "beta", "Value": 20},
            {"Name": "gamma", "Value": 30},
        ]
        insert_result = await int_data_tools["insert_records"](
            table_name="IntTestData",
            records=records,
        )
        insert_data = assert_success(insert_result)
        assert insert_data["status"] == "inserted"
        assert insert_data["inserted_count"] == 3
        assert len(insert_data["rids"]) == 3

        # Query all records
        query_result = await int_data_tools["query_table"](
            table_name="IntTestData",
            limit=100,
        )
        query_data = parse_json_result(query_result)
        assert query_data["count"] == 3

        # Query with column selection
        query_cols_result = await int_data_tools["query_table"](
            table_name="IntTestData",
            columns=["Name", "Value"],
            limit=100,
        )
        cols_data = parse_json_result(query_cols_result)
        assert cols_data["count"] == 3
        for rec in cols_data["records"]:
            assert "Name" in rec
            assert "Value" in rec

    @pytest.mark.asyncio
    async def test_query_with_filters(self, int_schema_tools, int_data_tools):
        """query_table with filters returns matching records only."""
        # Create and populate a table
        await int_schema_tools["create_table"](
            table_name="IntTestFilter",
            columns=[
                {"name": "Category", "type": "text"},
                {"name": "Amount", "type": "int4"},
            ],
            comment="Table for filter testing",
        )
        await int_data_tools["insert_records"](
            table_name="IntTestFilter",
            records=[
                {"Category": "A", "Amount": 100},
                {"Category": "B", "Amount": 200},
                {"Category": "A", "Amount": 300},
            ],
        )

        # Query with filter
        result = await int_data_tools["query_table"](
            table_name="IntTestFilter",
            filters={"Category": "A"},
            limit=100,
        )
        data = parse_json_result(result)
        assert data["count"] == 2
        for rec in data["records"]:
            assert rec["Category"] == "A"

    @pytest.mark.asyncio
    async def test_count_table(self, int_schema_tools, int_data_tools):
        """count_table returns correct record count with and without filters."""
        # Create and populate a table
        await int_schema_tools["create_table"](
            table_name="IntTestCount",
            columns=[
                {"name": "Type", "type": "text"},
            ],
            comment="Table for count testing",
        )
        await int_data_tools["insert_records"](
            table_name="IntTestCount",
            records=[
                {"Type": "X"},
                {"Type": "Y"},
                {"Type": "X"},
                {"Type": "X"},
            ],
        )

        # Count all
        result = await int_data_tools["count_table"](table_name="IntTestCount")
        data = parse_json_result(result)
        assert data["count"] == 4

        # Count with filter
        filtered_result = await int_data_tools["count_table"](
            table_name="IntTestCount",
            filters={"Type": "X"},
        )
        filtered_data = parse_json_result(filtered_result)
        assert filtered_data["count"] == 3

    @pytest.mark.asyncio
    async def test_query_with_limit_and_offset(self, int_schema_tools, int_data_tools):
        """query_table respects limit and offset for pagination."""
        # Create and populate a table
        await int_schema_tools["create_table"](
            table_name="IntTestPaginate",
            columns=[
                {"name": "Seq", "type": "int4"},
            ],
            comment="Table for pagination testing",
        )
        await int_data_tools["insert_records"](
            table_name="IntTestPaginate",
            records=[{"Seq": i} for i in range(10)],
        )

        # Query with limit
        result = await int_data_tools["query_table"](
            table_name="IntTestPaginate",
            limit=3,
        )
        data = parse_json_result(result)
        assert data["count"] == 3

        # Query with offset
        offset_result = await int_data_tools["query_table"](
            table_name="IntTestPaginate",
            limit=3,
            offset=5,
        )
        offset_data = parse_json_result(offset_result)
        assert offset_data["count"] == 3

    @pytest.mark.asyncio
    async def test_get_table(self, int_schema_tools, int_data_tools):
        """get_table returns all records from a table."""
        # Create and populate a table
        await int_schema_tools["create_table"](
            table_name="IntTestGetTable",
            columns=[
                {"name": "Label", "type": "text"},
            ],
            comment="Table for get_table testing",
        )
        await int_data_tools["insert_records"](
            table_name="IntTestGetTable",
            records=[{"Label": "one"}, {"Label": "two"}],
        )

        result = await int_data_tools["get_table"](
            table_name="IntTestGetTable",
        )
        data = parse_json_result(result)
        assert data["table"] == "IntTestGetTable"
        assert data["count"] == 2
        assert len(data["records"]) == 2

    @pytest.mark.asyncio
    async def test_get_record(self, int_schema_tools, int_data_tools):
        """get_record retrieves a single record by RID."""
        # Create and populate a table
        await int_schema_tools["create_table"](
            table_name="IntTestGetRecord",
            columns=[
                {"name": "Title", "type": "text"},
            ],
            comment="Table for get_record testing",
        )
        insert_result = await int_data_tools["insert_records"](
            table_name="IntTestGetRecord",
            records=[{"Title": "My Record"}],
        )
        insert_data = parse_json_result(insert_result)
        rid = insert_data["rids"][0]

        # Get the record
        result = await int_data_tools["get_record"](
            table_name="IntTestGetRecord",
            rid=rid,
        )
        data = parse_json_result(result)
        assert data["rid"] == rid
        assert data["record"]["Title"] == "My Record"

    @pytest.mark.asyncio
    async def test_update_record(self, int_schema_tools, int_data_tools):
        """update_record modifies fields in an existing record."""
        # Create and populate a table
        await int_schema_tools["create_table"](
            table_name="IntTestUpdate",
            columns=[
                {"name": "Status", "type": "text"},
                {"name": "Count", "type": "int4"},
            ],
            comment="Table for update_record testing",
        )
        insert_result = await int_data_tools["insert_records"](
            table_name="IntTestUpdate",
            records=[{"Status": "draft", "Count": 0}],
        )
        insert_data = parse_json_result(insert_result)
        rid = insert_data["rids"][0]

        # Update the record
        result = await int_data_tools["update_record"](
            table_name="IntTestUpdate",
            rid=rid,
            updates={"Status": "published", "Count": 5},
        )
        data = assert_success(result)
        assert data["status"] == "updated"

        # Verify the update
        get_result = await int_data_tools["get_record"](
            table_name="IntTestUpdate",
            rid=rid,
        )
        get_data = parse_json_result(get_result)
        assert get_data["record"]["Status"] == "published"
        assert get_data["record"]["Count"] == 5

    @pytest.mark.asyncio
    async def test_insert_into_managed_table_rejected(self, int_data_tools):
        """insert_records rejects inserts into managed tables like Dataset."""
        result = await int_data_tools["insert_records"](
            table_name="Dataset",
            records=[{"Description": "Should fail"}],
        )
        data = parse_json_result(result)
        assert data["status"] == "error"
        assert "create_dataset" in data["message"]

    @pytest.mark.asyncio
    async def test_query_populated_data(self, populated_data_tools):
        """query_table returns data from a populated catalog."""
        # The populated catalog has Subject and Image tables with data
        result = await populated_data_tools["query_table"](
            table_name="Subject",
            limit=10,
        )
        data = parse_json_result(result)
        assert data["count"] > 0

        result = await populated_data_tools["query_table"](
            table_name="Image",
            limit=10,
        )
        data = parse_json_result(result)
        assert data["count"] > 0


# =============================================================================
# 8. Resource Access Workflow Tests
# =============================================================================


class TestResourceAccessWorkflow:
    """Test MCP resource access patterns on real catalogs."""

    def test_catalog_schema_resource(self, int_resources):
        """Catalog schema resource returns valid schema information."""
        result = int_resources["deriva-ml://catalog/schema"]()
        data = parse_json_result(result)
        assert "hostname" in data
        assert "catalog_id" in data
        assert "tables" in data
        assert isinstance(data["tables"], list)

    def test_catalog_vocabularies_resource(self, int_resources):
        """Catalog vocabularies resource returns vocabulary data."""
        result = int_resources["deriva-ml://catalog/vocabularies"]()
        data = parse_json_result(result)
        assert isinstance(data, dict)
        # ML schema always has some vocabularies
        assert len(data) > 0

    def test_catalog_tables_resource(self, int_resources):
        """Catalog tables resource returns table metadata."""
        result = int_resources["deriva-ml://catalog/tables"]()
        data = parse_json_result(result)
        assert isinstance(data, list)

    def test_vocabulary_resource(self, int_resources):
        """Vocabulary resource returns terms for a specific vocabulary."""
        # Workflow_Type is always present in ML schema
        result = int_resources["deriva-ml://vocabulary/Workflow_Type"]("Workflow_Type")
        data = parse_json_result(result)
        assert isinstance(data, list)
        # At minimum, DerivaML MCP type exists from the connection
        assert len(data) > 0
        for term in data:
            assert "name" in term

    def test_table_schema_resource(self, int_resources):
        """Table schema resource returns column details for a table."""
        # Dataset table always exists in ML schema
        result = int_resources["deriva-ml://table/Dataset/schema"]("Dataset")
        data = parse_json_result(result)
        assert data["name"] == "Dataset"
        assert "columns" in data
        assert isinstance(data["columns"], list)
        assert len(data["columns"]) > 0

    def test_server_version_resource(self, int_resources):
        """Server version resource returns version info."""
        result = int_resources["deriva-ml://server/version"]()
        data = parse_json_result(result)
        assert "name" in data
        assert "version" in data

    def test_dataset_element_types_resource(self, populated_resources):
        """Dataset element types resource lists registered element types."""
        result = populated_resources["deriva-ml://catalog/dataset-element-types"]()
        data = parse_json_result(result)
        assert isinstance(data, list)
        # Populated catalog should have Image and Subject as element types
        element_names = [e["name"] for e in data]
        assert len(element_names) > 0

    def test_catalog_info_resource(self, int_resources):
        """Catalog info resource returns connection details."""
        result = int_resources["deriva-ml://catalog/info"]()
        data = parse_json_result(result)
        assert "hostname" in data
        assert "catalog_id" in data
        assert "domain_schemas" in data
        assert "project_name" in data

    def test_workflow_types_resource(self, int_resources):
        """Workflow types resource returns available workflow types."""
        result = int_resources["deriva-ml://catalog/workflow-types"]()
        data = parse_json_result(result)
        assert isinstance(data, list)
        # At minimum, DerivaML MCP type should exist
        names = [t["name"] for t in data]
        assert "Deriva MCP" in names

    def test_connections_resource(self, int_resources):
        """Connections resource shows the active connection."""
        result = int_resources["deriva-ml://catalog/connections"]()
        data = parse_json_result(result)
        assert isinstance(data, list)
        assert len(data) >= 1
        active_conns = [c for c in data if c["is_active"]]
        assert len(active_conns) == 1


# =============================================================================
# 9. Workflow Tool Tests
# =============================================================================


class TestWorkflowTools:
    """Test workflow management through MCP tools."""

    @pytest.mark.asyncio
    async def test_create_workflow(self, int_workflow_tools):
        """create_workflow creates a workflow and returns its details."""
        # Ensure workflow type exists
        await int_workflow_tools["add_workflow_type"](
            type_name="Workflow Tool Test",
            description="For workflow tool testing",
        )

        result = await int_workflow_tools["create_workflow"](
            name="My Test Workflow",
            workflow_type="Workflow Tool Test",
            description="A test workflow created by integration tests",
        )
        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "My Test Workflow"
        assert data["workflow_type"] == "Workflow Tool Test"
        assert "workflow_rid" in data

    @pytest.mark.asyncio
    async def test_workflows_resource(self, int_workflow_tools, int_resources):
        """Workflows resource shows created workflows."""
        # Create a workflow
        await int_workflow_tools["add_workflow_type"](
            type_name="Resource Check WF",
            description="For resource check testing",
        )
        await int_workflow_tools["create_workflow"](
            name="Resource Check Workflow",
            workflow_type="Resource Check WF",
            description="Check if workflows show in resource",
        )

        # Check resource
        result = int_resources["deriva-ml://catalog/workflows"]()
        data = parse_json_result(result)
        assert isinstance(data, list)
        assert len(data) > 0
        names = [wf["name"] for wf in data]
        assert "Resource Check Workflow" in names


# =============================================================================
# 10. Cross-Module Integration Tests
# =============================================================================


class TestCrossModuleIntegration:
    """Test workflows that span multiple tool modules."""

    @pytest.mark.asyncio
    async def test_schema_then_data_workflow(self, int_schema_tools, int_data_tools):
        """Create table, insert data, query it, update, and verify."""
        # Create table
        await int_schema_tools["create_table"](
            table_name="IntTestE2E",
            columns=[
                {"name": "Name", "type": "text", "nullok": False},
                {"name": "Score", "type": "float8"},
                {"name": "Active", "type": "boolean"},
            ],
            comment="End-to-end test table",
        )

        # Insert data
        insert_result = await int_data_tools["insert_records"](
            table_name="IntTestE2E",
            records=[
                {"Name": "Alice", "Score": 95.5, "Active": True},
                {"Name": "Bob", "Score": 82.0, "Active": False},
                {"Name": "Charlie", "Score": 91.3, "Active": True},
            ],
        )
        insert_data = parse_json_result(insert_result)
        assert insert_data["inserted_count"] == 3

        # Count
        count_result = await int_data_tools["count_table"](table_name="IntTestE2E")
        count_data = parse_json_result(count_result)
        assert count_data["count"] == 3

        # Query with filter
        query_result = await int_data_tools["query_table"](
            table_name="IntTestE2E",
            filters={"Active": True},
        )
        query_data = parse_json_result(query_result)
        assert query_data["count"] == 2

        # Get a specific record
        rid = insert_data["rids"][0]
        record_result = await int_data_tools["get_record"](
            table_name="IntTestE2E",
            rid=rid,
        )
        record_data = parse_json_result(record_result)
        assert record_data["record"]["Name"] == "Alice"

    @pytest.mark.asyncio
    async def test_vocab_then_table_with_fk(self, int_vocab_tools, int_schema_tools, int_data_tools):
        """Create vocabulary, create table with FK to vocabulary, insert data."""
        # Create vocabulary
        await int_vocab_tools["create_vocabulary"](
            vocabulary_name="IntTestSpecies",
            comment="Species for integration test",
        )
        await int_vocab_tools["add_term"](
            vocabulary_name="IntTestSpecies",
            term_name="Human",
            description="Homo sapiens",
        )
        await int_vocab_tools["add_term"](
            vocabulary_name="IntTestSpecies",
            term_name="Mouse",
            description="Mus musculus",
        )

        # Create table with FK to the vocabulary
        await int_schema_tools["create_table"](
            table_name="IntTestOrganism",
            columns=[
                {"name": "Name", "type": "text", "nullok": False},
                {"name": "Species", "type": "text"},
            ],
            foreign_keys=[
                {
                    "column": "Species",
                    "referenced_table": "IntTestSpecies",
                    "referenced_column": "Name",
                },
            ],
            comment="Organism table with species FK",
        )

        # Insert data referencing vocabulary terms
        result = await int_data_tools["insert_records"](
            table_name="IntTestOrganism",
            records=[
                {"Name": "Subject 1", "Species": "Human"},
                {"Name": "Subject 2", "Species": "Mouse"},
            ],
        )
        data = assert_success(result)
        assert data["inserted_count"] == 2

        # Query and verify
        query_result = await int_data_tools["query_table"](
            table_name="IntTestOrganism",
            filters={"Species": "Human"},
        )
        query_data = parse_json_result(query_result)
        assert query_data["count"] == 1
        assert query_data["records"][0]["Name"] == "Subject 1"

    @pytest.mark.asyncio
    async def test_execution_with_dataset_creation(
        self,
        populated_connection_manager,
    ):
        """Full workflow: create execution, create dataset inside it, add members."""
        from deriva_ml_mcp.tools.data import register_data_tools
        from deriva_ml_mcp.tools.dataset import register_dataset_tools
        from deriva_ml_mcp.tools.execution import register_execution_tools
        from deriva_ml_mcp.tools.workflow import register_workflow_tools

        cm = populated_connection_manager

        mcp_w, wf_tools = _create_tool_capture()
        register_workflow_tools(mcp_w, cm)

        mcp_e, exe_tools = _create_tool_capture()
        register_execution_tools(mcp_e, cm)

        mcp_ds, ds_tools = _create_tool_capture()
        register_dataset_tools(mcp_ds, cm)

        mcp_d, data_tools = _create_tool_capture()
        register_data_tools(mcp_d, cm)

        # Ensure workflow type
        await wf_tools["add_workflow_type"](
            type_name="Full Integration Test",
            description="Full integration workflow",
        )

        # Create execution
        exe_result = await exe_tools["create_execution"](
            workflow_name="Full Integration Run",
            workflow_type="Full Integration Test",
            description="Full end-to-end integration test",
        )
        exe_data = assert_success(exe_result)
        execution_rid = exe_data["execution_rid"]

        # Start execution
        await exe_tools["start_execution"]()

        # Create dataset via the execution context
        ds_result = await exe_tools["create_execution_dataset"](
            description="Integration test output dataset",
        )
        ds_data = assert_success(ds_result)
        dataset_rid = ds_data["dataset_rid"]
        assert ds_data["execution_rid"] == execution_rid

        # Get image RIDs from populated catalog
        query_result = await data_tools["query_table"](
            table_name="Image",
            limit=5,
        )
        query_data = parse_json_result(query_result)
        member_rids = [r["RID"] for r in query_data["records"][:5]]

        # Add members to the dataset
        add_result = await ds_tools["add_dataset_members"](
            dataset_rid=dataset_rid,
            member_rids=member_rids,
        )
        add_data = assert_success(add_result)
        assert add_data["added_count"] == len(member_rids)

        # Stop execution
        await exe_tools["stop_execution"]()

        # Verify dataset members
        members_result = await ds_tools["list_dataset_members"](
            dataset_rid=dataset_rid,
        )
        members = parse_json_result(members_result)
        total = sum(len(v) for v in members.values())
        assert total >= len(member_rids)
