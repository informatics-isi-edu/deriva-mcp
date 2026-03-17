"""Tests for RAG integration layers across MCP tools."""

import json
import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import _create_tool_capture, _create_resource_capture


class TestLayer2DataTools:
    """Layer 2: Error recovery on data tools."""

    def _make_conn_manager_with_rag(self):
        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.model.name_to_table.side_effect = Exception("Table 'Diagnoiss' not found in schema 'isa'")
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.hostname = "test.example.org"
        mock_conn_info.catalog_id = "1"
        mock_conn_info.schema_dirty = False
        conn_manager.get_active_connection_info.return_value = mock_conn_info
        return conn_manager

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_query_table_suggests_on_not_found(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis (vocabulary)", "relevance": 0.92,
             "source": "schema:test:1:abc", "section_heading": "isa.Diagnosis",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = self._make_conn_manager_with_rag()
        from deriva_mcp.tools.data import register_data_tools
        mcp, tools = _create_tool_capture()
        register_data_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["query_table"]("Diagnoiss")))
        assert result["status"] == "error"
        assert "suggestions" in result
        assert result["hint"].startswith("Did you mean:")

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_count_table_suggests_on_not_found(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis (vocabulary)", "relevance": 0.92,
             "source": "schema:test:1:abc", "section_heading": "isa.Diagnosis",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = self._make_conn_manager_with_rag()
        from deriva_mcp.tools.data import register_data_tools
        mcp, tools = _create_tool_capture()
        register_data_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["count_table"]("Diagnoiss")))
        assert result["status"] == "error"
        assert "suggestions" in result

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_get_table_suggests_on_not_found(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis (vocabulary)", "relevance": 0.92,
             "source": "schema:test:1:abc", "section_heading": "isa.Diagnosis",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = self._make_conn_manager_with_rag()
        # get_table uses get_table_as_dict not name_to_table, need to adjust
        conn_manager.get_active_or_raise.return_value.get_table_as_dict.side_effect = Exception(
            "Table 'Diagnoiss' not found in schema 'isa'"
        )
        # Also mock model.name_to_table to succeed for get_table (it doesn't call it)
        conn_manager.get_active_or_raise.return_value.model.name_to_table.side_effect = None

        from deriva_mcp.tools.data import register_data_tools
        mcp, tools = _create_tool_capture()
        register_data_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["get_table"]("Diagnoiss")))
        assert result["status"] == "error"
        assert "suggestions" in result

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_no_suggestions_when_rag_not_initialized(self, mock_get_rag):
        mock_get_rag.return_value = None  # RAG not initialized

        conn_manager = self._make_conn_manager_with_rag()
        from deriva_mcp.tools.data import register_data_tools
        mcp, tools = _create_tool_capture()
        register_data_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["query_table"]("Diagnoiss")))
        assert result["status"] == "error"
        assert "suggestions" not in result

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_no_suggestions_on_non_not_found_error(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        conn_manager.get_active_or_raise.side_effect = Exception("Network timeout error")
        conn_manager.get_active_connection_info.return_value = None

        from deriva_mcp.tools.data import register_data_tools
        mcp, tools = _create_tool_capture()
        register_data_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["query_table"]("SomeTable")))
        assert result["status"] == "error"
        assert "suggestions" not in result


class TestLayer3DuplicateDetection:
    """Layer 3: Duplicate detection on creation tools."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_create_table_warns_on_similar(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis_Type (vocabulary)", "relevance": 0.88,
             "source": "schema:test:1:abc", "section_heading": "isa.Diagnosis_Type",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_schema = MagicMock()
        mock_schema.name = "isa"
        mock_table = MagicMock()
        mock_table.name = "Diagnosis"
        mock_table.schema = mock_schema
        mock_table.columns = []
        mock_ml.create_table.return_value = mock_table
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.schema_dirty = False
        mock_conn_info._schema_reindex_at = 0.0
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.tools.schema import register_schema_tools
        mcp, tools = _create_tool_capture()
        register_schema_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["create_table"]("Diagnosis")))
        assert result["status"] == "created"
        assert "similar_existing" in result
        assert "warning" in result


class TestLayer3CreateVocabulary:
    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_create_vocabulary_warns_on_similar(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis_Type (vocabulary)", "relevance": 0.88,
             "source": "schema:test:1:abc", "section_heading": "isa.Diagnosis_Type",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_vocab_schema = MagicMock()
        mock_vocab_schema.name = "isa"
        mock_vocab_table = MagicMock()
        mock_vocab_table.name = "Diagnosis"
        mock_vocab_table.schema = mock_vocab_schema
        mock_ml.create_vocabulary.return_value = mock_vocab_table
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.schema_dirty = False
        mock_conn_info._schema_reindex_at = 0.0
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.tools.vocabulary import register_vocabulary_tools
        mcp, tools = _create_tool_capture()
        register_vocabulary_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["create_vocabulary"]("Diagnosis")))
        assert result["status"] == "created"
        assert "similar_existing" in result
        assert "warning" in result


class TestLayer3CreateFeature:
    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_create_feature_warns_on_similar(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Image_Quality (feature)", "relevance": 0.85,
             "source": "schema:test:1:abc", "section_heading": "isa.Image_Quality",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.create_feature.return_value = MagicMock()
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.schema_dirty = False
        mock_conn_info._schema_reindex_at = 0.0
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.tools.feature import register_feature_tools
        mcp, tools = _create_tool_capture()
        register_feature_tools(mcp, conn_manager)

        import asyncio
        result = json.loads(asyncio.run(tools["create_feature"]("Image_Score", "Image")))
        assert result["status"] == "created"
        assert "similar_existing" in result
        assert "warning" in result


class TestLayer4ResourceEnrichment:
    """Layer 4: _related_docs appears in resource responses."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_catalog_schema_includes_related_docs(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "Creating Tables Guide", "relevance": 0.92,
             "github_url": "https://example.com/tables-guide",
             "source": "docs", "section_heading": "Creating Tables",
             "doc_type": "documentation"},
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.model.get_schema_description.return_value = {"schemas": {}}
        mock_ml.host_name = "test.example.org"
        mock_ml.catalog_id = "1"
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        mock_conn_info.domain_schemas = ["isa"]
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.resources import register_resources
        mcp, resources = _create_resource_capture()
        register_resources(mcp, conn_manager)

        result = json.loads(resources["deriva://catalog/schema"]())
        assert "_related_docs" in result
        assert len(result["_related_docs"]) >= 1
        assert result["_related_docs"][0]["title"] == "Creating Tables"

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_catalog_datasets_includes_related_docs(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "Managing Datasets", "relevance": 0.85,
             "github_url": "https://example.com/datasets-guide",
             "source": "docs", "section_heading": "Managing Datasets",
             "doc_type": "documentation"},
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.find_datasets.return_value = []
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.schema_hash = "abc123"
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.resources import register_resources
        mcp, resources = _create_resource_capture()
        register_resources(mcp, conn_manager)

        result = json.loads(resources["deriva://catalog/datasets"]())
        assert "_related_docs" in result
        assert len(result["_related_docs"]) >= 1

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_related_docs_absent_when_rag_not_initialized(self, mock_get_rag):
        mock_get_rag.return_value = None  # RAG not initialized

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ml.model.get_schema_description.return_value = {"schemas": {}}
        conn_manager.get_active_or_raise.return_value = mock_ml
        conn_manager.get_active_connection_info.return_value = None

        from deriva_mcp.resources import register_resources
        mcp, resources = _create_resource_capture()
        register_resources(mcp, conn_manager)

        result = json.loads(resources["deriva://catalog/schema"]())
        assert "_related_docs" not in result


class TestLayer5RelatedData:
    """Layer 5: _related_data appears in dataset/execution resource responses."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_dataset_details_includes_related_data(self, mock_get_rag):
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## Dataset: Validation Set (RID: 2-XYZ)", "relevance": 0.85,
             "source": "data:test:1:user1", "section_heading": "## Dataset: Validation Set (RID: 2-XYZ)",
             "doc_type": "catalog-data"},
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ds = MagicMock()
        mock_ds.dataset_rid = "1-ABC"
        mock_ds.description = "Training dataset for lung segmentation"
        mock_ds.dataset_types = ["Training"]
        mock_ds.current_version = "1.0"
        mock_ds.list_dataset_members.return_value = {}
        mock_ds.dataset_history.return_value = []
        mock_ds.list_dataset_children.return_value = []
        mock_ds.list_dataset_parents.return_value = []
        mock_ml.lookup_dataset.return_value = mock_ds
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.user_id = "user1"
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        mock_conn_info.data_dirty = False
        mock_conn_info._data_reindex_at = 0.0
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.resources import register_resources
        from tests.conftest import _create_resource_capture
        mcp, resources = _create_resource_capture()
        register_resources(mcp, conn_manager)

        # The resource URI for dataset details is parameterized by RID
        # Resources are synchronous functions called directly with the parameter
        result = json.loads(resources["deriva://dataset/{dataset_rid}"]("1-ABC"))
        assert "_related_data" in result
        assert len(result["_related_data"]) >= 1

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_related_data_excludes_self(self, mock_get_rag):
        """Self-references (same RID) must be filtered out."""
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## Dataset: Same (RID: 1-ABC)", "relevance": 0.99,
             "source": "data:test:1:user1", "section_heading": "## Dataset: Same (RID: 1-ABC)",
             "doc_type": "catalog-data"},
        ]
        mock_get_rag.return_value = mock_manager

        conn_manager = MagicMock()
        mock_ml = MagicMock()
        mock_ds = MagicMock()
        mock_ds.dataset_rid = "1-ABC"
        mock_ds.description = "Training dataset"
        mock_ds.dataset_types = ["Training"]
        mock_ds.current_version = "1.0"
        mock_ds.list_dataset_members.return_value = {}
        mock_ds.dataset_history.return_value = []
        mock_ds.list_dataset_children.return_value = []
        mock_ds.list_dataset_parents.return_value = []
        mock_ml.lookup_dataset.return_value = mock_ds
        conn_manager.get_active_or_raise.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.user_id = "user1"
        mock_conn_info.hostname = "test"
        mock_conn_info.catalog_id = "1"
        mock_conn_info.data_dirty = False
        mock_conn_info._data_reindex_at = 0.0
        conn_manager.get_active_connection_info.return_value = mock_conn_info

        from deriva_mcp.resources import register_resources
        from tests.conftest import _create_resource_capture
        mcp, resources = _create_resource_capture()
        register_resources(mcp, conn_manager)

        result = json.loads(resources["deriva://dataset/{dataset_rid}"]("1-ABC"))
        related = result.get("_related_data", [])
        assert all(r.get("rid") != "1-ABC" for r in related)
