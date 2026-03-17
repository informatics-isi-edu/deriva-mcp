"""Tests for RAG integration layers across MCP tools."""

import json
import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import _create_tool_capture


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
