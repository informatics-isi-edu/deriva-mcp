"""Tests for RAG helper functions."""

import pytest


class TestIsNotFoundError:
    """Tests for the _is_not_found_error heuristic."""

    def test_table_not_found(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("Table 'Diagnoiss' not found in schema 'isa'") is True

    def test_no_such_table(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("no such table: Subject_Info") is True

    def test_does_not_exist(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("Table 'FooBar' does not exist") is True

    def test_could_not_find(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("could not find table Image_Data") is True

    def test_not_found_in_schema(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("'BadName' not found in schema") is True

    def test_generic_key_error_rejected(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("KeyError: 'missing_column'") is False

    def test_permission_error_rejected(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("Permission denied: cannot access table") is False

    def test_empty_string(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("") is False

    def test_connection_error_rejected(self):
        from deriva_mcp.rag.helpers import _is_not_found_error
        assert _is_not_found_error("Connection refused") is False


from unittest.mock import MagicMock, patch
import time


class TestTriggerSchemaReindex:
    def _make_conn_info(self):
        conn_info = MagicMock()
        conn_info.ml_instance = MagicMock()
        conn_info.hostname = "test.example.org"
        conn_info.catalog_id = "1"
        conn_info.schema_hash = "abc123"
        conn_info._schema_reindex_at = 0.0
        conn_info.schema_dirty = False
        return conn_info

    def test_noop_when_conn_info_none(self):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        trigger_schema_reindex(None)

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_noop_when_rag_not_initialized(self, mock_get_rag):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        mock_get_rag.return_value = None
        trigger_schema_reindex(self._make_conn_info())

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_debounce_skips_rapid_calls(self, mock_get_rag):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        mock_get_rag.return_value = MagicMock()
        conn_info = self._make_conn_info()
        conn_info._schema_reindex_at = time.time()
        trigger_schema_reindex(conn_info)

    @patch("deriva_mcp.rag.helpers.threading.Thread")
    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_spawns_thread_when_stale(self, mock_get_rag, mock_thread_cls):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        mock_get_rag.return_value = MagicMock()
        conn_info = self._make_conn_info()
        conn_info._schema_reindex_at = 0.0
        trigger_schema_reindex(conn_info)
        mock_thread_cls.assert_called_once()
        mock_thread_cls.return_value.start.assert_called_once()


class TestRagSuggestEntity:
    def _make_conn_info(self, schema_hash="abc123", schema_dirty=False):
        conn_info = MagicMock()
        conn_info.hostname = "test.example.org"
        conn_info.catalog_id = "1"
        conn_info.schema_hash = schema_hash
        conn_info.schema_dirty = schema_dirty
        conn_info._schema_reindex_at = 0.0
        conn_info.ml_instance = MagicMock()
        return conn_info

    def test_returns_empty_when_no_conn(self):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        assert rag_suggest_entity("Diagnosis", None) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_returns_empty_when_rag_not_init(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        mock_get_rag.return_value = None
        assert rag_suggest_entity("Diagnosis", self._make_conn_info()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_returns_empty_when_no_schema_hash(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        mock_get_rag.return_value = MagicMock()
        conn_info = self._make_conn_info(schema_hash=None)
        assert rag_suggest_entity("Diagnosis", conn_info) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_searches_schema_index(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "## isa.Diagnosis (vocabulary)\nDiagnosis terms", "relevance": 0.92,
             "source": "schema:test.example.org:1:abc123", "section_heading": "isa.Diagnosis",
             "doc_type": "catalog-schema"}
        ]
        mock_get_rag.return_value = mock_manager
        conn_info = self._make_conn_info()
        results = rag_suggest_entity("Diagnoiss", conn_info, limit=3)
        assert len(results) == 1
        assert results[0]["name"] == "Diagnosis"
        assert results[0]["relevance"] == 0.92

    @patch("deriva_mcp.rag.helpers.trigger_schema_reindex")
    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_triggers_reindex_when_dirty(self, mock_get_rag, mock_reindex):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        mock_manager = MagicMock()
        mock_manager.search.return_value = []
        mock_get_rag.return_value = mock_manager
        conn_info = self._make_conn_info(schema_dirty=True)
        rag_suggest_entity("Foo", conn_info)
        mock_reindex.assert_called_once_with(conn_info)
        assert conn_info.schema_dirty is False


class TestRagEnrichResource:
    def test_returns_empty_when_no_conn(self):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        assert rag_enrich_resource("tables", None) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_returns_empty_when_rag_not_init(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        mock_get_rag.return_value = None
        assert rag_enrich_resource("tables", MagicMock()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_filters_by_relevance(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "Creating Tables", "relevance": 0.9, "github_url": "https://example.com/tables",
             "source": "docs", "section_heading": "Creating Tables"},
            {"text": "Unrelated", "relevance": 0.5, "github_url": "https://example.com/other",
             "source": "docs", "section_heading": "Unrelated"},
        ]
        mock_get_rag.return_value = mock_manager
        results = rag_enrich_resource("tables", MagicMock())
        assert len(results) == 1
        assert results[0]["title"] == "Creating Tables"

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_deduplicates_by_url(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            {"text": "A", "relevance": 0.9, "github_url": "https://example.com/same",
             "source": "docs", "section_heading": "A"},
            {"text": "B", "relevance": 0.85, "github_url": "https://example.com/same",
             "source": "docs", "section_heading": "B"},
        ]
        mock_get_rag.return_value = mock_manager
        results = rag_enrich_resource("query", MagicMock())
        assert len(results) == 1


class TestTriggerDataReindex:
    def test_noop_when_conn_info_none(self):
        from deriva_mcp.rag.helpers import trigger_data_reindex
        trigger_data_reindex(None)


class TestRagSuggestRecord:
    def test_returns_empty_when_no_conn(self):
        from deriva_mcp.rag.helpers import rag_suggest_record
        assert rag_suggest_record("training", None) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_returns_empty_when_no_user_id(self, mock_get_rag):
        from deriva_mcp.rag.helpers import rag_suggest_record
        mock_get_rag.return_value = MagicMock()
        conn_info = MagicMock()
        conn_info.user_id = None
        assert rag_suggest_record("training", conn_info) == []


class TestGracefulDegradation:
    """Verify all helpers are no-ops when RAG is unavailable."""

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_suggest_entity_noop(self, _):
        from deriva_mcp.rag.helpers import rag_suggest_entity
        assert rag_suggest_entity("test", MagicMock()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_suggest_record_noop(self, _):
        from deriva_mcp.rag.helpers import rag_suggest_record
        assert rag_suggest_record("test", MagicMock()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_enrich_resource_noop(self, _):
        from deriva_mcp.rag.helpers import rag_enrich_resource
        assert rag_enrich_resource("test", MagicMock()) == []

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_trigger_schema_reindex_noop(self, _):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        trigger_schema_reindex(MagicMock())

    @patch("deriva_mcp.rag.helpers.get_rag_manager", return_value=None)
    def test_trigger_data_reindex_noop(self, _):
        from deriva_mcp.rag.helpers import trigger_data_reindex
        trigger_data_reindex(MagicMock())


class TestDebounce:
    """Verify debounce prevents rapid reindex."""

    @patch("deriva_mcp.rag.helpers.threading.Thread")
    @patch("deriva_mcp.rag.helpers.get_rag_manager")
    def test_rapid_calls_debounced(self, mock_get_rag, mock_thread_cls):
        from deriva_mcp.rag.helpers import trigger_schema_reindex
        mock_get_rag.return_value = MagicMock()

        conn_info = MagicMock()
        conn_info._schema_reindex_at = 0.0

        # First call should trigger
        trigger_schema_reindex(conn_info)
        assert mock_thread_cls.call_count == 1

        # Rapid second call should be debounced
        trigger_schema_reindex(conn_info)
        assert mock_thread_cls.call_count == 1  # Still 1
