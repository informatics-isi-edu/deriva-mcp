"""Unit tests for RAG documentation tools.

Tests all 7 RAG tools:
    - rag_search
    - rag_ingest
    - rag_update
    - rag_status
    - rag_add_source
    - rag_remove_source
    - rag_index_schema

These tools delegate to the RAGManager singleton. We mock:
    - `deriva_mcp.rag.get_rag_manager` — controls what RAGManager the tools see
    - `deriva_mcp.tasks.get_task_manager` — controls background task creation
    - `conn_manager` — controls active catalog connection info
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _create_tool_capture

# Patch target for the RAG manager singleton (source module)
RAG_MANAGER_PATCH = "deriva_mcp.rag.get_rag_manager"
TASK_MANAGER_PATCH = "deriva_mcp.tasks.get_task_manager"


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_rag_manager():
    """Create a mock RAGManager with default return values."""
    manager = MagicMock()
    manager.search.return_value = []
    manager.get_status.return_value = {
        "initialized": True,
        "total_chunks": 42,
        "sources": [],
        "catalog_schemas": [],
    }
    manager.add_source.return_value = {
        "status": "created",
        "source": {"name": "test-source"},
        "message": "Source 'test-source' registered.",
    }
    manager.remove_source.return_value = {
        "status": "removed",
        "source": "test-source",
        "chunks_deleted": 10,
    }
    manager.index_catalog_schema.return_value = {
        "source": "schema:test.example.org:1",
        "status": "indexed",
        "chunks_created": 5,
        "schema_hash": "abc123",
    }
    return manager


def _make_mock_task_manager():
    """Create a mock BackgroundTaskManager."""
    task_manager = MagicMock()
    mock_task = MagicMock()
    mock_task.task_id = "task-001"
    task_manager.create_task.return_value = mock_task
    return task_manager


def _make_search_results(n=2):
    """Create a list of mock search result dicts."""
    return [
        {
            "id": f"chunk-{i}",
            "text": f"Result text {i}",
            "relevance": 0.9 - i * 0.1,
            "source": "deriva-ml-docs",
            "doc_type": "user-guide",
        }
        for i in range(n)
    ]


def _make_schema_search_results(n=1):
    """Create mock schema search result dicts."""
    return [
        {
            "id": f"schema-chunk-{i}",
            "text": f"Table info {i}",
            "relevance": 0.85 - i * 0.1,
            "source": "schema:test.example.org:1",
            "doc_type": "catalog-schema",
        }
        for i in range(n)
    ]


def _register_rag_tools(conn_manager):
    """Register RAG tools with a given connection manager."""
    from deriva_mcp.tools.rag import register_rag_tools

    mcp, tools = _create_tool_capture()
    register_rag_tools(mcp, conn_manager)
    return tools


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_rag_manager():
    return _make_mock_rag_manager()


@pytest.fixture
def mock_task_manager():
    return _make_mock_task_manager()


@pytest.fixture
def mock_conn_manager():
    """ConnectionManager with active connection info."""
    conn_manager = MagicMock()
    conn_info = MagicMock()
    conn_info.hostname = "test.example.org"
    conn_info.catalog_id = "1"
    conn_info.ml_instance = MagicMock()
    conn_info.ml_instance.model.get_schema_description.return_value = {
        "domain_schemas": ["isa"],
        "default_schema": "isa",
        "schemas": {
            "isa": {
                "tables": {
                    "Diagnosis": {
                        "is_vocabulary": True,
                        "columns": [],
                        "foreign_keys": [],
                    },
                    "Image": {
                        "is_vocabulary": False,
                        "columns": [],
                        "foreign_keys": [],
                    },
                }
            }
        },
    }
    # Mock vocabulary terms
    mock_term = MagicMock()
    mock_term.name = "Normal"
    mock_term.description = "No pathology"
    mock_term.synonyms = ["Healthy", "NL"]
    conn_info.ml_instance.list_vocabulary_terms.return_value = [mock_term]

    conn_manager.get_active_connection_info.return_value = conn_info
    return conn_manager


@pytest.fixture
def disconnected_conn_manager():
    """ConnectionManager with no active connection."""
    conn_manager = MagicMock()
    conn_manager.get_active_connection_info.return_value = None
    return conn_manager


@pytest.fixture
def rag_tools(mock_conn_manager):
    """Capture RAG tools with a connected mock."""
    return _register_rag_tools(mock_conn_manager)


@pytest.fixture
def rag_tools_disconnected(disconnected_conn_manager):
    """Capture RAG tools with no connection."""
    return _register_rag_tools(disconnected_conn_manager)


# =============================================================================
# TestRagSearch
# =============================================================================


class TestRagSearch:
    """Tests for the rag_search tool."""

    def test_basic_search(self, rag_tools, mock_rag_manager):
        """rag_search delegates to manager.search and returns merged results."""
        doc_results = _make_search_results(2)
        schema_results = _make_schema_search_results(1)
        mock_rag_manager.search.side_effect = [doc_results, schema_results]

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_search"](query="how to create a dataset")

        assert result["query"] == "how to create a dataset"
        assert result["result_count"] == 3  # 2 doc + 1 schema
        assert len(result["results"]) == 3

    def test_search_with_source_filter(self, rag_tools, mock_rag_manager):
        """rag_search passes source filter directly to manager.search."""
        mock_rag_manager.search.return_value = _make_search_results(1)

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_search"](
                query="ermrest API",
                source="ermrest-docs",
            )

        mock_rag_manager.search.assert_called_once_with(
            query="ermrest API", limit=10, source="ermrest-docs", doc_type=None
        )
        assert result["result_count"] == 1

    def test_search_with_doc_type_filter(self, rag_tools, mock_rag_manager):
        """rag_search passes doc_type filter directly to manager.search."""
        mock_rag_manager.search.return_value = []

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_search"](
                query="API reference",
                doc_type="api-reference",
            )

        mock_rag_manager.search.assert_called_once_with(
            query="API reference", limit=10, source=None, doc_type="api-reference"
        )
        assert result["result_count"] == 0

    def test_search_with_custom_limit(self, rag_tools, mock_rag_manager):
        """rag_search respects the limit parameter."""
        mock_rag_manager.search.return_value = _make_search_results(5)

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_search"](query="dataset", limit=5)

        assert result["result_count"] == 5

    def test_search_merges_schema_results(self, rag_tools, mock_rag_manager):
        """rag_search merges doc and schema results when connected."""
        doc_results = _make_search_results(2)
        schema_results = _make_schema_search_results(1)

        # First call = doc search, second call = schema search
        mock_rag_manager.search.side_effect = [doc_results, schema_results]

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_search"](query="image table")

        assert result["result_count"] == 3
        # Results should be sorted by relevance (descending)
        relevances = [r["relevance"] for r in result["results"]]
        assert relevances == sorted(relevances, reverse=True)

    def test_search_respects_limit_after_merge(self, rag_tools, mock_rag_manager):
        """rag_search truncates merged results to the limit."""
        doc_results = _make_search_results(3)
        schema_results = _make_schema_search_results(3)
        mock_rag_manager.search.side_effect = [doc_results, schema_results]

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_search"](query="tables", limit=2)

        assert result["result_count"] == 2

    def test_search_no_schema_when_disconnected(self, rag_tools_disconnected, mock_rag_manager):
        """rag_search skips schema search when not connected to a catalog."""
        mock_rag_manager.search.return_value = _make_search_results(2)

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools_disconnected["rag_search"](query="tables")

        # Only one call (doc search), not two
        mock_rag_manager.search.assert_called_once_with(query="tables", limit=10)
        assert result["result_count"] == 2

    def test_search_include_schema_false(self, rag_tools, mock_rag_manager):
        """rag_search skips schema search when include_schema=False."""
        mock_rag_manager.search.return_value = _make_search_results(1)

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_search"](
                query="dataset",
                include_schema=False,
            )

        # Only one call (doc search)
        mock_rag_manager.search.assert_called_once_with(query="dataset", limit=10)
        assert result["result_count"] == 1

    def test_search_schema_source_name_correct(self, rag_tools, mock_rag_manager):
        """rag_search uses correct schema source name from connection info."""
        mock_rag_manager.search.side_effect = [[], []]

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            rag_tools["rag_search"](query="image columns")

        # Second call should use schema source
        calls = mock_rag_manager.search.call_args_list
        assert len(calls) == 2
        assert calls[1].kwargs.get("source") == "schema:test.example.org:1"

    def test_search_rag_not_initialized(self, rag_tools):
        """rag_search returns error when RAG manager is not initialized."""
        with patch(RAG_MANAGER_PATCH, return_value=None):
            result = rag_tools["rag_search"](query="test")

        assert "error" in result
        assert "not initialized" in result["error"]

    def test_search_empty_query(self, rag_tools, mock_rag_manager):
        """rag_search works with empty query string."""
        mock_rag_manager.search.return_value = []

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_search"](query="")

        assert result["result_count"] == 0
        assert result["query"] == ""


# =============================================================================
# TestRagIngest
# =============================================================================


class TestRagIngest:
    """Tests for the rag_ingest tool."""

    def test_ingest_all_sources(self, rag_tools, mock_rag_manager):
        """rag_ingest creates a background task for all sources."""
        mock_task_mgr = _make_mock_task_manager()

        with (
            patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager),
            patch(TASK_MANAGER_PATCH, return_value=mock_task_mgr),
        ):
            result = rag_tools["rag_ingest"]()

        assert result["status"] == "started"
        assert result["task_id"] == "task-001"
        assert "all sources" in result["message"]

    def test_ingest_specific_source(self, rag_tools, mock_rag_manager):
        """rag_ingest creates a task for a specific source."""
        mock_task_mgr = _make_mock_task_manager()

        with (
            patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager),
            patch(TASK_MANAGER_PATCH, return_value=mock_task_mgr),
        ):
            result = rag_tools["rag_ingest"](source_name="deriva-ml-docs")

        assert result["status"] == "started"
        assert "deriva-ml-docs" in result["message"]
        # Verify task was created with correct parameters
        call_kwargs = mock_task_mgr.create_task.call_args.kwargs
        assert call_kwargs["parameters"]["source_name"] == "deriva-ml-docs"

    def test_ingest_rag_not_initialized(self, rag_tools):
        """rag_ingest returns error when RAG manager is not initialized."""
        with patch(RAG_MANAGER_PATCH, return_value=None):
            result = rag_tools["rag_ingest"]()

        assert "error" in result
        assert "not initialized" in result["error"]

    def test_ingest_task_has_correct_type(self, rag_tools, mock_rag_manager):
        """rag_ingest creates task with RAG_INGEST type."""
        from deriva_mcp.tasks import TaskType

        mock_task_mgr = _make_mock_task_manager()

        with (
            patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager),
            patch(TASK_MANAGER_PATCH, return_value=mock_task_mgr),
        ):
            rag_tools["rag_ingest"]()

        call_kwargs = mock_task_mgr.create_task.call_args.kwargs
        assert call_kwargs["task_type"] == TaskType.RAG_INGEST


# =============================================================================
# TestRagUpdate
# =============================================================================


class TestRagUpdate:
    """Tests for the rag_update tool."""

    def test_update_all_sources(self, rag_tools, mock_rag_manager):
        """rag_update creates a background task for all sources."""
        mock_task_mgr = _make_mock_task_manager()

        with (
            patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager),
            patch(TASK_MANAGER_PATCH, return_value=mock_task_mgr),
        ):
            result = rag_tools["rag_update"]()

        assert result["status"] == "started"
        assert result["task_id"] == "task-001"
        assert "all sources" in result["message"]

    def test_update_specific_source(self, rag_tools, mock_rag_manager):
        """rag_update creates a task for a specific source."""
        mock_task_mgr = _make_mock_task_manager()

        with (
            patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager),
            patch(TASK_MANAGER_PATCH, return_value=mock_task_mgr),
        ):
            result = rag_tools["rag_update"](source_name="ermrest-docs")

        assert result["status"] == "started"
        assert "ermrest-docs" in result["message"]

    def test_update_rag_not_initialized(self, rag_tools):
        """rag_update returns error when RAG manager is not initialized."""
        with patch(RAG_MANAGER_PATCH, return_value=None):
            result = rag_tools["rag_update"]()

        assert "error" in result

    def test_update_task_has_correct_type(self, rag_tools, mock_rag_manager):
        """rag_update creates task with RAG_UPDATE type."""
        from deriva_mcp.tasks import TaskType

        mock_task_mgr = _make_mock_task_manager()

        with (
            patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager),
            patch(TASK_MANAGER_PATCH, return_value=mock_task_mgr),
        ):
            rag_tools["rag_update"]()

        call_kwargs = mock_task_mgr.create_task.call_args.kwargs
        assert call_kwargs["task_type"] == TaskType.RAG_UPDATE


# =============================================================================
# TestRagStatus
# =============================================================================


class TestRagStatus:
    """Tests for the rag_status tool."""

    def test_status_returns_manager_status(self, rag_tools, mock_rag_manager):
        """rag_status delegates to manager.get_status()."""
        mock_rag_manager.get_status.return_value = {
            "initialized": True,
            "total_chunks": 1500,
            "sources": [
                {"name": "deriva-ml-docs", "chunk_count": 800},
                {"name": "ermrest-docs", "chunk_count": 700},
            ],
            "catalog_schemas": [
                {"name": "schema:localhost:6", "chunk_count": 12},
            ],
        }

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_status"]()

        assert result["initialized"] is True
        assert result["total_chunks"] == 1500
        assert len(result["sources"]) == 2
        assert len(result["catalog_schemas"]) == 1
        mock_rag_manager.get_status.assert_called_once()

    def test_status_rag_not_initialized(self, rag_tools):
        """rag_status returns error when RAG manager is not initialized."""
        with patch(RAG_MANAGER_PATCH, return_value=None):
            result = rag_tools["rag_status"]()

        assert "error" in result


# =============================================================================
# TestRagAddSource
# =============================================================================


class TestRagAddSource:
    """Tests for the rag_add_source tool."""

    def test_add_source_defaults(self, rag_tools, mock_rag_manager):
        """rag_add_source passes all arguments to manager.add_source."""
        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_add_source"](
                name="my-docs",
                repo_owner="org",
                repo_name="repo",
            )

        assert result["status"] == "created"
        mock_rag_manager.add_source.assert_called_once_with(
            name="my-docs",
            repo_owner="org",
            repo_name="repo",
            branch="main",
            path_prefix="docs/",
            include_patterns=None,
            doc_type="user-guide",
        )

    def test_add_source_custom_args(self, rag_tools, mock_rag_manager):
        """rag_add_source passes custom arguments correctly."""
        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_add_source"](
                name="api-docs",
                repo_owner="informatics-isi-edu",
                repo_name="ermrest",
                branch="develop",
                path_prefix="api/",
                include_patterns=["*.md", "*.rst"],
                doc_type="api-reference",
            )

        mock_rag_manager.add_source.assert_called_once_with(
            name="api-docs",
            repo_owner="informatics-isi-edu",
            repo_name="ermrest",
            branch="develop",
            path_prefix="api/",
            include_patterns=["*.md", "*.rst"],
            doc_type="api-reference",
        )

    def test_add_source_duplicate(self, rag_tools, mock_rag_manager):
        """rag_add_source returns error when source already exists."""
        mock_rag_manager.add_source.return_value = {
            "error": "Source already exists: my-docs",
        }

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_add_source"](
                name="my-docs",
                repo_owner="org",
                repo_name="repo",
            )

        assert "error" in result
        assert "already exists" in result["error"]

    def test_add_source_rag_not_initialized(self, rag_tools):
        """rag_add_source returns error when RAG manager is not initialized."""
        with patch(RAG_MANAGER_PATCH, return_value=None):
            result = rag_tools["rag_add_source"](
                name="my-docs",
                repo_owner="org",
                repo_name="repo",
            )

        assert "error" in result


# =============================================================================
# TestRagRemoveSource
# =============================================================================


class TestRagRemoveSource:
    """Tests for the rag_remove_source tool."""

    def test_remove_source_success(self, rag_tools, mock_rag_manager):
        """rag_remove_source delegates to manager.remove_source."""
        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_remove_source"](name="test-source")

        assert result["status"] == "removed"
        assert result["chunks_deleted"] == 10
        mock_rag_manager.remove_source.assert_called_once_with("test-source")

    def test_remove_source_not_found(self, rag_tools, mock_rag_manager):
        """rag_remove_source returns error when source not found."""
        mock_rag_manager.remove_source.return_value = {
            "error": "Source not found: nonexistent",
        }

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_remove_source"](name="nonexistent")

        assert "error" in result
        assert "not found" in result["error"]

    def test_remove_source_rag_not_initialized(self, rag_tools):
        """rag_remove_source returns error when RAG manager is not initialized."""
        with patch(RAG_MANAGER_PATCH, return_value=None):
            result = rag_tools["rag_remove_source"](name="test")

        assert "error" in result


# =============================================================================
# TestRagIndexSchema
# =============================================================================


class TestRagIndexSchema:
    """Tests for the rag_index_schema tool."""

    def test_index_schema_success(self, rag_tools, mock_rag_manager, mock_conn_manager):
        """rag_index_schema fetches schema and vocab terms, then indexes."""
        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_index_schema"]()

        assert result["status"] == "indexed"
        assert result["chunks_created"] == 5

        # Verify index_catalog_schema was called with correct args
        call_args = mock_rag_manager.index_catalog_schema.call_args
        schema_info = call_args.args[0]
        assert "schemas" in schema_info
        hostname = call_args.args[1]
        assert hostname == "test.example.org"
        catalog_id = call_args.args[2]
        assert catalog_id == "1"

    def test_index_schema_includes_vocab_terms(self, rag_tools, mock_rag_manager, mock_conn_manager):
        """rag_index_schema fetches vocabulary terms with synonyms."""
        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            rag_tools["rag_index_schema"]()

        call_kwargs = mock_rag_manager.index_catalog_schema.call_args.kwargs
        vocab_terms = call_kwargs.get("vocabulary_terms", {})
        assert "Diagnosis" in vocab_terms
        terms = vocab_terms["Diagnosis"]
        assert len(terms) == 1
        assert terms[0]["Name"] == "Normal"
        assert terms[0]["Description"] == "No pathology"
        assert terms[0]["Synonyms"] == ["Healthy", "NL"]

    def test_index_schema_skips_non_vocab_tables(self, rag_tools, mock_rag_manager, mock_conn_manager):
        """rag_index_schema only fetches terms for vocabulary tables."""
        conn_info = mock_conn_manager.get_active_connection_info()
        ml = conn_info.ml_instance

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            rag_tools["rag_index_schema"]()

        # list_vocabulary_terms should be called once (for Diagnosis) not twice (not for Image)
        ml.list_vocabulary_terms.assert_called_once_with("Diagnosis")

    def test_index_schema_handles_vocab_error(self, rag_tools, mock_rag_manager, mock_conn_manager):
        """rag_index_schema silently skips vocab tables that fail to read."""
        conn_info = mock_conn_manager.get_active_connection_info()
        conn_info.ml_instance.list_vocabulary_terms.side_effect = Exception("Permission denied")

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_index_schema"]()

        # Should still succeed — vocab errors are caught
        assert result["status"] == "indexed"
        # Vocab terms should be empty
        call_kwargs = mock_rag_manager.index_catalog_schema.call_args.kwargs
        assert call_kwargs.get("vocabulary_terms") == {}

    def test_index_schema_not_connected(self, rag_tools_disconnected, mock_rag_manager):
        """rag_index_schema returns error when not connected to a catalog."""
        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools_disconnected["rag_index_schema"]()

        assert "error" in result
        assert "No active catalog" in result["error"]

    def test_index_schema_rag_not_initialized(self, rag_tools):
        """rag_index_schema returns error when RAG manager is not initialized."""
        with patch(RAG_MANAGER_PATCH, return_value=None):
            result = rag_tools["rag_index_schema"]()

        assert "error" in result
        assert "not initialized" in result["error"]

    def test_index_schema_unchanged(self, rag_tools, mock_rag_manager, mock_conn_manager):
        """rag_index_schema returns 'unchanged' when schema hash matches."""
        mock_rag_manager.index_catalog_schema.return_value = {
            "source": "schema:test.example.org:1",
            "status": "unchanged",
            "schema_hash": "abc123",
        }

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = rag_tools["rag_index_schema"]()

        assert result["status"] == "unchanged"

    def test_index_schema_vocab_term_serialization(self, rag_tools, mock_rag_manager, mock_conn_manager):
        """rag_index_schema properly converts vocab term objects to dicts."""
        conn_info = mock_conn_manager.get_active_connection_info()
        ml = conn_info.ml_instance

        # Return a term with None synonyms to test edge case
        mock_term_none_syns = MagicMock()
        mock_term_none_syns.name = "Abnormal"
        mock_term_none_syns.description = None
        mock_term_none_syns.synonyms = None
        ml.list_vocabulary_terms.return_value = [mock_term_none_syns]

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            rag_tools["rag_index_schema"]()

        call_kwargs = mock_rag_manager.index_catalog_schema.call_args.kwargs
        vocab_terms = call_kwargs["vocabulary_terms"]
        terms = vocab_terms["Diagnosis"]
        assert terms[0]["Name"] == "Abnormal"
        assert terms[0]["Description"] == ""
        assert terms[0]["Synonyms"] == []

    def test_index_schema_real_vocab_term_objects(self, mock_rag_manager):
        """rag_index_schema correctly serializes real VocabularyTerm objects."""
        from deriva_ml.core.ermrest import VocabularyTerm

        from deriva_mcp.tools.rag import register_rag_tools

        # Create real VocabularyTerm instances
        term_with_syns = VocabularyTerm(
            Name="Normal",
            Synonyms=["Healthy", "NL"],
            Description="No pathology detected",
            ID="DIAG:001",
            URI="http://example.org/DIAG/001",
            RID="1-AAAA",
        )
        term_no_syns = VocabularyTerm(
            Name="Abnormal",
            Synonyms=None,
            Description="Pathology present",
            ID="DIAG:002",
            URI="http://example.org/DIAG/002",
            RID="1-BBBB",
        )
        term_empty_desc = VocabularyTerm(
            Name="Unknown",
            Synonyms=[],
            Description=None,
            ID="DIAG:003",
            URI="http://example.org/DIAG/003",
            RID="1-CCCC",
        )

        conn_manager = MagicMock()
        conn_info = MagicMock()
        conn_info.hostname = "test.example.org"
        conn_info.catalog_id = "1"
        conn_info.ml_instance = MagicMock()
        conn_info.ml_instance.model.get_schema_description.return_value = {
            "schemas": {
                "isa": {
                    "tables": {
                        "Diagnosis": {"is_vocabulary": True, "columns": [], "foreign_keys": []},
                    }
                }
            },
        }
        conn_info.ml_instance.list_vocabulary_terms.return_value = [
            term_with_syns, term_no_syns, term_empty_desc,
        ]
        conn_manager.get_active_connection_info.return_value = conn_info

        mcp, tools = _create_tool_capture()
        register_rag_tools(mcp, conn_manager)

        with patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager):
            result = tools["rag_index_schema"]()

        assert result["status"] == "indexed"

        # Verify vocab terms were properly serialized from real objects
        call_kwargs = mock_rag_manager.index_catalog_schema.call_args.kwargs
        vocab_terms = call_kwargs["vocabulary_terms"]["Diagnosis"]

        assert len(vocab_terms) == 3

        # Term with synonyms
        assert vocab_terms[0] == {
            "Name": "Normal",
            "Description": "No pathology detected",
            "Synonyms": ["Healthy", "NL"],
        }

        # Term with None synonyms → empty list
        assert vocab_terms[1] == {
            "Name": "Abnormal",
            "Description": "Pathology present",
            "Synonyms": [],
        }

        # Term with None description → empty string, empty synonyms
        assert vocab_terms[2] == {
            "Name": "Unknown",
            "Description": "",
            "Synonyms": [],
        }

        # All term dicts must be JSON-serializable
        import json
        json.dumps(vocab_terms)  # Would raise TypeError if not serializable

    def test_all_return_values_json_serializable(self, mock_rag_manager):
        """All RAG tool return values must be JSON-serializable dicts."""
        import json

        from deriva_mcp.tools.rag import register_rag_tools

        conn_manager = MagicMock()
        conn_info = MagicMock()
        conn_info.hostname = "test.example.org"
        conn_info.catalog_id = "1"
        conn_info.ml_instance = MagicMock()
        conn_info.ml_instance.model.get_schema_description.return_value = {
            "schemas": {"isa": {"tables": {}}},
        }
        conn_manager.get_active_connection_info.return_value = conn_info

        mcp, tools = _create_tool_capture()
        register_rag_tools(mcp, conn_manager)

        mock_task_mgr = _make_mock_task_manager()
        mock_rag_manager.search.return_value = _make_search_results(2)

        with (
            patch(RAG_MANAGER_PATCH, return_value=mock_rag_manager),
            patch(TASK_MANAGER_PATCH, return_value=mock_task_mgr),
        ):
            # Test each tool returns JSON-serializable output
            results = [
                tools["rag_search"](query="test"),
                tools["rag_status"](),
                tools["rag_add_source"](name="x", repo_owner="o", repo_name="r"),
                tools["rag_remove_source"](name="test-source"),
                tools["rag_ingest"](),
                tools["rag_update"](),
                tools["rag_index_schema"](),
            ]

            for result in results:
                assert isinstance(result, dict), f"Expected dict, got {type(result)}"
                # Must not raise TypeError
                serialized = json.dumps(result)
                assert isinstance(serialized, str)
