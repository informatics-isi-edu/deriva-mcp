"""Tests for per-user data indexing."""

import pytest
from unittest.mock import MagicMock


class TestDataSourceName:
    def test_format(self):
        from deriva_mcp.rag.data import data_source_name
        assert data_source_name("dev.example.org", "52", "user123") == "data:dev.example.org:52:user123"

    def test_default_user(self):
        from deriva_mcp.rag.data import data_source_name
        assert data_source_name("host", "1", "default_user") == "data:host:1:default_user"


class TestRecordToMarkdown:
    def test_dataset_record(self):
        from deriva_mcp.rag.data import dataset_record_to_markdown
        record = {"RID": "1-ABC", "Description": "500 annotated lung CT images", "RCT": "2026-03-10T00:00:00"}
        md = dataset_record_to_markdown(record, types=["Training"], version="0.4.0")
        assert "1-ABC" in md
        assert "500 annotated lung CT images" in md
        assert "Training" in md
        assert "0.4.0" in md

    def test_execution_record(self):
        from deriva_mcp.rag.data import execution_record_to_markdown
        record = {"RID": "2-DEF", "Description": "Third training run", "RCT": "2026-03-12T00:00:00"}
        md = execution_record_to_markdown(record, workflow_name="Lung Seg", status="Completed")
        assert "2-DEF" in md
        assert "Third training run" in md
        assert "Lung Seg" in md
