"""Tests for catalog schema RAG indexing."""

import pytest

from deriva_mcp.rag.schema import (
    _schema_hash,
    _table_to_markdown,
    schema_source_name,
    schema_to_markdown,
)


SAMPLE_TABLE_INFO = {
    "comment": "Images captured during experiments",
    "is_vocabulary": False,
    "is_asset": True,
    "is_association": False,
    "columns": [
        {"name": "Filename", "type": "text", "nullok": False, "comment": "Original filename"},
        {"name": "Width", "type": "int4", "nullok": True, "comment": ""},
        {"name": "Height", "type": "int4", "nullok": True, "comment": ""},
    ],
    "foreign_keys": [
        {
            "columns": ["Subject"],
            "referenced_table": "isa.Subject",
            "referenced_columns": ["RID"],
        }
    ],
    "features": [
        {"name": "Image_Classification", "feature_table": "Image_Classification"},
    ],
}

SAMPLE_VOCAB_TABLE_INFO = {
    "comment": "Diagnosis terms",
    "is_vocabulary": True,
    "is_asset": False,
    "is_association": False,
    "columns": [
        {"name": "Name", "type": "text", "nullok": False, "comment": "Term name"},
        {"name": "Description", "type": "text", "nullok": True, "comment": ""},
    ],
    "foreign_keys": [],
}

SAMPLE_SCHEMA_INFO = {
    "domain_schemas": ["isa"],
    "default_schema": "isa",
    "ml_schema": "deriva-ml",
    "schemas": {
        "isa": {
            "tables": {
                "Image": SAMPLE_TABLE_INFO,
                "Diagnosis": SAMPLE_VOCAB_TABLE_INFO,
            }
        },
        "deriva-ml": {
            "tables": {
                "Dataset": {
                    "comment": "ML datasets",
                    "is_vocabulary": False,
                    "is_asset": False,
                    "is_association": False,
                    "columns": [
                        {"name": "Description", "type": "text", "nullok": True, "comment": ""},
                    ],
                    "foreign_keys": [],
                }
            }
        },
    },
}


class TestSchemaSourceName:
    def test_format(self):
        assert schema_source_name("dev.example.org", "42") == "schema:dev.example.org:42"

    def test_int_catalog_id(self):
        assert schema_source_name("localhost", 6) == "schema:localhost:6"


class TestSchemaHash:
    def test_deterministic(self):
        h1 = _schema_hash(SAMPLE_SCHEMA_INFO)
        h2 = _schema_hash(SAMPLE_SCHEMA_INFO)
        assert h1 == h2

    def test_different_on_change(self):
        modified = {**SAMPLE_SCHEMA_INFO, "default_schema": "other"}
        assert _schema_hash(SAMPLE_SCHEMA_INFO) != _schema_hash(modified)

    def test_length(self):
        h = _schema_hash(SAMPLE_SCHEMA_INFO)
        assert len(h) == 16  # sha256[:16]


class TestTableToMarkdown:
    def test_contains_table_name(self):
        md = _table_to_markdown("isa", "Image", SAMPLE_TABLE_INFO)
        assert "## isa.Image" in md

    def test_asset_tag(self):
        md = _table_to_markdown("isa", "Image", SAMPLE_TABLE_INFO)
        assert "(asset)" in md

    def test_vocabulary_tag(self):
        md = _table_to_markdown("isa", "Diagnosis", SAMPLE_VOCAB_TABLE_INFO)
        assert "(vocabulary)" in md

    def test_columns_section(self):
        md = _table_to_markdown("isa", "Image", SAMPLE_TABLE_INFO)
        assert "### Columns" in md
        assert "**Filename**" in md
        assert "NOT NULL" in md
        assert "Original filename" in md

    def test_foreign_keys_section(self):
        md = _table_to_markdown("isa", "Image", SAMPLE_TABLE_INFO)
        assert "### Foreign Keys" in md
        assert "Subject" in md
        assert "isa.Subject" in md

    def test_features_section(self):
        md = _table_to_markdown("isa", "Image", SAMPLE_TABLE_INFO)
        assert "### Features" in md
        assert "Image_Classification" in md

    def test_comment_included(self):
        md = _table_to_markdown("isa", "Image", SAMPLE_TABLE_INFO)
        assert "Images captured during experiments" in md

    def test_no_features_section_when_empty(self):
        md = _table_to_markdown("isa", "Diagnosis", SAMPLE_VOCAB_TABLE_INFO)
        assert "### Features" not in md


class TestTableToMarkdownWithVocabTerms:
    def test_vocab_terms_included(self):
        terms = [
            {"Name": "Normal", "Description": "No pathology detected"},
            {"Name": "Abnormal", "Description": "Pathology present"},
        ]
        md = _table_to_markdown("isa", "Diagnosis", SAMPLE_VOCAB_TABLE_INFO, terms)
        assert "### Terms" in md
        assert "**Normal**" in md
        assert "No pathology detected" in md
        assert "**Abnormal**" in md

    def test_no_terms_section_without_terms(self):
        md = _table_to_markdown("isa", "Image", SAMPLE_TABLE_INFO, None)
        assert "### Terms" not in md

    def test_term_without_description(self):
        terms = [{"Name": "Unknown", "Description": ""}]
        md = _table_to_markdown("isa", "Diagnosis", SAMPLE_VOCAB_TABLE_INFO, terms)
        assert "**Unknown**" in md
        assert "—" not in md.split("### Terms")[1]  # no dash for empty description


class TestSchemaToMarkdown:
    def test_contains_header(self):
        md = schema_to_markdown(SAMPLE_SCHEMA_INFO)
        assert "# Catalog Schema" in md

    def test_contains_domain_schemas(self):
        md = schema_to_markdown(SAMPLE_SCHEMA_INFO)
        assert "isa" in md

    def test_contains_all_tables(self):
        md = schema_to_markdown(SAMPLE_SCHEMA_INFO)
        assert "Image" in md
        assert "Diagnosis" in md
        assert "Dataset" in md

    def test_vocab_terms_in_full_schema(self):
        vocab = {"Diagnosis": [
            {"Name": "Normal", "Description": "Healthy"},
            {"Name": "Glaucoma", "Description": "Optic nerve damage"},
        ]}
        md = schema_to_markdown(SAMPLE_SCHEMA_INFO, vocabulary_terms=vocab)
        assert "Normal" in md
        assert "Glaucoma" in md
        assert "Optic nerve damage" in md
        # Non-vocabulary tables should NOT have terms
        assert md.count("### Terms") == 1  # only Diagnosis

    def test_chunks_properly(self):
        """Schema markdown should be chunkable."""
        from deriva_mcp.rag.chunker import chunk_markdown

        md = schema_to_markdown(SAMPLE_SCHEMA_INFO)
        chunks = chunk_markdown(md)
        assert len(chunks) > 0
        # All chunks should have text
        for chunk in chunks:
            assert chunk.text.strip()
