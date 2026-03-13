"""Tests for the RAG documentation service."""

import pytest

from deriva_mcp.rag.chunker import Chunk, chunk_markdown
from deriva_mcp.rag.config import DEFAULT_SOURCES, RAGConfig, SourceConfig


class TestSourceConfig:
    """Tests for SourceConfig serialization."""

    def test_round_trip(self):
        source = SourceConfig(
            name="test-source",
            repo_owner="org",
            repo_name="repo",
            branch="main",
            path_prefix="docs/",
            doc_type="user-guide",
        )
        data = source.to_dict()
        restored = SourceConfig.from_dict(data)
        assert restored.name == source.name
        assert restored.repo_owner == source.repo_owner
        assert restored.branch == source.branch
        assert restored.last_indexed_sha is None

    def test_default_sources_defined(self):
        assert len(DEFAULT_SOURCES) == 4
        names = {s.name for s in DEFAULT_SOURCES}
        assert "deriva-ml-docs" in names
        assert "ermrest-docs" in names
        assert "chaise-docs" in names
        assert "deriva-py-docs" in names


class TestRAGConfig:
    """Tests for RAGConfig defaults."""

    def test_defaults(self):
        config = RAGConfig()
        assert config.collection_name == "deriva_docs"
        assert config.chunk_size_target == 800
        assert config.default_search_limit == 10
        assert str(config.chroma_dir).endswith("rag/chroma")


class TestChunker:
    """Tests for markdown-aware chunking."""

    def test_empty_content(self):
        assert chunk_markdown("") == []
        assert chunk_markdown("   ") == []

    def test_single_section(self):
        content = "# Title\n\nSome content here."
        chunks = chunk_markdown(content)
        assert len(chunks) == 1
        assert "Some content" in chunks[0].text

    def test_heading_split(self):
        content = """# Title

Introduction paragraph.

## Section One

Content of section one.

## Section Two

Content of section two.
"""
        chunks = chunk_markdown(content)
        assert len(chunks) >= 2
        # Each section should be in a separate chunk
        texts = [c.text for c in chunks]
        assert any("Section One" in t or "section one" in t for t in texts)
        assert any("Section Two" in t or "section two" in t for t in texts)

    def test_code_blocks_preserved(self):
        content = """# Title

## Code Example

Here is some code:

```python
## This is NOT a heading
def foo():
    return "bar"
```

After the code block.
"""
        chunks = chunk_markdown(content)
        # The "## This is NOT a heading" inside the code block should NOT cause a split
        code_chunk = None
        for c in chunks:
            if "def foo():" in c.text:
                code_chunk = c
                break
        assert code_chunk is not None
        assert "## This is NOT a heading" in code_chunk.text

    def test_heading_hierarchy(self):
        content = """# Main Title

## Chapter One

### Sub Section

Content here.
"""
        chunks = chunk_markdown(content)
        # Find the chunk with "Sub Section"
        sub_chunk = None
        for c in chunks:
            if "Sub Section" in c.section_heading or "Content here" in c.text:
                sub_chunk = c
                break
        if sub_chunk:
            assert len(sub_chunk.heading_hierarchy) > 0

    def test_chunk_index_sequential(self):
        content = """# Title

## Section 1
Content 1.

## Section 2
Content 2.

## Section 3
Content 3.
"""
        chunks = chunk_markdown(content)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_large_section_splits(self):
        # Create a section with many paragraphs that exceeds the target
        paragraphs = ["Paragraph " + str(i) + ". " + "word " * 50 for i in range(20)]
        content = "# Title\n\n## Big Section\n\n" + "\n\n".join(paragraphs)
        chunks = chunk_markdown(content, chunk_size_target=200)
        # Should be split into multiple chunks
        assert len(chunks) > 1

    def test_estimated_tokens(self):
        chunk = Chunk(text="hello world foo bar baz", chunk_index=0)
        # 5 words * 1.3 = ~6-7 tokens
        assert chunk.estimated_tokens > 0
        assert chunk.estimated_tokens < 20

    def test_no_split_on_h4_and_deeper(self):
        content = """# Title

## Section

#### Deep heading

Content under deep heading.
"""
        chunks = chunk_markdown(content)
        # h4 should NOT cause a split (only h1-h3 do)
        # So "Deep heading" and "Content under deep heading" should be in same chunk as "Section"
        assert len(chunks) <= 2
