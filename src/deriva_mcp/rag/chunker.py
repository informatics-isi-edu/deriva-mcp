"""Markdown-aware document chunker for RAG indexing.

Splits markdown documents at heading boundaries while preserving
code blocks and tracking heading hierarchy for metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A chunk of document text with metadata."""

    text: str
    chunk_index: int
    section_heading: str = ""
    heading_hierarchy: list[str] = field(default_factory=list)
    start_line: int = 0

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (words * 1.3)."""
        return int(len(self.text.split()) * 1.3)


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    return int(len(text.split()) * 1.3)


def _find_code_block_ranges(lines: list[str]) -> set[int]:
    """Find line indices that are inside fenced code blocks."""
    inside = set()
    in_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_block:
                inside.add(i)  # closing fence is part of the block
                in_block = False
            else:
                in_block = True
                inside.add(i)  # opening fence is part of the block
        elif in_block:
            inside.add(i)
    return inside


def _is_heading(line: str) -> tuple[int, str] | None:
    """Check if a line is a markdown heading. Returns (level, title) or None."""
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if match:
        return len(match.group(1)), match.group(2).strip()
    return None


def _get_last_sentence(text: str) -> str:
    """Extract the last sentence from text for overlap."""
    text = text.strip()
    if not text:
        return ""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences:
        return sentences[-1].strip()
    return ""


def chunk_markdown(
    content: str,
    chunk_size_target: int = 800,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Split markdown content into chunks at heading boundaries.

    Rules:
    - Split at ## and ### headings (never inside fenced code blocks)
    - Target chunk_size_target tokens per chunk
    - If a section exceeds the target, split at paragraph boundaries
    - Maintain 1-sentence overlap between chunks
    - Track heading hierarchy for metadata

    Args:
        content: Raw markdown text
        chunk_size_target: Target tokens per chunk (default 800)
        overlap_sentences: Number of sentences to overlap (default 1)

    Returns:
        List of Chunk objects with text and metadata
    """
    if not content or not content.strip():
        return []

    lines = content.split("\n")
    code_blocks = _find_code_block_ranges(lines)

    # Build sections by splitting at headings
    sections: list[dict] = []
    current_section: dict = {
        "heading": "",
        "level": 0,
        "lines": [],
        "start_line": 0,
    }

    for i, line in enumerate(lines):
        if i not in code_blocks:
            heading_info = _is_heading(line)
            if heading_info and heading_info[0] <= 3:
                # Save current section if it has content
                if current_section["lines"]:
                    sections.append(current_section)
                # Start new section
                current_section = {
                    "heading": heading_info[1],
                    "level": heading_info[0],
                    "lines": [line],
                    "start_line": i,
                }
                continue

        current_section["lines"].append(line)

    # Don't forget the last section
    if current_section["lines"]:
        sections.append(current_section)

    if not sections:
        return []

    # Build heading hierarchy tracker
    heading_stack: list[str] = []
    chunks: list[Chunk] = []
    chunk_index = 0
    prev_overlap = ""

    for section in sections:
        section_text = "\n".join(section["lines"]).strip()
        if not section_text:
            continue

        # Update heading hierarchy
        level = section["level"]
        heading = section["heading"]
        if level > 0:
            # Pop headings at same or deeper level
            while heading_stack and len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(heading)

        # Check if section needs to be split further
        tokens = _estimate_tokens(section_text)
        if tokens <= chunk_size_target * 1.5:
            # Section fits in one chunk
            text = section_text
            if prev_overlap and overlap_sentences > 0:
                text = prev_overlap + "\n\n" + text

            chunks.append(
                Chunk(
                    text=text,
                    chunk_index=chunk_index,
                    section_heading=heading,
                    heading_hierarchy=list(heading_stack),
                    start_line=section["start_line"],
                )
            )
            chunk_index += 1
            if overlap_sentences > 0:
                prev_overlap = _get_last_sentence(section_text)
            else:
                prev_overlap = ""
        else:
            # Section too large — split at paragraph boundaries
            sub_chunks = _split_large_section(
                section_text,
                chunk_size_target,
                heading,
                heading_stack,
                section["start_line"],
                chunk_index,
                prev_overlap if overlap_sentences > 0 else "",
                overlap_sentences,
            )
            chunks.extend(sub_chunks)
            chunk_index += len(sub_chunks)
            if sub_chunks and overlap_sentences > 0:
                prev_overlap = _get_last_sentence(sub_chunks[-1].text)
            else:
                prev_overlap = ""

    return chunks


def _split_large_section(
    text: str,
    chunk_size_target: int,
    heading: str,
    heading_stack: list[str],
    start_line: int,
    start_index: int,
    prev_overlap: str,
    overlap_sentences: int,
) -> list[Chunk]:
    """Split a large section into smaller chunks at paragraph boundaries."""
    # Split on blank lines (paragraph boundaries)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0

    if prev_overlap:
        current_parts.append(prev_overlap)
        current_tokens = _estimate_tokens(prev_overlap)

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        # If adding this paragraph exceeds target and we already have content
        if current_tokens + para_tokens > chunk_size_target * 1.5 and current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_index=start_index + len(chunks),
                    section_heading=heading,
                    heading_hierarchy=list(heading_stack),
                    start_line=start_line,
                )
            )
            # Start new chunk with overlap
            if overlap_sentences > 0:
                overlap = _get_last_sentence(chunk_text)
                current_parts = [overlap] if overlap else []
                current_tokens = _estimate_tokens(overlap) if overlap else 0
            else:
                current_parts = []
                current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    # Remaining content
    if current_parts:
        chunks.append(
            Chunk(
                text="\n\n".join(current_parts),
                chunk_index=start_index + len(chunks),
                section_heading=heading,
                heading_hierarchy=list(heading_stack),
                start_line=start_line,
            )
        )

    return chunks
