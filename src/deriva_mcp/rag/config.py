"""Configuration for the RAG documentation service.

Defines source configurations for documentation repositories and
global RAG settings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("deriva-mcp")


@dataclass
class SourceConfig:
    """Configuration for a single documentation source."""

    name: str  # unique ID, e.g. "deriva-ml-docs"
    source_type: str = "github_repo"  # "github_repo" | "local_dir"
    repo_owner: str = ""
    repo_name: str = ""
    branch: str = "main"
    path_prefix: str = "docs/"
    include_patterns: list[str] = field(default_factory=lambda: ["*.md"])
    doc_type: str = "user-guide"  # metadata tag for filtering
    last_indexed_sha: str | None = None  # git tree SHA at last crawl
    last_indexed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        return {
            "name": self.name,
            "source_type": self.source_type,
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "branch": self.branch,
            "path_prefix": self.path_prefix,
            "include_patterns": self.include_patterns,
            "doc_type": self.doc_type,
            "last_indexed_sha": self.last_indexed_sha,
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceConfig:
        """Deserialize from dict."""
        last_indexed_at = None
        if data.get("last_indexed_at"):
            last_indexed_at = datetime.fromisoformat(data["last_indexed_at"])
        return cls(
            name=data["name"],
            source_type=data.get("source_type", "github_repo"),
            repo_owner=data.get("repo_owner", ""),
            repo_name=data.get("repo_name", ""),
            branch=data.get("branch", "main"),
            path_prefix=data.get("path_prefix", "docs/"),
            include_patterns=data.get("include_patterns", ["*.md"]),
            doc_type=data.get("doc_type", "user-guide"),
            last_indexed_sha=data.get("last_indexed_sha"),
            last_indexed_at=last_indexed_at,
        )


# Default documentation sources (match the 4 repos in github_docs.py)
DEFAULT_SOURCES: list[SourceConfig] = [
    SourceConfig(
        name="deriva-ml-docs",
        repo_owner="informatics-isi-edu",
        repo_name="deriva-ml",
        branch="main",
        path_prefix="docs/",
        include_patterns=["*.md"],
        doc_type="user-guide",
    ),
    SourceConfig(
        name="ermrest-docs",
        repo_owner="informatics-isi-edu",
        repo_name="ermrest",
        branch="master",
        path_prefix="docs/",
        include_patterns=["*.md"],
        doc_type="api-reference",
    ),
    SourceConfig(
        name="chaise-docs",
        repo_owner="informatics-isi-edu",
        repo_name="chaise",
        branch="master",
        path_prefix="docs/",
        include_patterns=["*.md"],
        doc_type="user-guide",
    ),
    SourceConfig(
        name="deriva-py-docs",
        repo_owner="informatics-isi-edu",
        repo_name="deriva-py",
        branch="master",
        path_prefix="docs/",
        include_patterns=["*.md"],
        doc_type="sdk-reference",
    ),
]


@dataclass
class RAGConfig:
    """Global configuration for the RAG service."""

    persist_dir: Path = field(default_factory=lambda: Path.home() / ".deriva-ml" / "rag")
    auto_update_on_start: bool = False
    collection_name: str = "deriva_docs"
    chunk_size_target: int = 800  # target tokens per chunk
    chunk_overlap_sentences: int = 1
    default_search_limit: int = 10

    @property
    def chroma_dir(self) -> Path:
        return self.persist_dir / "chroma"

    @property
    def sources_path(self) -> Path:
        return self.persist_dir / "sources.json"


def load_sources(path: Path) -> list[SourceConfig]:
    """Load source configurations from JSON file, falling back to defaults."""
    if not path.exists():
        return [SourceConfig(**s.__dict__) for s in DEFAULT_SOURCES]

    try:
        with open(path) as f:
            data = json.load(f)
        sources = [SourceConfig.from_dict(s) for s in data.get("sources", [])]
        if not sources:
            return [SourceConfig(**s.__dict__) for s in DEFAULT_SOURCES]
        return sources
    except Exception as e:
        logger.warning(f"Failed to load sources from {path}: {e}")
        return [SourceConfig(**s.__dict__) for s in DEFAULT_SOURCES]


def save_sources(path: Path, sources: list[SourceConfig]) -> None:
    """Save source configurations to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"sources": [s.to_dict() for s in sources], "updated_at": datetime.now(timezone.utc).isoformat()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
