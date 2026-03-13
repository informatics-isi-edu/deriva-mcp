"""GitHub repository crawler for RAG documentation indexing.

Uses the GitHub Trees API to discover documentation files in repositories,
with change detection via tree SHA comparison.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass

import httpx

from deriva_mcp.rag.config import SourceConfig

logger = logging.getLogger("deriva-mcp")

# GitHub API base URL
GITHUB_API = "https://api.github.com"


@dataclass
class FileInfo:
    """Information about a discovered file."""

    path: str  # relative path within repo
    sha: str  # blob SHA
    size: int  # file size in bytes


@dataclass
class CrawlResult:
    """Result of crawling a repository."""

    source_name: str
    tree_sha: str  # tree SHA of the crawled commit
    files: list[FileInfo]
    added: list[str] = None  # paths added since last crawl
    modified: list[str] = None  # paths modified since last crawl (SHA changed)
    deleted: list[str] = None  # paths deleted since last crawl
    unchanged: bool = False  # True if tree SHA matches last indexed

    def __post_init__(self):
        if self.added is None:
            self.added = []
        if self.modified is None:
            self.modified = []
        if self.deleted is None:
            self.deleted = []


def _matches_patterns(path: str, patterns: list[str]) -> bool:
    """Check if a file path matches any of the include patterns."""
    filename = path.rsplit("/", 1)[-1] if "/" in path else path
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)


def crawl_repo(
    source: SourceConfig,
    previous_files: dict[str, str] | None = None,
    github_token: str | None = None,
) -> CrawlResult:
    """Crawl a GitHub repository to discover documentation files.

    Uses the GitHub Trees API for efficient single-request file discovery.

    Args:
        source: Source configuration
        previous_files: Dict of {path: sha} from previous crawl for change detection
        github_token: Optional GitHub token for rate limiting

    Returns:
        CrawlResult with discovered files and change information
    """
    if source.source_type != "github_repo":
        raise ValueError(f"crawl_repo only supports github_repo sources, got: {source.source_type}")

    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    # Get the tree for the branch (recursive=1 gets all files in one call)
    url = f"{GITHUB_API}/repos/{source.repo_owner}/{source.repo_name}/git/trees/{source.branch}?recursive=1"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Failed to crawl {source.repo_owner}/{source.repo_name}: HTTP {e.response.status_code}"
        ) from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to crawl {source.repo_owner}/{source.repo_name}: {e}") from e

    tree_sha = data.get("sha", "")

    # Check if tree is unchanged
    if tree_sha and tree_sha == source.last_indexed_sha:
        return CrawlResult(
            source_name=source.name,
            tree_sha=tree_sha,
            files=[],
            unchanged=True,
        )

    # Filter files by path prefix and include patterns
    files: list[FileInfo] = []
    current_file_shas: dict[str, str] = {}

    for item in data.get("tree", []):
        if item.get("type") != "blob":
            continue

        path = item["path"]

        # Check path prefix
        if source.path_prefix and not path.startswith(source.path_prefix):
            continue

        # Check include patterns
        if not _matches_patterns(path, source.include_patterns):
            continue

        file_info = FileInfo(
            path=path,
            sha=item["sha"],
            size=item.get("size", 0),
        )
        files.append(file_info)
        current_file_shas[path] = item["sha"]

    # Compute diffs if we have previous state
    added = []
    modified = []
    deleted = []

    if previous_files is not None:
        prev_paths = set(previous_files.keys())
        curr_paths = set(current_file_shas.keys())

        added = list(curr_paths - prev_paths)
        deleted = list(prev_paths - curr_paths)

        # Check for modified files (same path, different SHA)
        for path in curr_paths & prev_paths:
            if current_file_shas[path] != previous_files[path]:
                modified.append(path)
    else:
        # First crawl — all files are "added"
        added = list(current_file_shas.keys())

    result = CrawlResult(
        source_name=source.name,
        tree_sha=tree_sha,
        files=files,
        added=added,
        modified=modified,
        deleted=deleted,
    )

    logger.info(
        f"Crawled {source.name}: {len(files)} files "
        f"(+{len(added)} ~{len(modified)} -{len(deleted)})"
    )
    return result


def fetch_file_content(
    source: SourceConfig,
    path: str,
    github_token: str | None = None,
) -> str:
    """Fetch raw file content from GitHub.

    Args:
        source: Source configuration
        path: File path within the repository
        github_token: Optional GitHub token

    Returns:
        File content as string
    """
    url = (
        f"https://raw.githubusercontent.com/"
        f"{source.repo_owner}/{source.repo_name}/{source.branch}/{path}"
    )

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch {path} from {source.name}: {e}") from e
