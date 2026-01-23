"""GitHub documentation fetcher for DerivaML MCP server.

This module provides utilities to fetch documentation from GitHub repositories
with caching to avoid repeated network requests.

Supported repositories:
- deriva-ml: DerivaML library documentation
- deriva-py: Core Deriva Python SDK documentation
- ermrest: ERMrest API documentation
- chaise: Chaise UI documentation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("deriva-mcp")

# GitHub raw content base URLs for public repositories
GITHUB_REPOS = {
    "deriva-ml": "https://raw.githubusercontent.com/informatics-isi-edu/deriva-ml/main",
    "deriva-py": "https://raw.githubusercontent.com/informatics-isi-edu/deriva-py/master",
    "ermrest": "https://raw.githubusercontent.com/informatics-isi-edu/ermrest/master",
    "chaise": "https://raw.githubusercontent.com/informatics-isi-edu/chaise/master",
}

# Cache TTL in seconds (1 hour)
CACHE_TTL = 3600


@dataclass
class CacheEntry:
    """Cache entry with content and expiration time."""
    content: str
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class GitHubDocFetcher:
    """Fetches and caches documentation from GitHub repositories."""

    def __init__(self, cache_ttl: int = CACHE_TTL):
        self._cache: dict[str, CacheEntry] = {}
        self._cache_ttl = cache_ttl
        self._client = httpx.Client(timeout=30.0)

    def _get_url(self, repo: str, path: str) -> str:
        """Build the raw GitHub URL for a file."""
        base = GITHUB_REPOS.get(repo)
        if base is None:
            raise ValueError(f"Unknown repository: {repo}. Valid repos: {list(GITHUB_REPOS.keys())}")
        return f"{base}/{path}"

    def fetch(self, repo: str, path: str, fallback: str | None = None) -> str:
        """Fetch a document from GitHub with caching.

        Args:
            repo: Repository name (e.g., "deriva-ml", "chaise")
            path: Path to file within repo (e.g., "docs/user-guide/datasets.md")
            fallback: Optional fallback content if fetch fails

        Returns:
            Document content as string, or fallback/error message on failure
        """
        cache_key = f"{repo}:{path}"

        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired:
                logger.debug(f"Cache hit for {cache_key}")
                return entry.content
            else:
                logger.debug(f"Cache expired for {cache_key}")

        # Fetch from GitHub
        url = self._get_url(repo, path)
        try:
            logger.info(f"Fetching {url}")
            response = self._client.get(url)
            response.raise_for_status()
            content = response.text

            # Cache the result
            self._cache[cache_key] = CacheEntry(
                content=content,
                expires_at=time.time() + self._cache_ttl
            )

            return content

        except httpx.HTTPStatusError as e:
            error_msg = f"Failed to fetch {url}: HTTP {e.response.status_code}"
            logger.warning(error_msg)
            if fallback is not None:
                return fallback
            return f"Error: {error_msg}"

        except httpx.RequestError as e:
            error_msg = f"Failed to fetch {url}: {e}"
            logger.warning(error_msg)
            if fallback is not None:
                return fallback
            return f"Error: {error_msg}"

    def fetch_json(self, repo: str, path: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch a JSON document from GitHub with caching.

        Args:
            repo: Repository name
            path: Path to JSON file
            fallback: Optional fallback dict if fetch fails

        Returns:
            Parsed JSON as dict, or fallback/error dict on failure
        """
        import json

        content = self.fetch(repo, path)

        if content.startswith("Error:"):
            if fallback is not None:
                return fallback
            return {"error": content}

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON from {repo}:{path}: {e}"
            logger.warning(error_msg)
            if fallback is not None:
                return fallback
            return {"error": error_msg}

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def clear_expired(self) -> int:
        """Clear only expired cache entries. Returns count of cleared entries."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)


# Global fetcher instance
_fetcher: GitHubDocFetcher | None = None


def get_fetcher() -> GitHubDocFetcher:
    """Get the global GitHubDocFetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = GitHubDocFetcher()
    return _fetcher


def fetch_doc(repo: str, path: str, fallback: str | None = None) -> str:
    """Convenience function to fetch a document.

    Args:
        repo: Repository name (e.g., "deriva-ml", "chaise")
        path: Path to file within repo
        fallback: Optional fallback content if fetch fails

    Returns:
        Document content as string
    """
    return get_fetcher().fetch(repo, path, fallback)


def fetch_json_doc(repo: str, path: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convenience function to fetch a JSON document.

    Args:
        repo: Repository name
        path: Path to JSON file
        fallback: Optional fallback dict if fetch fails

    Returns:
        Parsed JSON as dict
    """
    return get_fetcher().fetch_json(repo, path, fallback)
