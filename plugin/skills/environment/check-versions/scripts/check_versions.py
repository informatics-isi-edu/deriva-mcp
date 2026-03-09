#!/usr/bin/env python3
"""Check whether deriva-ml and deriva-mcp are up to date.

Compares installed package versions against the latest git tags on GitHub.
Outputs a JSON report with status for each component.

Usage:
    python check_versions.py [--json]

Exit codes:
    0 - All components up to date
    1 - One or more components are outdated
    2 - Error checking versions
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict


@dataclass
class VersionStatus:
    component: str
    installed: str | None
    latest: str | None
    up_to_date: bool | None
    message: str


def get_installed_version(package: str) -> str | None:
    """Get the installed version of a Python package using importlib.metadata."""
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c",
             f"from importlib.metadata import version; print(version('{package}'))"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_latest_git_tag(repo: str) -> str | None:
    """Get the latest semver tag from a GitHub repository.

    Args:
        repo: GitHub repo in 'owner/name' format.
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--tags", "--sort=-v:refname",
             f"https://github.com/{repo}.git", "v*"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse tags from ls-remote output
            for line in result.stdout.strip().split("\n"):
                ref = line.split("\t")[1] if "\t" in line else ""
                # Skip ^{} dereferenced tags
                if ref.endswith("^{}"):
                    continue
                tag = ref.replace("refs/tags/", "")
                # Match semver pattern
                if re.match(r"v\d+\.\d+\.\d+$", tag):
                    return tag
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def extract_base_version(version_str: str) -> str:
    """Extract the base semver version from a setuptools_scm version string.

    Examples:
        '1.18.0' -> '1.18.0'
        '1.17.17.post10+g95607b8ec' -> '1.17.17'
        '0.5.0.dev0' -> '0.5.0'
    """
    match = re.match(r"(\d+\.\d+\.\d+)", version_str)
    return match.group(1) if match else version_str


def is_dev_version(version_str: str) -> bool:
    """Check if a version string indicates a dev/post-release version."""
    return ".post" in version_str or ".dev" in version_str or "+" in version_str


def parse_semver(version: str) -> tuple[int, ...]:
    """Parse a semver string (with optional v prefix) into a tuple of ints."""
    version = version.lstrip("v")
    return tuple(int(x) for x in version.split("."))


def check_deriva_ml() -> VersionStatus:
    """Check if deriva-ml is up to date."""
    installed = get_installed_version("deriva_ml")
    if not installed:
        return VersionStatus("deriva-ml", None, None, None,
                             "Not installed")

    latest_tag = get_latest_git_tag("informatics-isi-edu/deriva-ml")
    if not latest_tag:
        return VersionStatus("deriva-ml", installed, None, None,
                             "Could not fetch latest version from GitHub")

    base_installed = extract_base_version(installed)
    latest_version = latest_tag.lstrip("v")

    try:
        installed_tuple = parse_semver(base_installed)
        latest_tuple = parse_semver(latest_version)
    except (ValueError, IndexError):
        return VersionStatus("deriva-ml", installed, latest_tag, None,
                             "Could not compare versions")

    if installed_tuple >= latest_tuple and not is_dev_version(installed):
        return VersionStatus("deriva-ml", installed, latest_tag, True,
                             "Up to date")
    elif installed_tuple >= latest_tuple:
        return VersionStatus("deriva-ml", installed, latest_tag, True,
                             f"Dev version (based on {base_installed}, latest release is {latest_tag})")
    else:
        return VersionStatus("deriva-ml", installed, latest_tag, False,
                             f"Outdated: installed {base_installed}, latest is {latest_tag}. "
                             f"Run: uv lock --upgrade-package deriva-ml && uv sync")


def check_deriva_mcp() -> VersionStatus:
    """Check if deriva-mcp is up to date."""
    # Try to read the MCP server version resource
    # The MCP server exposes deriva-ml://server/version
    # But we can also check the installed package version
    installed = get_installed_version("deriva_mcp")
    if not installed:
        # Try old package name
        installed = get_installed_version("deriva_ml_mcp")

    if not installed:
        return VersionStatus("deriva-mcp", None, None, None,
                             "Not installed (MCP server may be running externally)")

    latest_tag = get_latest_git_tag("informatics-isi-edu/deriva-mcp")
    if not latest_tag:
        return VersionStatus("deriva-mcp", installed, None, None,
                             "Could not fetch latest version from GitHub")

    base_installed = extract_base_version(installed)
    latest_version = latest_tag.lstrip("v")

    try:
        installed_tuple = parse_semver(base_installed)
        latest_tuple = parse_semver(latest_version)
    except (ValueError, IndexError):
        return VersionStatus("deriva-mcp", installed, latest_tag, None,
                             "Could not compare versions")

    if installed_tuple >= latest_tuple and not is_dev_version(installed):
        return VersionStatus("deriva-mcp", installed, latest_tag, True,
                             "Up to date")
    elif installed_tuple >= latest_tuple:
        return VersionStatus("deriva-mcp", installed, latest_tag, True,
                             f"Dev version (based on {base_installed}, latest release is {latest_tag})")
    else:
        return VersionStatus("deriva-mcp", installed, latest_tag, False,
                             f"Outdated: installed {base_installed}, latest is {latest_tag}. "
                             f"Update the deriva-mcp repository and rebuild.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check deriva ecosystem versions")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = [check_deriva_ml(), check_deriva_mcp()]

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        any_outdated = False
        for r in results:
            status = "UP TO DATE" if r.up_to_date else ("OUTDATED" if r.up_to_date is False else "UNKNOWN")
            if r.up_to_date is False:
                any_outdated = True
            print(f"  {r.component}: {status}")
            print(f"    Installed: {r.installed or 'N/A'}")
            print(f"    Latest:    {r.latest or 'N/A'}")
            print(f"    {r.message}")
            print()

        if any_outdated:
            print("Some components are outdated. See messages above for update instructions.")

    outdated = any(r.up_to_date is False for r in results)
    return 1 if outdated else 0


if __name__ == "__main__":
    sys.exit(main())
