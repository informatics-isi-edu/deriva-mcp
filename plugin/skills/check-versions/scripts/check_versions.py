#!/usr/bin/env python3
"""Check and update the DerivaML ecosystem.

Checks installed versions of deriva-ml, deriva-mcp skills plugin, and the MCP
server against the latest releases. Can optionally perform updates.

Usage:
    python check_versions.py              # Check only
    python check_versions.py --update     # Check and update outdated components
    python check_versions.py --json       # Output as JSON

Exit codes:
    0 - All components up to date (or successfully updated)
    1 - One or more components are outdated (check mode) or update failed
    2 - Error checking versions
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path


PLUGIN_CACHE_DIR = Path.home() / ".claude" / "plugins" / "cache" / "deriva-plugins" / "deriva"
GITHUB_REPO = "informatics-isi-edu/deriva-mcp"


@dataclass
class VersionStatus:
    component: str
    installed: str | None
    latest: str | None
    up_to_date: bool | None
    message: str
    update_commands: list[str] = field(default_factory=list)
    updated: bool = False
    update_message: str = ""


def run_cmd(
    cmd: list[str], capture: bool = True, timeout: int = 120, cwd: str | None = None
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(
        cmd, capture_output=capture, text=True, timeout=timeout, cwd=cwd
    )


def get_installed_version(package: str) -> str | None:
    """Get the installed version of a Python package using importlib.metadata."""
    try:
        result = run_cmd(
            ["uv", "run", "python", "-c",
             f"from importlib.metadata import version; print(version('{package}'))"],
            timeout=30,
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
        result = run_cmd(
            ["git", "ls-remote", "--tags", "--sort=-v:refname",
             f"https://github.com/{repo}.git", "v*"],
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                ref = line.split("\t")[1] if "\t" in line else ""
                if ref.endswith("^{}"):
                    continue
                tag = ref.replace("refs/tags/", "")
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


def version_is_outdated(installed: str, latest_tag: str) -> bool | None:
    """Compare installed version against latest tag. Returns True if outdated."""
    base_installed = extract_base_version(installed)
    latest_version = latest_tag.lstrip("v")
    try:
        return parse_semver(base_installed) < parse_semver(latest_version)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Component checks
# ---------------------------------------------------------------------------

def check_deriva_ml() -> VersionStatus:
    """Check if deriva-ml is up to date."""
    installed = get_installed_version("deriva_ml")
    if not installed:
        return VersionStatus("deriva-ml", None, None, None, "Not installed")

    latest_tag = get_latest_git_tag("informatics-isi-edu/deriva-ml")
    if not latest_tag:
        return VersionStatus("deriva-ml", installed, None, None,
                             "Could not fetch latest version from GitHub")

    outdated = version_is_outdated(installed, latest_tag)
    if outdated is None:
        return VersionStatus("deriva-ml", installed, latest_tag, None,
                             "Could not compare versions")

    base = extract_base_version(installed)
    if not outdated and not is_dev_version(installed):
        return VersionStatus("deriva-ml", installed, latest_tag, True, "Up to date")
    elif not outdated:
        return VersionStatus("deriva-ml", installed, latest_tag, True,
                             f"Dev version (based on {base}, latest release is {latest_tag})")
    else:
        return VersionStatus(
            "deriva-ml", installed, latest_tag, False,
            f"Outdated: installed {base}, latest is {latest_tag}",
            update_commands=["uv lock --upgrade-package deriva-ml", "uv sync"],
        )


def update_deriva_ml(status: VersionStatus) -> VersionStatus:
    """Update deriva-ml in the local venv."""
    print(f"  Updating deriva-ml to {status.latest}...")
    for cmd_str in status.update_commands:
        cmd = cmd_str.split()
        print(f"    $ {cmd_str}")
        result = run_cmd(cmd, timeout=300)
        if result.returncode != 0:
            status.update_message = f"Failed: {cmd_str}\n{result.stderr or result.stdout}"
            return status

    # Verify the update
    new_version = get_installed_version("deriva_ml")
    if new_version:
        new_base = extract_base_version(new_version)
        latest = status.latest.lstrip("v") if status.latest else ""
        if parse_semver(new_base) >= parse_semver(latest):
            status.updated = True
            status.up_to_date = True
            status.installed = new_version
            status.update_message = f"Updated to {new_version}"
        else:
            status.update_message = f"Update ran but version is still {new_version}"
    else:
        status.update_message = "Update ran but could not verify new version"
    return status


def _get_cached_plugin_version() -> str | None:
    """Get the installed plugin version from the Claude Code plugin cache.

    The plugin cache lives at ~/.claude/plugins/cache/deriva-plugins/deriva/<version>/
    with a plugin.json file containing the version metadata.
    """
    if not PLUGIN_CACHE_DIR.exists():
        return None

    # Find cached version directories (e.g., "0.1.0", "0.10.3")
    version_dirs = [d for d in PLUGIN_CACHE_DIR.iterdir() if d.is_dir()]
    if not version_dirs:
        return None

    # If multiple versions cached, pick the highest
    best_version = None
    best_tuple = (0, 0, 0)
    for d in version_dirs:
        plugin_json = d / ".claude-plugin" / "plugin.json"
        if plugin_json.exists():
            try:
                data = json.loads(plugin_json.read_text())
                ver = data.get("version", d.name)
                ver_tuple = parse_semver(ver)
                if ver_tuple > best_tuple:
                    best_version = ver
                    best_tuple = ver_tuple
            except (json.JSONDecodeError, ValueError):
                pass
        else:
            # Fall back to directory name
            try:
                ver_tuple = parse_semver(d.name)
                if ver_tuple > best_tuple:
                    best_version = d.name
                    best_tuple = ver_tuple
            except (ValueError, IndexError):
                pass

    return best_version


def check_skills() -> VersionStatus:
    """Check if the cached skills plugin is up to date.

    Compares the version in ~/.claude/plugins/cache/deriva-plugins/deriva/<ver>/
    against the latest release tag on GitHub.
    """
    installed = _get_cached_plugin_version()
    if not installed:
        return VersionStatus("skills", None, None, None,
                             "Not installed (run: /plugin install deriva)")

    latest_tag = get_latest_git_tag(GITHUB_REPO)
    if not latest_tag:
        return VersionStatus("skills", installed, None, None,
                             "Could not fetch latest version from GitHub")

    outdated = version_is_outdated(installed, latest_tag)
    if outdated is None:
        return VersionStatus("skills", installed, latest_tag, None,
                             "Could not compare versions")

    if not outdated:
        return VersionStatus("skills", installed, latest_tag, True, "Up to date")
    else:
        return VersionStatus(
            "skills", installed, latest_tag, False,
            f"Outdated: installed {installed}, latest is {latest_tag}",
            update_commands=["Tell Claude: /plugin install deriva"],
        )


def update_skills(status: VersionStatus) -> VersionStatus:
    """Update the skills plugin.

    Plugin updates require running '/plugin install deriva' in Claude Code.
    This function cannot do it programmatically, so it provides instructions.
    """
    status.update_message = (
        "Skills plugin must be updated from within Claude Code.\n"
        "    Run: /plugin install deriva"
    )
    return status


def _is_registry_image(image: str) -> bool:
    """Check if the image name looks like a registry image (ghcr.io/...)."""
    return "/" in image and ("ghcr.io" in image or "docker.io" in image
                            or "." in image.split("/")[0])


def check_mcp_server() -> VersionStatus:
    """Check if the MCP Docker container is up to date.

    Handles two deployment modes:
    - Local dev: image built from local repo (e.g., 'deriva-mcp:dev').
      Compares container creation time against latest repo commit.
    - Registry: image pulled from GHCR (e.g., 'ghcr.io/.../deriva-mcp:latest').
      Compares local image digest against remote registry digest.
    """
    # Check if the container is running
    result = run_cmd(
        ["docker", "ps", "--filter", "name=deriva-mcp", "--format", "{{.Image}}"],
    )
    if result.returncode != 0 or not result.stdout.strip():
        # No Docker container — check if MCP server runs natively
        return _check_native_mcp_server()

    image = result.stdout.strip()

    if _is_registry_image(image):
        return _check_registry_mcp_server(image)
    else:
        return _check_local_dev_mcp_server(image)


def _check_native_mcp_server() -> VersionStatus:
    """Check for a natively-running MCP server (no Docker)."""
    installed = get_installed_version("deriva_mcp")
    if not installed:
        installed = get_installed_version("deriva_ml_mcp")
    if not installed:
        return VersionStatus("mcp-server", None, None, None,
                             "No running Docker container or installed package found")

    latest_tag = get_latest_git_tag(GITHUB_REPO)
    if not latest_tag:
        return VersionStatus("mcp-server", installed, None, None,
                             "Could not fetch latest version from GitHub")

    outdated = version_is_outdated(installed, latest_tag)
    if outdated is None:
        return VersionStatus("mcp-server", installed, latest_tag, None,
                             "Could not compare versions")

    base = extract_base_version(installed)
    if not outdated:
        return VersionStatus("mcp-server", installed, latest_tag, True,
                             "Up to date")
    else:
        return VersionStatus(
            "mcp-server", installed, latest_tag, False,
            f"Outdated: installed {base}, latest is {latest_tag}",
            update_commands=["uv lock --upgrade-package deriva-mcp", "uv sync"],
        )


def _check_registry_mcp_server(image: str) -> VersionStatus:
    """Check a registry-pulled Docker image against the remote registry.

    When the deriva-mcp version is bumped, GitHub Actions builds and pushes
    a new image to ghcr.io. We compare the local image digest against the
    remote digest to detect updates.
    """
    # Get local image digest
    result = run_cmd(
        ["docker", "inspect", "--format", "{{.Id}}", image],
    )
    local_digest = result.stdout.strip() if result.returncode == 0 else None

    # Pull latest metadata from registry (doesn't download layers)
    result = run_cmd(
        ["docker", "manifest", "inspect", image],
        timeout=30,
    )
    if result.returncode != 0:
        # Fall back to checking the latest git tag
        latest_tag = get_latest_git_tag(GITHUB_REPO)
        return VersionStatus("mcp-server", image, latest_tag or "unknown", None,
                             "Could not check remote registry for updates. "
                             "Try: docker pull " + image)

    # If manifest inspect succeeds, suggest pulling to check
    # (We can't easily compare digests without pulling)
    return VersionStatus(
        "mcp-server", image, "check remote", None,
        "Registry image — run `docker pull` to check for updates",
        update_commands=[
            f"docker pull {image}",
            "docker restart deriva-mcp",
        ],
    )


def _check_local_dev_mcp_server(image: str) -> VersionStatus:
    """Check a locally-built Docker image against the repo commit history."""
    repo_dir = _find_repo_dir()
    if not repo_dir:
        return VersionStatus("mcp-server", image, None, None,
                             "Could not find deriva-mcp repo to check for updates")

    # Compare container creation time vs latest commit time
    result = run_cmd(
        ["docker", "inspect", "--format", "{{.Created}}", "deriva-mcp"],
    )
    container_created = result.stdout.strip() if result.returncode == 0 else None

    result = run_cmd(
        ["git", "log", "-1", "--format=%aI"],
        cwd=repo_dir,
    )
    latest_commit_time = result.stdout.strip() if result.returncode == 0 else None

    # Find compose file for rebuild command
    compose_file = None
    for candidate in ["docker-compose.dev.yaml", "docker-compose.mcp.yaml"]:
        if os.path.exists(os.path.join(repo_dir, candidate)):
            compose_file = candidate
            break

    if container_created and latest_commit_time:
        if latest_commit_time > container_created:
            rebuild_cmd = (
                f"docker compose -f {os.path.join(repo_dir, compose_file)} up -d --build"
                if compose_file
                else f"docker build -t {image} {repo_dir} && docker restart deriva-mcp"
            )
            return VersionStatus(
                "mcp-server", image, "repo has newer commits",
                False,
                "Container is older than latest repo commit",
                update_commands=[rebuild_cmd],
            )

    return VersionStatus("mcp-server", image, "current", True,
                         "Container appears up to date")


def _find_repo_dir() -> str | None:
    """Find the deriva-mcp git repo.

    Checks common locations since the script may run from the plugin cache
    (not inside the repo).
    """
    # First try walking up from script location (works when run from repo)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = script_dir
    for _ in range(10):
        if os.path.isdir(os.path.join(repo_dir, ".git")):
            # Verify it's actually the deriva-mcp repo
            plugin_json = os.path.join(repo_dir, "plugin", ".claude-plugin", "plugin.json")
            if os.path.exists(plugin_json):
                return repo_dir
            # It's a git repo but not deriva-mcp, keep looking
            break
        repo_dir = os.path.dirname(repo_dir)

    # Try common locations relative to home
    home = Path.home()
    for candidate in [
        home / "GitHub" / "deriva-mcp",
        home / "github" / "deriva-mcp",
        home / "src" / "deriva-mcp",
        home / "projects" / "deriva-mcp",
        home / "code" / "deriva-mcp",
    ]:
        if (candidate / ".git").is_dir():
            return str(candidate)

    return None


def update_mcp_server(status: VersionStatus) -> VersionStatus:
    """Rebuild and restart the MCP Docker container."""
    print("  Rebuilding MCP server...")
    for cmd_str in status.update_commands:
        # Use shell=True for compose commands with pipes
        print(f"    $ {cmd_str}")
        result = subprocess.run(
            cmd_str, shell=True, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            status.update_message = f"Failed: {cmd_str}\n{result.stderr or result.stdout}"
            return status

    status.updated = True
    status.up_to_date = True
    status.update_message = "Rebuilt and restarted MCP server"
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

UPDATERS = {
    "deriva-ml": update_deriva_ml,
    "skills": update_skills,
    "mcp-server": update_mcp_server,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and update DerivaML ecosystem")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--update", action="store_true",
                        help="Automatically update outdated components")
    parser.add_argument("--component", choices=["deriva-ml", "skills", "mcp-server"],
                        help="Check/update only this component")
    args = parser.parse_args()

    checks = {
        "deriva-ml": check_deriva_ml,
        "skills": check_skills,
        "mcp-server": check_mcp_server,
    }

    if args.component:
        checks = {args.component: checks[args.component]}

    results = []
    for name, check_fn in checks.items():
        status = check_fn()
        if args.update and status.up_to_date is False and status.update_commands:
            updater = UPDATERS.get(name)
            if updater:
                status = updater(status)
        results.append(status)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        any_outdated = False
        for r in results:
            if r.updated:
                label = "UPDATED"
            elif r.up_to_date is True:
                label = "UP TO DATE"
            elif r.up_to_date is False:
                label = "OUTDATED"
                any_outdated = True
            else:
                label = "UNKNOWN"

            print(f"  {r.component}: {label}")
            print(f"    Installed: {r.installed or 'N/A'}")
            print(f"    Latest:    {r.latest or 'N/A'}")
            print(f"    {r.message}")
            if r.updated:
                print(f"    -> {r.update_message}")
            elif r.up_to_date is False and r.update_commands:
                print(f"    Update: {' && '.join(r.update_commands)}")
            if r.update_message and not r.updated:
                print(f"    Note: {r.update_message}")
            print()

        if any_outdated:
            print("Some components are outdated. Run with --update to update them.")

    has_outdated = any(
        r.up_to_date is False and not r.updated for r in results
    )
    return 1 if has_outdated else 0


if __name__ == "__main__":
    sys.exit(main())
