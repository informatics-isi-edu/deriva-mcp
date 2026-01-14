"""MCP Tools for DerivaML development utilities.

This module provides tools for development workflow operations like
version management and Jupyter kernel installation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from deriva_ml_mcp.connection import ConnectionManager


def register_devtools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register development utility tools with the MCP server."""

    # =========================================================================
    # Version Management Tools
    # =========================================================================

    @mcp.tool()
    def bump_version(
        bump_type: str = "patch",
        start_version: str = "0.1.0",
        prefix: str = "v",
    ) -> str:
        """Bump the semantic version of the current repository and push to remote.

        This tool manages semantic versioning using git tags. It either seeds an
        initial version tag if none exists, or bumps the existing version using
        bump-my-version.

        **Semantic Versioning:**
        - **major**: Breaking changes (1.0.0 -> 2.0.0)
        - **minor**: New features, backward-compatible (1.0.0 -> 1.1.0)
        - **patch**: Bug fixes, backward-compatible (1.0.0 -> 1.0.1)

        **Requirements:**
        - Must be run from within a git repository
        - Repository must have at least one commit
        - `uv` and `git` must be available on PATH
        - `bump-my-version` should be configured in pyproject.toml

        **What it does:**
        1. Fetches existing tags from remote
        2. If no semver tag exists, creates initial tag (e.g., v0.1.0)
        3. If a tag exists, bumps the specified component
        4. Pushes the new tag and commits to remote

        Args:
            bump_type: Which version component to bump: "patch", "minor", or "major".
            start_version: Initial version if no tag exists (default: "0.1.0").
            prefix: Tag prefix (default: "v").

        Returns:
            JSON with status, previous_version, new_version, and any messages.

        Example:
            bump_version("patch")  # v1.0.0 -> v1.0.1
            bump_version("minor")  # v1.0.0 -> v1.1.0
            bump_version("major")  # v1.0.0 -> v2.0.0
        """
        if bump_type not in ("patch", "minor", "major"):
            return json.dumps({
                "status": "error",
                "error": f"Invalid bump_type: {bump_type}. Must be 'patch', 'minor', or 'major'."
            })

        # Check required tools
        for tool in ("git", "uv"):
            if shutil.which(tool) is None:
                return json.dumps({
                    "status": "error",
                    "error": f"Required tool '{tool}' not found on PATH."
                })

        # Check if in git repo
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            return json.dumps({
                "status": "error",
                "error": "Not inside a git repository."
            })

        # Check for commits
        try:
            subprocess.run(
                ["git", "log", "-1"],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            return json.dumps({
                "status": "error",
                "error": "No commits found. Commit something before tagging."
            })

        # Fetch tags
        try:
            subprocess.run(
                ["git", "fetch", "--tags", "--quiet"],
                check=False, capture_output=True, text=True
            )
        except Exception:
            pass  # Non-fatal

        # Find latest semver tag
        pattern = f"{prefix}[0-9]*.[0-9]*.[0-9]*"
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0", "--match", pattern],
                check=True, capture_output=True, text=True
            )
            current_tag = result.stdout.strip()
        except subprocess.CalledProcessError:
            current_tag = None

        if not current_tag:
            # Seed initial tag
            initial_tag = f"{prefix}{start_version}"
            try:
                subprocess.run(
                    ["git", "tag", initial_tag, "-m", f"Initial release {initial_tag}"],
                    check=True, capture_output=True, text=True
                )
                subprocess.run(
                    ["git", "push", "--tags"],
                    check=True, capture_output=True, text=True
                )
                return json.dumps({
                    "status": "success",
                    "action": "seeded",
                    "previous_version": None,
                    "new_version": initial_tag,
                    "message": f"No existing semver tag found. Seeded initial tag: {initial_tag}"
                })
            except subprocess.CalledProcessError as e:
                return json.dumps({
                    "status": "error",
                    "error": f"Failed to seed initial tag: {e.stderr}"
                })

        # Bump version using bump-my-version
        try:
            result = subprocess.run(
                ["uv", "run", "bump-my-version", "bump", bump_type, "--verbose"],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            return json.dumps({
                "status": "error",
                "error": f"bump-my-version failed: {e.stderr or e.stdout}"
            })

        # Push commits and tags
        try:
            subprocess.run(
                ["git", "push", "--follow-tags"],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to push: {e.stderr}"
            })

        # Get new version tag
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                check=True, capture_output=True, text=True
            )
            new_tag = result.stdout.strip()
        except subprocess.CalledProcessError:
            new_tag = "unknown"

        return json.dumps({
            "status": "success",
            "action": "bumped",
            "bump_type": bump_type,
            "previous_version": current_tag,
            "new_version": new_tag,
            "message": f"Version bumped from {current_tag} to {new_tag}"
        })

    @mcp.tool()
    def get_current_version() -> str:
        """Get the current semantic version from git tags.

        Uses git describe to find the current version based on tags.
        If running from a tagged commit, returns the clean version.
        If commits exist after the tag, returns version with distance info.

        Returns:
            JSON with version information including tag, distance from tag,
            and whether working tree is dirty.

        Example:
            get_current_version()
            # At tag: {"version": "v1.2.3", "tag": "v1.2.3", "distance": 0}
            # After tag: {"version": "v1.2.3-2-gabcdef", "tag": "v1.2.3", "distance": 2}
        """
        # Check if in git repo
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            return json.dumps({
                "status": "error",
                "error": "Not inside a git repository."
            })

        # Get full describe output
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--long", "--dirty"],
                check=True, capture_output=True, text=True
            )
            full_version = result.stdout.strip()
        except subprocess.CalledProcessError:
            return json.dumps({
                "status": "error",
                "error": "No version tags found. Run bump_version() to create initial tag."
            })

        # Get clean tag
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                check=True, capture_output=True, text=True
            )
            tag = result.stdout.strip()
        except subprocess.CalledProcessError:
            tag = None

        # Parse the full version to extract distance
        # Format: v1.2.3-0-gabcdef or v1.2.3-5-gabcdef-dirty
        dirty = full_version.endswith("-dirty")
        clean_version = full_version.replace("-dirty", "") if dirty else full_version

        # Extract distance (commits since tag)
        parts = clean_version.rsplit("-", 2)
        if len(parts) >= 2:
            try:
                distance = int(parts[-2])
            except ValueError:
                distance = 0
        else:
            distance = 0

        return json.dumps({
            "status": "success",
            "version": full_version,
            "tag": tag,
            "distance": distance,
            "dirty": dirty,
            "at_tag": distance == 0 and not dirty
        })

    # =========================================================================
    # Jupyter Kernel Tools
    # =========================================================================

    @mcp.tool()
    def install_jupyter_kernel(
        kernel_name: str | None = None,
        display_name: str | None = None,
    ) -> str:
        """Install a Jupyter kernel for the current virtual environment.

        This allows Jupyter notebooks to use the DerivaML environment with all
        its dependencies. The kernel will appear in Jupyter's kernel selector.

        **How it works:**
        1. Detects the virtual environment name from pyvenv.cfg
        2. Normalizes the name for Jupyter compatibility
        3. Registers the kernel with ipykernel
        4. The kernel becomes available in all Jupyter instances

        **Requirements:**
        - Must be run from within a virtual environment
        - ipykernel must be installed in the environment

        Args:
            kernel_name: Override the kernel directory name. If not provided,
                uses the virtual environment name (normalized).
            display_name: Override the display name shown in Jupyter. If not
                provided, uses "Python (<venv_name>)".

        Returns:
            JSON with status, kernel_name, display_name, and install location.

        Example:
            install_jupyter_kernel()
            # Uses venv name: "Python (my-ml-project)"

            install_jupyter_kernel("my-kernel", "My Custom Kernel")
            # Custom names
        """
        try:
            from ipykernel.kernelspec import install as ipykernel_install
        except ImportError:
            return json.dumps({
                "status": "error",
                "error": "ipykernel not installed. Run: uv add ipykernel"
            })

        # Check if in a virtual environment
        config_path = Path(sys.prefix) / "pyvenv.cfg"
        if not config_path.exists():
            return json.dumps({
                "status": "error",
                "error": "Not running in a virtual environment. Activate a venv first."
            })

        # Get venv name from pyvenv.cfg
        try:
            with config_path.open() as f:
                content = f.read()
                match = re.search(r"prompt *= *(?P<prompt>.*)", content)
                venv_name = match["prompt"].strip() if match else ""
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to read pyvenv.cfg: {e}"
            })

        if not venv_name:
            # Fall back to directory name
            venv_name = Path(sys.prefix).name

        # Normalize kernel name for Jupyter
        def normalize(name: str) -> str:
            name = name.strip().lower()
            name = re.sub(r"[^a-z0-9._-]+", "-", name)
            return name

        final_kernel_name = kernel_name or normalize(venv_name)
        final_display_name = display_name or f"Python ({venv_name})"

        # Install the kernel
        try:
            ipykernel_install(
                user=True,
                kernel_name=final_kernel_name,
                display_name=final_display_name,
            )
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to install kernel: {e}"
            })

        return json.dumps({
            "status": "success",
            "kernel_name": final_kernel_name,
            "display_name": final_display_name,
            "venv_name": venv_name,
            "prefix": str(sys.prefix),
            "message": f"Installed Jupyter kernel '{final_kernel_name}' with display name '{final_display_name}'"
        })

    @mcp.tool()
    def list_jupyter_kernels() -> str:
        """List all installed Jupyter kernels.

        Returns information about all kernels available to Jupyter,
        including their names, display names, and locations.

        Returns:
            JSON with list of installed kernels and their details.
        """
        try:
            from jupyter_client.kernelspec import KernelSpecManager
        except ImportError:
            return json.dumps({
                "status": "error",
                "error": "jupyter_client not installed. Run: uv add jupyter_client"
            })

        try:
            ksm = KernelSpecManager()
            specs = ksm.get_all_specs()

            kernels = []
            for name, spec_info in specs.items():
                spec = spec_info.get("spec", {})
                kernels.append({
                    "name": name,
                    "display_name": spec.get("display_name", name),
                    "language": spec.get("language", "unknown"),
                    "resource_dir": spec_info.get("resource_dir", ""),
                })

            return json.dumps({
                "status": "success",
                "kernels": kernels,
                "count": len(kernels)
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to list kernels: {e}"
            })
