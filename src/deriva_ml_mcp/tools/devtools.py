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

    # =========================================================================
    # Notebook Execution Tools
    # =========================================================================

    @mcp.tool()
    def inspect_notebook(notebook_path: str) -> str:
        """Inspect a Jupyter notebook to see its available parameters.

        Uses papermill to extract parameter cell information from a notebook.
        This shows what parameters can be injected when running the notebook.

        Args:
            notebook_path: Path to the notebook file (.ipynb).

        Returns:
            JSON with list of parameters, their types, and default values.

        Example:
            inspect_notebook("notebooks/train_model.ipynb")
            # Returns: {"parameters": [{"name": "learning_rate", "type": "float", "default": 0.001}, ...]}
        """
        try:
            import papermill as pm
        except ImportError:
            return json.dumps({
                "status": "error",
                "error": "papermill not installed. Run: uv add papermill"
            })

        notebook_file = Path(notebook_path)
        if not notebook_file.exists():
            return json.dumps({
                "status": "error",
                "error": f"Notebook file not found: {notebook_path}"
            })

        if notebook_file.suffix != ".ipynb":
            return json.dumps({
                "status": "error",
                "error": f"File must be a .ipynb notebook: {notebook_path}"
            })

        try:
            params = pm.inspect_notebook(notebook_file)
            parameters = []
            for name, info in params.items():
                parameters.append({
                    "name": name,
                    "type": info.get("inferred_type_name", "unknown"),
                    "default": info.get("default"),
                    "help": info.get("help", ""),
                })

            return json.dumps({
                "status": "success",
                "notebook": str(notebook_file),
                "parameters": parameters,
                "count": len(parameters)
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to inspect notebook: {e}"
            })

    @mcp.tool()
    def run_notebook(
        notebook_path: str,
        hostname: str,
        catalog_id: str,
        parameters: dict | None = None,
        kernel: str | None = None,
        log_output: bool = False,
    ) -> str:
        """Run a Jupyter notebook with DerivaML execution tracking.

        Executes a notebook using papermill while automatically tracking the
        execution in a Deriva catalog. The notebook is expected to use DerivaML's
        execution context to record its workflow.

        **What it does:**
        1. Sets environment variables for workflow provenance (URL, checksum, path)
        2. Executes the notebook with papermill, injecting parameters
        3. Converts executed notebook to Markdown format
        4. Uploads both outputs as execution assets to the catalog

        **Requirements:**
        - papermill, nbformat, nbconvert must be installed
        - The notebook should create a DerivaML execution during its run
        - nbstripout should be configured for clean notebook version control

        Args:
            notebook_path: Path to the notebook file (.ipynb).
            hostname: Deriva server hostname (e.g., "www.example.org").
            catalog_id: Catalog ID or number.
            parameters: Dictionary of parameters to inject into the notebook.
            kernel: Jupyter kernel name. If not provided, auto-detects from venv.
            log_output: If True, stream notebook cell outputs during execution.

        Returns:
            JSON with execution status, execution_rid, and output paths.

        Example:
            run_notebook(
                "notebooks/train_model.ipynb",
                "deriva.example.org",
                "42",
                parameters={"learning_rate": 0.001, "epochs": 100},
                kernel="my-ml-project"
            )
        """
        try:
            import papermill as pm
            import nbformat
            from nbconvert import MarkdownExporter
        except ImportError as e:
            return json.dumps({
                "status": "error",
                "error": f"Required package not installed: {e}. Run: uv add papermill nbformat nbconvert"
            })

        try:
            from deriva_ml import DerivaML, ExecAssetType, MLAsset
            from deriva_ml.execution import Execution, ExecutionConfiguration, Workflow
        except ImportError:
            return json.dumps({
                "status": "error",
                "error": "deriva_ml not installed properly"
            })

        import tempfile

        notebook_file = Path(notebook_path).resolve()
        if not notebook_file.exists():
            return json.dumps({
                "status": "error",
                "error": f"Notebook file not found: {notebook_path}"
            })

        if notebook_file.suffix != ".ipynb":
            return json.dumps({
                "status": "error",
                "error": f"File must be a .ipynb notebook: {notebook_path}"
            })

        # Check nbstripout status
        try:
            Workflow._check_nbstrip_status()
        except Exception as e:
            return json.dumps({
                "status": "warning",
                "warning": f"nbstripout check: {e}",
                "message": "Continuing anyway..."
            })

        # Build parameters dict
        params = parameters or {}
        params["host"] = hostname
        params["catalog"] = catalog_id

        # Auto-detect kernel if not provided
        if kernel is None:
            kernel = _find_kernel_for_venv()

        # Get workflow provenance info
        try:
            url, checksum = Workflow.get_url_and_checksum(notebook_file)
        except Exception:
            url = ""
            checksum = ""

        os.environ["DERIVA_ML_WORKFLOW_URL"] = url
        os.environ["DERIVA_ML_WORKFLOW_CHECKSUM"] = checksum
        os.environ["DERIVA_ML_NOTEBOOK_PATH"] = notebook_file.as_posix()

        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                notebook_output = Path(tmpdirname) / notebook_file.name
                execution_rid_path = Path(tmpdirname) / "execution_rid.json"
                os.environ["DERIVA_ML_SAVE_EXECUTION_RID"] = execution_rid_path.as_posix()

                # Execute the notebook
                pm.execute_notebook(
                    input_path=notebook_file,
                    output_path=notebook_output,
                    parameters=params,
                    kernel_name=kernel,
                    log_output=log_output,
                )

                # Read execution metadata
                if not execution_rid_path.exists():
                    return json.dumps({
                        "status": "error",
                        "error": "Notebook did not save execution metadata. Ensure notebook creates a DerivaML execution."
                    })

                with execution_rid_path.open("r") as f:
                    execution_config = json.load(f)

                execution_rid = execution_config["execution_rid"]
                exec_hostname = execution_config["hostname"]
                exec_catalog_id = execution_config["catalog_id"]

                # Create DerivaML instance
                ml_instance = DerivaML(
                    hostname=exec_hostname,
                    catalog_id=exec_catalog_id,
                    working_dir=tmpdirname
                )
                workflow_rid = ml_instance.retrieve_rid(execution_rid)["Workflow"]

                # Restore execution context
                execution = Execution(
                    configuration=ExecutionConfiguration(workflow=workflow_rid),
                    ml_object=ml_instance,
                    reload=execution_rid,
                )

                # Convert to Markdown
                notebook_output_md = notebook_output.with_suffix(".md")
                with notebook_output.open() as f:
                    nb = nbformat.read(f, as_version=4)
                exporter = MarkdownExporter()
                body, _ = exporter.from_notebook_node(nb)
                with notebook_output_md.open("w") as f:
                    f.write(body)

                # Register outputs
                execution.asset_file_path(
                    asset_name=MLAsset.execution_asset,
                    file_name=notebook_output,
                    asset_types=ExecAssetType.notebook_output,
                )
                execution.asset_file_path(
                    asset_name=MLAsset.execution_asset,
                    file_name=notebook_output_md,
                    asset_types=ExecAssetType.notebook_output,
                )

                # Upload outputs
                execution.upload_execution_outputs()

                # Get citation
                citation = ml_instance.cite(execution_rid)

                return json.dumps({
                    "status": "success",
                    "execution_rid": execution_rid,
                    "workflow_rid": workflow_rid,
                    "hostname": exec_hostname,
                    "catalog_id": exec_catalog_id,
                    "notebook_output": str(notebook_output),
                    "citation": citation,
                    "message": f"Notebook executed successfully. Execution RID: {execution_rid}"
                })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Notebook execution failed: {e}"
            })


def _find_kernel_for_venv() -> str | None:
    """Find a Jupyter kernel that matches the current virtual environment."""
    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except ImportError:
        return None

    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return None

    venv_path = Path(venv).resolve()
    try:
        ksm = KernelSpecManager()
        for name, spec in ksm.get_all_specs().items():
            kernel_json = spec.get("spec", {})
            argv = kernel_json.get("argv", [])
            for arg in argv:
                try:
                    if Path(arg).resolve() == venv_path.joinpath("bin", "python").resolve():
                        return name
                except Exception:
                    continue
    except Exception:
        pass
    return None
