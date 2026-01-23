"""MCP Tools for DerivaML development utilities.

This module provides tools for development workflow operations like
version management and Jupyter kernel installation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
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
        project_path: str | None = None,
    ) -> str:
        """Bump the semantic version of a DerivaML project and push to remote.

        Runs the `bump-version` CLI from deriva-ml to manage semantic versioning
        using git tags. This is the recommended way to version DerivaML projects
        before running significant experiments.

        **Semantic Versioning (semver.org):**

        Version format: MAJOR.MINOR.PATCH (e.g., v1.2.3)

        - **major**: Increment for incompatible/breaking API changes
          - Example: v1.0.0 → v2.0.0
          - Use when: Removing features, changing interfaces, breaking backwards compatibility

        - **minor**: Increment for new functionality in a backward-compatible manner
          - Example: v1.0.0 → v1.1.0
          - Use when: Adding new features, new model configs, new experiments

        - **patch**: Increment for backward-compatible bug fixes
          - Example: v1.0.0 → v1.0.1
          - Use when: Bug fixes, documentation updates, small tweaks

        **Dynamic Versioning with setuptools_scm:**

        DerivaML projects use setuptools_scm to derive versions from git tags:
        - At a tag: Version is clean (e.g., "1.2.3")
        - After a tag: Version includes distance (e.g., "1.2.3.post2+gabcdef")
        - Dirty tree: Adds ".dirty" suffix for uncommitted changes

        **What it does:**
        1. Fetches existing tags from remote
        2. If no semver tag exists, creates initial tag (default: v0.1.0)
        3. If a tag exists, bumps the specified component using bump-my-version
        4. Pushes the new tag and commits to remote

        **Requirements:**
        - Project must have deriva-ml installed
        - Must be a git repository with at least one commit
        - `uv` must be available on PATH

        Args:
            bump_type: Which version component to bump: "patch", "minor", or "major".
            project_path: Path to the project directory. If not provided, uses
                the current working directory.

        Returns:
            JSON with status, previous_version, new_version, and command output.

        Example:
            bump_version("patch")  # v1.0.0 -> v1.0.1 (bug fix)
            bump_version("minor")  # v1.0.0 -> v1.1.0 (new feature)
            bump_version("major")  # v1.0.0 -> v2.0.0 (breaking change)
            bump_version("patch", "/path/to/project")  # Bump specific project
        """
        if bump_type not in ("patch", "minor", "major"):
            return json.dumps(
                {"status": "error", "error": f"Invalid bump_type: {bump_type}. Must be 'patch', 'minor', or 'major'."}
            )

        # Check for uv
        if shutil.which("uv") is None:
            return json.dumps({"status": "error", "error": "Required tool 'uv' not found on PATH."})

        # Determine project directory
        if project_path:
            project_dir = Path(project_path).resolve()
            if not project_dir.exists():
                return json.dumps({"status": "error", "error": f"Project directory not found: {project_path}"})
        else:
            project_dir = Path.cwd()

        # Check for venv
        venv_path = project_dir / ".venv"
        if not venv_path.exists():
            return json.dumps(
                {"status": "error", "error": f"No .venv directory found in {project_dir}. Run 'uv sync' first."}
            )

        # Check if in git repo
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            return json.dumps({"status": "error", "error": f"Not a git repository: {project_dir}"})

        # Get current version before bump
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*.[0-9]*.[0-9]*"],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            previous_version = result.stdout.strip()
        except subprocess.CalledProcessError:
            previous_version = None

        # Run the deriva-ml bump-version CLI
        try:
            result = subprocess.run(
                ["uv", "run", "bump-version", bump_type],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"bump-version failed: {result.stderr or result.stdout}",
                        "returncode": result.returncode,
                    }
                )

        except subprocess.TimeoutExpired:
            return json.dumps({"status": "error", "error": "Command timed out after 120 seconds"})
        except Exception as e:
            return json.dumps({"status": "error", "error": f"Failed to run command: {e}"})

        # Get new version after bump
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"], cwd=project_dir, check=True, capture_output=True, text=True
            )
            new_version = result.stdout.strip()
        except subprocess.CalledProcessError:
            new_version = "unknown"

        return json.dumps(
            {
                "status": "success",
                "bump_type": bump_type,
                "previous_version": previous_version,
                "new_version": new_version,
                "project_path": str(project_dir),
                "message": f"Version bumped from {previous_version or 'none'} to {new_version}",
            }
        )

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
            subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            return json.dumps({"status": "error", "error": "Not inside a git repository."})

        # Get full describe output
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--long", "--dirty"], check=True, capture_output=True, text=True
            )
            full_version = result.stdout.strip()
        except subprocess.CalledProcessError:
            return json.dumps(
                {"status": "error", "error": "No version tags found. Run bump_version() to create initial tag."}
            )

        # Get clean tag
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"], check=True, capture_output=True, text=True
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

        return json.dumps(
            {
                "status": "success",
                "version": full_version,
                "tag": tag,
                "distance": distance,
                "dirty": dirty,
                "at_tag": distance == 0 and not dirty,
            }
        )

    # =========================================================================
    # Jupyter Kernel Tools
    # =========================================================================

    @mcp.tool()
    def install_jupyter_kernel(
        project_path: str | None = None,
    ) -> str:
        """Install a Jupyter kernel for a DerivaML project's virtual environment.

        Runs the `deriva-ml-install-kernel` command in the target project directory
        using `uv run`. This installs a Jupyter kernel that points to the project's
        virtual environment, making it available in Jupyter's kernel selector.

        **How it works:**
        1. Changes to the project directory (or uses current directory)
        2. Runs `uv run deriva-ml-install-kernel`
        3. The kernel name is derived from the venv's prompt setting
        4. The kernel becomes available in all Jupyter instances

        **Requirements:**
        - The project must have a virtual environment with deriva-ml installed
        - `uv` must be available on PATH

        Args:
            project_path: Path to the project directory containing the venv.
                If not provided, uses the current working directory.

        Returns:
            JSON with status and the command output.

        Example:
            install_jupyter_kernel("/path/to/my-ml-project")
            # Installs kernel "Python (my-ml-project)"

            install_jupyter_kernel()
            # Uses current directory
        """
        # Check for uv
        if shutil.which("uv") is None:
            return json.dumps({"status": "error", "error": "Required tool 'uv' not found on PATH."})

        # Determine project directory
        if project_path:
            project_dir = Path(project_path).resolve()
            if not project_dir.exists():
                return json.dumps({"status": "error", "error": f"Project directory not found: {project_path}"})
        else:
            project_dir = Path.cwd()

        # Check for venv
        venv_path = project_dir / ".venv"
        if not venv_path.exists():
            return json.dumps(
                {"status": "error", "error": f"No .venv directory found in {project_dir}. Run 'uv sync' first."}
            )

        # Run the deriva-ml-install-kernel command
        try:
            result = subprocess.run(
                ["uv", "run", "deriva-ml-install-kernel"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Failed to install kernel: {result.stderr or result.stdout}",
                        "returncode": result.returncode,
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "project_path": str(project_dir),
                    "output": result.stdout.strip(),
                    "message": "Jupyter kernel installed successfully",
                }
            )

        except subprocess.TimeoutExpired:
            return json.dumps({"status": "error", "error": "Command timed out after 60 seconds"})
        except Exception as e:
            return json.dumps({"status": "error", "error": f"Failed to run command: {e}"})

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
            return json.dumps({"status": "error", "error": "jupyter_client not installed. Run: uv add jupyter_client"})

        try:
            ksm = KernelSpecManager()
            specs = ksm.get_all_specs()

            kernels = []
            for name, spec_info in specs.items():
                spec = spec_info.get("spec", {})
                kernels.append(
                    {
                        "name": name,
                        "display_name": spec.get("display_name", name),
                        "language": spec.get("language", "unknown"),
                        "resource_dir": spec_info.get("resource_dir", ""),
                    }
                )

            return json.dumps({"status": "success", "kernels": kernels, "count": len(kernels)})
        except Exception as e:
            return json.dumps({"status": "error", "error": f"Failed to list kernels: {e}"})

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
            return json.dumps({"status": "error", "error": "papermill not installed. Run: uv add papermill"})

        notebook_file = Path(notebook_path)
        if not notebook_file.exists():
            return json.dumps({"status": "error", "error": f"Notebook file not found: {notebook_path}"})

        if notebook_file.suffix != ".ipynb":
            return json.dumps({"status": "error", "error": f"File must be a .ipynb notebook: {notebook_path}"})

        try:
            params = pm.inspect_notebook(notebook_file)
            parameters = []
            for name, info in params.items():
                parameters.append(
                    {
                        "name": name,
                        "type": info.get("inferred_type_name", "unknown"),
                        "default": info.get("default"),
                        "help": info.get("help", ""),
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "notebook": str(notebook_file),
                    "parameters": parameters,
                    "count": len(parameters),
                }
            )
        except Exception as e:
            return json.dumps({"status": "error", "error": f"Failed to inspect notebook: {e}"})

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
            import nbformat
            import papermill as pm
            from nbconvert import MarkdownExporter
        except ImportError as e:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Required package not installed: {e}. Run: uv add papermill nbformat nbconvert",
                }
            )

        try:
            from deriva_ml import DerivaML, ExecAssetType, MLAsset
            from deriva_ml.execution import Execution, ExecutionConfiguration, Workflow
        except ImportError:
            return json.dumps({"status": "error", "error": "deriva_ml not installed properly"})

        import tempfile

        notebook_file = Path(notebook_path).resolve()
        if not notebook_file.exists():
            return json.dumps({"status": "error", "error": f"Notebook file not found: {notebook_path}"})

        if notebook_file.suffix != ".ipynb":
            return json.dumps({"status": "error", "error": f"File must be a .ipynb notebook: {notebook_path}"})

        # Check nbstripout status
        try:
            Workflow._check_nbstrip_status()
        except Exception as e:
            return json.dumps(
                {"status": "warning", "warning": f"nbstripout check: {e}", "message": "Continuing anyway..."}
            )

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
                    return json.dumps(
                        {
                            "status": "error",
                            "error": "Notebook did not save execution metadata. Ensure notebook creates a DerivaML execution.",
                        }
                    )

                with execution_rid_path.open("r") as f:
                    execution_config = json.load(f)

                execution_rid = execution_config["execution_rid"]
                exec_hostname = execution_config["hostname"]
                exec_catalog_id = execution_config["catalog_id"]

                # Create DerivaML instance
                ml_instance = DerivaML(hostname=exec_hostname, catalog_id=exec_catalog_id, working_dir=tmpdirname)
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

                return json.dumps(
                    {
                        "status": "success",
                        "execution_rid": execution_rid,
                        "workflow_rid": workflow_rid,
                        "hostname": exec_hostname,
                        "catalog_id": exec_catalog_id,
                        "notebook_output": str(notebook_output),
                        "citation": citation,
                        "message": f"Notebook executed successfully. Execution RID: {execution_rid}",
                    }
                )

        except Exception as e:
            return json.dumps({"status": "error", "error": f"Notebook execution failed: {e}"})


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
