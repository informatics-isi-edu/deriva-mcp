"""Unit tests for development utility tools (devtools)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import assert_error, assert_success, parse_json_result


# =============================================================================
# bump_version
# =============================================================================


class TestBumpVersion:
    """Tests for the bump_version tool."""

    def test_bump_patch_success(self, devtools, tmp_path):
        """Bumping patch version returns success with version info."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if cmd[:3] == ["git", "describe", "--tags"] and "--abbrev=0" in cmd:
                if "--match" in cmd:
                    # Before bump: get current version
                    result.stdout = "v1.0.0\n"
                else:
                    # After bump: get new version
                    result.stdout = "v1.0.1\n"
                return result
            if cmd[:4] == ["uv", "run", "bump-version", "patch"]:
                result.stdout = "Bumped version to v1.0.1"
                return result
            return result

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run),
        ):
            result = devtools["bump_version"](
                bump_type="patch",
                project_path=str(project_dir),
            )

        data = assert_success(result)
        assert data["bump_type"] == "patch"
        assert data["previous_version"] == "v1.0.0"
        assert data["new_version"] == "v1.0.1"
        assert data["project_path"] == str(project_dir)

    def test_bump_minor_success(self, devtools, tmp_path):
        """Bumping minor version works correctly."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if cmd[:3] == ["git", "describe", "--tags"] and "--abbrev=0" in cmd:
                if "--match" in cmd:
                    result.stdout = "v1.0.0\n"
                else:
                    result.stdout = "v1.1.0\n"
                return result
            if cmd[:4] == ["uv", "run", "bump-version", "minor"]:
                result.stdout = "Bumped version to v1.1.0"
                return result
            return result

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run),
        ):
            result = devtools["bump_version"](
                bump_type="minor",
                project_path=str(project_dir),
            )

        data = assert_success(result)
        assert data["bump_type"] == "minor"
        assert data["new_version"] == "v1.1.0"

    def test_bump_major_success(self, devtools, tmp_path):
        """Bumping major version works correctly."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if cmd[:3] == ["git", "describe", "--tags"] and "--abbrev=0" in cmd:
                if "--match" in cmd:
                    result.stdout = "v1.2.3\n"
                else:
                    result.stdout = "v2.0.0\n"
                return result
            if cmd[:4] == ["uv", "run", "bump-version", "major"]:
                result.stdout = "Bumped"
                return result
            return result

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run),
        ):
            result = devtools["bump_version"](
                bump_type="major",
                project_path=str(project_dir),
            )

        data = assert_success(result)
        assert data["bump_type"] == "major"
        assert data["previous_version"] == "v1.2.3"
        assert data["new_version"] == "v2.0.0"

    def test_bump_invalid_type(self, devtools):
        """Passing an invalid bump_type returns an error."""
        result = devtools["bump_version"](bump_type="invalid")

        data = assert_error(result)
        assert "Invalid bump_type" in data["error"]
        assert "invalid" in data["error"]

    def test_bump_uv_not_found(self, devtools):
        """When uv is not on PATH, return an error."""
        with patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value=None):
            result = devtools["bump_version"](bump_type="patch")

        data = assert_error(result)
        assert "uv" in data["error"]
        assert "not found" in data["error"]

    def test_bump_project_path_not_found(self, devtools, tmp_path):
        """When project_path does not exist, return an error."""
        bad_path = str(tmp_path / "nonexistent")

        with patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"):
            result = devtools["bump_version"](
                bump_type="patch",
                project_path=bad_path,
            )

        data = assert_error(result)
        assert "not found" in data["error"]

    def test_bump_no_venv(self, devtools, tmp_path):
        """When .venv directory is missing, return an error."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        # No .venv created

        with patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"):
            result = devtools["bump_version"](
                bump_type="patch",
                project_path=str(project_dir),
            )

        data = assert_error(result)
        assert ".venv" in data["error"]

    def test_bump_not_git_repo(self, devtools, tmp_path):
        """When directory is not a git repo, return an error."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch(
                "deriva_ml_mcp.tools.devtools.subprocess.run",
                side_effect=subprocess.CalledProcessError(128, "git"),
            ),
        ):
            result = devtools["bump_version"](
                bump_type="patch",
                project_path=str(project_dir),
            )

        data = assert_error(result)
        assert "Not a git repository" in data["error"]

    def test_bump_no_previous_version(self, devtools, tmp_path):
        """When there is no previous version tag, previous_version is None."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        call_count = {"git_describe_match": 0, "git_describe_after": 0}

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if cmd[:3] == ["git", "describe", "--tags"] and "--abbrev=0" in cmd:
                if "--match" in cmd:
                    # No previous version tag
                    raise subprocess.CalledProcessError(128, "git")
                else:
                    # After bump: new version
                    result.stdout = "v0.1.0\n"
                    return result
            if cmd[:3] == ["uv", "run", "bump-version"]:
                result.stdout = "Created initial tag v0.1.0"
                return result
            return result

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run),
        ):
            result = devtools["bump_version"](
                bump_type="patch",
                project_path=str(project_dir),
            )

        data = assert_success(result)
        assert data["previous_version"] is None
        assert data["new_version"] == "v0.1.0"

    def test_bump_command_fails(self, devtools, tmp_path):
        """When bump-version command fails with non-zero exit, return an error."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if cmd[:3] == ["git", "describe", "--tags"] and "--match" in cmd:
                result.stdout = "v1.0.0\n"
                return result
            if cmd[:3] == ["uv", "run", "bump-version"]:
                result.returncode = 1
                result.stderr = "Error: dirty working tree"
                return result
            return result

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run),
        ):
            result = devtools["bump_version"](
                bump_type="patch",
                project_path=str(project_dir),
            )

        data = assert_error(result)
        assert "bump-version failed" in data["error"]
        assert data["returncode"] == 1

    def test_bump_command_timeout(self, devtools, tmp_path):
        """When bump-version command times out, return an error."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if cmd[:3] == ["git", "describe", "--tags"] and "--match" in cmd:
                result.stdout = "v1.0.0\n"
                return result
            if cmd[:3] == ["uv", "run", "bump-version"]:
                raise subprocess.TimeoutExpired(cmd, 120)
            return result

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run),
        ):
            result = devtools["bump_version"](
                bump_type="patch",
                project_path=str(project_dir),
            )

        data = assert_error(result)
        assert "timed out" in data["error"]

    def test_bump_command_generic_exception(self, devtools, tmp_path):
        """When bump-version raises an unexpected exception, return an error."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if cmd[:3] == ["git", "describe", "--tags"] and "--match" in cmd:
                result.stdout = "v1.0.0\n"
                return result
            if cmd[:3] == ["uv", "run", "bump-version"]:
                raise OSError("Permission denied")
            return result

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run),
        ):
            result = devtools["bump_version"](
                bump_type="patch",
                project_path=str(project_dir),
            )

        data = assert_error(result)
        assert "Failed to run command" in data["error"]

    def test_bump_new_version_unknown_on_failure(self, devtools, tmp_path):
        """When post-bump git describe fails, new_version is 'unknown'."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        describe_call_count = [0]

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if cmd[:3] == ["git", "describe", "--tags"] and "--abbrev=0" in cmd:
                if "--match" in cmd:
                    result.stdout = "v1.0.0\n"
                    return result
                else:
                    # Post-bump describe fails
                    raise subprocess.CalledProcessError(128, "git")
            if cmd[:3] == ["uv", "run", "bump-version"]:
                result.stdout = "Bumped"
                return result
            return result

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run),
        ):
            result = devtools["bump_version"](
                bump_type="patch",
                project_path=str(project_dir),
            )

        data = assert_success(result)
        assert data["new_version"] == "unknown"


# =============================================================================
# get_current_version
# =============================================================================


class TestGetCurrentVersion:
    """Tests for the get_current_version tool."""

    def test_get_version_at_tag(self, devtools):
        """When exactly at a tag, return clean version info."""
        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if "--long" in cmd and "--dirty" in cmd:
                result.stdout = "v1.2.3-0-gabcdef\n"
                return result
            if "--abbrev=0" in cmd:
                result.stdout = "v1.2.3\n"
                return result
            return result

        with patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run):
            result = devtools["get_current_version"]()

        data = assert_success(result)
        assert data["version"] == "v1.2.3-0-gabcdef"
        assert data["tag"] == "v1.2.3"
        assert data["distance"] == 0
        assert data["dirty"] is False
        assert data["at_tag"] is True

    def test_get_version_after_tag(self, devtools):
        """When commits exist after tag, distance is non-zero."""
        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if "--long" in cmd and "--dirty" in cmd:
                result.stdout = "v1.2.3-5-gabcdef\n"
                return result
            if "--abbrev=0" in cmd:
                result.stdout = "v1.2.3\n"
                return result
            return result

        with patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run):
            result = devtools["get_current_version"]()

        data = assert_success(result)
        assert data["version"] == "v1.2.3-5-gabcdef"
        assert data["tag"] == "v1.2.3"
        assert data["distance"] == 5
        assert data["dirty"] is False
        assert data["at_tag"] is False

    def test_get_version_dirty(self, devtools):
        """When working tree is dirty, dirty flag is True."""
        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if "--long" in cmd and "--dirty" in cmd:
                result.stdout = "v1.2.3-0-gabcdef-dirty\n"
                return result
            if "--abbrev=0" in cmd:
                result.stdout = "v1.2.3\n"
                return result
            return result

        with patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run):
            result = devtools["get_current_version"]()

        data = assert_success(result)
        assert data["version"] == "v1.2.3-0-gabcdef-dirty"
        assert data["dirty"] is True
        assert data["at_tag"] is False

    def test_get_version_dirty_with_distance(self, devtools):
        """When dirty and commits after tag, both flags reflect reality."""
        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if "--long" in cmd and "--dirty" in cmd:
                result.stdout = "v2.0.0-3-g1234567-dirty\n"
                return result
            if "--abbrev=0" in cmd:
                result.stdout = "v2.0.0\n"
                return result
            return result

        with patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run):
            result = devtools["get_current_version"]()

        data = assert_success(result)
        assert data["distance"] == 3
        assert data["dirty"] is True
        assert data["at_tag"] is False

    def test_get_version_not_git_repo(self, devtools):
        """When not inside a git repo, return an error."""
        with patch(
            "deriva_ml_mcp.tools.devtools.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            result = devtools["get_current_version"]()

        data = assert_error(result)
        assert "Not inside a git repository" in data["error"]

    def test_get_version_no_tags(self, devtools):
        """When no version tags exist, return an error suggesting bump_version."""
        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "true"
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                return result
            # git describe --tags --long --dirty fails if no tags
            raise subprocess.CalledProcessError(128, "git")

        with patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run):
            result = devtools["get_current_version"]()

        data = assert_error(result)
        assert "No version tags found" in data["error"]

    def test_get_version_tag_lookup_fails(self, devtools):
        """When clean tag lookup fails after getting full version, tag is None."""
        call_count = [0]

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
                result.stdout = "true"
                return result
            if "--long" in cmd and "--dirty" in cmd:
                result.stdout = "v1.0.0-0-gabc1234\n"
                return result
            if "--abbrev=0" in cmd:
                # Clean tag lookup fails
                raise subprocess.CalledProcessError(128, "git")
            return result

        with patch("deriva_ml_mcp.tools.devtools.subprocess.run", side_effect=mock_subprocess_run):
            result = devtools["get_current_version"]()

        data = assert_success(result)
        assert data["tag"] is None


# =============================================================================
# install_jupyter_kernel
# =============================================================================


class TestInstallJupyterKernel:
    """Tests for the install_jupyter_kernel tool."""

    def test_install_success(self, devtools, tmp_path):
        """Successfully installing a kernel returns status=success."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Installed kernelspec my-project in /usr/share/jupyter/kernels/my-project"
        mock_result.stderr = ""

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", return_value=mock_result),
        ):
            result = devtools["install_jupyter_kernel"](
                project_path=str(project_dir),
            )

        data = assert_success(result)
        assert data["project_path"] == str(project_dir)
        assert "Installed kernelspec" in data["output"]
        assert data["message"] == "Jupyter kernel installed successfully"

    def test_install_no_project_path_uses_cwd(self, devtools, tmp_path):
        """When no project_path is given, uses current working directory."""
        # Create .venv in the mocked cwd
        mock_cwd = tmp_path / "cwd_project"
        mock_cwd.mkdir()
        (mock_cwd / ".venv").mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Installed kernel"
        mock_result.stderr = ""

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", return_value=mock_result),
            patch("deriva_ml_mcp.tools.devtools.Path.cwd", return_value=mock_cwd),
        ):
            result = devtools["install_jupyter_kernel"]()

        data = assert_success(result)
        assert data["project_path"] == str(mock_cwd)

    def test_install_uv_not_found(self, devtools):
        """When uv is not on PATH, return an error."""
        with patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value=None):
            result = devtools["install_jupyter_kernel"]()

        data = assert_error(result)
        assert "uv" in data["error"]
        assert "not found" in data["error"]

    def test_install_project_path_not_found(self, devtools, tmp_path):
        """When project_path does not exist, return an error."""
        bad_path = str(tmp_path / "nonexistent")

        with patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"):
            result = devtools["install_jupyter_kernel"](project_path=bad_path)

        data = assert_error(result)
        assert "not found" in data["error"]

    def test_install_no_venv(self, devtools, tmp_path):
        """When .venv directory is missing, return an error."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        with patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"):
            result = devtools["install_jupyter_kernel"](project_path=str(project_dir))

        data = assert_error(result)
        assert ".venv" in data["error"]

    def test_install_command_fails(self, devtools, tmp_path):
        """When the install command returns non-zero, return an error."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "ERROR: kernel installation failed"

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch("deriva_ml_mcp.tools.devtools.subprocess.run", return_value=mock_result),
        ):
            result = devtools["install_jupyter_kernel"](
                project_path=str(project_dir),
            )

        data = assert_error(result)
        assert "Failed to install kernel" in data["error"]
        assert data["returncode"] == 1

    def test_install_timeout(self, devtools, tmp_path):
        """When the install command times out, return an error."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch(
                "deriva_ml_mcp.tools.devtools.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["uv"], 60),
            ),
        ):
            result = devtools["install_jupyter_kernel"](
                project_path=str(project_dir),
            )

        data = assert_error(result)
        assert "timed out" in data["error"]

    def test_install_generic_exception(self, devtools, tmp_path):
        """When an unexpected exception occurs, return an error."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".venv").mkdir()

        with (
            patch("deriva_ml_mcp.tools.devtools.shutil.which", return_value="/usr/bin/uv"),
            patch(
                "deriva_ml_mcp.tools.devtools.subprocess.run",
                side_effect=OSError("Cannot execute"),
            ),
        ):
            result = devtools["install_jupyter_kernel"](
                project_path=str(project_dir),
            )

        data = assert_error(result)
        assert "Failed to run command" in data["error"]


# =============================================================================
# list_jupyter_kernels
# =============================================================================


class TestListJupyterKernels:
    """Tests for the list_jupyter_kernels tool."""

    def test_list_kernels_success(self, devtools):
        """Listing kernels returns status=success with kernel info."""
        mock_specs = {
            "python3": {
                "spec": {
                    "display_name": "Python 3",
                    "language": "python",
                },
                "resource_dir": "/usr/share/jupyter/kernels/python3",
            },
            "my-ml-project": {
                "spec": {
                    "display_name": "Python (my-ml-project)",
                    "language": "python",
                },
                "resource_dir": "/usr/share/jupyter/kernels/my-ml-project",
            },
        }

        mock_ksm = MagicMock()
        mock_ksm.get_all_specs.return_value = mock_specs
        mock_ksm_cls = MagicMock(return_value=mock_ksm)

        with patch(
            "jupyter_client.kernelspec.KernelSpecManager",
            mock_ksm_cls,
        ):
            result = devtools["list_jupyter_kernels"]()

        data = assert_success(result)
        assert data["count"] == 2
        assert len(data["kernels"]) == 2

        kernel_names = [k["name"] for k in data["kernels"]]
        assert "python3" in kernel_names
        assert "my-ml-project" in kernel_names

        for kernel in data["kernels"]:
            if kernel["name"] == "python3":
                assert kernel["display_name"] == "Python 3"
                assert kernel["language"] == "python"
            elif kernel["name"] == "my-ml-project":
                assert kernel["display_name"] == "Python (my-ml-project)"

    def test_list_kernels_empty(self, devtools):
        """When no kernels are installed, return empty list."""
        mock_ksm = MagicMock()
        mock_ksm.get_all_specs.return_value = {}
        mock_ksm_cls = MagicMock(return_value=mock_ksm)

        with patch(
            "jupyter_client.kernelspec.KernelSpecManager",
            mock_ksm_cls,
        ):
            result = devtools["list_jupyter_kernels"]()

        data = assert_success(result)
        assert data["count"] == 0
        assert data["kernels"] == []

    def test_list_kernels_jupyter_not_installed(self, devtools):
        """When jupyter_client is not installed, return an error."""
        # The tool does `from jupyter_client.kernelspec import KernelSpecManager`
        # inside the function. We need to make that import fail.
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jupyter_client.kernelspec":
                raise ImportError("No module named 'jupyter_client'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = devtools["list_jupyter_kernels"]()

        data = assert_error(result)
        assert "jupyter_client" in data["error"]

    def test_list_kernels_exception(self, devtools):
        """When KernelSpecManager raises, return an error."""
        mock_ksm = MagicMock()
        mock_ksm.get_all_specs.side_effect = Exception("Failed to read kernel specs")
        mock_ksm_cls = MagicMock(return_value=mock_ksm)

        with patch(
            "jupyter_client.kernelspec.KernelSpecManager",
            mock_ksm_cls,
        ):
            result = devtools["list_jupyter_kernels"]()

        data = assert_error(result)
        assert "Failed to list kernels" in data["error"]

    def test_list_kernels_missing_spec_fields(self, devtools):
        """When spec fields are missing, defaults are used."""
        mock_specs = {
            "bare-kernel": {
                "spec": {},
                "resource_dir": "",
            },
        }

        mock_ksm = MagicMock()
        mock_ksm.get_all_specs.return_value = mock_specs
        mock_ksm_cls = MagicMock(return_value=mock_ksm)

        with patch(
            "jupyter_client.kernelspec.KernelSpecManager",
            mock_ksm_cls,
        ):
            result = devtools["list_jupyter_kernels"]()

        data = assert_success(result)
        assert data["count"] == 1
        kernel = data["kernels"][0]
        assert kernel["name"] == "bare-kernel"
        assert kernel["display_name"] == "bare-kernel"  # Falls back to name
        assert kernel["language"] == "unknown"


# =============================================================================
# inspect_notebook
# =============================================================================


class TestInspectNotebook:
    """Tests for the inspect_notebook tool."""

    def test_inspect_success(self, devtools, tmp_path):
        """Inspecting a valid notebook returns parameter info."""
        nb_path = tmp_path / "train.ipynb"
        nb_path.write_text("{}")  # Create file so it exists

        mock_params = {
            "learning_rate": {
                "inferred_type_name": "float",
                "default": "0.001",
                "help": "Learning rate for optimizer",
            },
            "epochs": {
                "inferred_type_name": "int",
                "default": "100",
                "help": "Number of training epochs",
            },
            "batch_size": {
                "inferred_type_name": "int",
                "default": "32",
                "help": "",
            },
        }

        with patch("papermill.inspect_notebook", return_value=mock_params):
            result = devtools["inspect_notebook"](notebook_path=str(nb_path))

        data = assert_success(result)
        assert data["count"] == 3
        assert len(data["parameters"]) == 3

        param_names = [p["name"] for p in data["parameters"]]
        assert "learning_rate" in param_names
        assert "epochs" in param_names
        assert "batch_size" in param_names

        for param in data["parameters"]:
            if param["name"] == "learning_rate":
                assert param["type"] == "float"
                assert param["default"] == "0.001"
                assert param["help"] == "Learning rate for optimizer"

    def test_inspect_no_parameters(self, devtools, tmp_path):
        """When notebook has no parameters, return empty list."""
        nb_path = tmp_path / "simple.ipynb"
        nb_path.write_text("{}")

        with patch("papermill.inspect_notebook", return_value={}):
            result = devtools["inspect_notebook"](notebook_path=str(nb_path))

        data = assert_success(result)
        assert data["count"] == 0
        assert data["parameters"] == []

    def test_inspect_file_not_found(self, devtools, tmp_path):
        """When notebook file does not exist, return an error."""
        nb_path = str(tmp_path / "nonexistent.ipynb")

        result = devtools["inspect_notebook"](notebook_path=nb_path)

        data = assert_error(result)
        assert "not found" in data["error"]

    def test_inspect_not_ipynb(self, devtools, tmp_path):
        """When file is not .ipynb, return an error."""
        py_path = tmp_path / "script.py"
        py_path.write_text("print('hello')")

        result = devtools["inspect_notebook"](notebook_path=str(py_path))

        data = assert_error(result)
        assert ".ipynb" in data["error"]

    def test_inspect_papermill_not_installed(self, devtools, tmp_path):
        """When papermill is not installed, return an error."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "papermill":
                raise ImportError("No module named 'papermill'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = devtools["inspect_notebook"](notebook_path="/some/notebook.ipynb")

        data = assert_error(result)
        assert "papermill" in data["error"]

    def test_inspect_papermill_raises(self, devtools, tmp_path):
        """When papermill.inspect_notebook raises, return an error."""
        nb_path = tmp_path / "broken.ipynb"
        nb_path.write_text("{}")

        with patch(
            "papermill.inspect_notebook",
            side_effect=Exception("Invalid notebook format"),
        ):
            result = devtools["inspect_notebook"](notebook_path=str(nb_path))

        data = assert_error(result)
        assert "Failed to inspect notebook" in data["error"]


# =============================================================================
# run_notebook
# =============================================================================


class TestRunNotebook:
    """Tests for the run_notebook tool."""

    def test_run_notebook_file_not_found(self, devtools, tmp_path):
        """When notebook file does not exist, return an error."""
        nb_path = str(tmp_path / "nonexistent.ipynb")

        result = devtools["run_notebook"](
            notebook_path=nb_path,
            hostname="test.example.org",
            catalog_id="1",
        )

        data = assert_error(result)
        assert "not found" in data["error"]

    def test_run_notebook_not_ipynb(self, devtools, tmp_path):
        """When file is not .ipynb, return an error."""
        py_path = tmp_path / "script.py"
        py_path.write_text("print('hello')")

        result = devtools["run_notebook"](
            notebook_path=str(py_path),
            hostname="test.example.org",
            catalog_id="1",
        )

        data = assert_error(result)
        assert ".ipynb" in data["error"]

    def test_run_notebook_import_error(self, devtools, tmp_path):
        """When required packages not installed, return an error."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "nbformat":
                raise ImportError("No module named 'nbformat'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = devtools["run_notebook"](
                notebook_path="/some/notebook.ipynb",
                hostname="test.example.org",
                catalog_id="1",
            )

        data = assert_error(result)
        assert "not installed" in data["error"]

    def test_run_notebook_deriva_ml_import_error(self, devtools, tmp_path):
        """When deriva_ml is not properly installed, return an error."""
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text("{}")

        import builtins
        original_import = builtins.__import__

        call_count = [0]

        def mock_import(name, *args, **kwargs):
            # Allow nbformat/papermill/nbconvert to import, but block deriva_ml
            if name == "deriva_ml" and call_count[0] == 0:
                call_count[0] += 1
                raise ImportError("No module named 'deriva_ml'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = devtools["run_notebook"](
                notebook_path=str(nb_path),
                hostname="test.example.org",
                catalog_id="1",
            )

        data = assert_error(result)
        assert "deriva_ml" in data["error"] or "error" in data["status"]

    def test_run_notebook_nbstrip_warning(self, devtools, tmp_path):
        """When nbstripout check raises, return a warning."""
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text("{}")

        mock_workflow_cls = MagicMock()
        mock_workflow_cls._check_nbstrip_status.side_effect = RuntimeError("nbstripout not configured")

        with patch.dict("sys.modules", {}):
            with (
                patch("deriva_ml.execution.Workflow", mock_workflow_cls),
            ):
                result = devtools["run_notebook"](
                    notebook_path=str(nb_path),
                    hostname="test.example.org",
                    catalog_id="1",
                )

        data = parse_json_result(result)
        assert data["status"] == "warning"
        assert "nbstripout" in data.get("warning", "")

    def test_run_notebook_execution_failed_exception(self, devtools, tmp_path):
        """When notebook execution raises an exception, return an error."""
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text("{}")

        mock_workflow_cls = MagicMock()
        mock_workflow_cls._check_nbstrip_status.return_value = None
        mock_workflow_cls.get_url_and_checksum.return_value = ("https://example.com/nb", "sha256:abc")

        with (
            patch("deriva_ml.execution.Workflow", mock_workflow_cls),
            patch("papermill.execute_notebook", side_effect=Exception("Cell execution error")),
            patch("deriva_ml_mcp.tools.devtools._find_kernel_for_venv", return_value="python3"),
        ):
            result = devtools["run_notebook"](
                notebook_path=str(nb_path),
                hostname="test.example.org",
                catalog_id="1",
            )

        data = assert_error(result)
        assert "Notebook execution failed" in data["error"]


# =============================================================================
# Tool Registration
# =============================================================================


class TestDevtoolRegistration:
    """Tests that all devtools are properly registered."""

    def test_all_tools_registered(self, devtools):
        """All 6 devtools should be registered."""
        expected = {
            "bump_version",
            "get_current_version",
            "install_jupyter_kernel",
            "list_jupyter_kernels",
            "inspect_notebook",
            "run_notebook",
        }
        assert set(devtools.keys()) == expected

    def test_tools_are_callable(self, devtools):
        """All registered tools should be callable."""
        for name, func in devtools.items():
            assert callable(func), f"Tool {name} is not callable"
