"""Unit tests for catalog connection and management tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import assert_error, assert_success, parse_json_result


# =============================================================================
# connect_catalog
# =============================================================================


class TestConnectCatalog:
    """Tests for the connect_catalog tool."""

    @pytest.mark.asyncio
    async def test_connect_success(self, catalog_tools, mock_conn_manager, mock_ml):
        """Connecting to a catalog returns status=connected with full details."""
        mock_conn_manager.connect.return_value = mock_ml

        result = await catalog_tools["connect_catalog"](
            hostname="test.example.org",
            catalog_id="1",
        )

        data = assert_success(result)
        assert data["status"] == "connected"
        assert data["hostname"] == "test.example.org"
        assert data["catalog_id"] == "1"
        assert "test_schema" in data["domain_schemas"]
        assert data["default_schema"] == "test_schema"
        assert data["project_name"] == "test_project"
        assert data["workflow_rid"] == "WF-TEST"
        assert data["execution_rid"] == "EXE-TEST"

        mock_conn_manager.connect.assert_called_once_with(
            "test.example.org", "1", None, default_schema=None,
        )

    @pytest.mark.asyncio
    async def test_connect_with_domain_schema(self, catalog_tools, mock_conn_manager, mock_ml):
        """Connecting with an explicit domain_schema passes it as a set."""
        mock_conn_manager.connect.return_value = mock_ml

        result = await catalog_tools["connect_catalog"](
            hostname="test.example.org",
            catalog_id="1",
            domain_schema="my_schema",
        )

        data = assert_success(result)
        assert data["status"] == "connected"
        mock_conn_manager.connect.assert_called_once_with(
            "test.example.org", "1", {"my_schema"}, default_schema=None,
        )

    @pytest.mark.asyncio
    async def test_connect_with_default_schema(self, catalog_tools, mock_conn_manager, mock_ml):
        """Connecting with default_schema passes it through."""
        mock_conn_manager.connect.return_value = mock_ml

        result = await catalog_tools["connect_catalog"](
            hostname="test.example.org",
            catalog_id="1",
            default_schema="isa",
        )

        data = assert_success(result)
        assert data["status"] == "connected"
        mock_conn_manager.connect.assert_called_once_with(
            "test.example.org", "1", None, default_schema="isa",
        )

    @pytest.mark.asyncio
    async def test_connect_no_execution(self, catalog_tools, mock_conn_manager, mock_ml):
        """When connection info has no execution, execution_rid is None."""
        mock_conn_manager.connect.return_value = mock_ml
        mock_conn_info = MagicMock()
        mock_conn_info.execution = None
        mock_conn_info.workflow_rid = "WF-TEST"
        mock_conn_manager.get_active_connection_info.return_value = mock_conn_info

        result = await catalog_tools["connect_catalog"](
            hostname="test.example.org",
            catalog_id="1",
        )

        data = assert_success(result)
        assert data["execution_rid"] is None
        assert data["workflow_rid"] == "WF-TEST"

    @pytest.mark.asyncio
    async def test_connect_no_conn_info(self, catalog_tools, mock_conn_manager, mock_ml):
        """When get_active_connection_info returns None, no workflow/execution in result."""
        mock_conn_manager.connect.return_value = mock_ml
        mock_conn_manager.get_active_connection_info.return_value = None

        result = await catalog_tools["connect_catalog"](
            hostname="test.example.org",
            catalog_id="1",
        )

        data = assert_success(result)
        assert data["status"] == "connected"
        assert "workflow_rid" not in data
        assert "execution_rid" not in data

    @pytest.mark.asyncio
    async def test_connect_failure(self, catalog_tools, mock_conn_manager):
        """When connect() raises, return an error with the exception message."""
        mock_conn_manager.connect.side_effect = Exception("Authentication failed")

        result = await catalog_tools["connect_catalog"](
            hostname="bad.host.org",
            catalog_id="999",
        )

        data = assert_error(result, expected_message="Authentication failed")


# =============================================================================
# disconnect_catalog
# =============================================================================


class TestDisconnectCatalog:
    """Tests for the disconnect_catalog tool."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, catalog_tools, mock_conn_manager):
        """When disconnect() returns True, status is 'disconnected'."""
        mock_conn_manager.disconnect.return_value = True

        result = await catalog_tools["disconnect_catalog"]()

        data = parse_json_result(result)
        assert data["status"] == "disconnected"
        mock_conn_manager.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_no_active(self, catalog_tools, mock_conn_manager):
        """When disconnect() returns False, status is 'no_active_connection'."""
        mock_conn_manager.disconnect.return_value = False

        result = await catalog_tools["disconnect_catalog"]()

        data = parse_json_result(result)
        assert data["status"] == "no_active_connection"


# =============================================================================
# set_active_catalog
# =============================================================================


class TestSetActiveCatalog:
    """Tests for the set_active_catalog tool."""

    @pytest.mark.asyncio
    async def test_set_active_success(self, catalog_tools, mock_conn_manager):
        """When set_active() returns True, report success with catalog key."""
        mock_conn_manager.set_active.return_value = True

        result = await catalog_tools["set_active_catalog"](
            hostname="test.example.org",
            catalog_id="42",
        )

        data = assert_success(result)
        assert data["status"] == "success"
        assert data["active_catalog"] == "test.example.org:42"
        mock_conn_manager.set_active.assert_called_once_with("test.example.org", "42")

    @pytest.mark.asyncio
    async def test_set_active_not_found(self, catalog_tools, mock_conn_manager):
        """When set_active() returns False, report error."""
        mock_conn_manager.set_active.return_value = False

        result = await catalog_tools["set_active_catalog"](
            hostname="unknown.host",
            catalog_id="999",
        )

        data = assert_error(result)
        assert "No connection found" in data["message"]


# =============================================================================
# set_default_schema
# =============================================================================


class TestSetDefaultSchema:
    """Tests for the set_default_schema tool."""

    @pytest.mark.asyncio
    async def test_set_default_schema_success(self, catalog_tools, mock_ml):
        """Setting a valid domain schema returns success."""
        mock_ml.domain_schemas = {"test_schema", "other_schema"}

        result = await catalog_tools["set_default_schema"](schema_name="other_schema")

        data = assert_success(result)
        assert data["status"] == "success"
        assert data["default_schema"] == "other_schema"
        assert sorted(data["domain_schemas"]) == ["other_schema", "test_schema"]
        assert mock_ml.model.default_schema == "other_schema"
        assert mock_ml.default_schema == "other_schema"

    @pytest.mark.asyncio
    async def test_set_default_schema_invalid(self, catalog_tools, mock_ml):
        """Setting a schema not in domain_schemas returns an error."""
        mock_ml.domain_schemas = {"test_schema"}

        result = await catalog_tools["set_default_schema"](schema_name="nonexistent")

        data = assert_error(result)
        assert "not a domain schema" in data["message"]
        assert "test_schema" in data["message"]

    @pytest.mark.asyncio
    async def test_set_default_schema_no_connection(self, catalog_tools_disconnected):
        """When not connected, return an error."""
        result = await catalog_tools_disconnected["set_default_schema"](
            schema_name="test_schema",
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# create_catalog
# =============================================================================


class TestCreateCatalog:
    """Tests for the create_catalog tool."""

    @pytest.mark.asyncio
    async def test_create_success(self, catalog_tools, mock_conn_manager, mock_ml):
        """Creating a catalog returns status=created with catalog details."""
        mock_catalog = MagicMock()
        mock_catalog.catalog_id = 42
        mock_model = MagicMock()
        mock_catalog.getCatalogModel.return_value = mock_model

        mock_conn_manager.connect.return_value = mock_ml

        with patch("deriva_ml.schema.create_ml_catalog", return_value=mock_catalog) as mock_create:
            result = await catalog_tools["create_catalog"](
                hostname="test.example.org",
                project_name="my_project",
            )

            mock_create.assert_called_once_with(
                "test.example.org", "my_project", catalog_alias=None,
            )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["hostname"] == "test.example.org"
        assert data["catalog_id"] == "42"
        assert data["project_name"] == "my_project"

        # Verify domain schema was created
        mock_model.create_schema.assert_called_once()

        # Verify auto-connect after creation
        mock_conn_manager.connect.assert_called_once_with(
            "test.example.org", "42", {"my_project"},
        )

    @pytest.mark.asyncio
    async def test_create_with_alias(self, catalog_tools, mock_conn_manager, mock_ml):
        """Creating a catalog with an alias passes it through and includes it in result."""
        mock_catalog = MagicMock()
        mock_catalog.catalog_id = 99
        mock_catalog.getCatalogModel.return_value = MagicMock()
        mock_conn_manager.connect.return_value = mock_ml

        with patch("deriva_ml.schema.create_ml_catalog", return_value=mock_catalog):
            result = await catalog_tools["create_catalog"](
                hostname="test.example.org",
                project_name="my_project",
                catalog_alias="my-alias",
            )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["catalog_alias"] == "my-alias"

    @pytest.mark.asyncio
    async def test_create_failure(self, catalog_tools, mock_conn_manager):
        """When create_ml_catalog raises, return an error."""
        with patch(
            "deriva_ml.schema.create_ml_catalog",
            side_effect=Exception("Permission denied"),
        ):
            result = await catalog_tools["create_catalog"](
                hostname="test.example.org",
                project_name="bad_project",
            )

        data = assert_error(result, expected_message="Permission denied")


# =============================================================================
# delete_catalog
# =============================================================================


class TestDeleteCatalog:
    """Tests for the delete_catalog tool."""

    @pytest.mark.asyncio
    async def test_delete_success(self, catalog_tools, mock_conn_manager, mock_ml):
        """Deleting a non-active catalog returns status=deleted."""
        # Active catalog has a different ID, so disconnect is not called
        mock_ml.catalog_id = "1"

        mock_server = MagicMock()
        with patch("deriva.core.DerivaServer", return_value=mock_server):
            result = await catalog_tools["delete_catalog"](
                hostname="test.example.org",
                catalog_id="99",
            )

        data = assert_success(result)
        assert data["status"] == "deleted"
        assert data["hostname"] == "test.example.org"
        assert data["catalog_id"] == "99"
        mock_server.delete.assert_called_once_with("/ermrest/catalog/99")
        mock_conn_manager.disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_active_catalog_disconnects_first(
        self, catalog_tools, mock_conn_manager, mock_ml,
    ):
        """When deleting the active catalog, disconnect before deleting."""
        mock_ml.catalog_id = "42"

        mock_server = MagicMock()
        with patch("deriva.core.DerivaServer", return_value=mock_server):
            result = await catalog_tools["delete_catalog"](
                hostname="test.example.org",
                catalog_id="42",
            )

        data = assert_success(result)
        assert data["status"] == "deleted"
        mock_conn_manager.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_no_active_connection(self, catalog_tools, mock_conn_manager):
        """When no active connection, delete proceeds without disconnect."""
        mock_conn_manager.get_active.return_value = None

        mock_server = MagicMock()
        with patch("deriva.core.DerivaServer", return_value=mock_server):
            result = await catalog_tools["delete_catalog"](
                hostname="test.example.org",
                catalog_id="5",
            )

        data = assert_success(result)
        assert data["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_failure(self, catalog_tools, mock_conn_manager):
        """When DerivaServer.delete() raises, return an error."""
        mock_conn_manager.get_active.return_value = None

        mock_server = MagicMock()
        mock_server.delete.side_effect = Exception("Catalog not found")
        with patch("deriva.core.DerivaServer", return_value=mock_server):
            result = await catalog_tools["delete_catalog"](
                hostname="test.example.org",
                catalog_id="999",
            )

        data = assert_error(result, expected_message="Catalog not found")


# =============================================================================
# apply_catalog_annotations
# =============================================================================


class TestApplyCatalogAnnotations:
    """Tests for the apply_catalog_annotations tool."""

    @pytest.mark.asyncio
    async def test_apply_defaults(self, catalog_tools, mock_ml):
        """Applying with defaults uses default navbar text and head title."""
        result = await catalog_tools["apply_catalog_annotations"]()

        data = assert_success(result)
        assert data["status"] == "success"
        assert data["navbar_brand_text"] == "ML Data Browser"
        assert data["head_title"] == "Catalog ML"
        assert "annotations applied" in data["message"].lower()

        mock_ml.apply_catalog_annotations.assert_called_once_with(
            navbar_brand_text="ML Data Browser",
            head_title="Catalog ML",
        )

    @pytest.mark.asyncio
    async def test_apply_custom(self, catalog_tools, mock_ml):
        """Applying with custom values passes them through."""
        result = await catalog_tools["apply_catalog_annotations"](
            navbar_brand_text="My ML Project",
            head_title="Project Dashboard",
        )

        data = assert_success(result)
        assert data["navbar_brand_text"] == "My ML Project"
        assert data["head_title"] == "Project Dashboard"

        mock_ml.apply_catalog_annotations.assert_called_once_with(
            navbar_brand_text="My ML Project",
            head_title="Project Dashboard",
        )

    @pytest.mark.asyncio
    async def test_apply_failure(self, catalog_tools, mock_ml):
        """When apply_catalog_annotations raises, return an error."""
        mock_ml.apply_catalog_annotations.side_effect = Exception("Missing schema")

        result = await catalog_tools["apply_catalog_annotations"]()

        data = assert_error(result, expected_message="Missing schema")

    @pytest.mark.asyncio
    async def test_apply_no_connection(self, catalog_tools_disconnected):
        """When not connected, return an error."""
        result = await catalog_tools_disconnected["apply_catalog_annotations"]()

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# create_catalog_alias
# =============================================================================


class TestCreateCatalogAlias:
    """Tests for the create_catalog_alias tool."""

    @pytest.mark.asyncio
    async def test_create_alias_success(self, catalog_tools):
        """Creating an alias returns status=created with details."""
        mock_server = MagicMock()

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value={"token": "abc"}),
        ):
            result = await catalog_tools["create_catalog_alias"](
                hostname="test.example.org",
                alias_name="my-alias",
                catalog_id="42",
                name="My Alias",
                description="A test alias",
            )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["hostname"] == "test.example.org"
        assert data["alias"] == "my-alias"
        assert data["target"] == "42"
        assert data["name"] == "My Alias"
        assert data["description"] == "A test alias"

        mock_server.create_ermrest_alias.assert_called_once_with(
            id="my-alias",
            alias_target="42",
            name="My Alias",
            description="A test alias",
        )

    @pytest.mark.asyncio
    async def test_create_alias_minimal(self, catalog_tools):
        """Creating an alias without optional name/description works."""
        mock_server = MagicMock()

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["create_catalog_alias"](
                hostname="test.example.org",
                alias_name="simple",
                catalog_id="10",
            )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] is None
        assert data["description"] is None

    @pytest.mark.asyncio
    async def test_create_alias_failure(self, catalog_tools):
        """When server raises, return an error."""
        mock_server = MagicMock()
        mock_server.create_ermrest_alias.side_effect = Exception("Alias already exists")

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["create_catalog_alias"](
                hostname="test.example.org",
                alias_name="duplicate",
                catalog_id="10",
            )

        data = assert_error(result, expected_message="Alias already exists")


# =============================================================================
# update_catalog_alias
# =============================================================================


class TestUpdateCatalogAlias:
    """Tests for the update_catalog_alias tool."""

    @pytest.mark.asyncio
    async def test_update_target(self, catalog_tools):
        """Updating the alias target calls alias.update with new target."""
        mock_alias = MagicMock()
        mock_alias.retrieve.return_value = {
            "alias": "my-alias",
            "target": "50",
            "owner": ["user1"],
        }
        mock_server = MagicMock()
        mock_server.connect_ermrest_alias.return_value = mock_alias

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["update_catalog_alias"](
                hostname="test.example.org",
                alias_name="my-alias",
                alias_target="50",
            )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["hostname"] == "test.example.org"
        assert data["target"] == "50"

        mock_alias.update.assert_called_once_with(alias_target="50")

    @pytest.mark.asyncio
    async def test_update_owner(self, catalog_tools):
        """Updating the owner ACL passes owner list."""
        mock_alias = MagicMock()
        mock_alias.retrieve.return_value = {"alias": "a", "owner": ["g1", "g2"]}
        mock_server = MagicMock()
        mock_server.connect_ermrest_alias.return_value = mock_alias

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["update_catalog_alias"](
                hostname="test.example.org",
                alias_name="a",
                owner=["g1", "g2"],
            )

        data = assert_success(result)
        mock_alias.update.assert_called_once_with(owner=["g1", "g2"])

    @pytest.mark.asyncio
    async def test_update_unbind(self, catalog_tools):
        """Passing empty string as target unbinds the alias (sets to None)."""
        mock_alias = MagicMock()
        mock_alias.retrieve.return_value = {"alias": "a", "target": None}
        mock_server = MagicMock()
        mock_server.connect_ermrest_alias.return_value = mock_alias

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["update_catalog_alias"](
                hostname="test.example.org",
                alias_name="a",
                alias_target="",
            )

        data = assert_success(result)
        # Empty string becomes None to unbind
        mock_alias.update.assert_called_once_with(alias_target=None)

    @pytest.mark.asyncio
    async def test_update_no_changes(self, catalog_tools):
        """When no parameters change, update is not called."""
        mock_alias = MagicMock()
        mock_alias.retrieve.return_value = {"alias": "a"}
        mock_server = MagicMock()
        mock_server.connect_ermrest_alias.return_value = mock_alias

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["update_catalog_alias"](
                hostname="test.example.org",
                alias_name="a",
            )

        data = assert_success(result)
        mock_alias.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_failure(self, catalog_tools):
        """When the server raises, return an error."""
        mock_server = MagicMock()
        mock_server.connect_ermrest_alias.side_effect = Exception("Alias not found")

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["update_catalog_alias"](
                hostname="test.example.org",
                alias_name="nonexistent",
            )

        data = assert_error(result, expected_message="Alias not found")


# =============================================================================
# delete_catalog_alias
# =============================================================================


class TestDeleteCatalogAlias:
    """Tests for the delete_catalog_alias tool."""

    @pytest.mark.asyncio
    async def test_delete_alias_success(self, catalog_tools):
        """Deleting an alias returns status=deleted."""
        mock_alias = MagicMock()
        mock_server = MagicMock()
        mock_server.connect_ermrest_alias.return_value = mock_alias

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["delete_catalog_alias"](
                hostname="test.example.org",
                alias_name="old-alias",
            )

        data = assert_success(result)
        assert data["status"] == "deleted"
        assert data["hostname"] == "test.example.org"
        assert data["alias"] == "old-alias"

        mock_alias.delete_ermrest_alias.assert_called_once_with(really=True)

    @pytest.mark.asyncio
    async def test_delete_alias_failure(self, catalog_tools):
        """When the server raises, return an error."""
        mock_server = MagicMock()
        mock_server.connect_ermrest_alias.side_effect = Exception("Not authorized")

        with (
            patch("deriva_ml_mcp.tools.catalog.DerivaServer", return_value=mock_server),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["delete_catalog_alias"](
                hostname="test.example.org",
                alias_name="protected",
            )

        data = assert_error(result, expected_message="Not authorized")


# =============================================================================
# clone_catalog
# =============================================================================


class TestCloneCatalog:
    """Tests for the clone_catalog tool."""

    def _make_clone_result(self, **overrides):
        """Build a mock CloneCatalogResult with sensible defaults."""
        result = MagicMock()
        result.hostname = overrides.get("hostname", "dest.example.org")
        result.catalog_id = overrides.get("catalog_id", "100")
        result.source_snapshot = overrides.get("source_snapshot", None)
        result.datasets_reinitialized = overrides.get("datasets_reinitialized", None)
        result.ml_schema_added = overrides.get("ml_schema_added", None)
        result.truncated_values = overrides.get("truncated_values", None)

        # Build a mock report
        report = MagicMock()
        summary = MagicMock()
        summary.orphan_rows_removed = 0
        summary.orphan_rows_nullified = 0
        summary.fkeys_pruned = 0
        summary.total_issues = 0
        summary.errors = 0
        summary.warnings = 0
        summary.tables_restored = 5
        summary.tables_failed = 0
        summary.tables_skipped = 0
        summary.total_rows_restored = 100
        summary.fkeys_applied = 10
        summary.fkeys_failed = 0
        report.summary = summary
        report.issues = []
        report.tables_restored = ["t1", "t2", "t3", "t4", "t5"]
        report.tables_failed = []
        report.tables_skipped = []
        report.orphan_details = {}
        report.to_text.return_value = "Clone completed successfully."
        result.report = overrides.get("report", report)

        return result

    @pytest.mark.asyncio
    async def test_clone_success(self, catalog_tools):
        """Cloning returns status=cloned with full details."""
        mock_result = self._make_clone_result()

        with (
            patch(
                "deriva_ml.catalog.create_ml_workspace",
                return_value=mock_result,
            ) as mock_create,
            patch(
                "deriva_ml_mcp.tools.catalog._resolve_hostname",
                side_effect=lambda h: h,
            ),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["clone_catalog"](
                source_hostname="source.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
                dest_hostname="dest.example.org",
            )

        data = assert_success(result)
        assert data["status"] == "cloned"
        assert data["source_hostname"] == "source.example.org"
        assert data["source_catalog_id"] == "1"
        assert data["dest_hostname"] == "dest.example.org"
        assert data["dest_catalog_id"] == "100"
        assert data["root_rid"] == "3-HXMC"
        assert data["asset_mode"] == "refs"
        assert "report" in data
        assert data["report"]["summary"]["tables_restored"] == 5

    @pytest.mark.asyncio
    async def test_clone_same_server(self, catalog_tools):
        """Cloning on the same server reports clone_type=same_server."""
        mock_result = self._make_clone_result(hostname="source.example.org")

        with (
            patch(
                "deriva_ml.catalog.create_ml_workspace",
                return_value=mock_result,
            ),
            patch(
                "deriva_ml_mcp.tools.catalog._resolve_hostname",
                side_effect=lambda h: h,
            ),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["clone_catalog"](
                source_hostname="source.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
            )

        data = assert_success(result)
        assert data["clone_type"] == "same_server"

    @pytest.mark.asyncio
    async def test_clone_with_alias(self, catalog_tools):
        """When alias is provided, it appears in the response."""
        mock_result = self._make_clone_result()

        with (
            patch(
                "deriva_ml.catalog.create_ml_workspace",
                return_value=mock_result,
            ),
            patch(
                "deriva_ml_mcp.tools.catalog._resolve_hostname",
                side_effect=lambda h: h,
            ),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["clone_catalog"](
                source_hostname="source.example.org",
                source_catalog_id="1",
                root_rid="3-HXMC",
                alias="my-clone",
            )

        data = assert_success(result)
        assert data["alias"] == "my-clone"

    @pytest.mark.asyncio
    async def test_clone_with_snapshot_and_reinit(self, catalog_tools):
        """When result has snapshot and datasets_reinitialized, they appear in response."""
        mock_result = self._make_clone_result(
            source_snapshot="2024-01-15T10:00:00",
            datasets_reinitialized=["DS-1", "DS-2"],
            ml_schema_added=True,
        )

        with (
            patch(
                "deriva_ml.catalog.create_ml_workspace",
                return_value=mock_result,
            ),
            patch(
                "deriva_ml_mcp.tools.catalog._resolve_hostname",
                side_effect=lambda h: h,
            ),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["clone_catalog"](
                source_hostname="source.example.org",
                source_catalog_id="1",
                root_rid="R-1",
            )

        data = assert_success(result)
        assert data["source_snapshot"] == "2024-01-15T10:00:00"
        assert data["datasets_reinitialized"] == ["DS-1", "DS-2"]
        assert data["ml_schema_added"] is True

    @pytest.mark.asyncio
    async def test_clone_hostname_resolution(self, catalog_tools):
        """When hostnames are resolved (Docker), credentials use the original name."""
        mock_result = self._make_clone_result()

        with (
            patch(
                "deriva_ml.catalog.create_ml_workspace",
                return_value=mock_result,
            ) as mock_create,
            patch(
                "deriva_ml_mcp.tools.catalog._resolve_hostname",
                side_effect=lambda h: "docker-host" if h == "localhost" else h,
            ),
            patch(
                "deriva_ml_mcp.tools.catalog.get_credential",
                return_value={"token": "xyz"},
            ) as mock_cred,
        ):
            result = await catalog_tools["clone_catalog"](
                source_hostname="localhost",
                source_catalog_id="1",
                root_rid="R-1",
                dest_hostname="localhost",
            )

        # Both hostnames resolved -> credentials fetched for both
        assert mock_cred.call_count == 2

    @pytest.mark.asyncio
    async def test_clone_failure(self, catalog_tools):
        """When create_ml_workspace raises, return an error."""
        with (
            patch(
                "deriva_ml.catalog.create_ml_workspace",
                side_effect=Exception("Source catalog not found"),
            ),
            patch(
                "deriva_ml_mcp.tools.catalog._resolve_hostname",
                side_effect=lambda h: h,
            ),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["clone_catalog"](
                source_hostname="bad.host",
                source_catalog_id="999",
                root_rid="R-BAD",
            )

        data = assert_error(result, expected_message="Source catalog not found")

    @pytest.mark.asyncio
    async def test_clone_no_report(self, catalog_tools):
        """When result.report is None, response still succeeds without report details."""
        mock_result = self._make_clone_result()
        mock_result.report = None

        with (
            patch(
                "deriva_ml.catalog.create_ml_workspace",
                return_value=mock_result,
            ),
            patch(
                "deriva_ml_mcp.tools.catalog._resolve_hostname",
                side_effect=lambda h: h,
            ),
            patch("deriva_ml_mcp.tools.catalog.get_credential", return_value=None),
        ):
            result = await catalog_tools["clone_catalog"](
                source_hostname="source.example.org",
                source_catalog_id="1",
                root_rid="R-1",
            )

        data = assert_success(result)
        assert data["status"] == "cloned"
        assert "report" not in data
        assert "clone_type" not in data


# =============================================================================
# validate_rids
# =============================================================================


class TestValidateRids:
    """Tests for the validate_rids tool."""

    @pytest.mark.asyncio
    async def test_validate_all_valid(self, catalog_tools, mock_ml):
        """When all RIDs are valid, is_valid is True with empty errors."""
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.validated_rids = {
            "datasets": {"1-ABC": {"name": "My Dataset"}},
        }
        mock_result.__str__ = lambda self: "All 1 RID(s) validated successfully."

        with patch(
            "deriva_ml.core.validation.validate_rids",
            return_value=mock_result,
        ) as mock_validate:
            result = await catalog_tools["validate_rids"](
                dataset_rids=["1-ABC"],
            )

        data = parse_json_result(result)
        assert data["is_valid"] is True
        assert data["errors"] == []
        assert data["warnings"] == []
        assert "1-ABC" in data["validated_rids"]["datasets"]

        mock_validate.assert_called_once_with(
            mock_ml,
            dataset_rids=["1-ABC"],
            asset_rids=None,
            dataset_versions=None,
            workflow_rids=None,
            execution_rids=None,
            warn_missing_descriptions=True,
        )

    @pytest.mark.asyncio
    async def test_validate_with_errors(self, catalog_tools, mock_ml):
        """When RIDs are invalid, is_valid is False with error messages."""
        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.errors = ["RID '1-BAD' not found in Dataset table"]
        mock_result.warnings = ["Dataset '2-OK' has no description"]
        mock_result.validated_rids = {}
        mock_result.__str__ = lambda self: "Validation failed."

        with patch(
            "deriva_ml.core.validation.validate_rids",
            return_value=mock_result,
        ):
            result = await catalog_tools["validate_rids"](
                dataset_rids=["1-BAD"],
                warn_missing_descriptions=True,
            )

        data = parse_json_result(result)
        assert data["is_valid"] is False
        assert len(data["errors"]) == 1
        assert "1-BAD" in data["errors"][0]

    @pytest.mark.asyncio
    async def test_validate_multiple_types(self, catalog_tools, mock_ml):
        """Passing multiple RID types sends them all to the validator."""
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.validated_rids = {}
        mock_result.__str__ = lambda self: "OK"

        with patch(
            "deriva_ml.core.validation.validate_rids",
            return_value=mock_result,
        ) as mock_validate:
            result = await catalog_tools["validate_rids"](
                dataset_rids=["D-1"],
                asset_rids=["A-1"],
                dataset_versions={"D-1": "0.4.0"},
                workflow_rids=["W-1"],
                execution_rids=["E-1"],
                warn_missing_descriptions=False,
            )

        mock_validate.assert_called_once_with(
            mock_ml,
            dataset_rids=["D-1"],
            asset_rids=["A-1"],
            dataset_versions={"D-1": "0.4.0"},
            workflow_rids=["W-1"],
            execution_rids=["E-1"],
            warn_missing_descriptions=False,
        )

    @pytest.mark.asyncio
    async def test_validate_no_connection(self, catalog_tools_disconnected):
        """When not connected, return an error."""
        result = await catalog_tools_disconnected["validate_rids"](
            dataset_rids=["1-ABC"],
        )

        assert_error(result, expected_message="No active catalog connection")


# =============================================================================
# cite
# =============================================================================


class TestCite:
    """Tests for the cite tool."""

    @pytest.mark.asyncio
    async def test_cite_snapshot(self, catalog_tools, mock_ml):
        """Default cite returns a snapshot URL with is_snapshot=True."""
        mock_ml.cite.return_value = "https://test.example.org/id/1/1-ABC@2024-01-15"

        result = await catalog_tools["cite"](rid="1-ABC")

        data = parse_json_result(result)
        assert data["url"] == "https://test.example.org/id/1/1-ABC@2024-01-15"
        assert data["rid"] == "1-ABC"
        assert data["is_snapshot"] is True

        mock_ml.cite.assert_called_once_with("1-ABC", current=False)

    @pytest.mark.asyncio
    async def test_cite_current(self, catalog_tools, mock_ml):
        """Citing with current=True returns a live URL with is_snapshot=False."""
        mock_ml.cite.return_value = "https://test.example.org/id/1/1-ABC"

        result = await catalog_tools["cite"](rid="1-ABC", current=True)

        data = parse_json_result(result)
        assert data["url"] == "https://test.example.org/id/1/1-ABC"
        assert data["is_snapshot"] is False

        mock_ml.cite.assert_called_once_with("1-ABC", current=True)

    @pytest.mark.asyncio
    async def test_cite_failure(self, catalog_tools, mock_ml):
        """When cite() raises, return an error."""
        mock_ml.cite.side_effect = Exception("RID not found")

        result = await catalog_tools["cite"](rid="INVALID")

        data = assert_error(result, expected_message="RID not found")

    @pytest.mark.asyncio
    async def test_cite_no_connection(self, catalog_tools_disconnected):
        """When not connected, return an error."""
        result = await catalog_tools_disconnected["cite"](rid="1-ABC")

        assert_error(result, expected_message="No active catalog connection")
