"""Tests for the connection manager."""

import pytest
from unittest.mock import MagicMock, patch

from deriva_ml_mcp.connection import (
    ConnectionManager,
    ConnectionInfo,
    MCP_WORKFLOW_TYPE,
    get_mcp_workflow_info,
)


class TestMCPWorkflowInfo:
    """Tests for MCP workflow info utilities."""

    def test_get_mcp_workflow_info_defaults(self):
        """Test default workflow info values."""
        info = get_mcp_workflow_info()
        assert info["workflow_name"] == "DerivaML MCP Server"
        assert info["workflow_type"] == MCP_WORKFLOW_TYPE
        assert info["in_docker"] is False

    @patch.dict("os.environ", {
        "DERIVAML_MCP_WORKFLOW_NAME": "Custom MCP",
        "DERIVAML_MCP_VERSION": "1.2.3",
        "DERIVAML_MCP_IN_DOCKER": "true",
    })
    def test_get_mcp_workflow_info_from_env(self):
        """Test workflow info from environment variables."""
        info = get_mcp_workflow_info()
        assert info["workflow_name"] == "Custom MCP"
        assert info["version"] == "1.2.3"
        assert info["in_docker"] is True


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    def test_init(self):
        """Test ConnectionManager initialization."""
        manager = ConnectionManager()
        assert manager._connections == {}
        assert manager._active_connection is None

    def test_connection_key(self):
        """Test connection key generation."""
        manager = ConnectionManager()
        key = manager._connection_key("example.org", "123")
        assert key == "example.org:123"

    def test_list_connections_empty(self):
        """Test listing connections when none exist."""
        manager = ConnectionManager()
        assert manager.list_connections() == []

    def test_get_active_none(self):
        """Test get_active returns None when no connection."""
        manager = ConnectionManager()
        assert manager.get_active() is None

    def test_get_active_execution_none(self):
        """Test get_active_execution returns None when no connection."""
        manager = ConnectionManager()
        assert manager.get_active_execution() is None

    def test_get_active_or_raise_no_connection(self):
        """Test get_active_or_raise raises when no connection."""
        from deriva_ml import DerivaMLException

        manager = ConnectionManager()
        with pytest.raises(DerivaMLException, match="No active catalog connection"):
            manager.get_active_or_raise()

    def test_get_active_execution_or_raise_no_connection(self):
        """Test get_active_execution_or_raise raises when no connection."""
        from deriva_ml import DerivaMLException

        manager = ConnectionManager()
        with pytest.raises(DerivaMLException, match="No active catalog connection"):
            manager.get_active_execution_or_raise()

    @patch("deriva_ml_mcp.connection.check_is_derivaml_catalog")
    @patch("deriva_ml.DerivaML")
    @patch("deriva_ml_mcp.connection.DerivaServer")
    def test_connect_success(self, mock_server, mock_derivaml, mock_check):
        """Test successful connection to a DerivaML catalog."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = "test_schema"
        mock_instance.host_name = "example.org"
        mock_instance.catalog_id = "123"
        mock_derivaml.return_value = mock_instance

        # Mock the server connection
        mock_catalog = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.connect_ermrest.return_value = mock_catalog
        mock_server.return_value = mock_server_instance

        # Mock the DerivaML check to return True
        mock_check.return_value = {
            "is_derivaml": True,
            "ml_schema": "deriva-ml",
            "domain_schema": "test_schema",
        }

        manager = ConnectionManager()
        # Mock _create_mcp_execution to avoid actual execution creation
        manager._create_mcp_execution = MagicMock(return_value=(None, None))

        result = manager.connect("example.org", "123")

        # Result is now ConnectionInfo, not DerivaML instance
        assert result.ml_instance == mock_instance
        assert result.is_derivaml is True
        assert manager._active_connection == "example.org:123"
        assert len(manager._connections) == 1

    @patch("deriva_ml_mcp.connection.check_is_derivaml_catalog")
    @patch("deriva_ml.DerivaML")
    @patch("deriva_ml_mcp.connection.DerivaServer")
    def test_connect_creates_workflow_and_execution(self, mock_server, mock_derivaml, mock_check):
        """Test that connect creates MCP workflow and execution."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = "test_schema"
        mock_instance.host_name = "example.org"
        mock_instance.catalog_id = "123"
        mock_derivaml.return_value = mock_instance

        # Mock the server connection
        mock_catalog = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.connect_ermrest.return_value = mock_catalog
        mock_server.return_value = mock_server_instance

        # Mock the DerivaML check to return True
        mock_check.return_value = {
            "is_derivaml": True,
            "ml_schema": "deriva-ml",
            "domain_schema": "test_schema",
        }

        mock_workflow = MagicMock()
        mock_execution = MagicMock()
        mock_execution.execution_rid = "EXE-123"

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=("WF-123", mock_execution))

        manager.connect("example.org", "123")

        conn_info = manager.get_active_connection_info()
        assert conn_info is not None
        assert conn_info.workflow_rid == "WF-123"
        assert conn_info.execution == mock_execution

    @patch("deriva_ml_mcp.connection.check_is_derivaml_catalog")
    @patch("deriva_ml.DerivaML")
    @patch("deriva_ml_mcp.connection.DerivaServer")
    def test_connect_reuses_existing(self, mock_server, mock_derivaml, mock_check):
        """Test that connecting to same catalog reuses connection."""
        mock_instance = MagicMock()
        mock_derivaml.return_value = mock_instance

        # Mock the server connection
        mock_catalog = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.connect_ermrest.return_value = mock_catalog
        mock_server.return_value = mock_server_instance

        # Mock the DerivaML check to return True
        mock_check.return_value = {
            "is_derivaml": True,
            "ml_schema": "deriva-ml",
            "domain_schema": "test_schema",
        }

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=(None, None))

        result1 = manager.connect("example.org", "123")
        result2 = manager.connect("example.org", "123")

        # Both results should be the same ConnectionInfo
        assert result1 == result2
        # DerivaML should only be called once (reuses existing connection)
        assert mock_derivaml.call_count == 1

    @patch("deriva_ml_mcp.connection.check_is_derivaml_catalog")
    @patch("deriva_ml.DerivaML")
    @patch("deriva_ml_mcp.connection.DerivaServer")
    def test_disconnect(self, mock_server, mock_derivaml, mock_check):
        """Test disconnecting from a catalog."""
        mock_instance = MagicMock()
        mock_derivaml.return_value = mock_instance

        # Mock the server connection
        mock_catalog = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.connect_ermrest.return_value = mock_catalog
        mock_server.return_value = mock_server_instance

        # Mock the DerivaML check to return True
        mock_check.return_value = {
            "is_derivaml": True,
            "ml_schema": "deriva-ml",
            "domain_schema": "test_schema",
        }

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=(None, None))
        manager.connect("example.org", "123")

        assert manager.disconnect() is True
        assert manager._active_connection is None
        assert len(manager._connections) == 0

    @patch("deriva_ml_mcp.connection.check_is_derivaml_catalog")
    @patch("deriva_ml.DerivaML")
    @patch("deriva_ml_mcp.connection.DerivaServer")
    def test_disconnect_closes_execution(self, mock_server, mock_derivaml, mock_check):
        """Test that disconnect closes the execution context."""
        mock_instance = MagicMock()
        mock_derivaml.return_value = mock_instance

        # Mock the server connection
        mock_catalog = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.connect_ermrest.return_value = mock_catalog
        mock_server.return_value = mock_server_instance

        # Mock the DerivaML check to return True
        mock_check.return_value = {
            "is_derivaml": True,
            "ml_schema": "deriva-ml",
            "domain_schema": "test_schema",
        }

        mock_execution = MagicMock()
        mock_execution.execution_rid = "EXE-123"

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=("WF-123", mock_execution))
        manager.connect("example.org", "123")

        manager.disconnect()

        # Verify execution context was closed
        mock_execution.__exit__.assert_called_once_with(None, None, None)

    def test_disconnect_no_connection(self):
        """Test disconnect when no connection exists."""
        manager = ConnectionManager()
        assert manager.disconnect() is False

    @patch("deriva_ml_mcp.connection.check_is_derivaml_catalog")
    @patch("deriva_ml.DerivaML")
    @patch("deriva_ml_mcp.connection.DerivaServer")
    def test_set_active(self, mock_server, mock_derivaml, mock_check):
        """Test setting active connection."""
        mock_instance = MagicMock()
        mock_derivaml.return_value = mock_instance

        # Mock the server connection
        mock_catalog = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.connect_ermrest.return_value = mock_catalog
        mock_server.return_value = mock_server_instance

        # Mock the DerivaML check to return True
        mock_check.return_value = {
            "is_derivaml": True,
            "ml_schema": "deriva-ml",
            "domain_schema": "test_schema",
        }

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=(None, None))
        manager.connect("example.org", "123", set_active=False)
        manager.connect("example.org", "456")

        assert manager._active_connection == "example.org:456"

        result = manager.set_active("example.org", "123")
        assert result is True
        assert manager._active_connection == "example.org:123"

    def test_set_active_not_found(self):
        """Test setting active to non-existent connection."""
        manager = ConnectionManager()
        result = manager.set_active("example.org", "123")
        assert result is False

    @patch("deriva_ml_mcp.connection.check_is_derivaml_catalog")
    @patch("deriva_ml.DerivaML")
    @patch("deriva_ml_mcp.connection.DerivaServer")
    def test_list_connections_includes_execution_info(self, mock_server, mock_derivaml, mock_check):
        """Test listing connections includes workflow and execution info."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = None
        mock_derivaml.return_value = mock_instance

        # Mock the server connection
        mock_catalog = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.connect_ermrest.return_value = mock_catalog
        mock_server.return_value = mock_server_instance

        # Mock the DerivaML check to return True
        mock_check.return_value = {
            "is_derivaml": True,
            "ml_schema": "deriva-ml",
            "domain_schema": None,
        }

        mock_execution = MagicMock()
        mock_execution.execution_rid = "EXE-123"

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=("WF-123", mock_execution))
        manager.connect("example.org", "123")

        connections = manager.list_connections()
        assert len(connections) == 1
        assert connections[0]["workflow_rid"] == "WF-123"
        assert connections[0]["execution_rid"] == "EXE-123"

    @patch("deriva_ml_mcp.connection.check_is_derivaml_catalog")
    @patch("deriva_ml.DerivaML")
    @patch("deriva_ml_mcp.connection.DerivaServer")
    def test_list_connections_multiple(self, mock_server, mock_derivaml, mock_check):
        """Test listing multiple connections."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = None
        mock_derivaml.return_value = mock_instance

        # Mock the server connection
        mock_catalog = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.connect_ermrest.return_value = mock_catalog
        mock_server.return_value = mock_server_instance

        # Mock the DerivaML check to return True
        mock_check.return_value = {
            "is_derivaml": True,
            "ml_schema": "deriva-ml",
            "domain_schema": None,
        }

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=(None, None))
        manager.connect("example.org", "123")
        manager.connect("example.org", "456")

        connections = manager.list_connections()
        assert len(connections) == 2

        # Check that exactly one is marked active
        active_count = sum(1 for c in connections if c["is_active"])
        assert active_count == 1
