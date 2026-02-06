"""Tests for the connection manager."""

import pytest
from unittest.mock import MagicMock, patch

from deriva_ml_mcp.connection import (
    ConnectionManager,
    ConnectionInfo,
    MCP_WORKFLOW_TYPE,
    derive_user_id,
    get_mcp_workflow_info,
)


class TestMCPWorkflowInfo:
    """Tests for MCP workflow info utilities."""

    def test_get_mcp_workflow_info_defaults(self):
        """Test default workflow info values."""
        info = get_mcp_workflow_info()
        assert info["workflow_name"] == "Deriva MCP Server"
        assert info["workflow_type"] == MCP_WORKFLOW_TYPE
        assert info["in_docker"] is False

    @patch.dict("os.environ", {
        "DERIVA_MCP_WORKFLOW_NAME": "Custom MCP",
        "DERIVA_MCP_VERSION": "1.2.3",
        "DERIVA_MCP_IN_DOCKER": "true",
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
        """Test connection key generation includes user_id."""
        manager = ConnectionManager()
        key = manager._connection_key("example.org", "123", "user_abc")
        assert key == "user_abc:example.org:123"

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
        with pytest.raises(DerivaMLException, match="No active execution context"):
            manager.get_active_execution_or_raise()

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_connect_success(self, mock_derivaml):
        """Test successful connection."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = "test_schema"
        mock_instance.host_name = "example.org"
        mock_instance.catalog_id = "123"
        mock_instance.credential = None  # No credential -> default_user
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
        # Mock _create_mcp_execution to avoid actual execution creation
        manager._create_mcp_execution = MagicMock(return_value=(None, None))

        result = manager.connect("example.org", "123")

        assert result == mock_instance
        assert manager._active_connection == "default_user:example.org:123"
        assert len(manager._connections) == 1

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_connect_creates_workflow_and_execution(self, mock_derivaml):
        """Test that connect creates MCP workflow and execution."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = "test_schema"
        mock_instance.host_name = "example.org"
        mock_instance.catalog_id = "123"
        mock_instance.credential = None
        mock_derivaml.return_value = mock_instance

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

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_connect_reuses_existing(self, mock_derivaml):
        """Test that connecting to same catalog reuses connection."""
        mock_instance = MagicMock()
        mock_instance.credential = None
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=(None, None))

        result1 = manager.connect("example.org", "123")
        result2 = manager.connect("example.org", "123")

        assert result1 == result2
        # DerivaML is called twice (to derive user_id), but only one connection is stored
        assert len(manager._connections) == 1

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_disconnect(self, mock_derivaml):
        """Test disconnecting from a catalog."""
        mock_instance = MagicMock()
        mock_instance.credential = None
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=(None, None))
        manager.connect("example.org", "123")

        assert manager.disconnect() is True
        assert manager._active_connection is None
        assert len(manager._connections) == 0

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_disconnect_closes_execution(self, mock_derivaml):
        """Test that disconnect closes the execution context."""
        mock_instance = MagicMock()
        mock_instance.credential = None
        mock_derivaml.return_value = mock_instance

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

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_set_active(self, mock_derivaml):
        """Test setting active connection."""
        mock_instance = MagicMock()
        mock_instance.credential = None
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=(None, None))
        manager.connect("example.org", "123", set_active=False)
        manager.connect("example.org", "456")

        assert manager._active_connection == "default_user:example.org:456"

        result = manager.set_active("example.org", "123")
        assert result is True
        assert manager._active_connection == "default_user:example.org:123"

    def test_set_active_not_found(self):
        """Test setting active to non-existent connection."""
        manager = ConnectionManager()
        result = manager.set_active("example.org", "123")
        assert result is False

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_list_connections_includes_execution_info(self, mock_derivaml):
        """Test listing connections includes workflow and execution info."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = None
        mock_instance.credential = None
        mock_derivaml.return_value = mock_instance

        mock_execution = MagicMock()
        mock_execution.execution_rid = "EXE-123"

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=("WF-123", mock_execution))
        manager.connect("example.org", "123")

        connections = manager.list_connections()
        assert len(connections) == 1
        assert connections[0]["workflow_rid"] == "WF-123"
        assert connections[0]["execution_rid"] == "EXE-123"

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_list_connections_multiple(self, mock_derivaml):
        """Test listing multiple connections."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = None
        mock_instance.credential = None
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
        manager._create_mcp_execution = MagicMock(return_value=(None, None))
        manager.connect("example.org", "123")
        manager.connect("example.org", "456")

        connections = manager.list_connections()
        assert len(connections) == 2

        # Check that exactly one is marked active
        active_count = sum(1 for c in connections if c["is_active"])
        assert active_count == 1
