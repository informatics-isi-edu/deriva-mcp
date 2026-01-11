"""Tests for the connection manager."""

import pytest
from unittest.mock import MagicMock, patch

from deriva_ml_mcp.connection import ConnectionManager, ConnectionInfo


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

    def test_get_active_or_raise_no_connection(self):
        """Test get_active_or_raise raises when no connection."""
        from deriva_ml import DerivaMLException

        manager = ConnectionManager()
        with pytest.raises(DerivaMLException, match="No active catalog connection"):
            manager.get_active_or_raise()

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_connect_success(self, mock_derivaml):
        """Test successful connection."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = "test_schema"
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
        result = manager.connect("example.org", "123")

        assert result == mock_instance
        assert manager._active_connection == "example.org:123"
        assert len(manager._connections) == 1

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_connect_reuses_existing(self, mock_derivaml):
        """Test that connecting to same catalog reuses connection."""
        mock_instance = MagicMock()
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
        result1 = manager.connect("example.org", "123")
        result2 = manager.connect("example.org", "123")

        assert result1 == result2
        # DerivaML should only be called once
        assert mock_derivaml.call_count == 1

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_disconnect(self, mock_derivaml):
        """Test disconnecting from a catalog."""
        mock_instance = MagicMock()
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
        manager.connect("example.org", "123")

        assert manager.disconnect() is True
        assert manager._active_connection is None
        assert len(manager._connections) == 0

    def test_disconnect_no_connection(self):
        """Test disconnect when no connection exists."""
        manager = ConnectionManager()
        assert manager.disconnect() is False

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_set_active(self, mock_derivaml):
        """Test setting active connection."""
        mock_instance = MagicMock()
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
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

    @patch("deriva_ml_mcp.connection.DerivaML")
    def test_list_connections(self, mock_derivaml):
        """Test listing multiple connections."""
        mock_instance = MagicMock()
        mock_instance.domain_schema = None
        mock_derivaml.return_value = mock_instance

        manager = ConnectionManager()
        manager.connect("example.org", "123")
        manager.connect("example.org", "456")

        connections = manager.list_connections()
        assert len(connections) == 2

        # Check that exactly one is marked active
        active_count = sum(1 for c in connections if c["is_active"])
        assert active_count == 1
