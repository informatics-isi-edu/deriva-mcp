"""Unit tests for workflow management tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.conftest import assert_error, assert_success, parse_json_result


class TestLookupWorkflowByUrl:
    """Tests for the lookup_workflow_by_url tool."""

    @pytest.mark.asyncio
    async def test_lookup_found(self, workflow_tools, mock_ml):
        """When the URL matches an existing workflow, return found=True with details."""
        mock_workflow = MagicMock()
        mock_workflow.rid = "WF-001"
        mock_workflow.name = "Training Pipeline"
        mock_workflow.workflow_type = "Training"
        mock_workflow.description = "Trains a ResNet model"
        mock_workflow.url = "https://github.com/org/repo/blob/main/train.py"
        mock_workflow.checksum = "abc123"
        mock_workflow.version = "1.0"
        mock_ml.lookup_workflow_by_url.return_value = mock_workflow

        result = await workflow_tools["lookup_workflow_by_url"](
            url="https://github.com/org/repo/blob/main/train.py",
        )

        data = parse_json_result(result)
        assert data["found"] is True
        wf = data["workflow"]
        assert wf["rid"] == "WF-001"
        assert wf["name"] == "Training Pipeline"
        assert wf["workflow_type"] == "Training"
        assert wf["description"] == "Trains a ResNet model"
        assert wf["url"] == "https://github.com/org/repo/blob/main/train.py"
        assert wf["checksum"] == "abc123"
        assert wf["version"] == "1.0"
        mock_ml.lookup_workflow_by_url.assert_called_once_with(
            "https://github.com/org/repo/blob/main/train.py",
        )

    @pytest.mark.asyncio
    async def test_lookup_not_found(self, workflow_tools, mock_ml):
        """When no workflow matches the URL, return found=False."""
        mock_ml.lookup_workflow_by_url.return_value = None

        result = await workflow_tools["lookup_workflow_by_url"](
            url="https://github.com/org/repo/blob/main/nonexistent.py",
        )

        data = parse_json_result(result)
        assert data["found"] is False
        assert data["workflow"] is None

    @pytest.mark.asyncio
    async def test_lookup_no_connection(self, workflow_tools_disconnected):
        """When not connected to a catalog, return an error."""
        result = await workflow_tools_disconnected["lookup_workflow_by_url"](
            url="https://github.com/org/repo/blob/main/train.py",
        )

        assert_error(result, expected_message="No active catalog connection")


class TestCreateWorkflow:
    """Tests for the create_workflow tool."""

    @pytest.mark.asyncio
    async def test_create_success(self, workflow_tools, mock_ml):
        """Creating a workflow returns status=created with its RID and metadata."""
        mock_workflow = MagicMock()
        mock_workflow.name = "Inference Pipeline"
        mock_workflow.workflow_type = "Inference"
        mock_workflow.description = "Runs model inference"
        mock_ml.create_workflow.return_value = mock_workflow
        mock_ml.add_workflow.return_value = "WF-002"

        result = await workflow_tools["create_workflow"](
            name="Inference Pipeline",
            workflow_type="Inference",
            description="Runs model inference",
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["workflow_rid"] == "WF-002"
        assert data["name"] == "Inference Pipeline"
        assert data["workflow_type"] == "Inference"
        assert data["description"] == "Runs model inference"
        mock_ml.create_workflow.assert_called_once_with(
            name="Inference Pipeline",
            workflow_type="Inference",
            description="Runs model inference",
        )
        mock_ml.add_workflow.assert_called_once_with(mock_workflow)

    @pytest.mark.asyncio
    async def test_create_no_connection(self, workflow_tools_disconnected):
        """When not connected, create_workflow returns an error."""
        result = await workflow_tools_disconnected["create_workflow"](
            name="Pipeline",
            workflow_type="Training",
        )

        assert_error(result, expected_message="No active catalog connection")


class TestSetWorkflowDescription:
    """Tests for the set_workflow_description tool."""

    @pytest.mark.asyncio
    async def test_set_description_success(self, workflow_tools, mock_ml):
        """Setting a description returns status=updated with the new text."""
        mock_workflow = MagicMock()
        mock_ml.lookup_workflow.return_value = mock_workflow

        result = await workflow_tools["set_workflow_description"](
            workflow_rid="WF-003",
            description="Updated description for the workflow",
        )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["workflow_rid"] == "WF-003"
        assert data["description"] == "Updated description for the workflow"
        mock_ml.lookup_workflow.assert_called_once_with("WF-003")
        assert mock_workflow.description == "Updated description for the workflow"

    @pytest.mark.asyncio
    async def test_set_description_no_connection(self, workflow_tools_disconnected):
        """When not connected, set_workflow_description returns an error."""
        result = await workflow_tools_disconnected["set_workflow_description"](
            workflow_rid="WF-003",
            description="New description",
        )

        assert_error(result, expected_message="No active catalog connection")


class TestAddWorkflowType:
    """Tests for the add_workflow_type tool."""

    @pytest.mark.asyncio
    async def test_add_type_success(self, workflow_tools, mock_ml):
        """Adding a workflow type returns status=created with term details."""
        mock_term = MagicMock()
        mock_term.name = "Data Augmentation"
        mock_term.description = "Workflows that augment training data"
        mock_term.rid = "WT-010"
        mock_ml.add_term.return_value = mock_term

        result = await workflow_tools["add_workflow_type"](
            type_name="Data Augmentation",
            description="Workflows that augment training data",
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "Data Augmentation"
        assert data["description"] == "Workflows that augment training data"
        assert data["rid"] == "WT-010"

        # Verify add_term was called with exists_ok=True
        call_kwargs = mock_ml.add_term.call_args
        assert call_kwargs.kwargs["term_name"] == "Data Augmentation"
        assert call_kwargs.kwargs["description"] == "Workflows that augment training data"
        assert call_kwargs.kwargs["exists_ok"] is True

    @pytest.mark.asyncio
    async def test_add_type_no_connection(self, workflow_tools_disconnected):
        """When not connected, add_workflow_type returns an error."""
        result = await workflow_tools_disconnected["add_workflow_type"](
            type_name="Custom Type",
            description="A custom workflow type",
        )

        assert_error(result, expected_message="No active catalog connection")
