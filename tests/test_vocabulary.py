"""Unit tests for vocabulary management tools.

Tests all 6 vocabulary tools:
    - add_term
    - create_vocabulary
    - add_synonym
    - remove_synonym
    - update_term_description
    - delete_term
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.conftest import assert_error, assert_success, parse_json_result


# =============================================================================
# Helpers
# =============================================================================


def _make_term(
    name: str = "Training",
    description: str = "Data used for training",
    synonyms: tuple[str, ...] | None = None,
    rid: str = "1-0001",
) -> MagicMock:
    """Create a mock vocabulary term object."""
    term = MagicMock()
    term.name = name
    term.description = description
    term.synonyms = synonyms if synonyms is not None else ()
    term.rid = rid
    return term


def _make_table(name: str = "Quality_Level", schema_name: str = "test_schema") -> MagicMock:
    """Create a mock vocabulary table object."""
    table = MagicMock()
    table.name = name
    table.schema = MagicMock()
    table.schema.name = schema_name
    return table


# =============================================================================
# TestAddTerm
# =============================================================================


class TestAddTerm:
    """Tests for the add_term tool."""

    @pytest.mark.asyncio
    async def test_success_no_synonyms(self, vocab_tools, mock_ml):
        """add_term creates a term and returns its details."""
        mock_term = _make_term(name="Validation", description="Held-out data", rid="2-0001")
        mock_ml.add_term.return_value = mock_term

        result = await vocab_tools["add_term"](
            vocabulary_name="Dataset_Type",
            term_name="Validation",
            description="Held-out data",
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "Validation"
        assert data["description"] == "Held-out data"
        assert data["synonyms"] == []
        assert data["rid"] == "2-0001"

        mock_ml.add_term.assert_called_once_with(
            table="Dataset_Type",
            term_name="Validation",
            description="Held-out data",
            synonyms=[],
            exists_ok=False,
        )

    @pytest.mark.asyncio
    async def test_success_with_synonyms(self, vocab_tools, mock_ml):
        """add_term passes synonyms through and returns them."""
        mock_term = _make_term(
            name="Training",
            description="Training data",
            synonyms=("train", "trn"),
            rid="3-0001",
        )
        mock_ml.add_term.return_value = mock_term

        result = await vocab_tools["add_term"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="Training data",
            synonyms=["train", "trn"],
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "Training"
        assert data["synonyms"] == ["train", "trn"]
        assert data["rid"] == "3-0001"

        mock_ml.add_term.assert_called_once_with(
            table="Dataset_Type",
            term_name="Training",
            description="Training data",
            synonyms=["train", "trn"],
            exists_ok=False,
        )

    @pytest.mark.asyncio
    async def test_term_already_exists_with_lookup(self, vocab_tools, mock_ml):
        """add_term returns 'exists' status when term already exists and lookup succeeds."""
        mock_ml.add_term.side_effect = Exception("Term 'Training' already exists in Dataset_Type")

        existing_term = _make_term(name="Training", description="Existing desc", rid="4-0001")
        mock_ml.lookup_term.return_value = existing_term

        result = await vocab_tools["add_term"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="Training data",
        )

        data = parse_json_result(result)
        assert data["status"] == "exists"
        assert data["name"] == "Training"
        assert data["description"] == "Existing desc"
        assert data["rid"] == "4-0001"

        mock_ml.lookup_term.assert_called_once_with("Dataset_Type", "Training")

    @pytest.mark.asyncio
    async def test_term_already_exists_lookup_fails(self, vocab_tools, mock_ml):
        """add_term returns error when term already exists but lookup also fails."""
        mock_ml.add_term.side_effect = Exception("Term 'Training' already exists in Dataset_Type")
        mock_ml.lookup_term.side_effect = Exception("Lookup failed")

        result = await vocab_tools["add_term"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="Training data",
        )

        data = assert_error(result)
        assert "already exists" in data["message"]

    @pytest.mark.asyncio
    async def test_generic_exception(self, vocab_tools, mock_ml):
        """add_term returns error for non-'already exists' exceptions."""
        mock_ml.add_term.side_effect = Exception("Connection timeout")

        result = await vocab_tools["add_term"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="Training data",
        )

        data = assert_error(result, "Connection timeout")

    @pytest.mark.asyncio
    async def test_no_connection(self, vocab_tools_disconnected):
        """add_term returns error when not connected to a catalog."""
        result = await vocab_tools_disconnected["add_term"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="Training data",
        )

        assert_error(result, "No active")

    @pytest.mark.asyncio
    async def test_synonyms_none_becomes_empty_list(self, vocab_tools, mock_ml):
        """add_term converts None synonyms to empty list when calling ml.add_term."""
        mock_term = _make_term()
        mock_ml.add_term.return_value = mock_term

        await vocab_tools["add_term"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="Training data",
            synonyms=None,
        )

        _, kwargs = mock_ml.add_term.call_args
        assert kwargs["synonyms"] == []

    @pytest.mark.asyncio
    async def test_term_with_none_synonyms_in_response(self, vocab_tools, mock_ml):
        """add_term returns empty list when the returned term has None synonyms."""
        mock_term = _make_term()
        mock_term.synonyms = None
        mock_ml.add_term.return_value = mock_term

        result = await vocab_tools["add_term"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="Training data",
        )

        data = assert_success(result)
        assert data["synonyms"] == []


# =============================================================================
# TestCreateVocabulary
# =============================================================================


class TestCreateVocabulary:
    """Tests for the create_vocabulary tool."""

    @pytest.mark.asyncio
    async def test_success_default_schema(self, vocab_tools, mock_ml):
        """create_vocabulary creates a vocabulary table with default schema."""
        mock_table = _make_table(name="Quality_Level", schema_name="test_schema")
        mock_ml.create_vocabulary.return_value = mock_table

        result = await vocab_tools["create_vocabulary"](
            vocabulary_name="Quality_Level",
            comment="Image quality ratings",
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "Quality_Level"
        assert data["schema"] == "test_schema"
        assert data["comment"] == "Image quality ratings"

        mock_ml.create_vocabulary.assert_called_once_with(
            vocab_name="Quality_Level",
            comment="Image quality ratings",
            schema=None,
        )

    @pytest.mark.asyncio
    async def test_success_explicit_schema(self, vocab_tools, mock_ml):
        """create_vocabulary passes schema argument when provided."""
        mock_table = _make_table(name="My_Vocab", schema_name="custom_schema")
        mock_ml.create_vocabulary.return_value = mock_table

        result = await vocab_tools["create_vocabulary"](
            vocabulary_name="My_Vocab",
            comment="A custom vocabulary",
            schema="custom_schema",
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["name"] == "My_Vocab"
        assert data["schema"] == "custom_schema"
        assert data["comment"] == "A custom vocabulary"

        mock_ml.create_vocabulary.assert_called_once_with(
            vocab_name="My_Vocab",
            comment="A custom vocabulary",
            schema="custom_schema",
        )

    @pytest.mark.asyncio
    async def test_success_empty_comment(self, vocab_tools, mock_ml):
        """create_vocabulary works with default empty comment."""
        mock_table = _make_table(name="Simple_Vocab")
        mock_ml.create_vocabulary.return_value = mock_table

        result = await vocab_tools["create_vocabulary"](
            vocabulary_name="Simple_Vocab",
        )

        data = assert_success(result)
        assert data["status"] == "created"
        assert data["comment"] == ""

        mock_ml.create_vocabulary.assert_called_once_with(
            vocab_name="Simple_Vocab",
            comment="",
            schema=None,
        )

    @pytest.mark.asyncio
    async def test_exception(self, vocab_tools, mock_ml):
        """create_vocabulary returns error when ml.create_vocabulary raises."""
        mock_ml.create_vocabulary.side_effect = Exception("Table already exists")

        result = await vocab_tools["create_vocabulary"](
            vocabulary_name="Quality_Level",
        )

        assert_error(result, "Table already exists")

    @pytest.mark.asyncio
    async def test_no_connection(self, vocab_tools_disconnected):
        """create_vocabulary returns error when not connected."""
        result = await vocab_tools_disconnected["create_vocabulary"](
            vocabulary_name="Quality_Level",
        )

        assert_error(result, "No active")


# =============================================================================
# TestAddSynonym
# =============================================================================


class TestAddSynonym:
    """Tests for the add_synonym tool."""

    @pytest.mark.asyncio
    async def test_success(self, vocab_tools, mock_ml):
        """add_synonym adds a new synonym to an existing term."""
        mock_term = _make_term(name="Training", synonyms=("train",), rid="5-0001")
        mock_ml.lookup_term.return_value = mock_term

        # The tool does:
        #   current = list(term.synonyms)  -> ["train"]
        #   if "trn" not in current:       -> True
        #       term.synonyms = tuple(current + ["trn"])  -> sets to ("train", "trn")
        #   return ... list(term.synonyms) -> ["train", "trn"]
        # Since mock_term.synonyms is a plain attribute, the assignment works naturally.

        result = await vocab_tools["add_synonym"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            synonym="trn",
        )

        data = assert_success(result)
        assert data["status"] == "added"
        assert data["name"] == "Training"
        assert data["synonyms"] == ["train", "trn"]
        assert data["rid"] == "5-0001"

        mock_ml.lookup_term.assert_called_once_with("Dataset_Type", "Training")

    @pytest.mark.asyncio
    async def test_duplicate_synonym_not_added(self, vocab_tools, mock_ml):
        """add_synonym does not duplicate an existing synonym."""
        mock_term = _make_term(name="Training", synonyms=("train",), rid="5-0001")
        mock_ml.lookup_term.return_value = mock_term

        result = await vocab_tools["add_synonym"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            synonym="train",
        )

        data = assert_success(result)
        assert data["status"] == "added"
        assert data["name"] == "Training"
        assert data["synonyms"] == ["train"]
        assert data["rid"] == "5-0001"

    @pytest.mark.asyncio
    async def test_exception_on_lookup(self, vocab_tools, mock_ml):
        """add_synonym returns error when lookup_term fails."""
        mock_ml.lookup_term.side_effect = Exception("Term not found: Unknown")

        result = await vocab_tools["add_synonym"](
            vocabulary_name="Dataset_Type",
            term_name="Unknown",
            synonym="unk",
        )

        assert_error(result, "Term not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, vocab_tools_disconnected):
        """add_synonym returns error when not connected."""
        result = await vocab_tools_disconnected["add_synonym"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            synonym="train",
        )

        assert_error(result, "No active")

    @pytest.mark.asyncio
    async def test_add_synonym_to_term_with_no_synonyms(self, vocab_tools, mock_ml):
        """add_synonym works when term has no existing synonyms."""
        mock_term = _make_term(name="Testing", synonyms=(), rid="6-0001")
        mock_ml.lookup_term.return_value = mock_term

        result = await vocab_tools["add_synonym"](
            vocabulary_name="Dataset_Type",
            term_name="Testing",
            synonym="test",
        )

        data = assert_success(result)
        assert data["status"] == "added"
        assert data["name"] == "Testing"
        assert data["synonyms"] == ["test"]
        assert data["rid"] == "6-0001"


# =============================================================================
# TestRemoveSynonym
# =============================================================================


class TestRemoveSynonym:
    """Tests for the remove_synonym tool."""

    @pytest.mark.asyncio
    async def test_success(self, vocab_tools, mock_ml):
        """remove_synonym removes an existing synonym from a term."""
        mock_term = _make_term(name="Training", synonyms=("train", "trn"), rid="7-0001")
        mock_ml.lookup_term.return_value = mock_term

        # The tool does:
        #   current = list(term.synonyms)  -> ["train", "trn"]
        #   if "train" in current:         -> True
        #       term.synonyms = tuple(s for s in current if s != "train")  -> ("trn",)
        #   return ... list(term.synonyms) -> ["trn"]

        result = await vocab_tools["remove_synonym"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            synonym="train",
        )

        data = assert_success(result)
        assert data["status"] == "removed"
        assert data["name"] == "Training"
        assert data["synonyms"] == ["trn"]
        assert data["rid"] == "7-0001"

        mock_ml.lookup_term.assert_called_once_with("Dataset_Type", "Training")

    @pytest.mark.asyncio
    async def test_synonym_not_present(self, vocab_tools, mock_ml):
        """remove_synonym succeeds silently when synonym is not present."""
        mock_term = _make_term(name="Training", synonyms=("train",), rid="7-0002")
        mock_ml.lookup_term.return_value = mock_term

        result = await vocab_tools["remove_synonym"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            synonym="nonexistent",
        )

        data = assert_success(result)
        assert data["status"] == "removed"
        assert data["name"] == "Training"
        assert data["synonyms"] == ["train"]
        assert data["rid"] == "7-0002"

    @pytest.mark.asyncio
    async def test_exception_on_lookup(self, vocab_tools, mock_ml):
        """remove_synonym returns error when lookup_term fails."""
        mock_ml.lookup_term.side_effect = Exception("Vocabulary not found: BadVocab")

        result = await vocab_tools["remove_synonym"](
            vocabulary_name="BadVocab",
            term_name="Training",
            synonym="train",
        )

        assert_error(result, "Vocabulary not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, vocab_tools_disconnected):
        """remove_synonym returns error when not connected."""
        result = await vocab_tools_disconnected["remove_synonym"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            synonym="train",
        )

        assert_error(result, "No active")

    @pytest.mark.asyncio
    async def test_remove_last_synonym(self, vocab_tools, mock_ml):
        """remove_synonym works when removing the only synonym."""
        mock_term = _make_term(name="Training", synonyms=("train",), rid="7-0003")
        mock_ml.lookup_term.return_value = mock_term

        result = await vocab_tools["remove_synonym"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            synonym="train",
        )

        data = assert_success(result)
        assert data["status"] == "removed"
        assert data["name"] == "Training"
        assert data["synonyms"] == []
        assert data["rid"] == "7-0003"


# =============================================================================
# TestUpdateTermDescription
# =============================================================================


class TestUpdateTermDescription:
    """Tests for the update_term_description tool."""

    @pytest.mark.asyncio
    async def test_success(self, vocab_tools, mock_ml):
        """update_term_description sets the description and returns updated info."""
        mock_term = _make_term(name="Training", description="Old description", rid="8-0001")
        mock_ml.lookup_term.return_value = mock_term

        # The tool does: term.description = description, then reads term.description
        # MagicMock will store the set value and return it on read.

        result = await vocab_tools["update_term_description"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="New description for training data",
        )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["name"] == "Training"
        assert data["description"] == "New description for training data"
        assert data["rid"] == "8-0001"

        mock_ml.lookup_term.assert_called_once_with("Dataset_Type", "Training")

    @pytest.mark.asyncio
    async def test_empty_description(self, vocab_tools, mock_ml):
        """update_term_description works with an empty description string."""
        mock_term = _make_term(name="Training", description="Old", rid="8-0002")
        mock_ml.lookup_term.return_value = mock_term

        result = await vocab_tools["update_term_description"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="",
        )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["description"] == ""

    @pytest.mark.asyncio
    async def test_exception_on_lookup(self, vocab_tools, mock_ml):
        """update_term_description returns error when term not found."""
        mock_ml.lookup_term.side_effect = Exception("Term 'Missing' not found")

        result = await vocab_tools["update_term_description"](
            vocabulary_name="Dataset_Type",
            term_name="Missing",
            description="Some description",
        )

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, vocab_tools_disconnected):
        """update_term_description returns error when not connected."""
        result = await vocab_tools_disconnected["update_term_description"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description="New description",
        )

        assert_error(result, "No active")

    @pytest.mark.asyncio
    async def test_description_with_special_characters(self, vocab_tools, mock_ml):
        """update_term_description handles descriptions with special characters."""
        mock_term = _make_term(name="Training", rid="8-0003")
        mock_ml.lookup_term.return_value = mock_term

        special_desc = 'Data with "quotes", newlines\nand unicode: \u00e9\u00e8\u00ea'

        result = await vocab_tools["update_term_description"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
            description=special_desc,
        )

        data = assert_success(result)
        assert data["status"] == "updated"
        assert data["description"] == special_desc


# =============================================================================
# TestDeleteTerm
# =============================================================================


class TestDeleteTerm:
    """Tests for the delete_term tool."""

    @pytest.mark.asyncio
    async def test_success(self, vocab_tools, mock_ml):
        """delete_term deletes a term and returns confirmation."""
        mock_ml.delete_term.return_value = None

        result = await vocab_tools["delete_term"](
            vocabulary_name="Dataset_Type",
            term_name="Obsolete",
        )

        data = assert_success(result)
        assert data["status"] == "deleted"
        assert data["vocabulary"] == "Dataset_Type"
        assert data["name"] == "Obsolete"

        mock_ml.delete_term.assert_called_once_with("Dataset_Type", "Obsolete")

    @pytest.mark.asyncio
    async def test_term_in_use(self, vocab_tools, mock_ml):
        """delete_term returns error when term is referenced by other records."""
        mock_ml.delete_term.side_effect = Exception(
            "Cannot delete term 'Training': referenced by 5 records"
        )

        result = await vocab_tools["delete_term"](
            vocabulary_name="Dataset_Type",
            term_name="Training",
        )

        assert_error(result, "referenced by 5 records")

    @pytest.mark.asyncio
    async def test_term_not_found(self, vocab_tools, mock_ml):
        """delete_term returns error when the term does not exist."""
        mock_ml.delete_term.side_effect = Exception("Term 'Ghost' not found in Dataset_Type")

        result = await vocab_tools["delete_term"](
            vocabulary_name="Dataset_Type",
            term_name="Ghost",
        )

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, vocab_tools_disconnected):
        """delete_term returns error when not connected."""
        result = await vocab_tools_disconnected["delete_term"](
            vocabulary_name="Dataset_Type",
            term_name="Obsolete",
        )

        assert_error(result, "No active")

    @pytest.mark.asyncio
    async def test_generic_exception(self, vocab_tools, mock_ml):
        """delete_term returns error on unexpected exception."""
        mock_ml.delete_term.side_effect = RuntimeError("Server unavailable")

        result = await vocab_tools["delete_term"](
            vocabulary_name="Dataset_Type",
            term_name="SomeTerm",
        )

        assert_error(result, "Server unavailable")
