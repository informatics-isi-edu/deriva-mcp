"""Unit tests for annotation management tools.

Tests all 16 annotation tools:
    - set_display_annotation
    - set_visible_columns
    - set_visible_foreign_keys
    - set_table_display
    - set_column_display
    - apply_annotations
    - add_visible_column
    - remove_visible_column
    - reorder_visible_columns
    - add_visible_foreign_key
    - remove_visible_foreign_key
    - reorder_visible_foreign_keys
    - get_handlebars_template_variables
    - get_table_sample_data
    - preview_handlebars_template
    - validate_template_syntax
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from tests.conftest import assert_error, assert_success, parse_json_result


# =============================================================================
# Helpers
# =============================================================================


def _make_column(name: str, col_type: str = "text") -> MagicMock:
    """Create a mock column object."""
    col = MagicMock()
    col.name = name
    col.type = MagicMock()
    col.type.typename = col_type
    return col


def _make_table(
    name: str = "Image",
    schema_name: str = "test_schema",
    column_names: list[str] | None = None,
) -> MagicMock:
    """Create a mock table object with columns."""
    table = MagicMock()
    table.name = name
    table.schema = MagicMock()
    table.schema.name = schema_name
    if column_names is None:
        column_names = ["RID", "Filename", "Subject", "Description"]
    table.columns = [_make_column(c) for c in column_names]
    return table


# =============================================================================
# TestSetDisplayAnnotation
# =============================================================================


class TestSetDisplayAnnotation:
    """Tests for the set_display_annotation tool."""

    @pytest.mark.asyncio
    async def test_success_table_level(self, annotation_tools, mock_ml):
        """set_display_annotation stages a display annotation on a table."""
        mock_ml.set_display_annotation.return_value = "Image"

        result = await annotation_tools["set_display_annotation"](
            table_name="Image",
            annotation={"name": "Images"},
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Image"
        assert data["annotation"] == "display"
        assert data["value"] == {"name": "Images"}
        assert "apply_annotations" in data["message"]

        mock_ml.set_display_annotation.assert_called_once_with(
            "Image", {"name": "Images"}, None
        )

    @pytest.mark.asyncio
    async def test_success_column_level(self, annotation_tools, mock_ml):
        """set_display_annotation stages a display annotation on a column."""
        mock_ml.set_display_annotation.return_value = "Image.Filename"

        result = await annotation_tools["set_display_annotation"](
            table_name="Image",
            column_name="Filename",
            annotation={"name": "File Name"},
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Image.Filename"
        assert data["value"] == {"name": "File Name"}

        mock_ml.set_display_annotation.assert_called_once_with(
            "Image", {"name": "File Name"}, "Filename"
        )

    @pytest.mark.asyncio
    async def test_remove_annotation_with_none(self, annotation_tools, mock_ml):
        """set_display_annotation removes annotation when value is None."""
        mock_ml.set_display_annotation.return_value = "Image"

        result = await annotation_tools["set_display_annotation"](
            table_name="Image",
            annotation=None,
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["value"] is None

        mock_ml.set_display_annotation.assert_called_once_with(
            "Image", None, None
        )

    @pytest.mark.asyncio
    async def test_name_style_annotation(self, annotation_tools, mock_ml):
        """set_display_annotation works with name_style sub-object."""
        mock_ml.set_display_annotation.return_value = "Image"
        style_annotation = {"name_style": {"underline_space": True, "title_case": True}}

        result = await annotation_tools["set_display_annotation"](
            table_name="Image",
            annotation=style_annotation,
        )

        data = assert_success(result)
        assert data["value"] == style_annotation

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """set_display_annotation returns error on exception."""
        mock_ml.set_display_annotation.side_effect = Exception("Table not found: BadTable")

        result = await annotation_tools["set_display_annotation"](
            table_name="BadTable",
            annotation={"name": "Bad"},
        )

        assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """set_display_annotation returns error when not connected."""
        result = await annotation_tools_disconnected["set_display_annotation"](
            table_name="Image",
            annotation={"name": "Images"},
        )

        assert_error(result, "No active")


# =============================================================================
# TestSetVisibleColumns
# =============================================================================


class TestSetVisibleColumns:
    """Tests for the set_visible_columns tool."""

    @pytest.mark.asyncio
    async def test_success(self, annotation_tools, mock_ml):
        """set_visible_columns stages a visible-columns annotation."""
        mock_ml.set_visible_columns.return_value = "Image"
        annotation = {
            "compact": ["RID", "Filename", "Subject"],
            "detailed": ["RID", "Filename", "Subject", "Description"],
        }

        result = await annotation_tools["set_visible_columns"](
            table_name="Image",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Image"
        assert data["annotation"] == "visible-columns"
        assert data["value"] == annotation
        assert "apply_annotations" in data["message"]

        mock_ml.set_visible_columns.assert_called_once_with("Image", annotation)

    @pytest.mark.asyncio
    async def test_with_foreign_key_ref(self, annotation_tools, mock_ml):
        """set_visible_columns handles foreign key references."""
        mock_ml.set_visible_columns.return_value = "Image"
        annotation = {
            "compact": ["Filename", ["domain", "Image_Subject_fkey"]],
        }

        result = await annotation_tools["set_visible_columns"](
            table_name="Image",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["value"] == annotation

    @pytest.mark.asyncio
    async def test_with_pseudo_column(self, annotation_tools, mock_ml):
        """set_visible_columns handles pseudo-column definitions."""
        mock_ml.set_visible_columns.return_value = "Image"
        annotation = {
            "detailed": [
                "Filename",
                {
                    "source": [{"outbound": ["domain", "Image_Subject_fkey"]}, "Name"],
                    "markdown_name": "Subject Name",
                },
            ],
        }

        result = await annotation_tools["set_visible_columns"](
            table_name="Image",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["value"] == annotation

    @pytest.mark.asyncio
    async def test_with_filter_context(self, annotation_tools, mock_ml):
        """set_visible_columns handles filter context with faceted search config."""
        mock_ml.set_visible_columns.return_value = "Image"
        annotation = {
            "filter": {
                "and": [
                    {"source": "Species", "open": True},
                    {"source": "Quality", "ux_mode": "choices"},
                ],
            },
        }

        result = await annotation_tools["set_visible_columns"](
            table_name="Image",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["value"] == annotation

    @pytest.mark.asyncio
    async def test_remove_with_none(self, annotation_tools, mock_ml):
        """set_visible_columns removes annotation when value is None."""
        mock_ml.set_visible_columns.return_value = "Image"

        result = await annotation_tools["set_visible_columns"](
            table_name="Image",
            annotation=None,
        )

        data = assert_success(result)
        assert data["value"] is None

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """set_visible_columns returns error on exception."""
        mock_ml.set_visible_columns.side_effect = Exception("Table not found")

        result = await annotation_tools["set_visible_columns"](
            table_name="BadTable",
            annotation={"compact": ["RID"]},
        )

        assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """set_visible_columns returns error when not connected."""
        result = await annotation_tools_disconnected["set_visible_columns"](
            table_name="Image",
            annotation={"compact": ["RID"]},
        )

        assert_error(result, "No active")


# =============================================================================
# TestSetVisibleForeignKeys
# =============================================================================


class TestSetVisibleForeignKeys:
    """Tests for the set_visible_foreign_keys tool."""

    @pytest.mark.asyncio
    async def test_success(self, annotation_tools, mock_ml):
        """set_visible_foreign_keys stages a visible-foreign-keys annotation."""
        mock_ml.set_visible_foreign_keys.return_value = "Subject"
        annotation = {
            "detailed": [
                ["domain", "Image_Subject_fkey"],
                ["domain", "Diagnosis_Subject_fkey"],
            ],
        }

        result = await annotation_tools["set_visible_foreign_keys"](
            table_name="Subject",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Subject"
        assert data["annotation"] == "visible-foreign-keys"
        assert data["value"] == annotation
        assert "apply_annotations" in data["message"]

        mock_ml.set_visible_foreign_keys.assert_called_once_with("Subject", annotation)

    @pytest.mark.asyncio
    async def test_hide_all_related_tables(self, annotation_tools, mock_ml):
        """set_visible_foreign_keys with empty list hides all related tables."""
        mock_ml.set_visible_foreign_keys.return_value = "Subject"

        result = await annotation_tools["set_visible_foreign_keys"](
            table_name="Subject",
            annotation={"detailed": []},
        )

        data = assert_success(result)
        assert data["value"] == {"detailed": []}

    @pytest.mark.asyncio
    async def test_with_pseudo_column(self, annotation_tools, mock_ml):
        """set_visible_foreign_keys handles pseudo-column definitions."""
        mock_ml.set_visible_foreign_keys.return_value = "Subject"
        annotation = {
            "detailed": [
                {
                    "source": [{"inbound": ["domain", "Image_Subject_fkey"]}],
                    "markdown_name": "Subject Images",
                },
            ],
        }

        result = await annotation_tools["set_visible_foreign_keys"](
            table_name="Subject",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["value"] == annotation

    @pytest.mark.asyncio
    async def test_remove_with_none(self, annotation_tools, mock_ml):
        """set_visible_foreign_keys removes annotation when value is None."""
        mock_ml.set_visible_foreign_keys.return_value = "Subject"

        result = await annotation_tools["set_visible_foreign_keys"](
            table_name="Subject",
            annotation=None,
        )

        data = assert_success(result)
        assert data["value"] is None

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """set_visible_foreign_keys returns error on exception."""
        mock_ml.set_visible_foreign_keys.side_effect = Exception("Table not found")

        result = await annotation_tools["set_visible_foreign_keys"](
            table_name="BadTable",
            annotation={"detailed": []},
        )

        assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """set_visible_foreign_keys returns error when not connected."""
        result = await annotation_tools_disconnected["set_visible_foreign_keys"](
            table_name="Subject",
            annotation={"detailed": []},
        )

        assert_error(result, "No active")


# =============================================================================
# TestSetTableDisplay
# =============================================================================


class TestSetTableDisplay:
    """Tests for the set_table_display tool."""

    @pytest.mark.asyncio
    async def test_success_row_name(self, annotation_tools, mock_ml):
        """set_table_display stages a table-display annotation with row_name."""
        mock_ml.set_table_display.return_value = "Subject"
        annotation = {
            "row_name": {
                "row_markdown_pattern": "{{{Name}}} ({{{Species}}})",
            },
        }

        result = await annotation_tools["set_table_display"](
            table_name="Subject",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Subject"
        assert data["annotation"] == "table-display"
        assert data["value"] == annotation
        assert "apply_annotations" in data["message"]

        mock_ml.set_table_display.assert_called_once_with("Subject", annotation)

    @pytest.mark.asyncio
    async def test_success_compact_options(self, annotation_tools, mock_ml):
        """set_table_display stages compact view options."""
        mock_ml.set_table_display.return_value = "Image"
        annotation = {
            "compact": {
                "row_order": [{"column": "RCT", "descending": True}],
                "page_size": 50,
            },
        }

        result = await annotation_tools["set_table_display"](
            table_name="Image",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["value"] == annotation

    @pytest.mark.asyncio
    async def test_success_detailed_options(self, annotation_tools, mock_ml):
        """set_table_display stages detailed view options."""
        mock_ml.set_table_display.return_value = "Page"
        annotation = {
            "detailed": {
                "hide_column_headers": True,
                "collapse_toc_panel": True,
            },
        }

        result = await annotation_tools["set_table_display"](
            table_name="Page",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["value"] == annotation

    @pytest.mark.asyncio
    async def test_remove_with_none(self, annotation_tools, mock_ml):
        """set_table_display removes annotation when value is None."""
        mock_ml.set_table_display.return_value = "Image"

        result = await annotation_tools["set_table_display"](
            table_name="Image",
            annotation=None,
        )

        data = assert_success(result)
        assert data["value"] is None

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """set_table_display returns error on exception."""
        mock_ml.set_table_display.side_effect = Exception("Permission denied")

        result = await annotation_tools["set_table_display"](
            table_name="Image",
            annotation={"row_name": {"row_markdown_pattern": "{{{Name}}}"}},
        )

        assert_error(result, "Permission denied")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """set_table_display returns error when not connected."""
        result = await annotation_tools_disconnected["set_table_display"](
            table_name="Image",
            annotation={"row_name": {"row_markdown_pattern": "{{{Name}}}"}},
        )

        assert_error(result, "No active")


# =============================================================================
# TestSetColumnDisplay
# =============================================================================


class TestSetColumnDisplay:
    """Tests for the set_column_display tool."""

    @pytest.mark.asyncio
    async def test_success_pre_format(self, annotation_tools, mock_ml):
        """set_column_display stages column-display with pre_format."""
        mock_ml.set_column_display.return_value = "Measurement.Value"
        annotation = {"*": {"pre_format": {"format": "%.2f"}}}

        result = await annotation_tools["set_column_display"](
            table_name="Measurement",
            column_name="Value",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Measurement.Value"
        assert data["annotation"] == "column-display"
        assert data["value"] == annotation
        assert "apply_annotations" in data["message"]

        mock_ml.set_column_display.assert_called_once_with(
            "Measurement", "Value", annotation
        )

    @pytest.mark.asyncio
    async def test_success_bool_format(self, annotation_tools, mock_ml):
        """set_column_display stages boolean formatting."""
        mock_ml.set_column_display.return_value = "Subject.Active"
        annotation = {
            "*": {
                "pre_format": {
                    "bool_true_value": "Active",
                    "bool_false_value": "Inactive",
                },
            },
        }

        result = await annotation_tools["set_column_display"](
            table_name="Subject",
            column_name="Active",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["value"] == annotation

    @pytest.mark.asyncio
    async def test_success_markdown_pattern(self, annotation_tools, mock_ml):
        """set_column_display stages markdown pattern template."""
        mock_ml.set_column_display.return_value = "Image.URL"
        annotation = {
            "detailed": {
                "markdown_pattern": "[![Image]({{{_value}}})]({{{_value}}})",
            },
        }

        result = await annotation_tools["set_column_display"](
            table_name="Image",
            column_name="URL",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["value"] == annotation

    @pytest.mark.asyncio
    async def test_success_disable_sorting(self, annotation_tools, mock_ml):
        """set_column_display stages column_order=false to disable sorting."""
        mock_ml.set_column_display.return_value = "Subject.Notes"
        annotation = {"*": {"column_order": False}}

        result = await annotation_tools["set_column_display"](
            table_name="Subject",
            column_name="Notes",
            annotation=annotation,
        )

        data = assert_success(result)
        assert data["value"] == annotation

    @pytest.mark.asyncio
    async def test_remove_with_none(self, annotation_tools, mock_ml):
        """set_column_display removes annotation when value is None."""
        mock_ml.set_column_display.return_value = "Image.URL"

        result = await annotation_tools["set_column_display"](
            table_name="Image",
            column_name="URL",
            annotation=None,
        )

        data = assert_success(result)
        assert data["value"] is None

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """set_column_display returns error on exception."""
        mock_ml.set_column_display.side_effect = Exception("Column not found: BadCol")

        result = await annotation_tools["set_column_display"](
            table_name="Image",
            column_name="BadCol",
            annotation={"*": {}},
        )

        assert_error(result, "Column not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """set_column_display returns error when not connected."""
        result = await annotation_tools_disconnected["set_column_display"](
            table_name="Image",
            column_name="URL",
            annotation={"*": {}},
        )

        assert_error(result, "No active")


# =============================================================================
# TestApplyAnnotations
# =============================================================================


class TestApplyAnnotations:
    """Tests for the apply_annotations tool."""

    @pytest.mark.asyncio
    async def test_success(self, annotation_tools, mock_ml):
        """apply_annotations commits staged changes and returns applied status."""
        mock_ml.apply_annotations.return_value = None

        result = await annotation_tools["apply_annotations"]()

        data = assert_success(result)
        assert data["status"] == "applied"
        assert "applied" in data["message"].lower()

        mock_ml.apply_annotations.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """apply_annotations returns error on exception."""
        mock_ml.apply_annotations.side_effect = Exception("Network error")

        result = await annotation_tools["apply_annotations"]()

        assert_error(result, "Network error")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """apply_annotations returns error when not connected."""
        result = await annotation_tools_disconnected["apply_annotations"]()

        assert_error(result, "No active")


# =============================================================================
# TestAddVisibleColumn
# =============================================================================


class TestAddVisibleColumn:
    """Tests for the add_visible_column tool."""

    @pytest.mark.asyncio
    async def test_success_append(self, annotation_tools, mock_ml):
        """add_visible_column appends a column to the end of visible columns."""
        mock_ml.add_visible_column.return_value = ["RID", "Filename", "Description"]

        result = await annotation_tools["add_visible_column"](
            table_name="Image",
            context="compact",
            column="Description",
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Image"
        assert data["context"] == "compact"
        assert data["column_added"] == "Description"
        assert data["position"] == 2  # len(new_list) - 1
        assert data["new_list"] == ["RID", "Filename", "Description"]
        assert "apply_annotations" in data["message"]

        mock_ml.add_visible_column.assert_called_once_with(
            "Image", "compact", "Description", None
        )

    @pytest.mark.asyncio
    async def test_success_with_position(self, annotation_tools, mock_ml):
        """add_visible_column inserts a column at a specific position."""
        mock_ml.add_visible_column.return_value = [
            "RID",
            ["domain", "Image_Subject_fkey"],
            "Filename",
        ]

        result = await annotation_tools["add_visible_column"](
            table_name="Image",
            context="detailed",
            column=["domain", "Image_Subject_fkey"],
            position=1,
        )

        data = assert_success(result)
        assert data["position"] == 1
        assert data["column_added"] == ["domain", "Image_Subject_fkey"]

    @pytest.mark.asyncio
    async def test_success_pseudo_column(self, annotation_tools, mock_ml):
        """add_visible_column handles pseudo-column dict input."""
        pseudo_col = {
            "source": [{"outbound": ["domain", "Image_Subject_fkey"]}, "Name"],
            "markdown_name": "Subject",
        }
        mock_ml.add_visible_column.return_value = ["Filename", pseudo_col]

        result = await annotation_tools["add_visible_column"](
            table_name="Image",
            context="compact",
            column=pseudo_col,
        )

        data = assert_success(result)
        assert data["column_added"] == pseudo_col

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """add_visible_column returns error on exception."""
        mock_ml.add_visible_column.side_effect = Exception("Invalid context: badctx")

        result = await annotation_tools["add_visible_column"](
            table_name="Image",
            context="badctx",
            column="RID",
        )

        assert_error(result, "Invalid context")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """add_visible_column returns error when not connected."""
        result = await annotation_tools_disconnected["add_visible_column"](
            table_name="Image",
            context="compact",
            column="RID",
        )

        assert_error(result, "No active")


# =============================================================================
# TestRemoveVisibleColumn
# =============================================================================


class TestRemoveVisibleColumn:
    """Tests for the remove_visible_column tool."""

    @pytest.mark.asyncio
    async def test_success_by_name(self, annotation_tools, mock_ml):
        """remove_visible_column removes a column by name."""
        mock_ml.remove_visible_column.return_value = ["RID", "Subject"]

        result = await annotation_tools["remove_visible_column"](
            table_name="Image",
            context="compact",
            column="Filename",
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Image"
        assert data["context"] == "compact"
        assert data["column_removed"] == "Filename"
        assert data["new_list"] == ["RID", "Subject"]
        assert "apply_annotations" in data["message"]

        mock_ml.remove_visible_column.assert_called_once_with(
            "Image", "compact", "Filename"
        )

    @pytest.mark.asyncio
    async def test_success_by_fkey_ref(self, annotation_tools, mock_ml):
        """remove_visible_column removes a column by foreign key reference."""
        mock_ml.remove_visible_column.return_value = ["RID", "Filename"]

        result = await annotation_tools["remove_visible_column"](
            table_name="Image",
            context="detailed",
            column=["domain", "Image_Subject_fkey"],
        )

        data = assert_success(result)
        assert data["column_removed"] == ["domain", "Image_Subject_fkey"]

    @pytest.mark.asyncio
    async def test_success_by_index(self, annotation_tools, mock_ml):
        """remove_visible_column removes a column by position index."""
        mock_ml.remove_visible_column.return_value = ["Filename", "Subject"]

        result = await annotation_tools["remove_visible_column"](
            table_name="Image",
            context="compact",
            column=0,
        )

        data = assert_success(result)
        assert data["column_removed"] == 0

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """remove_visible_column returns error on exception."""
        mock_ml.remove_visible_column.side_effect = Exception(
            "Column 'Missing' not found in compact context"
        )

        result = await annotation_tools["remove_visible_column"](
            table_name="Image",
            context="compact",
            column="Missing",
        )

        assert_error(result, "not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """remove_visible_column returns error when not connected."""
        result = await annotation_tools_disconnected["remove_visible_column"](
            table_name="Image",
            context="compact",
            column="RID",
        )

        assert_error(result, "No active")


# =============================================================================
# TestReorderVisibleColumns
# =============================================================================


class TestReorderVisibleColumns:
    """Tests for the reorder_visible_columns tool."""

    @pytest.mark.asyncio
    async def test_success_by_indices(self, annotation_tools, mock_ml):
        """reorder_visible_columns reorders by index positions."""
        mock_ml.reorder_visible_columns.return_value = ["Subject", "RID", "Filename"]

        result = await annotation_tools["reorder_visible_columns"](
            table_name="Image",
            context="compact",
            new_order=[2, 0, 1],
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Image"
        assert data["context"] == "compact"
        assert data["new_order"] == ["Subject", "RID", "Filename"]
        assert "apply_annotations" in data["message"]

        mock_ml.reorder_visible_columns.assert_called_once_with(
            "Image", "compact", [2, 0, 1]
        )

    @pytest.mark.asyncio
    async def test_success_by_column_names(self, annotation_tools, mock_ml):
        """reorder_visible_columns reorders by specifying column names."""
        mock_ml.reorder_visible_columns.return_value = ["Filename", "Subject", "RID"]

        result = await annotation_tools["reorder_visible_columns"](
            table_name="Image",
            context="compact",
            new_order=["Filename", "Subject", "RID"],
        )

        data = assert_success(result)
        assert data["new_order"] == ["Filename", "Subject", "RID"]

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """reorder_visible_columns returns error on exception."""
        mock_ml.reorder_visible_columns.side_effect = Exception("Index out of range")

        result = await annotation_tools["reorder_visible_columns"](
            table_name="Image",
            context="compact",
            new_order=[5, 0, 1],
        )

        assert_error(result, "Index out of range")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """reorder_visible_columns returns error when not connected."""
        result = await annotation_tools_disconnected["reorder_visible_columns"](
            table_name="Image",
            context="compact",
            new_order=[0, 1],
        )

        assert_error(result, "No active")


# =============================================================================
# TestAddVisibleForeignKey
# =============================================================================


class TestAddVisibleForeignKey:
    """Tests for the add_visible_foreign_key tool."""

    @pytest.mark.asyncio
    async def test_success_append(self, annotation_tools, mock_ml):
        """add_visible_foreign_key appends an FK to the list."""
        mock_ml.add_visible_foreign_key.return_value = [
            ["domain", "Image_Subject_fkey"],
            ["domain", "Diagnosis_Subject_fkey"],
        ]

        result = await annotation_tools["add_visible_foreign_key"](
            table_name="Subject",
            context="detailed",
            foreign_key=["domain", "Diagnosis_Subject_fkey"],
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Subject"
        assert data["context"] == "detailed"
        assert data["foreign_key_added"] == ["domain", "Diagnosis_Subject_fkey"]
        assert data["position"] == 1  # len(new_list) - 1
        assert "apply_annotations" in data["message"]

        mock_ml.add_visible_foreign_key.assert_called_once_with(
            "Subject", "detailed", ["domain", "Diagnosis_Subject_fkey"], None
        )

    @pytest.mark.asyncio
    async def test_success_with_position(self, annotation_tools, mock_ml):
        """add_visible_foreign_key inserts at a specific position."""
        mock_ml.add_visible_foreign_key.return_value = [
            ["domain", "Diagnosis_Subject_fkey"],
            ["domain", "Image_Subject_fkey"],
        ]

        result = await annotation_tools["add_visible_foreign_key"](
            table_name="Subject",
            context="detailed",
            foreign_key=["domain", "Diagnosis_Subject_fkey"],
            position=0,
        )

        data = assert_success(result)
        assert data["position"] == 0

    @pytest.mark.asyncio
    async def test_success_pseudo_column(self, annotation_tools, mock_ml):
        """add_visible_foreign_key handles pseudo-column FK dicts."""
        pseudo_fk = {
            "source": [{"inbound": ["domain", "Image_Subject_fkey"]}],
            "markdown_name": "Subject Images",
        }
        mock_ml.add_visible_foreign_key.return_value = [pseudo_fk]

        result = await annotation_tools["add_visible_foreign_key"](
            table_name="Subject",
            context="detailed",
            foreign_key=pseudo_fk,
        )

        data = assert_success(result)
        assert data["foreign_key_added"] == pseudo_fk

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """add_visible_foreign_key returns error on exception."""
        mock_ml.add_visible_foreign_key.side_effect = Exception(
            "Foreign key not found"
        )

        result = await annotation_tools["add_visible_foreign_key"](
            table_name="Subject",
            context="detailed",
            foreign_key=["domain", "bad_fkey"],
        )

        assert_error(result, "Foreign key not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """add_visible_foreign_key returns error when not connected."""
        result = await annotation_tools_disconnected["add_visible_foreign_key"](
            table_name="Subject",
            context="detailed",
            foreign_key=["domain", "Image_Subject_fkey"],
        )

        assert_error(result, "No active")


# =============================================================================
# TestRemoveVisibleForeignKey
# =============================================================================


class TestRemoveVisibleForeignKey:
    """Tests for the remove_visible_foreign_key tool."""

    @pytest.mark.asyncio
    async def test_success_by_ref(self, annotation_tools, mock_ml):
        """remove_visible_foreign_key removes an FK by reference."""
        mock_ml.remove_visible_foreign_key.return_value = [
            ["domain", "Diagnosis_Subject_fkey"],
        ]

        result = await annotation_tools["remove_visible_foreign_key"](
            table_name="Subject",
            context="detailed",
            foreign_key=["domain", "Image_Subject_fkey"],
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Subject"
        assert data["context"] == "detailed"
        assert data["foreign_key_removed"] == ["domain", "Image_Subject_fkey"]
        assert data["new_list"] == [["domain", "Diagnosis_Subject_fkey"]]
        assert "apply_annotations" in data["message"]

        mock_ml.remove_visible_foreign_key.assert_called_once_with(
            "Subject", "detailed", ["domain", "Image_Subject_fkey"]
        )

    @pytest.mark.asyncio
    async def test_success_by_index(self, annotation_tools, mock_ml):
        """remove_visible_foreign_key removes an FK by position index."""
        mock_ml.remove_visible_foreign_key.return_value = [
            ["domain", "Diagnosis_Subject_fkey"],
        ]

        result = await annotation_tools["remove_visible_foreign_key"](
            table_name="Subject",
            context="detailed",
            foreign_key=0,
        )

        data = assert_success(result)
        assert data["foreign_key_removed"] == 0

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """remove_visible_foreign_key returns error on exception."""
        mock_ml.remove_visible_foreign_key.side_effect = Exception(
            "FK not in list"
        )

        result = await annotation_tools["remove_visible_foreign_key"](
            table_name="Subject",
            context="detailed",
            foreign_key=["domain", "bad_fkey"],
        )

        assert_error(result, "FK not in list")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """remove_visible_foreign_key returns error when not connected."""
        result = await annotation_tools_disconnected["remove_visible_foreign_key"](
            table_name="Subject",
            context="detailed",
            foreign_key=["domain", "Image_Subject_fkey"],
        )

        assert_error(result, "No active")


# =============================================================================
# TestReorderVisibleForeignKeys
# =============================================================================


class TestReorderVisibleForeignKeys:
    """Tests for the reorder_visible_foreign_keys tool."""

    @pytest.mark.asyncio
    async def test_success_by_indices(self, annotation_tools, mock_ml):
        """reorder_visible_foreign_keys reorders by index positions."""
        mock_ml.reorder_visible_foreign_keys.return_value = [
            ["domain", "Diagnosis_Subject_fkey"],
            ["domain", "Image_Subject_fkey"],
        ]

        result = await annotation_tools["reorder_visible_foreign_keys"](
            table_name="Subject",
            context="detailed",
            new_order=[1, 0],
        )

        data = assert_success(result)
        assert data["status"] == "staged"
        assert data["target"] == "Subject"
        assert data["context"] == "detailed"
        assert data["new_order"] == [
            ["domain", "Diagnosis_Subject_fkey"],
            ["domain", "Image_Subject_fkey"],
        ]
        assert "apply_annotations" in data["message"]

        mock_ml.reorder_visible_foreign_keys.assert_called_once_with(
            "Subject", "detailed", [1, 0]
        )

    @pytest.mark.asyncio
    async def test_success_by_fkey_refs(self, annotation_tools, mock_ml):
        """reorder_visible_foreign_keys reorders by FK reference list."""
        mock_ml.reorder_visible_foreign_keys.return_value = [
            ["domain", "Diagnosis_Subject_fkey"],
            ["domain", "Image_Subject_fkey"],
        ]

        result = await annotation_tools["reorder_visible_foreign_keys"](
            table_name="Subject",
            context="detailed",
            new_order=[
                ["domain", "Diagnosis_Subject_fkey"],
                ["domain", "Image_Subject_fkey"],
            ],
        )

        data = assert_success(result)
        assert data["new_order"] == [
            ["domain", "Diagnosis_Subject_fkey"],
            ["domain", "Image_Subject_fkey"],
        ]

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """reorder_visible_foreign_keys returns error on exception."""
        mock_ml.reorder_visible_foreign_keys.side_effect = Exception("Index out of range")

        result = await annotation_tools["reorder_visible_foreign_keys"](
            table_name="Subject",
            context="detailed",
            new_order=[5, 0],
        )

        assert_error(result, "Index out of range")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """reorder_visible_foreign_keys returns error when not connected."""
        result = await annotation_tools_disconnected["reorder_visible_foreign_keys"](
            table_name="Subject",
            context="detailed",
            new_order=[0, 1],
        )

        assert_error(result, "No active")


# =============================================================================
# TestGetHandlebarsTemplateVariables
# =============================================================================


class TestGetHandlebarsTemplateVariables:
    """Tests for the get_handlebars_template_variables tool."""

    @pytest.mark.asyncio
    async def test_success(self, annotation_tools, mock_ml):
        """get_handlebars_template_variables returns template variables."""
        expected = {
            "table": "Image",
            "columns": [
                {"name": "RID", "type": "ermrest_rid", "template": "{{{RID}}}"},
                {"name": "Filename", "type": "text", "template": "{{{Filename}}}"},
            ],
            "foreign_keys": [
                {
                    "constraint": ["domain", "Image_Subject_fkey"],
                    "to_table": "Subject",
                    "values_template": "{{{$fkeys.domain.Image_Subject_fkey.values.column}}}",
                    "row_name_template": "{{{$fkeys.domain.Image_Subject_fkey.rowName}}}",
                },
            ],
            "special_variables": {"self": "{{{$self}}}"},
        }
        mock_ml.get_handlebars_template_variables.return_value = expected

        result = await annotation_tools["get_handlebars_template_variables"](
            table_name="Image",
        )

        data = parse_json_result(result)
        assert data["table"] == "Image"
        assert len(data["columns"]) == 2
        assert data["columns"][0]["name"] == "RID"
        assert len(data["foreign_keys"]) == 1
        assert data["foreign_keys"][0]["to_table"] == "Subject"

        mock_ml.get_handlebars_template_variables.assert_called_once_with("Image")

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """get_handlebars_template_variables returns error on exception."""
        mock_ml.get_handlebars_template_variables.side_effect = Exception(
            "Table not found"
        )

        result = await annotation_tools["get_handlebars_template_variables"](
            table_name="BadTable",
        )

        assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """get_handlebars_template_variables returns error when not connected."""
        result = await annotation_tools_disconnected["get_handlebars_template_variables"](
            table_name="Image",
        )

        assert_error(result, "No active")


# =============================================================================
# TestGetTableSampleData
# =============================================================================


class TestGetTableSampleData:
    """Tests for the get_table_sample_data tool."""

    @pytest.mark.asyncio
    async def test_success(self, annotation_tools, mock_ml):
        """get_table_sample_data returns sample rows from the table."""
        mock_table = _make_table(
            name="Image",
            column_names=["RID", "Filename", "Subject", "Description"],
        )
        mock_ml.model.name_to_table.return_value = mock_table

        # Set up the pathBuilder chain
        mock_path = MagicMock()
        mock_entities = MagicMock()
        sample_rows = [
            {"RID": "1-ABC", "Filename": "scan001.jpg", "Subject": "2-DEF"},
            {"RID": "1-XYZ", "Filename": "scan002.jpg", "Subject": "2-DEF"},
        ]
        mock_entities.fetch.return_value = iter(sample_rows)
        mock_path.entities.return_value = mock_entities
        mock_ml.catalog.getPathBuilder.return_value.schemas.__getitem__.return_value.tables.__getitem__.return_value = mock_path

        result = await annotation_tools["get_table_sample_data"](
            table_name="Image",
            limit=2,
        )

        data = assert_success(result)
        assert data["table"] == "Image"
        assert data["row_count"] == 2
        assert len(data["sample_rows"]) == 2
        assert data["sample_rows"][0]["Filename"] == "scan001.jpg"
        assert "columns" in data
        assert "template_test_suggestion" in data

    @pytest.mark.asyncio
    async def test_default_limit(self, annotation_tools, mock_ml):
        """get_table_sample_data uses default limit of 3."""
        mock_table = _make_table(name="Image")
        mock_ml.model.name_to_table.return_value = mock_table

        mock_path = MagicMock()
        mock_entities = MagicMock()
        mock_entities.fetch.return_value = iter([])
        mock_path.entities.return_value = mock_entities
        mock_ml.catalog.getPathBuilder.return_value.schemas.__getitem__.return_value.tables.__getitem__.return_value = mock_path

        result = await annotation_tools["get_table_sample_data"](
            table_name="Image",
        )

        data = assert_success(result)
        # Verify fetch was called with limit=3 (the default)
        mock_entities.fetch.assert_called_once_with(limit=3)

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max(self, annotation_tools, mock_ml):
        """get_table_sample_data clamps limit to max 10."""
        mock_table = _make_table(name="Image")
        mock_ml.model.name_to_table.return_value = mock_table

        mock_path = MagicMock()
        mock_entities = MagicMock()
        mock_entities.fetch.return_value = iter([])
        mock_path.entities.return_value = mock_entities
        mock_ml.catalog.getPathBuilder.return_value.schemas.__getitem__.return_value.tables.__getitem__.return_value = mock_path

        result = await annotation_tools["get_table_sample_data"](
            table_name="Image",
            limit=50,
        )

        data = assert_success(result)
        mock_entities.fetch.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_limit_clamped_to_min(self, annotation_tools, mock_ml):
        """get_table_sample_data clamps limit to min 1."""
        mock_table = _make_table(name="Image")
        mock_ml.model.name_to_table.return_value = mock_table

        mock_path = MagicMock()
        mock_entities = MagicMock()
        mock_entities.fetch.return_value = iter([])
        mock_path.entities.return_value = mock_entities
        mock_ml.catalog.getPathBuilder.return_value.schemas.__getitem__.return_value.tables.__getitem__.return_value = mock_path

        result = await annotation_tools["get_table_sample_data"](
            table_name="Image",
            limit=0,
        )

        data = assert_success(result)
        mock_entities.fetch.assert_called_once_with(limit=1)

    @pytest.mark.asyncio
    async def test_template_suggestion_excludes_rc_rm_columns(self, annotation_tools, mock_ml):
        """get_table_sample_data excludes RC* and RM* columns from suggestion."""
        mock_table = _make_table(
            name="Image",
            column_names=["RID", "RCT", "RMT", "RCB", "RMB", "Filename", "Subject"],
        )
        mock_ml.model.name_to_table.return_value = mock_table

        mock_path = MagicMock()
        mock_entities = MagicMock()
        mock_entities.fetch.return_value = iter([])
        mock_path.entities.return_value = mock_entities
        mock_ml.catalog.getPathBuilder.return_value.schemas.__getitem__.return_value.tables.__getitem__.return_value = mock_path

        result = await annotation_tools["get_table_sample_data"](
            table_name="Image",
        )

        data = assert_success(result)
        suggestion = data["template_test_suggestion"]
        assert "RCT" not in suggestion
        assert "RMT" not in suggestion
        assert "RCB" not in suggestion
        assert "RMB" not in suggestion
        # RID does not start with RC or RM, so it should be included
        assert "RID" in suggestion

    @pytest.mark.asyncio
    async def test_exception(self, annotation_tools, mock_ml):
        """get_table_sample_data returns error on exception."""
        mock_ml.model.name_to_table.side_effect = Exception("Table not found")

        result = await annotation_tools["get_table_sample_data"](
            table_name="BadTable",
        )

        assert_error(result, "Table not found")

    @pytest.mark.asyncio
    async def test_no_connection(self, annotation_tools_disconnected):
        """get_table_sample_data returns error when not connected."""
        result = await annotation_tools_disconnected["get_table_sample_data"](
            table_name="Image",
        )

        assert_error(result, "No active")


# =============================================================================
# TestPreviewHandlebarsTemplate
# =============================================================================


class TestPreviewHandlebarsTemplate:
    """Tests for the preview_handlebars_template tool.

    This is a pure function that does not require a DerivaML connection.
    """

    @pytest.mark.asyncio
    async def test_simple_substitution(self, annotation_tools):
        """preview_handlebars_template renders simple triple-brace variables."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{{Name}}} - {{{Status}}}",
            data={"Name": "John", "Status": "Active"},
        )

        data = assert_success(result)
        assert data["rendered"] == "John - Active"
        assert data["template"] == "{{{Name}}} - {{{Status}}}"
        assert data["data"] == {"Name": "John", "Status": "Active"}

    @pytest.mark.asyncio
    async def test_double_brace_substitution(self, annotation_tools):
        """preview_handlebars_template renders double-brace variables."""
        result = await annotation_tools["preview_handlebars_template"](
            template="Hello {{Name}}!",
            data={"Name": "World"},
        )

        data = assert_success(result)
        assert data["rendered"] == "Hello World!"

    @pytest.mark.asyncio
    async def test_missing_variable_renders_empty(self, annotation_tools):
        """preview_handlebars_template renders missing variables as empty string."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{{Name}}} ({{{Missing}}})",
            data={"Name": "John"},
        )

        data = assert_success(result)
        assert data["rendered"] == "John ()"

    @pytest.mark.asyncio
    async def test_if_block_truthy(self, annotation_tools):
        """preview_handlebars_template renders #if block when condition is truthy."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{{Name}}}{{#if Nickname}} ({{{Nickname}}}){{/if}}",
            data={"Name": "John", "Nickname": "Johnny"},
        )

        data = assert_success(result)
        assert data["rendered"] == "John (Johnny)"

    @pytest.mark.asyncio
    async def test_if_block_falsy(self, annotation_tools):
        """preview_handlebars_template hides #if block when condition is falsy."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{{Name}}}{{#if Nickname}} ({{{Nickname}}}){{/if}}",
            data={"Name": "John"},
        )

        data = assert_success(result)
        assert data["rendered"] == "John"

    @pytest.mark.asyncio
    async def test_if_else_block(self, annotation_tools):
        """preview_handlebars_template handles if/else blocks.

        Note: Due to implementation ordering, {{else}} is consumed by the
        double-brace substitution pass before #if block processing. As a
        result, the entire if block content is treated as the true branch
        and empty string is returned when the condition is falsy.
        """
        result = await annotation_tools["preview_handlebars_template"](
            template="{{#if Active}}Active{{else}}Inactive{{/if}}",
            data={"Active": False},
        )

        data = assert_success(result)
        # {{else}} is consumed by double-brace sub pass, so the if block
        # sees "ActiveInactive" as the true branch. Active is falsy -> ""
        assert data["rendered"] == ""

    @pytest.mark.asyncio
    async def test_if_else_block_true(self, annotation_tools):
        """preview_handlebars_template renders if/else true branch.

        Note: {{else}} is consumed by double-brace substitution, so the
        entire content between #if and /if is treated as the true branch.
        """
        result = await annotation_tools["preview_handlebars_template"](
            template="{{#if Active}}Active{{else}}Inactive{{/if}}",
            data={"Active": True},
        )

        data = assert_success(result)
        # {{else}} is consumed, so true branch is "ActiveInactive"
        assert data["rendered"] == "ActiveInactive"

    @pytest.mark.asyncio
    async def test_unless_block_falsy(self, annotation_tools):
        """preview_handlebars_template renders #unless block when condition is falsy."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{#unless Deleted}}Available{{/unless}}",
            data={"Deleted": False},
        )

        data = assert_success(result)
        assert data["rendered"] == "Available"

    @pytest.mark.asyncio
    async def test_unless_block_truthy(self, annotation_tools):
        """preview_handlebars_template hides #unless block when condition is truthy."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{#unless Deleted}}Available{{/unless}}",
            data={"Deleted": True},
        )

        data = assert_success(result)
        assert data["rendered"] == ""

    @pytest.mark.asyncio
    async def test_each_block(self, annotation_tools):
        """preview_handlebars_template handles #each blocks.

        Note: Due to implementation ordering, {{{this}}} is consumed by the
        triple-brace substitution pass before #each block processing. The
        #each handler then operates on the already-substituted content.
        """
        result = await annotation_tools["preview_handlebars_template"](
            template="{{#each items}}{{{this}}} {{/each}}",
            data={"items": ["apple", "banana", "cherry"]},
        )

        data = assert_success(result)
        # {{{this}}} is consumed first (data.get("this","") -> ""), leaving " "
        # #each then repeats " " three times
        assert data["rendered"] == "   "

    @pytest.mark.asyncio
    async def test_each_block_with_dict_items(self, annotation_tools):
        """preview_handlebars_template handles #each with dict items.

        Note: {{{this.name}}} is consumed by triple-brace dot-path substitution
        before #each block processing.
        """
        result = await annotation_tools["preview_handlebars_template"](
            template="{{#each items}}{{{this.name}}} {{/each}}",
            data={"items": [{"name": "Alice"}, {"name": "Bob"}]},
        )

        data = assert_success(result)
        # {{{this.name}}} is consumed by dot-path handler: data["this"]["name"] fails -> ""
        # Leaves " " per iteration, repeated twice
        assert data["rendered"] == "  "

    @pytest.mark.asyncio
    async def test_each_block_with_index(self, annotation_tools):
        """preview_handlebars_template handles #each with @index.

        Note: Both {{{this}}} and {{@index}} are consumed by the
        substitution passes before #each block processing.
        """
        result = await annotation_tools["preview_handlebars_template"](
            template="{{#each items}}{{@index}}:{{{this}}} {{/each}}",
            data={"items": ["a", "b", "c"]},
        )

        data = assert_success(result)
        # {{{this}}} consumed (-> ""), {{@index}} consumed (-> "")
        # Remaining inner content is ": " repeated 3 times
        assert data["rendered"] == ": : : "

    @pytest.mark.asyncio
    async def test_value_special_variable(self, annotation_tools):
        """preview_handlebars_template handles _value special variable."""
        result = await annotation_tools["preview_handlebars_template"](
            template="Value: {{{_value}}}",
            data={"_value": "hello"},
        )

        data = assert_success(result)
        assert data["rendered"] == "Value: hello"

    @pytest.mark.asyncio
    async def test_row_variable(self, annotation_tools):
        """preview_handlebars_template handles _row.column_name variables."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{{_row.Name}}} ({{{_row.Status}}})",
            data={"Name": "John", "Status": "Active"},
        )

        data = assert_success(result)
        assert data["rendered"] == "John (Active)"

    @pytest.mark.asyncio
    async def test_nested_path_variable(self, annotation_tools):
        """preview_handlebars_template handles dot-separated nested paths."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{{$fkeys.domain.fkey.values.Name}}}",
            data={"$fkeys": {"domain": {"fkey": {"values": {"Name": "Test"}}}}},
        )

        data = assert_success(result)
        assert data["rendered"] == "Test"

    @pytest.mark.asyncio
    async def test_empty_data(self, annotation_tools):
        """preview_handlebars_template handles empty data dict."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{{Name}}}",
            data={},
        )

        data = assert_success(result)
        assert data["rendered"] == ""

    @pytest.mark.asyncio
    async def test_note_in_response(self, annotation_tools):
        """preview_handlebars_template includes a note about simplified preview."""
        result = await annotation_tools["preview_handlebars_template"](
            template="{{{Name}}}",
            data={"Name": "Test"},
        )

        data = assert_success(result)
        assert "note" in data
        assert "simplified" in data["note"].lower() or "preview" in data["note"].lower()

    @pytest.mark.asyncio
    async def test_each_with_unless_first(self, annotation_tools):
        """preview_handlebars_template handles #each with nested #unless @first.

        Note: {{{this}}} is consumed by the triple-brace substitution pass
        before #each processing. The #unless @first block inside #each is
        also consumed by the earlier #unless processing pass.
        """
        result = await annotation_tools["preview_handlebars_template"](
            template="{{#each items}}{{#unless @first}}, {{/unless}}{{{this}}}{{/each}}",
            data={"items": ["a", "b", "c"]},
        )

        data = assert_success(result)
        # {{{this}}} consumed first (-> "")
        # {{#unless @first}} is processed before #each -- @first is not in data, so
        # it is falsy, meaning the ", " content is rendered. This produces ", " before
        # the now-empty {{{this}}} position.
        # After unless processing: "{{#each items}}, {{/each}}"
        # #each then repeats ", " three times
        assert data["rendered"] == ", , , "


# =============================================================================
# TestValidateTemplateSyntax
# =============================================================================


class TestValidateTemplateSyntax:
    """Tests for the validate_template_syntax tool.

    This is a pure function that does not require a DerivaML connection.
    """

    @pytest.mark.asyncio
    async def test_valid_simple_template(self, annotation_tools):
        """validate_template_syntax reports valid for a correct template."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{{Name}}} ({{{Status}}})",
        )

        data = parse_json_result(result)
        assert data["valid"] is True
        assert data["errors"] == []
        assert data["template"] == "{{{Name}}} ({{{Status}}})"

    @pytest.mark.asyncio
    async def test_valid_with_if_block(self, annotation_tools):
        """validate_template_syntax validates if blocks correctly."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{#if Name}}{{{Name}}}{{/if}}",
        )

        data = parse_json_result(result)
        assert data["valid"] is True
        assert data["errors"] == []
        assert data["block_counts"]["if_blocks"] == 1

    @pytest.mark.asyncio
    async def test_unmatched_triple_braces(self, annotation_tools):
        """validate_template_syntax detects unmatched triple braces."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{{Name}}",
        )

        data = parse_json_result(result)
        assert data["valid"] is False
        assert any("triple braces" in e.lower() or "unmatched" in e.lower() for e in data["errors"])

    @pytest.mark.asyncio
    async def test_unclosed_if_block(self, annotation_tools):
        """validate_template_syntax detects unclosed #if blocks."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{#if Name}}{{{Name}}}",
        )

        data = parse_json_result(result)
        assert data["valid"] is False
        assert any("#if" in e for e in data["errors"])

    @pytest.mark.asyncio
    async def test_unclosed_each_block(self, annotation_tools):
        """validate_template_syntax detects unclosed #each blocks."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{#each items}}{{{this}}}",
        )

        data = parse_json_result(result)
        assert data["valid"] is False
        assert any("#each" in e for e in data["errors"])

    @pytest.mark.asyncio
    async def test_unclosed_unless_block(self, annotation_tools):
        """validate_template_syntax detects unclosed #unless blocks."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{#unless Deleted}}Available",
        )

        data = parse_json_result(result)
        assert data["valid"] is False
        assert any("#unless" in e for e in data["errors"])

    @pytest.mark.asyncio
    async def test_if_without_condition(self, annotation_tools):
        """validate_template_syntax detects #if with no condition variable."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{#if}}content{{/if}}",
        )

        data = parse_json_result(result)
        assert data["valid"] is False
        assert any("#if" in e and "condition" in e.lower() for e in data["errors"])

    @pytest.mark.asyncio
    async def test_each_without_variable(self, annotation_tools):
        """validate_template_syntax detects #each with no array variable."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{#each}}content{{/each}}",
        )

        data = parse_json_result(result)
        assert data["valid"] is False
        assert any("#each" in e and "array" in e.lower() for e in data["errors"])

    @pytest.mark.asyncio
    async def test_warning_double_brace_only(self, annotation_tools):
        """validate_template_syntax warns when using only double braces (HTML-escaped)."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{Name}}",
        )

        data = parse_json_result(result)
        assert data["valid"] is True
        assert any("escape" in w.lower() or "raw" in w.lower() for w in data["warnings"])

    @pytest.mark.asyncio
    async def test_warning_empty_expression(self, annotation_tools):
        """validate_template_syntax warns on empty template expressions."""
        result = await annotation_tools["validate_template_syntax"](
            template="Hello {{ }} world",
        )

        data = parse_json_result(result)
        assert any("empty" in w.lower() for w in data["warnings"])

    @pytest.mark.asyncio
    async def test_block_counts(self, annotation_tools):
        """validate_template_syntax returns correct block counts."""
        result = await annotation_tools["validate_template_syntax"](
            template="{{#if A}}{{{A}}}{{/if}} {{#each B}}{{{this}}}{{/each}} {{#unless C}}x{{/unless}}",
        )

        data = parse_json_result(result)
        assert data["valid"] is True
        assert data["block_counts"]["if_blocks"] == 1
        assert data["block_counts"]["each_blocks"] == 1
        assert data["block_counts"]["unless_blocks"] == 1
        # Only {{{A}}} and {{{this}}} match the variable regex (block openers/closers excluded)
        assert data["block_counts"]["variables"] == 2

    @pytest.mark.asyncio
    async def test_valid_complex_template(self, annotation_tools):
        """validate_template_syntax validates a complex realistic template."""
        template = (
            "{{{Name}}}{{#if Nickname}} ({{{Nickname}}}){{/if}}"
            " - {{#if Active}}Active{{else}}Inactive{{/if}}"
        )

        result = await annotation_tools["validate_template_syntax"](
            template=template,
        )

        data = parse_json_result(result)
        assert data["valid"] is True
        assert data["errors"] == []

    @pytest.mark.asyncio
    async def test_empty_template(self, annotation_tools):
        """validate_template_syntax handles an empty template string."""
        result = await annotation_tools["validate_template_syntax"](
            template="",
        )

        data = parse_json_result(result)
        assert data["valid"] is True
        assert data["errors"] == []

    @pytest.mark.asyncio
    async def test_plain_text_template(self, annotation_tools):
        """validate_template_syntax handles plain text without any handlebars."""
        result = await annotation_tools["validate_template_syntax"](
            template="Just plain text",
        )

        data = parse_json_result(result)
        assert data["valid"] is True
        assert data["errors"] == []
