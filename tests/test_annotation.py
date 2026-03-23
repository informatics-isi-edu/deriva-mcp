"""Unit tests for annotation management tools.

Tests all 14 annotation tools:
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
