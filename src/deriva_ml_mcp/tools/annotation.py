"""Annotation management tools for DerivaML MCP server.

This module provides tools to read and modify Deriva catalog annotations,
focusing on display, visible-columns, visible-foreign-keys, table-display,
and column-display annotations.

Annotation Tag URIs:
    - display: tag:isrd.isi.edu,2015:display
    - visible-columns: tag:isrd.isi.edu,2016:visible-columns
    - visible-foreign-keys: tag:isrd.isi.edu,2016:visible-foreign-keys
    - table-display: tag:isrd.isi.edu,2016:table-display
    - column-display: tag:isrd.isi.edu,2016:column-display
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-mcp")

# Annotation tag URIs
DISPLAY_TAG = "tag:isrd.isi.edu,2015:display"
VISIBLE_COLUMNS_TAG = "tag:isrd.isi.edu,2016:visible-columns"
VISIBLE_FOREIGN_KEYS_TAG = "tag:isrd.isi.edu,2016:visible-foreign-keys"
TABLE_DISPLAY_TAG = "tag:isrd.isi.edu,2016:table-display"
COLUMN_DISPLAY_TAG = "tag:isrd.isi.edu,2016:column-display"


def register_annotation_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register annotation management tools with the MCP server."""

    @mcp.tool()
    async def set_display_annotation(
        table_name: str,
        column_name: str | None = None,
        annotation: dict[str, Any] | None = None,
    ) -> str:
        """Set the display annotation on a table or column.

        The display annotation controls basic naming and display options.
        Changes are staged locally until apply_annotations() is called.

        Args:
            table_name: Name of the table.
            column_name: Name of the column (optional). If provided, sets the
                annotation on the column; otherwise sets it on the table.
            annotation: The display annotation value. Set to null/None to remove.

        **Display Annotation Schema** (tag:isrd.isi.edu,2015:display):
        ```json
        {
            "name": "string",           // Display name (mutually exclusive with markdown_name)
            "markdown_name": "string",  // Markdown-formatted name (mutually exclusive with name)
            "name_style": {
                "underline_space": true,  // Replace underscores with spaces
                "title_case": true,       // Apply title case
                "markdown": true          // Interpret as markdown
            },
            "comment": "string",        // Tooltip/description text
            "show_null": {              // How to display null values per context
                "*": true,              // true = show "No value", false = hide, "string" = custom
                "compact": false,
                "detailed": "\"N/A\""
            },
            "show_foreign_key_link": {  // Show FK as link per context
                "*": true,
                "compact": false
            }
        }
        ```

        **Valid contexts for show_null/show_foreign_key_link:**
        - "*" (all contexts)
        - "compact", "compact/select", "compact/brief", "compact/brief/inline"
        - "detailed"

        Returns:
            JSON with status and the target (table or column).

        Examples:
            # Set table display name
            set_display_annotation("Image", annotation={"name": "Images"})

            # Set column display name
            set_display_annotation("Image", "Filename", {"name": "File Name"})

            # Set name style to replace underscores with spaces
            set_display_annotation("Image", annotation={"name_style": {"underline_space": true}})

            # Remove display annotation
            set_display_annotation("Image", annotation=null)
        """
        try:
            ml = conn_manager.get_active_or_raise()
            target = ml.set_display_annotation(table_name, annotation, column_name)

            return json.dumps({
                "status": "staged",
                "target": target,
                "annotation": "display",
                "value": annotation,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to set display annotation: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_visible_columns(
        table_name: str,
        annotation: dict[str, Any] | None = None,
    ) -> str:
        """Set the visible-columns annotation on a table.

        Controls which columns appear in different UI contexts and their order.
        Changes are staged locally until apply_annotations() is called.

        Args:
            table_name: Name of the table.
            annotation: The visible-columns annotation value. Set to null/None to remove.

        **Visible-Columns Annotation Schema** (tag:isrd.isi.edu,2016:visible-columns):
        ```json
        {
            "compact": [...],           // Columns for compact/list view
            "detailed": [...],          // Columns for detailed/record view
            "entry": [...],             // Columns for data entry (create/edit)
            "entry/create": [...],      // Columns for create only
            "entry/edit": [...],        // Columns for edit only
            "export": [...],            // Columns for export
            "filter": {                 // Faceted search configuration
                "and": [...]
            },
            "*": [...]                  // Default for all contexts
        }
        ```

        **Column directive formats** (items in column lists):

        1. **Simple column name** (string):
           ```json
           "RID"
           ```

        2. **Foreign key reference** (array of [schema, constraint_name]):
           ```json
           ["schema_name", "fkey_constraint_name"]
           ```

        3. **Pseudo-column** (object with source path):
           ```json
           {
               "source": "column_name",           // Simple column
               "source": [                        // Or path through foreign keys
                   {"outbound": ["schema", "fkey"]},
                   "target_column"
               ],
               "sourcekey": "predefined_key",    // OR reference to source-definitions
               "entity": true,                   // Show as entity (row) vs scalar value
               "aggregate": "array",             // Aggregation: min, max, cnt, cnt_d, array, array_d
               "self_link": true,                // Link to current row
               "markdown_name": "Display Name",  // Custom column header
               "comment": "Tooltip text",        // Column tooltip
               "display": {                      // Display options
                   "markdown_pattern": "{{{value}}}",
                   "template_engine": "handlebars",  // or "mustache"
                   "show_foreign_key_link": true,
                   "array_ux_mode": "csv"           // raw, csv, olist, ulist
               }
           }
           ```

        **Filter context** (for faceted search):
        ```json
        {
            "filter": {
                "and": [
                    {
                        "source": "column_name",
                        "markdown_name": "Filter Label",
                        "open": true,              // Expand by default
                        "ux_mode": "choices",      // choices, ranges, check_presence
                        "bar_plot": true,          // Show distribution chart
                        "hide_null_choice": true,  // Hide "No value" option
                        "choices": ["val1", "val2"], // Preset choices
                        "ranges": [{"min": 0, "max": 100}]  // Preset ranges
                    }
                ]
            }
        }
        ```

        Returns:
            JSON with status and the table name.

        Examples:
            # Simple column list for compact view
            set_visible_columns("Image", {
                "compact": ["RID", "Filename", "Subject"],
                "detailed": ["RID", "Filename", "Subject", "Description", "URL"]
            })

            # Include foreign key as a column
            set_visible_columns("Image", {
                "compact": ["Filename", ["domain", "Image_Subject_fkey"]]
            })

            # Pseudo-column traversing foreign key
            set_visible_columns("Image", {
                "detailed": [
                    "Filename",
                    {
                        "source": [{"outbound": ["domain", "Image_Subject_fkey"]}, "Name"],
                        "markdown_name": "Subject Name"
                    }
                ]
            })

            # Configure faceted search
            set_visible_columns("Image", {
                "filter": {
                    "and": [
                        {"source": "Species", "open": true},
                        {"source": "Quality", "ux_mode": "choices"}
                    ]
                }
            })
        """
        try:
            ml = conn_manager.get_active_or_raise()
            target = ml.set_visible_columns(table_name, annotation)

            return json.dumps({
                "status": "staged",
                "target": target,
                "annotation": "visible-columns",
                "value": annotation,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to set visible-columns annotation: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_visible_foreign_keys(
        table_name: str,
        annotation: dict[str, Any] | None = None,
    ) -> str:
        """Set the visible-foreign-keys annotation on a table.

        Controls which related tables (via inbound foreign keys) appear in
        different UI contexts and their order. These show as "related tables"
        sections in the detailed view.

        Changes are staged locally until apply_annotations() is called.

        Args:
            table_name: Name of the table.
            annotation: The visible-foreign-keys annotation value. Set to null/None to remove.

        **Visible-Foreign-Keys Annotation Schema** (tag:isrd.isi.edu,2016:visible-foreign-keys):
        ```json
        {
            "detailed": [...],  // Related tables in detailed view
            "*": [...]          // Default for all contexts
        }
        ```

        **Foreign key directive formats** (items in the lists):

        1. **Inbound foreign key reference** (array of [schema, constraint_name]):
           ```json
           ["schema_name", "fkey_constraint_name"]
           ```
           The constraint must be an INBOUND foreign key (i.e., another table
           references this table).

        2. **Pseudo-column for related entities** (object):
           ```json
           {
               "source": [
                   {"inbound": ["schema", "fkey_to_this_table"]},
                   {"outbound": ["schema", "fkey_to_related"]},
                   "column_name"
               ],
               "sourcekey": "predefined_key",    // OR reference to source-definitions
               "markdown_name": "Related Items",
               "comment": "Tooltip text",
               "display": {
                   "markdown_pattern": "...",
                   "template_engine": "handlebars"
               }
           }
           ```

        Returns:
            JSON with status and the table name.

        Examples:
            # Show specific related tables in detailed view
            set_visible_foreign_keys("Subject", {
                "detailed": [
                    ["domain", "Image_Subject_fkey"],
                    ["domain", "Diagnosis_Subject_fkey"]
                ]
            })

            # Hide all related tables
            set_visible_foreign_keys("Subject", {"detailed": []})

            # Pseudo-column for complex relationship
            set_visible_foreign_keys("Subject", {
                "detailed": [
                    {
                        "source": [{"inbound": ["domain", "Image_Subject_fkey"]}],
                        "markdown_name": "Subject Images"
                    }
                ]
            })
        """
        try:
            ml = conn_manager.get_active_or_raise()
            target = ml.set_visible_foreign_keys(table_name, annotation)

            return json.dumps({
                "status": "staged",
                "target": target,
                "annotation": "visible-foreign-keys",
                "value": annotation,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to set visible-foreign-keys annotation: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_table_display(
        table_name: str,
        annotation: dict[str, Any] | None = None,
    ) -> str:
        """Set the table-display annotation on a table.

        Controls table-level display options like row naming patterns,
        page size, and row ordering.

        Changes are staged locally until apply_annotations() is called.

        Args:
            table_name: Name of the table.
            annotation: The table-display annotation value. Set to null/None to remove.

        **Table-Display Annotation Schema** (tag:isrd.isi.edu,2016:table-display):
        ```json
        {
            "row_name": {                              // How to display row identifiers
                "row_markdown_pattern": "{{{Name}}}",  // Template for row display
                "template_engine": "handlebars"        // or "mustache"
            },
            "detailed": {                              // Options for detailed view
                "hide_column_headers": true,           // Hide column headers
                "collapse_toc_panel": true             // Collapse table of contents
            },
            "compact": {                               // Options for compact/list view
                "page_size": 25,                       // Rows per page
                "row_order": [                         // Default sort order
                    {"column": "RCT", "descending": true},
                    "Name"
                ]
            },
            "*": {                                     // Default for all contexts
                "page_size": 10,
                "row_order": ["Name"]
            }
        }
        ```

        **Context-specific options:**
        - `row_name`: Special context for row identifier display
        - `detailed`: Detailed/record view
        - `compact`, `compact/select`, `compact/brief`: List views
        - `*`: Default for unspecified contexts

        **Available options per context:**
        - `row_order`: Array of sort keys (column name or {column, descending})
        - `page_size`: Number of rows per page
        - `collapse_toc_panel`: Boolean to collapse TOC (detailed only)
        - `hide_column_headers`: Boolean to hide headers (detailed only)
        - `row_markdown_pattern`: Template string using {{{column}}} syntax
        - `page_markdown_pattern`: Template for entire page
        - `separator_markdown`: Separator between rows
        - `prefix_markdown`: Content before rows
        - `suffix_markdown`: Content after rows
        - `template_engine`: "handlebars" or "mustache"

        Returns:
            JSON with status and the table name.

        Examples:
            # Set row name pattern
            set_table_display("Subject", {
                "row_name": {
                    "row_markdown_pattern": "{{{Name}}} ({{{Species}}})"
                }
            })

            # Set default sort order and page size
            set_table_display("Image", {
                "compact": {
                    "row_order": [{"column": "RCT", "descending": true}],
                    "page_size": 50
                }
            })

            # Configure detailed view
            set_table_display("Page", {
                "detailed": {
                    "hide_column_headers": true,
                    "collapse_toc_panel": true
                }
            })
        """
        try:
            ml = conn_manager.get_active_or_raise()
            target = ml.set_table_display(table_name, annotation)

            return json.dumps({
                "status": "staged",
                "target": target,
                "annotation": "table-display",
                "value": annotation,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to set table-display annotation: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def set_column_display(
        table_name: str,
        column_name: str,
        annotation: dict[str, Any] | None = None,
    ) -> str:
        """Set the column-display annotation on a column.

        Controls how a column's values are rendered, including custom
        formatting and markdown patterns.

        Changes are staged locally until apply_annotations() is called.

        Args:
            table_name: Name of the table containing the column.
            column_name: Name of the column.
            annotation: The column-display annotation value. Set to null/None to remove.

        **Column-Display Annotation Schema** (tag:isrd.isi.edu,2016:column-display):
        ```json
        {
            "*": {                                    // Default for all contexts
                "pre_format": {
                    "format": "%.2f",                 // printf-style format string
                    "bool_true_value": "Yes",         // Display for true
                    "bool_false_value": "No"          // Display for false
                },
                "markdown_pattern": "**{{{value}}}**",  // Markdown template
                "template_engine": "handlebars",        // or "mustache"
                "column_order": false                   // Disable sorting, or specify sort
            },
            "compact": {...},                         // Compact/list view options
            "detailed": {...},                        // Detailed/record view options
            "entry": {...},                           // Entry form options
            "entry/create": {...},                    // Create form options
            "entry/edit": {...}                       // Edit form options
        }
        ```

        **Available options:**
        - `pre_format`: Pre-processing before display
          - `format`: printf-style format (e.g., "%.2f" for 2 decimal places)
          - `bool_true_value`: Text to show for boolean true
          - `bool_false_value`: Text to show for boolean false
        - `markdown_pattern`: Template using {{{column_name}}} substitution
        - `template_engine`: "handlebars" (default) or "mustache"
        - `column_order`: Sort configuration or false to disable sorting

        **Template variables in markdown_pattern:**
        - `{{{_value}}}` or `{{{value}}}`: The column's value
        - `{{{_row.column_name}}}`: Another column's value from same row
        - `{{{$fkeys.schema.fkey.values.column}}}`: Value from related table

        Returns:
            JSON with status and the target column.

        Examples:
            # Format numbers with 2 decimal places
            set_column_display("Measurement", "Value", {
                "*": {"pre_format": {"format": "%.2f"}}
            })

            # Display boolean as Yes/No
            set_column_display("Subject", "Active", {
                "*": {
                    "pre_format": {
                        "bool_true_value": "Active",
                        "bool_false_value": "Inactive"
                    }
                }
            })

            # Custom markdown pattern
            set_column_display("Image", "URL", {
                "detailed": {
                    "markdown_pattern": "[![Image]({{{_value}}})]({{{_value}}})"
                }
            })

            # Disable column sorting
            set_column_display("Subject", "Notes", {
                "*": {"column_order": false}
            })
        """
        try:
            ml = conn_manager.get_active_or_raise()
            target = ml.set_column_display(table_name, column_name, annotation)

            return json.dumps({
                "status": "staged",
                "target": target,
                "annotation": "column-display",
                "value": annotation,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to set column-display annotation: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def apply_annotations() -> str:
        """Apply all staged annotation changes to the catalog.

        This commits any annotation changes made via set_display_annotation,
        set_visible_columns, set_visible_foreign_keys, set_table_display,
        or set_column_display to the remote catalog.

        Returns:
            JSON with status and details of the apply operation.

        Example workflow:
            1. get_table_annotations("Image")           # Check current state
            2. set_display_annotation("Image", annotation={"name": "Images"})
            3. set_visible_columns("Image", {"compact": ["RID", "Filename"]})
            4. apply_annotations()                      # Commit all changes
        """
        try:
            ml = conn_manager.get_active_or_raise()
            ml.apply_annotations()

            return json.dumps({
                "status": "applied",
                "message": "All staged annotation changes have been applied to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to apply annotations: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_visible_column(
        table_name: str,
        context: str,
        column: str | list[str] | dict[str, Any],
        position: int | None = None,
    ) -> str:
        """Add a column to the visible-columns list for a specific context.

        This is a convenience tool for adding columns without replacing the
        entire visible-columns annotation. Changes are staged until
        apply_annotations() is called.

        Args:
            table_name: Name of the table.
            context: The context to modify. See **Contexts** below.
            column: Column to add. Can be:
                - String: column name (e.g., "Filename")
                - List: foreign key reference (e.g., ["schema", "fkey_name"])
                - Dict: pseudo-column definition (see set_visible_columns)
            position: Position to insert at (0-indexed). If None, appends to end.

        **Contexts for visible-columns:**

        | Context | Description | When Used |
        |---------|-------------|-----------|
        | `*` | Default for all contexts | Fallback when specific context not set |
        | `compact` | List/table view | Main record list, search results |
        | `compact/brief` | Abbreviated list | Inline previews, tooltips |
        | `compact/brief/inline` | Minimal inline | Foreign key cell display |
        | `compact/select` | Selection modal | Picker dialogs for foreign keys |
        | `detailed` | Full record view | Single record page |
        | `entry` | Data entry forms | Both create and edit forms |
        | `entry/create` | Create form only | New record creation |
        | `entry/edit` | Edit form only | Editing existing records |
        | `export` | Data export | CSV/JSON export |
        | `filter` | Faceted search | Search sidebar (uses different format) |

        Returns:
            JSON with the updated column list for the context.

        Examples:
            # Add column to end of compact view
            add_visible_column("Image", "compact", "Description")

            # Add foreign key reference at position 1
            add_visible_column("Image", "detailed", ["domain", "Image_Subject_fkey"], 1)

            # Add pseudo-column
            add_visible_column("Image", "compact", {
                "source": [{"outbound": ["domain", "Image_Subject_fkey"]}, "Name"],
                "markdown_name": "Subject"
            })
        """
        try:
            ml = conn_manager.get_active_or_raise()
            new_list = ml.add_visible_column(table_name, context, column, position)

            return json.dumps({
                "status": "staged",
                "target": table_name,
                "context": context,
                "column_added": column,
                "position": position if position is not None else len(new_list) - 1,
                "new_list": new_list,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to add visible column: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def remove_visible_column(
        table_name: str,
        context: str,
        column: str | list[str] | int,
    ) -> str:
        """Remove a column from the visible-columns list for a specific context.

        This is a convenience tool for removing columns without replacing the
        entire visible-columns annotation. Changes are staged until
        apply_annotations() is called.

        Args:
            table_name: Name of the table.
            context: The context to modify (e.g., "compact", "detailed").
            column: Column to remove. Can be:
                - String: column name to find and remove
                - List: foreign key reference [schema, constraint] to find and remove
                - Integer: index position to remove (0-indexed)

        Returns:
            JSON with the updated column list for the context.

        Examples:
            # Remove by column name
            remove_visible_column("Image", "compact", "Description")

            # Remove by foreign key reference
            remove_visible_column("Image", "detailed", ["domain", "Image_Subject_fkey"])

            # Remove by position (first column)
            remove_visible_column("Image", "compact", 0)
        """
        try:
            ml = conn_manager.get_active_or_raise()
            new_list = ml.remove_visible_column(table_name, context, column)

            return json.dumps({
                "status": "staged",
                "target": table_name,
                "context": context,
                "column_removed": column,
                "new_list": new_list,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to remove visible column: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def reorder_visible_columns(
        table_name: str,
        context: str,
        new_order: list[int] | list[str | list[str]],
    ) -> str:
        """Reorder columns in the visible-columns list for a specific context.

        This is a convenience tool for reordering columns without manually
        reconstructing the list. Changes are staged until apply_annotations()
        is called.

        Args:
            table_name: Name of the table.
            context: The context to modify (e.g., "compact", "detailed").
            new_order: The new order specification. Can be:
                - List of indices: [2, 0, 1, 3] reorders by current positions
                - List of column names/refs: ["Name", "RID", ...] specifies exact order

        Returns:
            JSON with the reordered column list for the context.

        Examples:
            # Reorder by indices (move item at index 2 to front)
            reorder_visible_columns("Image", "compact", [2, 0, 1, 3, 4])

            # Reorder by specifying exact column order
            reorder_visible_columns("Image", "compact", ["Filename", "Subject", "RID"])

            # Note: When using column names, all columns must be included
            # or unmentioned columns will be removed from the list
        """
        try:
            ml = conn_manager.get_active_or_raise()
            new_list = ml.reorder_visible_columns(table_name, context, new_order)

            return json.dumps({
                "status": "staged",
                "target": table_name,
                "context": context,
                "new_order": new_list,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to reorder visible columns: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_visible_foreign_key(
        table_name: str,
        context: str,
        foreign_key: list[str] | dict[str, Any],
        position: int | None = None,
    ) -> str:
        """Add a foreign key to the visible-foreign-keys list for a specific context.

        This is a convenience tool for adding related tables without replacing the
        entire visible-foreign-keys annotation. Changes are staged until
        apply_annotations() is called.

        Args:
            table_name: Name of the table.
            context: The context to modify (typically "detailed" or "*").
            foreign_key: Foreign key to add. Can be:
                - List: inbound foreign key reference (e.g., ["schema", "Other_Table_fkey"])
                - Dict: pseudo-column definition for complex relationships
            position: Position to insert at (0-indexed). If None, appends to end.

        **Contexts for visible-foreign-keys:**

        | Context | Description | When Used |
        |---------|-------------|-----------|
        | `*` | Default for all contexts | Fallback when specific context not set |
        | `detailed` | Full record view | Related tables shown on single record page |

        **Important:** Only INBOUND foreign keys are valid - these are foreign keys
        from OTHER tables that reference THIS table. Use list_foreign_keys() to
        see which inbound foreign keys are available.

        Returns:
            JSON with the updated foreign key list for the context.

        Examples:
            # Add inbound foreign key to detailed view
            add_visible_foreign_key("Subject", "detailed", ["domain", "Image_Subject_fkey"])

            # Add at specific position
            add_visible_foreign_key("Subject", "detailed", ["domain", "Diagnosis_Subject_fkey"], 0)

            # Add pseudo-column for complex relationship
            add_visible_foreign_key("Subject", "detailed", {
                "source": [{"inbound": ["domain", "Image_Subject_fkey"]}],
                "markdown_name": "Subject Images"
            })
        """
        try:
            ml = conn_manager.get_active_or_raise()
            new_list = ml.add_visible_foreign_key(table_name, context, foreign_key, position)

            return json.dumps({
                "status": "staged",
                "target": table_name,
                "context": context,
                "foreign_key_added": foreign_key,
                "position": position if position is not None else len(new_list) - 1,
                "new_list": new_list,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to add visible foreign key: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def remove_visible_foreign_key(
        table_name: str,
        context: str,
        foreign_key: list[str] | int,
    ) -> str:
        """Remove a foreign key from the visible-foreign-keys list for a specific context.

        This is a convenience tool for removing related tables without replacing the
        entire visible-foreign-keys annotation. Changes are staged until
        apply_annotations() is called.

        Args:
            table_name: Name of the table.
            context: The context to modify (e.g., "detailed", "*").
            foreign_key: Foreign key to remove. Can be:
                - List: foreign key reference [schema, constraint] to find and remove
                - Integer: index position to remove (0-indexed)

        Returns:
            JSON with the updated foreign key list for the context.

        Examples:
            # Remove by foreign key reference
            remove_visible_foreign_key("Subject", "detailed", ["domain", "Image_Subject_fkey"])

            # Remove by position (first foreign key)
            remove_visible_foreign_key("Subject", "detailed", 0)
        """
        try:
            ml = conn_manager.get_active_or_raise()
            new_list = ml.remove_visible_foreign_key(table_name, context, foreign_key)

            return json.dumps({
                "status": "staged",
                "target": table_name,
                "context": context,
                "foreign_key_removed": foreign_key,
                "new_list": new_list,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to remove visible foreign key: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def reorder_visible_foreign_keys(
        table_name: str,
        context: str,
        new_order: list[int] | list[list[str]],
    ) -> str:
        """Reorder foreign keys in the visible-foreign-keys list for a specific context.

        This is a convenience tool for reordering related tables without manually
        reconstructing the list. Changes are staged until apply_annotations()
        is called.

        Args:
            table_name: Name of the table.
            context: The context to modify (e.g., "detailed", "*").
            new_order: The new order specification. Can be:
                - List of indices: [2, 0, 1] reorders by current positions
                - List of foreign key refs: [["schema", "fkey1"], ...] specifies exact order

        Returns:
            JSON with the reordered foreign key list for the context.

        Examples:
            # Reorder by indices (move item at index 2 to front)
            reorder_visible_foreign_keys("Subject", "detailed", [2, 0, 1])

            # Reorder by specifying exact foreign key order
            reorder_visible_foreign_keys("Subject", "detailed", [
                ["domain", "Diagnosis_Subject_fkey"],
                ["domain", "Image_Subject_fkey"]
            ])
        """
        try:
            ml = conn_manager.get_active_or_raise()
            new_list = ml.reorder_visible_foreign_keys(table_name, context, new_order)

            return json.dumps({
                "status": "staged",
                "target": table_name,
                "context": context,
                "new_order": new_list,
                "message": "Use apply_annotations() to commit changes to the catalog.",
            })
        except Exception as e:
            logger.error(f"Failed to reorder visible foreign keys: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    # =========================================================================
    # Handlebars Template Tools
    # =========================================================================

    @mcp.tool()
    async def get_handlebars_template_variables(table_name: str) -> str:
        """Get all available template variables for a table.

        Returns the columns, foreign keys, and special variables that can be
        used in Handlebars templates (row_markdown_pattern, markdown_pattern, etc.)
        for the specified table.

        Args:
            table_name: Name of the table to get variables for.

        Returns:
            JSON with columns, foreign_keys, and special variables available
            for use in templates.

        Example:
            get_handlebars_template_variables("Image") -> {
                "table": "Image",
                "columns": [
                    {"name": "RID", "type": "ermrest_rid", "template": "{{{RID}}}"},
                    {"name": "Filename", "type": "text", "template": "{{{Filename}}}"},
                    ...
                ],
                "foreign_keys": [
                    {
                        "constraint": ["domain", "Image_Subject_fkey"],
                        "to_table": "Subject",
                        "values_template": "{{{$fkeys.domain.Image_Subject_fkey.values.column}}}",
                        "row_name_template": "{{{$fkeys.domain.Image_Subject_fkey.rowName}}}"
                    }
                ],
                "special_variables": {...}
            }
        """
        try:
            ml = conn_manager.get_active_or_raise()
            return json.dumps(ml.get_handlebars_template_variables(table_name))
        except Exception as e:
            logger.error(f"Failed to get template variables: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def get_table_sample_data(
        table_name: str,
        limit: int = 3,
    ) -> str:
        """Get sample row data from a table for template testing.

        Retrieves a few sample rows from the table that can be used to test
        Handlebars templates. Use this to see real values that would be
        available in templates.

        Args:
            table_name: Name of the table.
            limit: Number of sample rows to return (default: 3, max: 10).

        Returns:
            JSON with sample rows and their column values.

        Example:
            get_table_sample_data("Image", 2) -> {
                "table": "Image",
                "sample_rows": [
                    {"RID": "1-ABC", "Filename": "scan001.jpg", "Subject": "2-DEF", ...},
                    {"RID": "1-XYZ", "Filename": "scan002.jpg", "Subject": "2-DEF", ...}
                ],
                "template_test_suggestion": "Try: {{{Filename}}} - Subject: {{{Subject}}}"
            }
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.model.name_to_table(table_name)

            # Limit to reasonable number
            limit = min(max(1, limit), 10)

            # Query sample data
            path = ml.catalog.getPathBuilder().schemas[table.schema.name].tables[table.name]
            results = list(path.entities().fetch(limit=limit))

            # Build template suggestion based on columns
            columns = [col.name for col in table.columns if not col.name.startswith('RC') and not col.name.startswith('RM')]
            suggestion_cols = columns[:3] if columns else ['RID']
            suggestion = " - ".join(["{{{" + c + "}}}" for c in suggestion_cols])

            return json.dumps({
                "table": table_name,
                "row_count": len(results),
                "sample_rows": results,
                "template_test_suggestion": f"Try: {suggestion}",
                "columns": [col.name for col in table.columns],
            })
        except Exception as e:
            logger.error(f"Failed to get sample data: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def preview_handlebars_template(
        template: str,
        data: dict[str, Any],
    ) -> str:
        """Preview a Handlebars template with sample data.

        Renders a Handlebars template using the provided data dictionary.
        This is useful for testing templates before applying them to annotations.

        Note: This is a simplified preview - the actual Deriva UI may render
        templates slightly differently with additional context.

        Args:
            template: Handlebars template string (e.g., "{{{Name}}} ({{{Status}}})")
            data: Dictionary of values to use in the template.

        Returns:
            JSON with the rendered output and any errors.

        **Template Syntax Quick Reference:**
        - `{{{variable}}}` - Output value (raw, no escaping)
        - `{{variable}}` - Output value (HTML escaped)
        - `{{#if var}}...{{/if}}` - Conditional
        - `{{#if var}}...{{else}}...{{/if}}` - If/else
        - `{{#unless var}}...{{/unless}}` - Inverse conditional
        - `{{#each array}}...{{/each}}` - Iteration
        - `{{{_value}}}` - Current column value (in column_display)
        - `{{{_row.column}}}` - Other column from same row

        Examples:
            # Simple template
            preview_handlebars_template(
                "{{{Name}}} - {{{Status}}}",
                {"Name": "John", "Status": "Active"}
            ) -> "John - Active"

            # Conditional template
            preview_handlebars_template(
                "{{{Name}}}{{#if Nickname}} ({{{Nickname}}}){{/if}}",
                {"Name": "John", "Nickname": "Johnny"}
            ) -> "John (Johnny)"

            # Without the optional value
            preview_handlebars_template(
                "{{{Name}}}{{#if Nickname}} ({{{Nickname}}}){{/if}}",
                {"Name": "John"}
            ) -> "John"
        """
        try:
            # Simple Handlebars-like template rendering
            # This is a basic implementation for preview purposes
            result = template

            # Handle {{{variable}}} and {{variable}} substitution
            import re

            # First handle triple braces (raw output)
            def replace_triple(match: re.Match) -> str:
                var_name = match.group(1).strip()
                # Handle _value special case
                if var_name == "_value" or var_name == "value":
                    return str(data.get("_value", data.get("value", "")))
                # Handle _row.column
                if var_name.startswith("_row."):
                    col = var_name[5:]
                    return str(data.get(col, ""))
                # Handle nested paths like $fkeys.schema.name.values.col
                if "." in var_name:
                    parts = var_name.split(".")
                    value = data
                    for part in parts:
                        if isinstance(value, dict):
                            value = value.get(part, "")
                        else:
                            value = ""
                            break
                    return str(value)
                return str(data.get(var_name, ""))

            result = re.sub(r'\{\{\{([^}]+)\}\}\}', replace_triple, result)

            # Handle double braces (same for preview, would be escaped in real render)
            result = re.sub(r'\{\{([^#/][^}]*)\}\}', replace_triple, result)

            # Handle {{#if variable}}...{{/if}} blocks
            def handle_if_block(match: re.Match) -> str:
                condition_var = match.group(1).strip()
                content = match.group(2)

                # Check for {{else}}
                if "{{else}}" in content:
                    parts = content.split("{{else}}")
                    true_content = parts[0]
                    false_content = parts[1] if len(parts) > 1 else ""
                else:
                    true_content = content
                    false_content = ""

                # Evaluate condition
                value = data.get(condition_var)
                is_truthy = bool(value) and value != "" and value != [] and value != {}

                chosen = true_content if is_truthy else false_content

                # Recursively process the chosen content
                chosen = re.sub(r'\{\{\{([^}]+)\}\}\}', replace_triple, chosen)
                chosen = re.sub(r'\{\{([^#/][^}]*)\}\}', replace_triple, chosen)

                return chosen

            # Process #if blocks (non-greedy, innermost first)
            max_iterations = 10
            for _ in range(max_iterations):
                new_result = re.sub(
                    r'\{\{#if\s+([^}]+)\}\}((?:(?!\{\{#if).)*?)\{\{/if\}\}',
                    handle_if_block,
                    result,
                    flags=re.DOTALL
                )
                if new_result == result:
                    break
                result = new_result

            # Handle {{#unless variable}}...{{/unless}} blocks
            def handle_unless_block(match: re.Match) -> str:
                condition_var = match.group(1).strip()
                content = match.group(2)

                value = data.get(condition_var)
                is_falsy = not value or value == "" or value == [] or value == {}

                if is_falsy:
                    content = re.sub(r'\{\{\{([^}]+)\}\}\}', replace_triple, content)
                    content = re.sub(r'\{\{([^#/][^}]*)\}\}', replace_triple, content)
                    return content
                return ""

            result = re.sub(
                r'\{\{#unless\s+([^}]+)\}\}(.*?)\{\{/unless\}\}',
                handle_unless_block,
                result,
                flags=re.DOTALL
            )

            # Handle {{#each array}}...{{/each}} blocks
            def handle_each_block(match: re.Match) -> str:
                array_var = match.group(1).strip()
                item_template = match.group(2)

                array_value = data.get(array_var, [])
                if not isinstance(array_value, list):
                    return ""

                results_list = []
                for i, item in enumerate(array_value):
                    item_result = item_template

                    # Replace {{{this}}} with current item
                    if isinstance(item, dict):
                        for k, v in item.items():
                            item_result = item_result.replace("{{{this." + k + "}}}", str(v))
                            item_result = item_result.replace("{{this." + k + "}}", str(v))
                        item_result = item_result.replace("{{{this}}}", str(item))
                        item_result = item_result.replace("{{this}}", str(item))
                    else:
                        item_result = item_result.replace("{{{this}}}", str(item))
                        item_result = item_result.replace("{{this}}", str(item))

                    # Handle @index, @first, @last
                    item_result = item_result.replace("{{@index}}", str(i))
                    item_result = re.sub(
                        r'\{\{#if @first\}\}(.*?)\{\{/if\}\}',
                        r'\1' if i == 0 else '',
                        item_result
                    )
                    item_result = re.sub(
                        r'\{\{#if @last\}\}(.*?)\{\{/if\}\}',
                        r'\1' if i == len(array_value) - 1 else '',
                        item_result
                    )
                    item_result = re.sub(
                        r'\{\{#unless @first\}\}(.*?)\{\{/unless\}\}',
                        '' if i == 0 else r'\1',
                        item_result
                    )
                    item_result = re.sub(
                        r'\{\{#unless @last\}\}(.*?)\{\{/unless\}\}',
                        '' if i == len(array_value) - 1 else r'\1',
                        item_result
                    )

                    results_list.append(item_result)

                return "".join(results_list)

            result = re.sub(
                r'\{\{#each\s+([^}]+)\}\}(.*?)\{\{/each\}\}',
                handle_each_block,
                result,
                flags=re.DOTALL
            )

            return json.dumps({
                "status": "success",
                "template": template,
                "data": data,
                "rendered": result,
                "note": "This is a simplified preview. Actual Deriva rendering may differ slightly."
            })
        except Exception as e:
            logger.error(f"Failed to preview template: {e}")
            return json.dumps({
                "status": "error",
                "template": template,
                "message": str(e)
            })

    @mcp.tool()
    async def validate_template_syntax(template: str) -> str:
        """Validate the syntax of a Handlebars template.

        Checks a template for common syntax errors like unmatched braces,
        unclosed blocks, and invalid helper usage.

        Args:
            template: Handlebars template string to validate.

        Returns:
            JSON with validation results including any errors or warnings.

        Example:
            validate_template_syntax("{{#if Name}}{{{Name}}}{{/if}}")
            -> {"valid": true, "errors": [], "warnings": []}

            validate_template_syntax("{{#if Name}}{{{Name}}}")
            -> {"valid": false, "errors": ["Unclosed #if block"]}
        """
        import re

        errors = []
        warnings = []

        # Check for unmatched braces
        triple_open = len(re.findall(r'\{\{\{', template))
        triple_close = len(re.findall(r'\}\}\}', template))
        if triple_open != triple_close:
            errors.append(f"Unmatched triple braces: {triple_open} opening vs {triple_close} closing")

        # Remove triple braces to check double braces
        temp = re.sub(r'\{\{\{.*?\}\}\}', '', template)
        double_open = len(re.findall(r'\{\{(?!\{)', temp))
        double_close = len(re.findall(r'(?<!\})\}\}', temp))
        if double_open != double_close:
            errors.append(f"Unmatched double braces: {double_open} opening vs {double_close} closing")

        # Check for block helpers
        block_helpers = ['if', 'unless', 'each', 'with']
        for helper in block_helpers:
            opens = len(re.findall(rf'\{{\{{#{helper}\s', template))
            closes = len(re.findall(rf'\{{\{{/{helper}\}}\}}', template))
            if opens != closes:
                errors.append(f"Unclosed #{helper} block: {opens} opening vs {closes} closing")

        # Check for common mistakes
        if '{{#if}}' in template:
            errors.append("#if block requires a condition variable")
        if '{{#each}}' in template:
            errors.append("#each block requires an array variable")

        # Warnings for potential issues
        if '{{' in template and '{{{' not in template:
            warnings.append("Using {{...}} will HTML-escape output. Consider {{{...}}} for raw output.")

        if re.search(r'\{\{\s*\}\}', template):
            warnings.append("Empty template expression found: {{}}")

        # Check for valid variable names
        var_pattern = re.findall(r'\{\{(?:#\w+\s+)?([^}#/]+)(?:\}\}|\}\}\})', template)
        for var in var_pattern:
            var = var.strip()
            if var and not re.match(r'^[\w.$@_]+$', var.split()[0]):
                warnings.append(f"Unusual variable name: {var}")

        return json.dumps({
            "valid": len(errors) == 0,
            "template": template,
            "errors": errors,
            "warnings": warnings,
            "block_counts": {
                "if_blocks": len(re.findall(r'\{\{#if\s', template)),
                "unless_blocks": len(re.findall(r'\{\{#unless\s', template)),
                "each_blocks": len(re.findall(r'\{\{#each\s', template)),
                "variables": len(re.findall(r'\{\{\{?[^#/}]+\}\}\}?', template)),
            }
        })
