# Common Annotation Recipes and Patterns

## Column Display Formatting

Control how individual column values are rendered:

```
set_column_display(
    table_name="Image",
    column_name="URL",
    annotation={"compact": {"markdown_pattern": "[Download]({{{URL}}})"}}
)

set_column_display(
    table_name="Measurement",
    column_name="Value",
    annotation={"compact": {"markdown_pattern": "{{{Value}}} {{{Units}}}"}}
)
```

## Common Recipes

### Hide system columns from compact view
```
remove_visible_column(table_name="Image", context="compact", column="RID")
remove_visible_column(table_name="Image", context="compact", column="RCT")
remove_visible_column(table_name="Image", context="compact", column="RMT")
remove_visible_column(table_name="Image", context="compact", column="RCB")
remove_visible_column(table_name="Image", context="compact", column="RMB")
```

### Make a table's compact view show key info only
```
set_visible_columns(
    table_name="Subject",
    annotation={"compact": ["Name", "Age", "Sex", "Species", "Diagnosis"]}
)
set_row_name_pattern(table_name="Subject", pattern="{{{Name}}}")
apply_annotations()
```

### Configure a vocabulary table display
```
set_visible_columns(
    table_name="Diagnosis",
    annotation={"compact": ["Name", "Description", "Synonyms"]}
)
set_row_name_pattern(table_name="Diagnosis", pattern="{{{Name}}}")
apply_annotations()
```
