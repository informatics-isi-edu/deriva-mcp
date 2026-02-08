"""MCP Prompts for DerivaML.

This module provides prompt registration functions that expose
workflow guides and procedures as MCP prompts for LLM applications.

Prompts provide interactive, step-by-step guidance for common tasks:
- ML execution lifecycle (training, inference workflows)
- Dataset preparation for ML training
- Annotation customization workflows
- Feature engineering workflows
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from deriva_ml_mcp.connection import ConnectionManager


def register_prompts(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register all DerivaML prompts with the MCP server."""

    # =========================================================================
    # ML Execution Workflow Prompts
    # =========================================================================

    @mcp.prompt(
        name="run-ml-execution",
        description="Python API and MCP tools guide for ML execution lifecycle - use for custom scripts and interactive work",
    )
    def run_ml_execution_prompt() -> str:
        """Guide for running an ML execution with provenance."""
        return """# Running an ML Execution with Provenance

**Use this prompt for**: Custom Python scripts, interactive work, or when NOT using
the `deriva-ml-run` CLI. For CLI-based experiments with full git verification,
use the `run-experiment` prompt instead.

Follow these steps to run an ML workflow (training, inference, preprocessing, etc.)
with full provenance tracking in DerivaML.

## Prerequisites
- Connected to a DerivaML catalog (use `connect_catalog` if not connected)
- Input dataset(s) available (read `deriva-ml://catalog/datasets` resource to find them)

## Python API: Using the Context Manager (Recommended)

The recommended way to run executions in Python is using the context manager,
which automatically handles start/stop timing:

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration, Workflow

# Connect to catalog
ml = DerivaML(hostname="example.org", catalog_id="1")

# Configure the execution
config = ExecutionConfiguration(
    workflow=Workflow(
        name="ResNet Training Run 1",
        workflow_type="Training",
        description="Train ResNet50 on CIFAR-10"
    ),
    datasets=["1-ABC"],  # Input dataset RIDs
    assets=["2-XYZ"],    # Optional: input asset RIDs
)

# Use context manager - automatically starts/stops timing
with ml.create_execution(config) as exe:
    # Download input data
    exe.download_execution_dataset("1-ABC")

    # Run your ML code here...

    # Register output files
    model_path = exe.asset_file_path("Model", "model.pt")
    # Write model to model_path...

    metrics_path = exe.asset_file_path("Execution_Metadata", "metrics.json")
    # Write metrics to metrics_path...

# IMPORTANT: Upload AFTER exiting the context manager
exe.upload_execution_outputs()
```

**Key points:**
- The `with` block automatically calls `start_execution()` on entry
- The `with` block automatically calls `stop_execution()` on exit
- You MUST call `upload_execution_outputs()` AFTER the `with` block

## MCP Tools

MCP tools mirror the Python API. Use the execution tools in this order:

### Step 0 (Optional): Find or Reuse a Workflow

Before creating an execution, optionally check for existing workflows:

```
# List available workflow types - use the resource
# deriva-ml://catalog/workflow-types

# Search for similar workflows by type or description
# Read the deriva-ml://catalog/workflows resource to see all workflows

# Check if workflow exists by URL/checksum (for code-tracked workflows)
lookup_workflow_by_url("https://github.com/org/repo/blob/main/train.py")
```

If a suitable workflow exists, you can reuse it. The `create_execution` tool will
create a new workflow automatically if the exact name/type/description doesn't exist.

### Step 1: Create the Execution

```
create_execution(
    workflow_name="<descriptive name>",  # e.g., "ResNet Training Run 1"
    workflow_type="<type>",              # e.g., "Training", "Inference", "Preprocessing"
    description="<what this run does>",
    dataset_rids=["<input-dataset-rid>"],  # Input datasets for provenance
    asset_rids=["<input-asset-rid>"]       # Optional: specific input assets
)
```

Use the `deriva-ml://catalog/workflow-types` resource to see available workflow types.
Use `add_workflow_type()` to create a new type if needed.

### Step 2: Do Your ML Work

- **Download input data**: Use `download_execution_dataset(dataset_rid)` to get data as a DatasetBag
- **Process/train/infer**: Run your ML code
- **Register outputs**: For each output file, call `asset_file_path()`:

```
asset_file_path(
    asset_name="<asset-table>",  # e.g., "Model", "Execution_Metadata"
    file_name="<path-or-name>",  # Existing file path OR new filename
    asset_types=["<type>"]       # Optional: from Asset_Type vocabulary
)
```

### Step 3: Upload Outputs (REQUIRED)

```
upload_execution_outputs()
```

**Important**: Files are NOT persisted until this is called!

## Example: MCP Tool Workflow

```
# 1. Create execution
create_execution("CIFAR-10 ResNet", "Training", "Train ResNet50 on CIFAR-10", ["1-ABC"])

# 2. Get training data
download_execution_dataset("1-ABC")

# 3. [Run training code here...]

# 4. Register model output
asset_file_path("Model", "/tmp/model.pt", ["Trained Model"])

# 5. Register metrics
asset_file_path("Execution_Metadata", "metrics.json")

# 6. Upload everything
upload_execution_outputs()
```

## Tips

- Use `get_execution_info()` to check current execution status
- Use `update_execution_status(status, message)` for progress updates
- Use `restore_execution(rid)` to resume a previous execution
- Read `deriva-ml://catalog/executions` resource to see past workflow runs
- Read `deriva-ml://catalog/workflows` resource to discover existing workflows before creating new ones
- Read `deriva-ml://catalog/workflow-types` resource to see available workflow type categories
- In Python, always prefer the context manager over manual start/stop
"""

    @mcp.prompt(
        name="prepare-training-data",
        description="Step-by-step guide to prepare dataset for ML training using denormalization",
    )
    def prepare_training_data_prompt() -> str:
        """Guide for preparing ML training data from datasets."""
        return """# Preparing Training Data from DerivaML Datasets

Follow these steps to extract and prepare data from DerivaML datasets
for use in ML training pipelines.

## Prerequisites
- Connected to a DerivaML catalog
- Know your dataset RID (read `deriva-ml://catalog/datasets` resource to find it)

## Step 1: Explore Your Dataset

First, understand what data is available:

```
# Read the deriva-ml://catalog/datasets resource to list all datasets

# Get details about your dataset - use the resource
# deriva-ml://dataset/<dataset-rid>

# See which tables have data in this dataset - use the resource
# deriva-ml://dataset/<dataset-rid>/members
```

## Step 2: Understand Table Structure

For each table you want to use:

```
# See columns and their types - use the resource
# deriva-ml://table/<table-name>/schema
```

## Step 3: Choose Your Approach

### Option A: Denormalization (Recommended for Training)

Join multiple tables into a flat structure suitable for ML:

```
denormalize_dataset(
    dataset_rid="<rid>",
    include_tables=["Image", "Subject", "Diagnosis"],  # Tables to join
    limit=1000  # Start small for testing
)
```

This returns:
- `columns`: List like ["Image.RID", "Image.Filename", "Subject.Name", "Diagnosis.Label"]
- `rows`: Array of dictionaries with all values

### Option B: Query Table Data

Query records from a specific table (not dataset-filtered):

```
query_table("<table-name>", limit=1000)
```

### Option C: Full Download (Production)

For large datasets, download locally:

```
download_dataset("<dataset-rid>", materialize=True)
```

This returns a local path to a BDBag with all data and files.

## Step 4: Use the Data

The denormalized data is ready for ML frameworks:

**Example columns from denormalize_dataset:**
- `Image.URL` - Path/URL to image file
- `Image.Filename` - Image filename
- `Subject.Age` - Subject metadata
- `Diagnosis.Label` - Classification label

**Common patterns:**
- Image classification: Join [Image, Label]
- Medical imaging: Join [Image, Subject, Diagnosis]
- Multi-modal: Join [Image, Scan, Subject]

## Step 5: Version Pinning (Reproducibility)

For reproducible training, pin to a specific dataset version:

```
# Check version history - use the resource
# deriva-ml://dataset/<dataset-rid>/versions

# Use specific version
denormalize_dataset("<rid>", ["Image", "Label"], version="1.2.0")
```

## Tips

- Start with `limit=100` to test your joins before fetching all data
- Column names are prefixed with table name to avoid collisions
- Tables must be related through foreign keys to produce join results
- For very large datasets, use `download_dataset()` for local processing
"""

    @mcp.prompt(
        name="customize-display",
        description="MCP tools approach: Customize Chaise web UI display using individual annotation tools",
    )
    def customize_display_prompt() -> str:
        """Guide for customizing catalog and table display annotations."""
        return """# Customizing Display in Chaise (MCP Tools)

**Approach**: This prompt uses individual MCP tools for annotation changes. For a
Python-first approach with type-safe builder classes, see `use-annotation-builders`.

Follow these steps to customize how your catalog and tables appear in the
Chaise web interface using Deriva annotations.

## Prerequisites
- Connected to a DerivaML catalog
- Admin access to modify annotations

## Quick Start: Apply Default Catalog Annotations

Set up standard navigation and display settings for the entire catalog:

```
apply_catalog_annotations(
    navbar_brand_text="My ML Project",  # Text in nav bar
    head_title="ML Catalog"              # Browser tab title
)
```

This automatically configures:
- Navigation bar with organized dropdown menus
- Display settings (underscores as spaces, system columns visible)
- Default landing page (Dataset table)
- Bulk upload support for asset tables

## Step 1: Check Current Annotations

See what's already configured using the resource:

```
deriva-ml://table/<table-name>/annotations
```

This shows current settings for:
- `display`: Name and display options
- `visible_columns`: Which columns appear in each context
- `visible_foreign_keys`: Which related tables are shown
- `table_display`: Row naming and display settings

## Step 2: Understand Contexts

Annotations can be set per-context. Common contexts:

| Context | When Used |
|---------|-----------|
| `*` | Default for all contexts |
| `compact` | Table listings, search results |
| `compact/brief` | Inline foreign key displays |
| `compact/select` | Foreign key selection dropdowns |
| `detailed` | Single record view |
| `entry` | Record creation/editing |
| `entry/create` | New record form |
| `entry/edit` | Edit existing record form |

See full context documentation: `deriva-ml://docs/annotation-contexts`

## Step 3: Customize Display Name

Set a friendly name for the table:

```
set_display_annotation("<table>", {"name": "Friendly Name"})
```

Or for a column:

```
set_display_annotation("<table>", {"name": "Column Label"}, column_name="<column>")
```

## Step 4: Configure Visible Columns

Control which columns appear and their order:

### Quick Method: Add/Remove/Reorder

```
# Add a column to compact view
add_visible_column("<table>", "compact", "<column-name>")

# Remove a column
remove_visible_column("<table>", "compact", "<column-name>")

# Reorder columns (list of indices)
reorder_visible_columns("<table>", "compact", [2, 0, 1, 3])
```

### Full Control: Set All Columns

```
set_visible_columns("<table>", {
    "compact": ["RID", "Name", "Description"],
    "detailed": ["RID", "Name", "Description", "Created", "Modified"],
    "entry": ["Name", "Description"]
})
```

## Step 5: Configure Row Names

Set how rows are identified in listings:

```
set_table_display("<table>", {
    "row_name": {
        "row_markdown_pattern": "{{{Name}}} ({{{RID}}})"
    }
})
```

Use Handlebars syntax: `{{{column_name}}}` for values.

## Step 6: Configure Visible Foreign Keys

Control which related tables appear:

```
# First, find available foreign keys using the resource
deriva-ml://table/<table>/foreign-keys

# Add an FK to detailed view
add_visible_foreign_key("<table>", "detailed", ["schema", "FK_constraint_name"])
```

## Step 7: Apply Changes (REQUIRED)

Commit all staged changes to the catalog:

```
apply_annotations()
```

**Important**: Changes are NOT saved until this is called!

## Complete Example

```
# 1. Check current state (use resource)
deriva-ml://table/Image/annotations

# 2. Set friendly name
set_display_annotation("Image", {"name": "Images"})

# 3. Configure compact view columns
set_visible_columns("Image", {
    "compact": ["RID", "Filename", "Subject", "Created"]
})

# 4. Set row name pattern
set_table_display("Image", {
    "row_name": {
        "row_markdown_pattern": "{{{Filename}}}"
    }
})

# 5. Add related table to detailed view
# Check deriva-ml://table/Image/foreign-keys for FK names
add_visible_foreign_key("Image", "detailed", ["domain", "Image_Subject_fkey"])

# 6. Commit changes
apply_annotations()
```

## Tips

- Always check current annotations before making changes
- Use `*` context for defaults, specific contexts for overrides
- Test changes in Chaise after applying
- Use `deriva-ml://chaise-url/<table>` resource to get the URL to view changes
"""

    @mcp.prompt(
        name="use-annotation-builders",
        description="Python approach: Type-safe annotation builder classes for production code",
    )
    def use_annotation_builders_prompt() -> str:
        """Guide for using Python annotation builder classes."""
        return """# Using DerivaML Annotation Builders (Python)

**Approach**: This prompt covers Python builder classes for annotations. For
quick edits using MCP tools, see `customize-display`.

DerivaML provides **annotation builder classes** that simplify annotation creation
with IDE autocompletion, type safety, and validation. This guide shows how to use
them in Python code.

## When to Use Builders vs MCP Tools

- **MCP Tools** (`customize-display`): Quick edits, exploratory work, CI/CD scripts
- **Python Builders** (this prompt): Production code, complex annotations, type-safe development

## Quick Start

```python
from deriva_ml import DerivaML
from deriva_ml.model import (
    TableHandle, Display, VisibleColumns, TableDisplay,
    PseudoColumn, OutboundFK, InboundFK, fk_constraint, SortKey,
    Aggregate, CONTEXT_COMPACT, CONTEXT_DETAILED
)

# Connect and get table handle
ml = DerivaML(hostname="example.org", catalog_id="1")
table = ml.model.name_to_table("Subject")
handle = TableHandle(table)

# Set display name
handle.set_annotation(Display(
    name="Research Subjects",
    comment="Individuals enrolled in the study"
))

# Configure visible columns
vc = VisibleColumns()
vc.compact(["RID", "Name", "Species", "Age"])
vc.detailed(["RID", "Name", "Species", "Age", "Notes"])
handle.set_annotation(vc)

# Set row name pattern
td = TableDisplay()
td.row_name("{{{Name}}} ({{{Species}}})")
handle.set_annotation(td)
```

## Available Builders

### Display - Table/Column names and descriptions

```python
from deriva_ml.model import Display, NameStyle

# Simple name
Display(name="Friendly Name")

# With markdown (mutually exclusive with name)
Display(markdown_name="**Bold** Name")

# With description (shows as tooltip)
Display(name="Subjects", comment="Research subjects enrolled in the study")

# With name styling
Display(name_style=NameStyle(underline_space=True, title_case=True))
```

### VisibleColumns - Which columns appear

```python
from deriva_ml.model import VisibleColumns, PseudoColumn, fk_constraint

vc = VisibleColumns()

# Simple column lists
vc.compact(["RID", "Name", "Status"])
vc.detailed(["RID", "Name", "Status", "Description"])
vc.entry(["Name", "Status", "Description"])

# Include FK references
vc.compact([
    "RID",
    "Name",
    fk_constraint("domain", "Subject_Species_fkey"),
])

# Include pseudo-columns (computed values)
vc.detailed([
    "RID",
    "Name",
    PseudoColumn(source="Description", markdown_name="Notes"),
])

# Method chaining
vc = (VisibleColumns()
    .compact(["RID", "Name"])
    .detailed(["RID", "Name", "Description"])
    .entry(["Name", "Description"]))
```

### VisibleForeignKeys - Related tables in detail view

```python
from deriva_ml.model import VisibleForeignKeys, fk_constraint

vfk = VisibleForeignKeys()
vfk.detailed([
    fk_constraint("domain", "Image_Subject_fkey"),
    fk_constraint("domain", "Diagnosis_Subject_fkey"),
])
```

### TableDisplay - Row naming and ordering

```python
from deriva_ml.model import TableDisplay, TableDisplayOptions, SortKey

td = TableDisplay()

# Row name pattern (used in dropdowns)
td.row_name("{{{Name}}} ({{{RID}}})")

# Compact view options
td.compact(TableDisplayOptions(
    row_order=[
        SortKey("Name"),                      # Ascending
        SortKey("Created", descending=True),  # Descending
    ],
    page_size=50
))

# Detailed view options
td.detailed(TableDisplayOptions(
    collapse_toc_panel=True,
    hide_column_headers=False
))
```

### ColumnDisplay - Column value formatting

```python
from deriva_ml.model import ColumnDisplay, ColumnDisplayOptions, PreFormat

cd = ColumnDisplay()

# Number formatting
cd.default(ColumnDisplayOptions(
    pre_format=PreFormat(format="%.2f")
))

# Boolean formatting
cd.default(ColumnDisplayOptions(
    pre_format=PreFormat(bool_true_value="Yes", bool_false_value="No")
))

# Markdown pattern (make URLs clickable)
cd.default(ColumnDisplayOptions(
    markdown_pattern="[Link]({{{_value}}})"
))

# Apply to column
col_handle = handle.column("URL")
col_handle.annotations[ColumnDisplay.tag] = cd.to_dict()
col_handle.apply()
```

## PseudoColumns - Computed and FK-traversed values

### Basic pseudo-column
```python
PseudoColumn(source="Internal_ID", markdown_name="ID")
```

### FK traversal - Get values from related tables
```python
from deriva_ml.model import PseudoColumn, OutboundFK, InboundFK

# Outbound: Follow FK to get value from referenced table
# Image -> Subject (get Subject name)
PseudoColumn(
    source=[OutboundFK("domain", "Image_Subject_fkey"), "Name"],
    markdown_name="Subject Name"
)

# Inbound: Aggregate values from referencing table
# Subject <- Images (count images)
PseudoColumn(
    source=[InboundFK("domain", "Image_Subject_fkey"), "RID"],
    aggregate=Aggregate.CNT,
    markdown_name="Image Count"
)

# Multi-hop: Chain FKs
# Image -> Subject -> Species
PseudoColumn(
    source=[
        OutboundFK("domain", "Image_Subject_fkey"),
        OutboundFK("domain", "Subject_Species_fkey"),
        "Name"
    ],
    markdown_name="Species"
)
```

### Aggregates
```python
from deriva_ml.model import Aggregate

# Available aggregates
Aggregate.CNT       # Count
Aggregate.CNT_D     # Count distinct
Aggregate.MIN       # Minimum
Aggregate.MAX       # Maximum
Aggregate.ARRAY     # Array of values
Aggregate.ARRAY_D   # Array of distinct values
```

## Faceted Search Configuration

```python
from deriva_ml.model import Facet, FacetList, FacetRange, FacetUxMode, OutboundFK

facets = FacetList()

# Simple choice facet
facets.add(Facet(source="Status", open=True, markdown_name="Status"))

# FK-based facet
facets.add(Facet(
    source=[OutboundFK("domain", "Subject_Species_fkey"), "Name"],
    markdown_name="Species",
    open=True
))

# Range facet
facets.add(Facet(
    source="Age",
    ux_mode=FacetUxMode.RANGES,
    ranges=[
        FacetRange(min=0, max=18),
        FacetRange(min=18, max=65),
        FacetRange(min=65),  # 65+
    ],
    markdown_name="Age Group"
))

# Check presence facet (has value / no value)
facets.add(Facet(
    source="Notes",
    ux_mode=FacetUxMode.CHECK_PRESENCE,
    markdown_name="Has Notes"
))

# Apply to visible columns
vc = VisibleColumns()
vc.compact(["RID", "Name"])
vc._contexts["filter"] = facets.to_dict()
handle.set_annotation(vc)
```

## Handlebars Templates

Templates use `{{{column_name}}}` syntax for values.

```python
# Get available variables for a table
vars = ml.get_handlebars_template_variables("Subject")

# Common patterns
"{{{Name}}}"                              # Simple value
"{{{Name}}} ({{{RID}}})"                  # Multiple values
"{{#if Notes}}{{{Notes}}}{{else}}N/A{{/if}}"  # Conditional
"[{{{Filename}}}]({{{URL}}})"             # Link
"{{formatDate RCT 'YYYY-MM-DD'}}"         # Date formatting

# FK values
"{{{$fkeys.domain.Subject_Species_fkey.values.Name}}}"
```

## Context Constants

```python
from deriva_ml.model import (
    CONTEXT_DEFAULT,    # "*" - fallback for all
    CONTEXT_COMPACT,    # "compact" - table/list views
    CONTEXT_DETAILED,   # "detailed" - single record view
    CONTEXT_ENTRY,      # "entry" - create/edit forms
    CONTEXT_FILTER,     # "filter" - facet panel
)
```

## Complete Example

```python
from deriva_ml import DerivaML
from deriva_ml.model import (
    TableHandle, Display, VisibleColumns, VisibleForeignKeys,
    TableDisplay, TableDisplayOptions, PseudoColumn, Facet, FacetList,
    OutboundFK, InboundFK, fk_constraint, SortKey, Aggregate, FacetUxMode
)

ml = DerivaML(hostname="example.org", catalog_id="1")
table = ml.model.name_to_table("Subject")
handle = TableHandle(table)

# 1. Display annotation
handle.set_annotation(Display(
    name="Research Subjects",
    comment="Individuals enrolled in research studies"
))

# 2. Table display
td = TableDisplay()
td.row_name("{{{Name}}} ({{{Subject_ID}}})")
td.compact(TableDisplayOptions(row_order=[SortKey("Name")], page_size=25))
handle.set_annotation(td)

# 3. Visible columns with pseudo-column
vc = VisibleColumns()
vc.compact([
    "RID", "Subject_ID", "Name",
    fk_constraint("domain", "Subject_Species_fkey"),
    "Age",
    PseudoColumn(
        source=[InboundFK("domain", "Sample_Subject_fkey"), "RID"],
        aggregate=Aggregate.CNT,
        markdown_name="Samples"
    ),
])
vc.detailed([
    "RID", "Subject_ID", "Name",
    fk_constraint("domain", "Subject_Species_fkey"),
    "Age", "Sex", "Enrollment_Date", "Notes"
])
vc.entry(["Subject_ID", "Name", "Species", "Age", "Sex", "Notes"])

# 4. Faceted search
facets = FacetList()
facets.add(Facet(
    source=[OutboundFK("domain", "Subject_Species_fkey"), "Name"],
    markdown_name="Species",
    open=True
))
facets.add(Facet(source="Sex", open=True))
facets.add(Facet(source="Age", ux_mode=FacetUxMode.RANGES))
vc._contexts["filter"] = facets.to_dict()
handle.set_annotation(vc)

# 5. Visible foreign keys
vfk = VisibleForeignKeys()
vfk.detailed([
    fk_constraint("domain", "Sample_Subject_fkey"),
    fk_constraint("domain", "Diagnosis_Subject_fkey"),
])
handle.set_annotation(vfk)

# 6. Column annotations
age_col = handle.column("Age")
age_col.description = "Age in years at enrollment"
age_col.set_display_name("Age (years)")

print("Annotations configured!")
print(f"View at: {ml.get_chaise_url('Subject')}")
```

## Tips

- Use `handle.set_annotation()` with `apply=True` (default) to apply immediately
- Builders have `to_dict()` method if you need the raw dictionary
- Raw dictionaries still work for edge cases not covered by builders
- Test changes in Chaise after applying
- Use `get_handlebars_template_variables()` to see available template variables
"""

    @mcp.prompt(
        name="create-feature",
        description="Step-by-step guide to create and populate a feature for ML labeling",
    )
    def create_feature_prompt() -> str:
        """Guide for creating and using features."""
        return """# Creating Features for ML Labeling

Features associate labels, scores, or derived data with domain objects (like Images)
for use in ML workflows. Follow these steps to create and populate features.

## Prerequisites
- Connected to a DerivaML catalog
- Know which table you want to add features to (e.g., "Image")
- Have vocabulary terms for labels OR asset table for derived files

## Understanding Features

A Feature links:
- **Target table**: The domain object being labeled (e.g., Image)
- **Feature values**: Labels from vocabularies OR references to asset files
- **Execution**: Provenance tracking (which workflow created the label)

## Step 1: Check Existing Features

See what features already exist:

```
# Read the deriva-ml://table/{table_name}/features resource to find features for a table

# Get details about a specific feature
# Read deriva-ml://feature/{table_name}/{feature_name} resource

# List all registered feature names - use the resource
# deriva-ml://catalog/features
```

## Step 2: Create Vocabulary (if needed)

If your labels come from a controlled vocabulary:

```
# Create the vocabulary table
create_vocabulary("Diagnosis_Type", "Types of diagnoses for images")

# Add terms
add_term("Diagnosis_Type", "Normal", "No abnormality detected")
add_term("Diagnosis_Type", "Abnormal", "Abnormality present")
add_term("Diagnosis_Type", "Uncertain", "Requires further review")
```

## Step 3: Create the Feature

Create a feature definition:

```
create_feature(
    table_name="Image",           # Target table
    feature_name="Diagnosis",     # Feature name
    comment="Clinical diagnosis label",
    terms=["Diagnosis_Type"]      # Vocabulary tables for values
)
```

For features that reference asset files (e.g., segmentation masks):

```
create_feature(
    table_name="Image",
    feature_name="Segmentation",
    comment="Segmentation mask image",
    assets=["Segmentation_Mask"]  # Asset tables for values
)
```

## Step 4: Add Feature Values

Add labels within an execution context for provenance:

### Simple Labels (Single Value)

**Python API (recommended):**
```python
config = ExecutionConfiguration(
    workflow=Workflow(name="Labeling Run", workflow_type="Annotation", description="Manual image labeling")
)

with ml.create_execution(config) as exe:
    exe.add_feature_value("Image", "Diagnosis", "<image-rid>", "Normal")
    exe.add_feature_value("Image", "Diagnosis", "<image-rid-2>", "Abnormal")

# Upload AFTER exiting context manager
exe.upload_execution_outputs()
```

**MCP Tools:**
```
# Create execution
create_execution("Labeling Run", "Annotation", "Manual image labeling")

# Add labels
add_feature_value("Image", "Diagnosis", "<image-rid>", "Normal")
add_feature_value("Image", "Diagnosis", "<image-rid-2>", "Abnormal")

# Upload outputs
upload_execution_outputs()
```

### Complex Labels (Multiple Fields)

For features with multiple columns (read `deriva-ml://feature/{table}/{feature}` resource):

```
add_feature_value_record(
    table_name="Image",
    feature_name="Diagnosis",
    target_rid="<image-rid>",
    values={
        "Diagnosis_Type": "Normal",
        "confidence": 0.95
    }
)
```

## Step 5: Query Feature Values

Retrieve labels for analysis or training:

```
# Get all values for a feature
# Read deriva-ml://feature/Image/Diagnosis/values resource
```

## Complete Example: Image Classification Labels

**Python API (recommended):**
```python
# 1. Create vocabulary
ml.create_vocabulary("Image_Class", "Image classification categories")
ml.add_term("Image_Class", "Cat", "Image contains a cat")
ml.add_term("Image_Class", "Dog", "Image contains a dog")
ml.add_term("Image_Class", "Other", "Other content")

# 2. Create feature
ml.create_feature("Image", "Classification", "Image class label", terms=["Image_Class"])

# 3. Label images within execution context
config = ExecutionConfiguration(
    workflow=Workflow(name="Manual Labeling", workflow_type="Annotation", description="Label training images")
)

with ml.create_execution(config) as exe:
    # Add labels (typically in a loop over images)
    exe.add_feature_value("Image", "Classification", "1-ABC", "Cat")
    exe.add_feature_value("Image", "Classification", "1-DEF", "Dog")
    exe.add_feature_value("Image", "Classification", "1-GHI", "Cat")

# Upload AFTER exiting context manager
exe.upload_execution_outputs()

# 4. Query for training
ml.# Read deriva-ml://feature/Image/Classification/values resource
```

**MCP Tools:**
```
# 1. Create vocabulary
create_vocabulary("Image_Class", "Image classification categories")
add_term("Image_Class", "Cat", "Image contains a cat")
add_term("Image_Class", "Dog", "Image contains a dog")
add_term("Image_Class", "Other", "Other content")

# 2. Create feature
create_feature("Image", "Classification", "Image class label", terms=["Image_Class"])

# 3. Create labeling execution
create_execution("Manual Labeling", "Annotation", "Label training images")

# 4. Add labels (typically in a loop over images)
add_feature_value("Image", "Classification", "1-ABC", "Cat")
add_feature_value("Image", "Classification", "1-DEF", "Dog")
add_feature_value("Image", "Classification", "1-GHI", "Cat")

# 5. Upload outputs
upload_execution_outputs()

# 6. Query for training
# Read deriva-ml://feature/Image/Classification/values resource
```

## Tips

- Always add feature values within an execution for provenance
- **Always provide a `comment`** for features to describe what they represent
- **Always provide `description`** for vocabulary terms to explain their meaning
- Read `deriva-ml://feature/{table}/{feature}` resource to see the feature's column structure
- Features can reference both vocabulary terms AND assets
- Feature values track which execution created them
- Use `delete_feature` to remove a feature definition (and all values)
"""

    @mcp.prompt(
        name="create-table",
        description="Step-by-step guide to create domain tables with columns and foreign keys",
    )
    def create_table_prompt() -> str:
        """Guide for creating tables in the domain schema."""
        return """# Creating Tables in DerivaML

Tables store your domain data (subjects, samples, experiments, etc.). Follow this
guide to create tables with proper column types and relationships.

## Prerequisites
- Connected to a DerivaML catalog (use `connect_catalog` if not connected)
- Know what data you want to store and how tables relate

## Table Types

DerivaML has two main table types:

1. **Standard Tables** (`create_table`): For storing structured data like subjects,
   experiments, protocols
2. **Asset Tables** (`create_asset_table`): For files with automatic URL, checksum,
   and provenance tracking (images, models, etc.)

This guide covers standard tables. See `create_asset_table` for file storage.

## Step 1: Plan Your Table Structure

Before creating a table, determine:
- **Table name**: Use singular nouns with underscores (e.g., "Subject", "Tissue_Sample")
- **Columns**: What data fields do you need?
- **Foreign keys**: Does this table reference other tables?

## Step 2: Choose Column Types

Available column types:

| Type | Use For | Example |
|------|---------|---------|
| `text` | Names, identifiers, short strings | Name, ID, Code |
| `markdown` | Long text with formatting | Description, Notes |
| `int2` | Small integers (-32768 to 32767) | Age, Count |
| `int4` | Standard integers | Quantity, Score |
| `int8` | Large integers | File size, timestamps |
| `float4` | Single precision decimals | Temperature, Weight |
| `float8` | Double precision decimals | Precise measurements |
| `boolean` | True/False values | Is_Active, Has_Consent |
| `date` | Dates only | Birth_Date, Collection_Date |
| `timestamp` | Date and time (no timezone) | Created_At |
| `timestamptz` | Date and time with timezone | Event_Time |
| `json` / `jsonb` | Structured data | Metadata, Config |

## Step 3: Create a Simple Table

Create a table with basic columns. **Always provide descriptions** for both the table
and each column to make your catalog self-documenting:

```
create_table(
    "Subject",
    columns=[
        {"name": "Name", "type": "text", "nullok": false, "comment": "Full name of the research subject"},
        {"name": "Age", "type": "int4", "comment": "Age in years at enrollment"},
        {"name": "Notes", "type": "markdown", "comment": "Additional observations or comments"}
    ],
    comment="Research subjects enrolled in the study"
)
```

**Column options:**
- `name` (required): Column name
- `type`: Data type (default: "text")
- `nullok`: Allow NULL values (default: true)
- `comment`: **Always provide** - describes what the column stores, units, valid ranges, etc.

## Step 4: Create Tables with Foreign Keys

To link tables together, use foreign keys:

```
# First, create the parent table
create_table(
    "Subject",
    columns=[
        {"name": "Name", "type": "text", "nullok": false, "comment": "Subject identifier"},
        {"name": "Species", "type": "text", "comment": "Species name (e.g., 'Homo sapiens', 'Mus musculus')"}
    ],
    comment="Research subjects enrolled in the study"
)

# Then create a child table that references it
create_table(
    "Sample",
    columns=[
        {"name": "Name", "type": "text", "nullok": false, "comment": "Sample identifier"},
        {"name": "Subject", "type": "text", "nullok": false, "comment": "Reference to the source subject"},
        {"name": "Collection_Date", "type": "date", "comment": "Date the sample was collected"},
        {"name": "Tissue_Type", "type": "text", "comment": "Type of tissue (e.g., 'blood', 'biopsy')"}
    ],
    foreign_keys=[
        {
            "column": "Subject",
            "referenced_table": "Subject",
            "on_delete": "CASCADE"
        }
    ],
    comment="Biological samples collected from subjects"
)
```

**Foreign key options:**
- `column` (required): Column in this table
- `referenced_table` (required): Table to reference
- `referenced_column`: Column in referenced table (default: "RID")
- `on_delete`: What happens when referenced row is deleted
  - `"NO ACTION"` (default): Prevent deletion if references exist
  - `"CASCADE"`: Delete this row too
  - `"SET NULL"`: Set the foreign key column to NULL

## Step 5: Verify Your Table

After creation, verify the table structure:

```
# List all tables - use the resource
# deriva-ml://catalog/tables

# Get detailed schema - use the resource
# deriva-ml://table/Subject/schema

# Get Chaise URL to view in browser - use the resource
# deriva-ml://chaise-url/Subject
```

## Common Patterns

### Subject -> Sample -> Measurement Hierarchy

```
# Subject (top level)
create_table("Subject", columns=[
    {"name": "Name", "type": "text", "nullok": false, "comment": "Subject identifier"},
    {"name": "Age", "type": "int4", "comment": "Age in years at enrollment"}
], comment="Research subjects enrolled in the study")

# Sample references Subject
create_table("Sample", columns=[
    {"name": "Name", "type": "text", "nullok": false, "comment": "Sample identifier"},
    {"name": "Subject", "type": "text", "nullok": false, "comment": "Reference to the source subject"},
    {"name": "Collection_Date", "type": "date", "comment": "Date the sample was collected"}
], foreign_keys=[
    {"column": "Subject", "referenced_table": "Subject", "on_delete": "CASCADE"}
], comment="Biological samples collected from subjects")

# Measurement references Sample
create_table("Measurement", columns=[
    {"name": "Sample", "type": "text", "nullok": false, "comment": "Reference to the measured sample"},
    {"name": "Value", "type": "float8", "nullok": false, "comment": "Measured value"},
    {"name": "Unit", "type": "text", "comment": "Unit of measurement (e.g., 'mg/L', 'mmHg')"},
    {"name": "Measured_At", "type": "timestamptz", "comment": "Timestamp when measurement was taken"}
], foreign_keys=[
    {"column": "Sample", "referenced_table": "Sample", "on_delete": "CASCADE"}
], comment="Quantitative measurements from samples")
```

### Protocol with Versioning

```
create_table("Protocol", columns=[
    {"name": "Name", "type": "text", "nullok": false, "comment": "Protocol name"},
    {"name": "Version", "type": "text", "nullok": false, "comment": "Version identifier (e.g., '1.0', '2.1')"},
    {"name": "Description", "type": "markdown", "comment": "Detailed protocol steps and instructions"},
    {"name": "Is_Active", "type": "boolean", "comment": "Whether this protocol version is currently in use"}
], comment="Experimental protocols with version tracking")

## Documentation Best Practices

**Always provide descriptions** for tables and columns. Good documentation:
- Makes catalogs self-explanatory without needing external documentation
- Helps users understand data provenance and meaning
- Appears in the Chaise UI as tooltips and help text

For every table:
- Provide a `comment` explaining what the table stores and its role

For every column:
- Provide a `comment` explaining what the column contains
- Include units of measurement where applicable (e.g., "Age in years")
- Describe valid values or ranges when relevant
- Note relationships to other data

After creating tables, you can also:
- Use `set_table_display_name` for user-friendly table names in the UI
- Use `set_column_display_name` for user-friendly column names
- Use `set_row_name_pattern` to control how rows appear in dropdowns

## Tips

- **Naming conventions**: Use singular nouns (Subject, not Subjects)
- **RID column**: Every table automatically gets an RID (unique identifier)
- **Required fields**: Set `nullok: false` for required columns
- **Always document**: Add comments to every table and column
- **Foreign keys**: The referenced table must exist before creating the foreign key
- **Navbar update**: Tables are automatically added to the navigation bar

## Next Steps

After creating tables:
1. **Add data**: Use `insert_records` to add rows
2. **Create vocabularies**: Use `create_vocabulary` for controlled terms
3. **Add features**: Use `create_feature` to add ML labels/annotations
4. **Create asset tables**: Use `create_asset_table` for file storage
"""

    @mcp.prompt(
        name="create-dataset",
        description="Step-by-step guide to create and populate a dataset for ML workflows",
    )
    def create_dataset_prompt() -> str:
        """Guide for creating and managing datasets."""
        return """# Creating and Managing Datasets

Datasets are versioned, reproducible collections of data for ML workflows.
Follow these steps to create and populate datasets.

## Prerequisites
- Connected to a DerivaML catalog
- Data already exists in domain tables (e.g., Images, Subjects)

## Understanding Datasets

Key concepts:
- **Dataset Elements**: Records from domain tables that belong to the dataset
- **Dataset Types**: Labels like "Training", "Testing", "Validation" from a controlled vocabulary
- **Nested Datasets**: Parent datasets can contain child datasets
- **Versioning**: Semantic versioning (major.minor.patch) for reproducibility

## Understanding Dataset Types

Dataset Types are **controlled vocabulary terms** that categorize datasets for ML workflows.
A dataset can have multiple types (e.g., both "Training" and "Labeled").

### Standard Dataset Types

Check available types with the `deriva-ml://catalog/dataset-types` resource. Common types include:

| Type | Purpose |
|------|---------|
| Training | Data used to train models |
| Testing | Held-out data for final evaluation |
| Validation | Data for hyperparameter tuning |
| Complete | Full dataset before splitting |
| Labeled | Dataset includes ground truth labels |
| Unlabeled | Dataset without ground truth (for inference) |

### When to Use Multiple Types

Datasets often need multiple types to fully describe them:
- `["Training", "Labeled"]` - Training data with ground truth
- `["Testing", "Unlabeled"]` - Test data without labels (blind evaluation)
- `["Complete", "Labeled"]` - Full labeled dataset before train/test split

### Creating Custom Dataset Types

If standard types don't fit your workflow, create custom ones:

```
# Check existing types first
# deriva-ml://catalog/dataset-types

# Create a new type term
create_dataset_type_term("Augmented", "Data created through augmentation")
create_dataset_type_term("Curated", "Manually reviewed and cleaned data")
```

### Best Practices for Dataset Types

1. **Be consistent** - Use the same type names across your project
2. **Document custom types** - Always provide clear descriptions
3. **Use "Labeled" vs "Unlabeled"** - Critical for knowing if ground truth is available
4. **Consider workflow stages** - Types should reflect how data is used

## Understanding Dataset Element Types

**Dataset Element Types** define which domain tables can contribute records to datasets.
Before adding records from a table (e.g., Image, Subject), the table must be registered.

```
# Check registered element types
# deriva-ml://catalog/dataset-element-types

# Register a new table as an element type
add_dataset_element_type("Image")
```

This is a one-time setup per table. Once registered, records from that table can be added
to any dataset using `add_dataset_members()`.

## Understanding Nested Datasets

**Nested Datasets** create parent-child relationships, useful for:
- **Train/test splits**: Parent "Complete" dataset with "Training" and "Testing" children
- **Hierarchical organization**: Group related datasets under a common parent
- **Versioned subsets**: Children can have independent version histories

```
# Create hierarchy
add_dataset_child("<parent-rid>", "<child-rid>")

# Query relationships
list_dataset_children("<parent-rid>")
list_dataset_parents("<child-rid>")

# View in dataset details
# deriva-ml://dataset/<rid>  (includes children and parents)
```

Child datasets share provenance with their parent but maintain their own members and versions.

## Step 1: Check for Existing Workflow (Optional)

If you've created datasets before, you may already have a workflow:

```
# Read deriva-ml://catalog/workflows resource
```

If a suitable workflow exists (e.g., "Dataset Curation" with type "Preprocessing"),
you can reuse it. Otherwise, a new one will be created in Step 2.

## Step 2: Create Dataset via Execution

Datasets must be created through an execution for provenance.

**Python API (recommended):**
```python
config = ExecutionConfiguration(
    workflow=Workflow(name="Create Training Set", workflow_type="Preprocessing", description="Curate training data")
)

with ml.create_execution(config) as exe:
    # Create the dataset
    dataset = exe.create_execution_dataset(
        description="Training images for model v2",
        dataset_types=["Training"]
    )
    dataset_rid = dataset.rid
    # ... add members inside context ...

# Upload AFTER exiting context manager
exe.upload_execution_outputs()
```

**MCP Tools:**
```
# Create an execution for dataset creation
create_execution("Create Training Set", "Preprocessing", "Curate training data")

# Create the dataset
create_dataset(
    description="Training images for model v2",
    dataset_types=["Training"]
)
```

Note the returned `dataset_rid` for adding members.

## Step 3: Register Element Types (if needed)

Before adding records, their table must be registered as an element type:

```
# Check which tables are registered - use the resource
# deriva-ml://catalog/dataset-element-types

# Register a new table
add_dataset_element_type("Image")
add_dataset_element_type("Subject")
```

## Step 4: Add Dataset Members

Add records to the dataset:

```
# Add records by their RIDs
add_dataset_members("<dataset-rid>", [
    "<image-rid-1>",
    "<image-rid-2>",
    "<image-rid-3>"
])
```

Adding members automatically increments the minor version.

## Step 5: Upload Outputs (REQUIRED)

**Python API:** The context manager handles timing automatically. Call upload after exiting:
```python
# After exiting the `with` block:
exe.upload_execution_outputs()
```

**MCP Tools:**
```
upload_execution_outputs()
```

## Step 6: Manage Dataset Types

Types can be assigned at creation time or modified later:

```
# View available types - use the resource
# deriva-ml://catalog/dataset-types

# Add a type to existing dataset
add_dataset_type("<dataset-rid>", "Curated")

# Remove a type
remove_dataset_type("<dataset-rid>", "Draft")

# Create a new type term (if needed)
create_dataset_type_term("Augmented", "Data created through augmentation")
```

**Tip**: Assign types at creation when possible:
```
create_dataset(description="...", dataset_types=["Training", "Labeled"])
```

## Step 7: Split Dataset into Train/Test (Optional)

**Use `split_dataset` to create train/test splits automatically.** This is the preferred
method - it handles execution context, provenance, dataset hierarchy, and versioning.

**MCP Tool (recommended):**
```
# Simple random 80/20 split
split_dataset("<source-dataset-rid>", test_size=0.2, seed=42)

# Stratified split preserving class distribution
split_dataset("<source-dataset-rid>",
              test_size=0.2,
              seed=42,
              stratify_by_column="Image_Classification_Image_Class",
              include_tables=["Image", "Image_Classification"])

# Labeled split (both train and test have ground truth)
split_dataset("<source-dataset-rid>",
              test_size=0.2,
              seed=42,
              training_types=["Labeled"],
              testing_types=["Labeled"])

# Preview without modifying catalog
split_dataset("<source-dataset-rid>", test_size=0.2, dry_run=True)
```

**Python API:**
```python
from deriva_ml.dataset.split import split_dataset

result = split_dataset(ml, "<source-dataset-rid>", test_size=0.2, seed=42)
print(f"Training: {result['training']} ({result['train_count']} samples)")
print(f"Testing:  {result['testing']} ({result['test_count']} samples)")
```

`split_dataset` creates a three-level dataset hierarchy:
```
Split (parent, type: "Split")
├── Training (child, type: "Training")
└── Testing (child, type: "Testing")
```

The `stratify_by_column` uses denormalized column naming: `{TableName}_{ColumnName}`.
Use `denormalize_dataset` to discover available column names before stratifying.

**IMPORTANT - Code Provenance:**
If using a custom script instead of `split_dataset`, commit the script BEFORE running it
to ensure valid code provenance. The `split_dataset` tool handles provenance automatically.

## Step 8: Version Management

```
# View version history - use the resource
# deriva-ml://dataset/<dataset-rid>/versions

# Manually increment version
increment_dataset_version("<dataset-rid>", "major", "Schema change")

# Query specific version
list_dataset_members("<dataset-rid>", version="1.0.0")
```

## Complete Example: Create Dataset and Split

**Python API (recommended):**
```python
from deriva_ml.dataset.split import split_dataset

config = ExecutionConfiguration(
    workflow=Workflow(name="Dataset Curation", workflow_type="Preprocessing", description="Create ML datasets")
)

with ml.create_execution(config) as exe:
    # Ensure element types registered
    ml.add_dataset_element_type("Image")

    # Create complete dataset
    complete_ds = exe.create_execution_dataset("Complete Image Set", ["Complete"])

    # Add all images
    ml.add_dataset_members(complete_ds.rid, ["2-D01", "2-D02", "2-D03", ..., "2-D100"])

# Upload AFTER exiting context manager
exe.upload_execution_outputs()

# Split into train/test (creates its own execution for provenance)
result = split_dataset(ml, complete_ds.rid, test_size=0.2, seed=42,
                       training_types=["Labeled"], testing_types=["Labeled"])
print(f"Training: {result['training']} ({result['train_count']} samples)")
print(f"Testing:  {result['testing']} ({result['test_count']} samples)")
```

**MCP Tools:**
```
# 1. Create execution
create_execution("Dataset Curation", "Preprocessing", "Create ML datasets")

# 2. Ensure element types registered
add_dataset_element_type("Image")

# 3. Create complete dataset
create_dataset(description="Complete Image Set", dataset_types=["Complete"])
# Returns dataset_rid: "1-ABC"

# 4. Add all images
add_dataset_members("1-ABC", ["2-D01", "2-D02", "2-D03", ..., "2-D100"])

# 5. Upload outputs
upload_execution_outputs()

# 6. Split into train/test (uses split_dataset tool)
split_dataset("1-ABC", test_size=0.2, seed=42,
              training_types=["Labeled"], testing_types=["Labeled"])
# Returns: {"split": "1-XYZ", "training": "1-DEF", "testing": "1-GHI", ...}

# 7. Verify
list_dataset_children("1-XYZ")
```

## Tips

- Always create datasets within an execution for provenance
- **Always provide descriptive `description` values** - explains what the dataset contains and its purpose
- Read `deriva-ml://catalog/datasets` resource to find existing datasets
- Use semantic versioning: patch=metadata, minor=elements, major=breaking
- **Use `split_dataset` for train/test splits** - handles provenance, hierarchy, and versioning automatically
- Use `stratify_by_column` to preserve class distribution across splits
- Use `dry_run=True` to preview split sizes before modifying the catalog
- Use `restructure_assets` after downloading to organize files for ML frameworks (e.g., PyTorch ImageFolder)
- Pin versions for reproducible training
- Use `download_dataset` to get local copies for ML training

## Provenance Tracking

**Find executions that used a dataset:**
```
list_dataset_executions("<dataset-rid>")
```
Returns all executions that used this dataset as input.

**Find executions that created/used an asset:**
```
list_asset_executions("<asset-rid>")
list_asset_executions("<asset-rid>", "Output")  # Only creating execution
list_asset_executions("<asset-rid>", "Input")   # Only using executions
```
Returns executions with their role (Input/Output).
"""

    # =========================================================================
    # Dataset Versioning Prompt
    # =========================================================================

    @mcp.prompt(
        name="dataset-versioning",
        description="CRITICAL: Dataset version management rules - explicit versions for experiments, increment after catalog changes",
    )
    def dataset_versioning_prompt() -> str:
        """Comprehensive guide for dataset versioning best practices."""
        return """# Dataset Versioning Best Practices

**CRITICAL**: Dataset versioning is essential for reproducible ML experiments.
Follow these rules carefully to ensure experiments are reproducible and results
can be traced back to exact data snapshots.

---

## Core Versioning Rules

### Rule 1: Always Use Explicit Versions for Real Experiments

**NEVER use "current" or "latest" version for production experiments.**

```python
# CORRECT: Explicit version for reproducibility
DatasetSpecConfig(rid="1-ABC", version="1.2.0")

# WRONG: Implicit version - results may not be reproducible
DatasetSpecConfig(rid="1-ABC")  # Gets "current" version - AVOID for real runs
```

**When to use explicit versions:**
- Running actual training/inference experiments
- Creating results you want to reference later
- Publishing or sharing experiment results
- Any run that should be reproducible

**When "current" version is acceptable (development only):**
- Quick debugging or testing configuration
- Exploring data before committing to a version
- Dry runs (`dry_run=true`)

### Rule 2: Increment Version After Catalog Changes

**Dataset versions are snapshots of catalog state at version creation time.**

If you modify the catalog (add features, fix labels, add images), those changes
are NOT visible in existing dataset versions. You MUST increment the version
to capture changes.

```
Workflow:
1. User adds quality labels to images in catalog
2. Existing dataset version 1.0.0 does NOT include new labels
3. Call increment_dataset_version() to create version 1.1.0
4. Version 1.1.0 now includes the new labels
5. Update config files to use version 1.1.0
```

**MCP Tool:**
```
increment_dataset_version(
    "<dataset-rid>",
    "Added quality labels to 500 images for QA workflow",  # REQUIRED description
    "minor"  # or "major" for breaking changes, "patch" for fixes
)
```

### Rule 3: Always Provide Version Descriptions

When incrementing versions, the description is **REQUIRED** and should explain:
- What changed in this version
- Why the change was made
- Impact on users of this dataset

**Good descriptions:**
```
"Added 500 new training images from batch 3 collection"
"Fixed incorrect class labels on 23 images identified in QA review"
"Added Image_Quality feature with Good/Fair/Poor classifications"
"Schema change: added 'acquisition_date' column to Image table"
"Removed 15 duplicate images found during deduplication"
```

**Bad descriptions:**
```
"Updated"  # Too vague
"v2"       # Not descriptive
""         # Empty - NEVER do this
```

---

## Version Management Workflow

### After Creating a Dataset

```
1. Create dataset → returns RID and initial version (e.g., "1.0.0")
2. Add to config file with EXPLICIT version:

   datasets_store(
       [DatasetSpecConfig(rid="2-XYZ", version="1.0.0", materialize=True)],
       name="my_training_data",
   )
```

### After Modifying Catalog Data

```
1. Make catalog changes (add features, fix labels, add images)
2. Increment dataset version with description:

   increment_dataset_version(
       "2-XYZ",
       "Added quality labels and fixed 12 mislabeled images",
       "minor"
   )

3. Update config file to new version:

   # OLD
   DatasetSpecConfig(rid="2-XYZ", version="1.0.0")

   # NEW
   DatasetSpecConfig(rid="2-XYZ", version="1.1.0")

4. Commit config changes before running experiments
```

### Updating Config Files to Current Version

If you need to update a config file to use the current dataset version:

```
1. Look up current version using the resource:
   deriva-ml://dataset/<rid>  → returns current_version

2. Update config:
   DatasetSpecConfig(rid="<rid>", version="<current_version>")

3. Commit the change before running experiments
```

**Prompt for LLM:**
"Update my dataset configs to use current versions" should:
1. Read the dataset config file
2. For each DatasetSpecConfig, read `deriva-ml://dataset/<rid>` to get current_version
3. Update the version parameter to match
4. Show the changes to the user for approval
5. Remind user to commit before running experiments

---

## Version Components (Semantic Versioning)

| Component | When to Use | Example Change |
|-----------|-------------|----------------|
| **major** | Breaking changes, schema modifications | Added new required column |
| **minor** | New data, features, or non-breaking changes | Added 500 images, new labels |
| **patch** | Bug fixes, label corrections | Fixed 10 mislabeled images |

---

## Checking Version History

```
# Get full version history with descriptions - use the resource
# deriva-ml://dataset/<dataset-rid>/versions

# Returns:
[
  {"version": "1.2.0", "created": "2024-01-15", "description": "Added quality labels"},
  {"version": "1.1.0", "created": "2024-01-10", "description": "Fixed mislabeled images"},
  {"version": "1.0.0", "created": "2024-01-01", "description": "Initial dataset"}
]
```

---

## Common Mistakes to Avoid

### Mistake 1: Running experiments without explicit version
```python
# WRONG - version will vary, results not reproducible
datasets = [DatasetSpecConfig(rid="1-ABC")]
```

### Mistake 2: Expecting changes to appear in old versions
```
# Catalog change made
add_feature_value(...)

# WRONG - version 1.0.0 doesn't include new feature values
# Must increment version first
```

### Mistake 3: Empty or vague version descriptions
```python
# WRONG
increment_dataset_version("1-ABC", "")  # No description
increment_dataset_version("1-ABC", "update")  # Too vague
```

### Mistake 4: Not updating config after version increment
```
# Incremented version to 1.1.0
# But config still says version="1.0.0"
# Experiment will use old data!
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Check current version | `deriva-ml://dataset/<rid>` resource |
| View version history | `deriva-ml://dataset/<rid>/versions` resource |
| Increment version | `increment_dataset_version("<rid>", "<description>", "<component>")` |
| Query specific version | `list_dataset_members("<rid>", version="X.Y.Z")` |
| Download specific version | `download_dataset("<rid>", version="X.Y.Z")` |

---

## Summary Checklist

Before running an experiment:
- [ ] Dataset version is explicitly specified (not "current")
- [ ] Config file has been updated if version was incremented
- [ ] Code changes are committed (for provenance tracking)

After modifying catalog data:
- [ ] Called `increment_dataset_version()` with meaningful description
- [ ] Updated config files to new version
- [ ] Committed config changes
"""

    @mcp.prompt(
        name="configure-experiment",
        description="Comprehensive guide to configure ML experiments, multiruns, and sweeps with hydra-zen and DerivaML",
    )
    def configure_experiment_prompt() -> str:
        """Guide for configuring ML experiments, presets, and sweeps with hydra-zen."""
        return """# Configuring ML Experiments with Hydra-Zen and DerivaML

**Use this prompt for**: Setting up config files (datasets.py, experiments.py, etc.)
for use with the `deriva-ml-run` CLI.

**Related prompts:**
- `run-experiment`: Pre-flight checklist and CLI commands (after configs are set up)
- `run-ml-execution`: For custom Python scripts or interactive MCP tool usage

---

This guide explains how to set up reproducible ML experiments, experiment presets,
and multirun sweeps using hydra-zen configuration management with DerivaML.

## Overview

The configuration system uses **hydra-zen** to define composable configurations
organized into groups:

| Config Group | Purpose | Example |
|--------------|---------|---------|
| `deriva_ml` | Catalog connection settings | hostname, catalog_id |
| `datasets` | Which datasets to use | RIDs and versions |
| `assets` | Additional input files | Pre-trained weights |
| `workflow` | Workflow metadata | Name, type, description |
| `model_config` | Model hyperparameters | epochs, learning_rate |
| `experiments` | Preset combinations | Quick test, full training |

## Project Structure

```
src/
├── configs/                    # Configuration modules
│   ├── __init__.py            # Auto-loads all config modules
│   ├── deriva.py              # Catalog connection configs
│   ├── datasets.py            # Dataset specifications
│   ├── assets.py              # Asset (file) references
│   ├── workflow.py            # Workflow definitions
│   ├── my_model.py            # Model-specific configs
│   └── experiments.py         # Experiment presets
├── models/
│   └── my_model.py            # Model implementation
└── deriva_run.py              # Main entry point
```

## Step 1: Configure Deriva Connection

Create `configs/deriva.py` to define catalog connections:

```python
from hydra_zen import store
from deriva_ml import DerivaMLConfig

deriva_store = store(group="deriva_ml")

# Local development catalog
deriva_store(
    DerivaMLConfig,
    name="local",
    hostname="localhost",
    catalog_id=4,
)

# Production catalog
deriva_store(
    DerivaMLConfig,
    name="production",
    hostname="www.my-project.org",
    catalog_id="my-catalog",
)
```

## Step 2: Configure Datasets

Create `configs/datasets.py` to specify input data. **Always include descriptions**
to help users and AI assistants understand what each dataset is for.

```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

datasets_store = store(group="datasets")

# Training dataset - pinned version for reproducibility
# Use with_description() to add metadata about the dataset
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="1-ABC", version="1.0.0")],
        "Training dataset with 5,000 labeled images. "
        "Use for model training. Source: CIFAR-10 training partition."
    ),
    name="training",
)

# Full dataset - all available data
datasets_store(
    with_description(
        [
            DatasetSpecConfig(rid="1-ABC"),  # Latest version
            DatasetSpecConfig(rid="1-DEF"),  # Additional data
        ],
        "Complete dataset with all 10,000 images. "
        "Combines training and testing partitions."
    ),
    name="full",
)

# Default for quick testing
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="1-ABC", version="1.0.0")],
        "Default dataset for quick testing."
    ),
    name="default_dataset",
)
```

**Getting descriptions from the catalog:**
Use the `deriva-ml://dataset/<rid>` resource to see the dataset's description and types.
This is a good starting point for your configuration description.

## Step 3: Configure Assets (Optional)

Create `configs/assets.py` for pre-trained weights or other files.
**Always include descriptions** explaining what each asset contains and when to use it.

```python
from hydra_zen import store
from deriva_ml.execution import with_description

asset_store = store(group="assets")

# Pre-trained weights - describe source and architecture
asset_store(
    with_description(
        ["2-XYZ"],
        "Model weights (model.pt) from extended training: 50 epochs, "
        "64→128 channels, dropout 0.25. Use for inference or fine-tuning."
    ),
    name="pretrained_weights",
)

# Classification probabilities from evaluation
asset_store(
    with_description(
        ["2-ABC"],
        "Probability outputs (probabilities.csv) from test evaluation. "
        "Contains per-class confidence scores for 5,000 test images."
    ),
    name="evaluation_results",
)

# No assets needed (default)
asset_store(
    with_description([], "No assets - empty default configuration"),
    name="default_asset",
)
```

**Getting descriptions from the catalog:**
Read the `deriva-ml://asset/<rid>` resource to see the asset's filename, description,
types, and which execution created it. This helps write accurate config descriptions.

## Step 4: Configure Workflow

Create `configs/workflow.py` to define workflow metadata:

```python
from hydra_zen import store, builds
from deriva_ml.execution import Workflow

workflow_store = store(group="workflow")

# Define your workflow
MyWorkflow = builds(
    Workflow,
    name="My Model Training",
    workflow_type="Training",  # Must exist in Workflow_Type vocabulary
    description=\"\"\"
    Train my model on the dataset.

    ## Architecture
    - Description of model architecture

    ## Expected Outputs
    - model.pt: Trained model weights
    - metrics.json: Training metrics
    \"\"\".strip(),
    populate_full_signature=True,
)

workflow_store(MyWorkflow, name="default_workflow")
workflow_store(MyWorkflow, name="training")
```

## Step 5: Configure Model Parameters

Create `configs/my_model.py` for model hyperparameters. Use `zen_meta` to add
descriptions to each model configuration explaining its purpose and parameters.

```python
from hydra_zen import builds, store
from models.my_model import train_model

model_store = store(group="model_config")

# Base configuration using builds()
MyModelConfig = builds(
    train_model,
    # Architecture
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
    # Training
    learning_rate=1e-3,
    epochs=10,
    batch_size=32,
    # Hydra-zen settings
    populate_full_signature=True,
    zen_partial=True,  # Execution context added at runtime
)

# Register variations with descriptions using zen_meta
model_store(
    MyModelConfig,
    name="default_model",
    zen_meta={
        "description": (
            "Default config: 128 hidden, 2 layers, 10 epochs, lr=1e-3. "
            "Balanced for standard training runs."
        )
    },
)

# Quick test - fewer epochs
model_store(
    MyModelConfig,
    name="quick",
    epochs=2,
    batch_size=64,
    zen_meta={
        "description": (
            "Quick test: 2 epochs, batch 64. Use for rapid iteration "
            "and verifying the training pipeline works."
        )
    },
)

# Extended training
model_store(
    MyModelConfig,
    name="extended",
    epochs=50,
    hidden_size=256,
    dropout=0.2,
    zen_meta={
        "description": (
            "Extended training: 50 epochs, 256 hidden, dropout 0.2. "
            "Use for production training when accuracy is the priority."
        )
    },
)
```

**Note:** `zen_meta` attaches metadata to the config without affecting instantiation.
The description is accessible via `config._zen_meta["description"]` if needed.

## Step 6: Define Experiments

Create `configs/experiments.py` to combine configurations:

```python
from hydra_zen import make_config, store

# Get the base application config
app_config = store[None]
app_name = next(iter(app_config))
base_config = store[None][app_name]

experiment_store = store(group="experiments")

# Quick test experiment
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "training"},
            {"override /model_config": "quick"},
        ],
        bases=(base_config,)
    ),
    name="quick_test",
)

# Full training experiment
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "full"},
            {"override /model_config": "extended"},
            {"override /assets": "pretrained_weights"},
        ],
        bases=(base_config,)
    ),
    name="full_training",
)
```

## Step 7: Implement the Model

Create `models/my_model.py`:

```python
from deriva_ml import DerivaML
from deriva_ml.execution import Execution

def train_model(
    # Model parameters (from config)
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    # Runtime context (provided by runner)
    ml_instance: DerivaML,
    execution: Execution,
) -> None:
    \"\"\"Train the model within DerivaML execution context.\"\"\"

    # 1. Load data from execution's downloaded datasets
    data = execution.get_training_data()

    # 2. Train model
    model = MyModel(hidden_size, num_layers, dropout)
    train(model, data, epochs, learning_rate, batch_size)

    # 3. Save outputs using execution's asset_file_path
    model_path = execution.asset_file_path("Model", "model.pt")
    torch.save(model.state_dict(), model_path)

    metrics_path = execution.asset_file_path("Execution_Metadata", "metrics.json")
    save_metrics(metrics_path)
```

## Running Experiments

### Basic Usage

```bash
# Run with all defaults
python deriva_run.py

# Run a specific experiment
python deriva_run.py +experiment=quick_test

# Override parameters
python deriva_run.py model_config.epochs=20 model_config.learning_rate=1e-4

# Use different dataset
python deriva_run.py datasets=full

# Dry run (test config without execution)
python deriva_run.py dry_run=True
```

### Parameter Sweeps

```bash
# Sweep over learning rates
python deriva_run.py -m model_config.learning_rate=1e-2,1e-3,1e-4

# Sweep over multiple parameters
python deriva_run.py -m model_config.epochs=10,20,50 model_config.dropout=0.0,0.2
```

### Multiple Experiments

```bash
# Run multiple experiments
python deriva_run.py -m +experiment=quick_test,full_training
```

## Configuration Precedence

Configurations are composed in order (later overrides earlier):
1. Defaults from `hydra_defaults` list
2. Experiment overrides
3. Command-line overrides

## Configuration Descriptions

**Why descriptions matter:**
- Help users understand what each configuration does
- Enable AI assistants to discover and select appropriate configs
- Serve as inline documentation in the repository

**Two mechanisms for descriptions:**

| Config Type | Mechanism | Example |
|-------------|-----------|---------|
| Lists (assets, datasets) | `with_description()` | `with_description(["RID"], "desc")` |
| Model configs (builds) | `zen_meta` | `zen_meta={"description": "desc"}` |

**Getting descriptions from the catalog:**
When adding a new configuration, use MCP resources to find good starting descriptions:

```
# For datasets - includes description and types
# deriva-ml://dataset/<dataset-rid>

# For assets - includes filename, description, types, source execution
# deriva-ml://asset/<asset-rid>

# Then use these details in your configuration
```

If a dataset or asset lacks a good description in the catalog, consider adding one:
```
# Update dataset description
update_record("deriva-ml", "Dataset", {"RID": "<rid>", "Description": "..."})
```

---

# Part 2: Experiments and Multiruns

## Experiments vs Multiruns

| Feature | Experiments | Multiruns |
|---------|-------------|-----------|
| Purpose | Single preset configuration | Multiple runs with variations |
| Invocation | `+experiment=name` | `+multirun=name` |
| Output | One execution | Multiple executions (parent + children) |
| Use case | Reproducible single runs | Hyperparameter sweeps, comparisons |

**Decision Guide:**
- Use **experiments** when you want a named, reproducible configuration preset
- Use **multiruns** when you want to sweep parameters or compare configurations
- **Best practice**: Build multiruns on top of existing experiments

## Step 1: Define Your Base Configuration

First, ensure you have a base configuration that your experiments inherit from.
This is typically defined in `configs/base.py`:

```python
from hydra_zen import builds, store
from model_runner import run_model

# Build the base application config
DerivaModelConfig = builds(
    run_model,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "default_deriva"},
        {"datasets": "default_dataset"},
        {"assets": "default_asset"},
        {"workflow": "default_workflow"},
        {"model_config": "default_model"},
    ],
)

# Store as the main app config
store(DerivaModelConfig, name="deriva_model_app")
```

## Step 2: Create Experiment Presets

Create `configs/experiments.py` to define named experiment configurations:

```python
from hydra_zen import make_config, store
from configs.base import DerivaModelConfig

# Use _global_ package for root-level overrides
experiment_store = store(group="experiment", package="_global_")

# Quick validation experiment
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "quick"},
            {"override /datasets": "small_split"},
        ],
        description="Quick training: 3 epochs for fast validation",
        bases=(DerivaModelConfig,),
    ),
    name="my_quick",
)

# Extended training experiment
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "extended"},
            {"override /datasets": "full_split"},
        ],
        description="Extended training: 50 epochs with regularization",
        bases=(DerivaModelConfig,),
    ),
    name="my_extended",
)

# Test-only experiment (no training, just evaluation)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "test_only"},
            {"override /datasets": "testing"},
            {"override /assets": "pretrained_weights"},
        ],
        description="Evaluation only: load weights and evaluate",
        bases=(DerivaModelConfig,),
    ),
    name="my_test_only",
)
```

**Key patterns:**
- `package="_global_"` allows experiments to override root-level configs
- `{"override /group": "name"}` selects which config to use from each group
- `bases=(BaseConfig,)` inherits all settings from the base config
- `description` documents the experiment purpose

## Step 3: Create Multirun Configurations

Create `configs/multiruns.py` for sweep configurations:

```python
from deriva_ml.execution import multirun_config

# Compare two experiments side by side
multirun_config(
    "quick_vs_extended",
    overrides=[
        "+experiment=my_quick,my_extended",
    ],
    description=\"\"\"## Quick vs Extended Comparison

**Objective:** Compare fast vs full training.
\"\"\",
)

# Learning rate sweep (build on existing experiment)
multirun_config(
    "lr_sweep",
    overrides=[
        "+experiment=my_quick",                    # Start from experiment
        "model_config.epochs=10",                  # Override epochs
        "model_config.learning_rate=0.0001,0.001,0.01,0.1",
    ],
    description=\"\"\"## Learning Rate Sweep

**Objective:** Find optimal learning rate.

| Learning Rate | Expected Behavior |
|---------------|-------------------|
| 0.0001 | Slow convergence, stable |
| 0.001 | Balanced (default) |
| 0.01 | Fast convergence, may overshoot |
| 0.1 | Likely unstable |
\"\"\",
)

# Grid search (multiple parameters)
multirun_config(
    "lr_batch_grid",
    overrides=[
        "+experiment=my_quick",
        "model_config.learning_rate=0.001,0.01",
        "model_config.batch_size=64,128",
    ],
    description=\"\"\"## LR x Batch Size Grid Search

**Total runs:** 4 (2 x 2)
\"\"\",
)
```

**Key patterns:**
- `multirun_config()` from `deriva_ml.execution`
- Build on experiments: `"+experiment=name"` in overrides
- Comma-separated values create multiple runs
- Multiple comma-separated params create grid search
- Markdown description becomes parent execution documentation

## Step 4: Running Experiments and Multiruns

### Single Experiments

```bash
# Run with defaults
uv run deriva-ml-run

# Run a specific experiment
uv run deriva-ml-run +experiment=my_quick

# Override experiment parameters
uv run deriva-ml-run +experiment=my_quick model_config.epochs=5
```

### Multiruns

```bash
# Run a named multirun
uv run deriva-ml-run +multirun=lr_sweep

# Override multirun parameters
uv run deriva-ml-run +multirun=lr_sweep model_config.epochs=20

# Show available configs
uv run deriva-ml-run --info
```

### Ad-hoc Sweeps

```bash
# Quick sweep without defining multirun config
uv run deriva-ml-run --multirun +experiment=my_quick \\
    model_config.learning_rate=0.001,0.01
```

## Common Patterns

### Pattern 1: Experiment Hierarchy

Define a base experiment and extend it:

```python
# Base quick experiment
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "quick"},
            {"override /datasets": "small_split"},
        ],
        bases=(DerivaModelConfig,),
    ),
    name="quick",
)

# Quick on full dataset (override just the dataset)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "quick"},
            {"override /datasets": "full_split"},  # Different dataset
        ],
        bases=(DerivaModelConfig,),
    ),
    name="quick_full",
)
```

### Pattern 2: Multirun Comparison

Compare experiments with same parameters:

```python
multirun_config(
    "architecture_comparison",
    overrides=[
        "+experiment=small_model,medium_model,large_model",
        "model_config.epochs=20",  # Same epochs for fair comparison
    ],
    description="Compare model architectures with same training budget",
)
```

### Pattern 3: Sweep on Experiment

Build parameter sweep on an existing experiment:

```python
multirun_config(
    "regularization_sweep",
    overrides=[
        "+experiment=my_extended",
        "model_config.dropout_rate=0.0,0.1,0.25,0.5",
        "model_config.weight_decay=0.0,1e-4,1e-3",
    ],
    description="Grid search for regularization hyperparameters",
)
```

---

# Best Practices

1. **Pin dataset versions** for reproducibility
2. **Use meaningful names** for configurations (e.g., `cifar10_quick` not `exp1`)
3. **Document workflows** with markdown descriptions
4. **Create experiment presets** for common configurations
5. **Build multiruns on experiments**: Reuse experiment settings
6. **Use `dry_run=True`** to test config before long runs
7. **Register all configs** with hydra-zen store for discoverability
8. **Always add descriptions** to configurations using `with_description()` or `zen_meta`
9. **Use markdown in descriptions**: Tables and formatting render in Chaise
10. **Use --info to discover**: `uv run deriva-ml-run --info` shows all options

# Troubleshooting

## "Config not found" Error
- Ensure config module is imported in `configs/__init__.py`
- Check spelling of config name
- Use `--info` to see available configs

## Experiment Not Overriding Correctly
- Verify `package="_global_"` in experiment_store
- Check `hydra_defaults` uses `"override /group"` syntax
- Ensure `bases=(BaseConfig,)` is set

## Multirun Creates Wrong Number of Runs
- Count comma-separated values: `a,b,c` = 3 runs
- Grid search multiplies: `a,b` x `c,d` = 4 runs
- Check no spaces after commas in overrides

# Tips

- Use the workflow-types resource to see available workflow types
- Read `deriva-ml://catalog/datasets` resource to find dataset RIDs
- Read `deriva-ml://dataset/{rid}` and `deriva-ml://asset/{rid}` resources for descriptions
- Group related configs (e.g., all CIFAR-10 configs in one file)
- Use `zen_partial=True` for model configs that need runtime context
- Store experiments in `experiments.py` for easy discovery
"""

    @mcp.prompt(
        name="add-config-descriptions",
        description="Hydra-zen specific: How to add descriptions to Python config files using with_description() and zen_meta",
    )
    def add_config_descriptions_prompt() -> str:
        """Guide for adding descriptions to configuration files."""
        return """# Adding Descriptions to Hydra-zen Configurations

**Focus**: This prompt covers the **Python syntax** for adding descriptions to
hydra-zen configuration files. For guidance on **what to write** in descriptions
and how to generate them from context, see the `generate-descriptions` prompt.

Configuration descriptions help users and AI assistants understand and discover
the right configurations for their tasks.

## Quick Reference

**Simple rule: How is the config stored?**

| If storing... | Use this pattern |
|---------------|------------------|
| A **list** (assets, datasets) | `with_description(items, "description")` |
| A **builds()** config (models) | `zen_meta={"description": "..."}` |

```python
# LISTS → with_description()
from deriva_ml.execution import with_description

asset_store(with_description(["RID1", "RID2"], "Description here"), name="my_assets")
dataset_store(with_description([DatasetSpecConfig(...)], "Description"), name="my_dataset")

# BUILDS → zen_meta
model_store(MyModelConfig, name="my_model", zen_meta={"description": "Description here"})
```

## Why Add Descriptions?

1. **Discoverability**: AI assistants can understand and select appropriate configs
2. **Documentation**: Inline docs in the repository for human users
3. **Context**: Explain what each config is for and when to use it

## Two Mechanisms for Descriptions

| Config Type | Mechanism | Use When |
|-------------|-----------|----------|
| List configs (assets, datasets) | `with_description()` | Config is a list of RIDs |
| Built configs (models, workflows) | `zen_meta` | Config uses `builds()` |

## Step 1: Gather Information from the Catalog

Before writing descriptions, get details from the catalog:

### For Datasets
```
# Use the resource: deriva-ml://dataset/<dataset-rid>
```
Returns: description, dataset_types, version, children/parents

**Good dataset description includes:**
- Size (number of images/records)
- Labeling status (labeled, unlabeled, ground truth available)
- Purpose (training, testing, validation)
- Any special characteristics

### For Assets
```
# Read deriva-ml://asset/<asset-rid> resource
```
Returns: filename, description, asset_types, execution that created it

**Good asset description includes:**
- Filename and what it contains
- Source execution (which training run produced it)
- Key parameters (epochs, architecture, etc.)
- When to use it (inference, fine-tuning, analysis)

## Step 2: Add Descriptions to Datasets

Use `with_description()` from `deriva_ml.execution`:

```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

datasets_store = store(group="datasets")

# Basic pattern
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="28D4", version="0.22.0")],
        "Split dataset with 10,000 images (5,000 train + 5,000 test). "
        "Testing images are unlabeled. Use for standard train/test workflows."
    ),
    name="cifar10_split",
)

# Multiple datasets in one config
datasets_store(
    with_description(
        [
            DatasetSpecConfig(rid="28DC", version="0.22.0"),
            DatasetSpecConfig(rid="28DP", version="0.22.0"),
        ],
        "Combined training and testing datasets for full evaluation. "
        "Training: 5,000 labeled, Testing: 5,000 unlabeled."
    ),
    name="cifar10_combined",
)
```

## Step 3: Add Descriptions to Assets

Use `with_description()` for asset lists:

```python
from hydra_zen import store
from deriva_ml.execution import with_description

asset_store = store(group="assets")

# Model weights
asset_store(
    with_description(
        ["3WMG"],
        "Model weights (cifar10_cnn_weights.pt) from cifar10_quick: "
        "3 epochs, 32→64 channels, batch 128. Source: execution 3WKE."
    ),
    name="quick_weights",
)

# Probability outputs
asset_store(
    with_description(
        ["3WNA"],
        "Classification probabilities (probabilities.csv) from cifar10_quick "
        "on small testing dataset. Contains per-class confidence scores."
    ),
    name="quick_probabilities",
)

# Multiple related assets
asset_store(
    with_description(
        ["3WMG", "3WNA", "3WN8"],
        "Complete outputs from cifar10_quick experiment: "
        "weights, probabilities, and training log."
    ),
    name="quick_complete",
)

# Empty default
asset_store(
    with_description([], "No assets - empty default configuration"),
    name="default_asset",
)
```

## Step 4: Add Descriptions to Model Configs

Use `zen_meta` parameter when storing `builds()` configs:

```python
from hydra_zen import builds, store
from models.my_model import train_model

model_store = store(group="model_config")

MyModelConfig = builds(
    train_model,
    hidden_size=128,
    epochs=10,
    learning_rate=1e-3,
    populate_full_signature=True,
    zen_partial=True,
)

# Add description via zen_meta
model_store(
    MyModelConfig,
    name="default_model",
    zen_meta={
        "description": (
            "Default config: 128 hidden, 10 epochs, lr=1e-3. "
            "Balanced for standard training runs."
        )
    },
)

# Variant with description
model_store(
    MyModelConfig,
    name="quick",
    epochs=3,
    batch_size=128,
    zen_meta={
        "description": (
            "Quick training: 3 epochs, batch 128. "
            "Use for rapid iteration and debugging."
        )
    },
)
```

## Step 5: Update Catalog Descriptions (Optional)

If an asset or dataset lacks a good description in the catalog, add one:

```
# Update dataset description
update_record(
    "deriva-ml",
    "Dataset",
    {"RID": "<dataset-rid>", "Description": "Clear description of the dataset..."}
)

# Update asset description (requires knowing the asset table)
update_record(
    "<schema>",
    "<AssetTable>",
    {"RID": "<asset-rid>", "Description": "Clear description of the asset..."}
)
```

## Description Writing Guidelines

**Good descriptions include:**
- What it is (dataset size, asset contents)
- Where it came from (source execution, creation process)
- When to use it (training, testing, inference, debugging)
- Key parameters (epochs, learning rate, architecture)

**Examples:**

❌ Bad: `"Training dataset"`
✅ Good: `"Training dataset with 5,000 labeled CIFAR-10 images (32x32 RGB). All images have ground truth classifications."`

❌ Bad: `"Model weights"`
✅ Good: `"Model weights (model.pt) from extended training: 50 epochs, 64→128 channels, dropout 0.25. Use for inference or fine-tuning."`

❌ Bad: `"Quick config"`
✅ Good: `"Quick training: 3 epochs, batch 128. Use for rapid iteration and verifying the training pipeline works."`

## Workflow: Adding Descriptions to Existing Configs

1. **Inventory**: List all configs that need descriptions
2. **Research**: Read `deriva-ml://dataset/{rid}` and `deriva-ml://asset/{rid}` resources
3. **Write**: Create clear, informative descriptions
4. **Import**: Add `from deriva_ml.execution import with_description`
5. **Update**: Wrap list configs with `with_description()`
6. **Model configs**: Add `zen_meta={"description": "..."}` to store calls
7. **Test**: Run `--info` to verify configs still load
8. **Commit**: Track description changes in version control
"""

    @mcp.prompt(
        name="derivaml-coding-guidelines",
        description="Recommended workflow and coding standards for DerivaML projects",
    )
    def derivaml_coding_guidelines_prompt() -> str:
        """Coding guidelines for DerivaML projects."""
        return """# DerivaML Coding Guidelines and Best Practices

These guidelines ensure DerivaML projects are robust, reproducible, and well-tracked.
Follow these standards when building ML workflows with DerivaML.

## Project Configuration

### Repository Setup
- Each model should live in its own repository following the DerivaML template
- Use **uv** to manage all dependencies
- The `uv.lock` file **MUST** be committed to enable reproducible environments

### Environment Management
```bash
# Install dependencies
uv sync

# Rebuild environment from lock file
uv sync --frozen
```

## Git Workflow

### Branch Strategy
- **SHOULD** work in Git branches and create pull requests, even for solo projects
- Rebase your branch regularly to stay current with main

### Commit Requirements
- **MUST** commit all code changes before running ML workflows
- This maximizes DerivaML's ability to track the exact code used to produce results
- No change is too small to properly track in GitHub and DerivaML

### Example Workflow
```bash
# 1. Create feature branch
git checkout -b feature/model-improvement

# 2. Make changes and commit
git add .
git commit -m "Update learning rate scheduler"

# 3. Bump version before running
uv run bump-version patch

# 4. Run your ML workflow (code is now trackable)
uv run python deriva_run.py

# 5. Create PR when ready
git push -u origin feature/model-improvement
```

## Coding Standards

### Documentation
- **SHOULD** use Google docstring format for all code
- Reference: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

```python
def train_model(data: Dataset, epochs: int) -> Model:
    \"\"\"Train a model on the provided dataset.

    Args:
        data: The training dataset containing images and labels.
        epochs: Number of training epochs.

    Returns:
        The trained model with optimized weights.

    Raises:
        ValueError: If epochs is less than 1.
    \"\"\"
```

### Type Hints
- **SHOULD** use type hints wherever possible

```python
def process_batch(
    images: list[np.ndarray],
    labels: list[int],
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

## Versioning and Releases

### Semantic Versioning
DerivaML uses semantic versioning (major.minor.patch):
- **Major**: Breaking changes to model architecture or data format
- **Minor**: New features or significant improvements
- **Patch**: Bug fixes and small tweaks

### Version Management
```bash
# Create version tag before running
uv run bump-version major|minor|patch

# Check current version
uv run python -m setuptools_scm
```

### Version Workflow
1. Make code changes
2. Commit changes
3. Bump version (`uv run bump-version patch`)
4. Push tags (`git push --tags`)
5. Run ML workflow

## Notebook Guidelines

### Output Management
- **MUST NOT** commit notebooks with output cells
- Install and enable `nbstripout` to automatically strip outputs

```bash
# Install nbstripout
uv add nbstripout --dev
uv run nbstripout --install
```

### Notebook Focus
- Notebooks **SHOULD** focus on a single task (analysis, visualization)
- Prefer Python scripts for model training workflows

### Execution Requirements
- **MUST** ensure notebooks can run start-to-finish without intervention
- Test your notebook completely before uploading to DerivaML

### Running Notebooks with DerivaML
```bash
uv run deriva-ml-run-notebook notebooks/analysis.ipynb \\
    --host <HOST> \\
    --catalog <CATALOG_ID> \\
    --kernel <repository-name>
```

This uploads the executed notebook to the catalog with full provenance.

## Executions and Experiments

### Hydra-Zen Configuration
- **MUST** always run code from hydra-zen configuration files
- **SHOULD** commit code before running

### Debugging with Dry Run
During development, use `dry_run` to test without creating records:

```python
# In your config or command line
python deriva_run.py dry_run=True

# Or programmatically
execution = ml.create_execution(config, dry_run=True)
```

Dry run behavior:
- Downloads input datasets
- Does NOT create Execution records
- Does NOT upload results

### Production Workflow
1. Debug with `dry_run=True`
2. Remove `dry_run` when ready
3. Bump version (`uv run bump-version patch`)
4. Run full execution

## Data Management

### Data Storage
- **SHOULD NOT** commit data files to Git
- Store all data in DerivaML catalogs instead

```python
# Good: Reference data by RID
datasets = [DatasetSpecConfig(rid="1-ABC", version="1.0.0")]

# Bad: Don't check in data files
# data/training_images/  # Never commit this
```

### Dataset Versioning
Pin dataset versions for reproducibility:

```python
# Always specify version for production runs
DatasetSpecConfig(rid="1-ABC", version="1.0.0")

# Latest version OK for development
DatasetSpecConfig(rid="1-ABC")  # Gets latest
```

## Extensibility

### Domain-Specific Extensions
DerivaML is designed for extension via inheritance:

```python
from deriva_ml import DerivaML

class MedicalImagingML(DerivaML):
    \"\"\"Domain-specific ML class for medical imaging.\"\"\"

    def load_dicom_dataset(self, dataset_rid: str) -> DicomDataset:
        \"\"\"Load DICOM images from a dataset.\"\"\"
        ...

    def compute_image_metrics(self, images: list[Image]) -> dict:
        \"\"\"Compute domain-specific image quality metrics.\"\"\"
        ...
```

Instantiate your domain class in scripts and notebooks for specialized functionality.

## Summary Checklist

Before running an ML workflow:
- [ ] Code changes committed to Git
- [ ] Version bumped (`uv run bump-version`)
- [ ] `uv.lock` committed
- [ ] Notebooks stripped of outputs
- [ ] Configuration files use hydra-zen
- [ ] Dataset versions pinned for production
- [ ] No data files in repository
- [ ] Docstrings and type hints present

## Quick Reference

| Task | Command |
|------|---------|
| Install deps | `uv sync` |
| Bump version | `uv run bump-version patch` |
| Check version | `uv run python -m setuptools_scm` |
| Run notebook | `uv run deriva-ml-run-notebook <notebook> --host <HOST> --catalog <ID>` |
| Test config | `python deriva_run.py dry_run=True` |
| Run experiment | `python deriva_run.py +experiment=<name>` |
"""

    @mcp.prompt(
        name="run-experiment",
        description="Pre-flight checklist and CLI commands for deriva-ml-run - assumes configs are already set up",
    )
    def run_experiment_prompt() -> str:
        """Checklist for running an ML experiment with full provenance."""
        return """# Running an ML Experiment (Pre-Flight Checklist)

**Use this prompt for**: Running experiments via the `deriva-ml-run` CLI when
config files are already set up.

**Related prompts:**
- `configure-experiment`: How to set up config files (datasets.py, experiments.py, etc.)
- `run-ml-execution`: For custom Python scripts or interactive MCP tool usage

---

## Pre-Flight Checklist (REQUIRED)

### Step 1.1: Check Git Status

Run this command and report the results to the user:

```bash
git status --porcelain
```

**If output is EMPTY (clean):**
✅ Working tree is clean - proceed to Step 1.2

**If output shows changes:**
⚠️ **STOP** - There are uncommitted changes!

Report to the user which files are modified/untracked.
Ask: "There are uncommitted changes. Should I commit these before proceeding?"

If user confirms:
```bash
git add .
git commit -m "<descriptive message>"
```

### Step 1.2: Check Current Version

```bash
git describe --tags --always 2>/dev/null || echo "No tags found"
```

**If version ends with `-g<hash>` (e.g., `v1.2.3-5-gabcd123`):**
⚠️ There are commits since the last version tag.

Ask the user: "The current version is `<version>`. Should I bump the version?"

If user confirms:
```bash
uv run bump-my-version bump patch   # For bug fixes
uv run bump-my-version bump minor   # For new features
uv run bump-my-version bump major   # For breaking changes

git push && git push --tags
```

**If version is a clean tag (e.g., `v1.2.3`):**
✅ Version is current - proceed

### Step 1.3: Verify Lock File

```bash
uv lock
git status --porcelain uv.lock
```

If `uv.lock` changed:
```bash
git add uv.lock
git commit -m "Update uv.lock"
```

### Step 1.4: User Confirmation

**You MUST ask the user explicitly:**

> I have verified the repository state:
> - **Commit**: `<commit hash and message>`
> - **Version**: `<version tag>`
> - **Status**: Clean working tree
>
> **Is it OK to proceed with the ML execution?**

**Wait for explicit user confirmation** before proceeding.

---

## Verify Configuration

Before running, ensure configs are set up. Use `uv run deriva-ml-run --info` to see available configs.

**If configs need to be created or modified**, see the `configure-experiment` prompt for:
- Repository structure
- Dataset configuration (`configs/datasets.py`)
- Asset configuration (`configs/assets.py`)
- Model parameters (`configs/<model>.py`)
- Experiment presets (`configs/experiments.py`)

---

## Run the Experiment

The `deriva-ml-run` CLI is provided by deriva-ml. It automatically loads configs from
`src/configs/` and supports both `--host`/`--catalog` arguments and Hydra config defaults.

### Test First (Dry Run)
```bash
uv run deriva-ml-run --info                           # Show available configs
uv run deriva-ml-run dry_run=True                     # Test without records
uv run deriva-ml-run +experiment=quick_test dry_run=True
```

### Production Run
```bash
# Run with defaults (uses host/catalog from Hydra config)
uv run deriva-ml-run

# Run specific experiment
uv run deriva-ml-run +experiment=full_training

# Override parameters
uv run deriva-ml-run \\
    datasets=training \\
    model_config=full \\
    model_config.epochs=100

# Override host/catalog from command line
uv run deriva-ml-run --host localhost --catalog 45 +experiment=cifar10_quick

# Run parameter sweep (multirun mode)
uv run deriva-ml-run --multirun \\
    model_config.learning_rate=1e-2,1e-3,1e-4

# Run multiple experiments
uv run deriva-ml-run --multirun +experiment=cifar10_quick,cifar10_extended
```

**Common CLI options:**
| Option | Purpose |
|--------|---------|
| `--host <host>` | Override catalog hostname |
| `--catalog <id>` | Override catalog ID |
| `--info` | Show available config groups |
| `--multirun`, `-m` | Enable parameter sweep mode |
| `--config-dir` | Custom configs location |

**Common Hydra overrides:**
| Override | Purpose |
|----------|---------|
| `+experiment=<name>` | Use experiment preset |
| `datasets=<name>` | Select dataset config |
| `model_config=<name>` | Select model config |
| `assets=<name>` | Select asset config |
| `dry_run=True` | Test without records |

---

## Verify Results

After the run completes:

```
# Check recent executions - use the resource
# deriva-ml://catalog/executions

# View in Chaise - use the resource
# deriva-ml://chaise-url/Execution
```

**Checklist:**
- [ ] Execution record created in catalog
- [ ] Output assets uploaded
- [ ] Provenance links correct (datasets, code version)

---

## Quick Reference

| Check | Command | Expected |
|-------|---------|----------|
| Clean tree | `git status --porcelain` | Empty output |
| Version | `git describe --tags` | Clean tag (no `-g<hash>`) |
| Available configs | `uv run deriva-ml-run --info` | Lists all config groups |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Dataset not found | Check `deriva-ml://dataset/<rid>` resource |
| Asset not found | Check `deriva-ml://asset/<rid>` resource |
| Config not found | Run `--info`, check `configs/__init__.py` |
| Execution failed | Try `dry_run=True`, run `uv sync` |

**For config file issues**, see the `configure-experiment` prompt.
"""

    @mcp.prompt(
        name="setup-notebook-environment",
        description="Step-by-step guide to set up environment for running DerivaML notebooks",
    )
    def setup_notebook_environment_prompt() -> str:
        """Guide for setting up notebook development environment."""
        return """# Setting Up a DerivaML Notebook Environment

Follow these steps to set up your environment for developing and running
Jupyter notebooks with DerivaML tracking.

## Prerequisites
- Repository created from deriva-ml-model-template
- `uv` installed (https://docs.astral.sh/uv/)

## Step 1: Initialize the Virtual Environment

Create the Python environment and install dependencies:

```bash
# From repository root
uv sync
```

This creates a virtual environment and generates `uv.lock`. Commit the lock file
for reproducible environments.

## Step 2: Install Jupyter Dependencies

Install the Jupyter-specific dependency group:

```bash
uv sync --group=jupyter
```

This installs:
- jupyter / jupyterlab
- ipykernel
- nbstripout
- papermill
- nbconvert

## Step 3: Install nbstripout

Configure nbstripout to automatically strip output cells from notebooks on commit:

```bash
uv run nbstripout --install
```

**Why this matters:**
- Prevents large output cells from bloating Git history
- Ensures notebooks can be compared meaningfully in diffs
- Required by DerivaML coding guidelines
- Notebooks MUST NOT be committed with output cells

**Verify installation:**
```bash
# Check Git hooks are installed
cat .git/hooks/pre-commit | grep nbstripout
```

## Step 4: Install the Jupyter Kernel

Register a Jupyter kernel for your virtual environment:

```bash
uv run deriva-ml-install-kernel
```

This creates a kernel named after your project (from `pyvenv.cfg` prompt).
The kernel will appear in Jupyter's kernel selector.

**Verify kernel installation:**
```bash
uv run jupyter kernelspec list
```

You should see your project kernel listed (e.g., `deriva-model-template`).

**MCP Tool alternative:**
```
install_jupyter_kernel()
list_jupyter_kernels()
```

## Step 5: Authenticate to Deriva

Before accessing catalog data, authenticate with Globus:

```bash
uv run deriva-globus-auth-utils login --host <HOSTNAME>
```

Replace `<HOSTNAME>` with your Deriva server (e.g., `www.eye-ai.org`).

## Step 6: Verify Setup

Test your environment:

```bash
# Start Jupyter Lab
uv run jupyter lab

# Or start classic notebook
uv run jupyter notebook
```

In Jupyter:
1. Create a new notebook
2. Select your project kernel from the kernel dropdown
3. Test DerivaML import:
   ```python
   from deriva_ml import DerivaML
   print("DerivaML ready!")
   ```

## Optional: Install ML Framework Dependencies

Install additional dependency groups as needed:

```bash
# PyTorch
uv sync --group=pytorch

# TensorFlow
uv sync --group=tensorflow

# All groups
uv sync --all-groups
```

To make these permanent, add them to `default-groups` in `pyproject.toml`:
```toml
[tool.uv]
default-groups = ["jupyter", "pytorch"]
```

## Complete Setup Checklist

- [ ] `uv sync` completed successfully
- [ ] `uv sync --group=jupyter` completed
- [ ] `nbstripout --install` configured Git hooks
- [ ] `deriva-ml-install-kernel` registered kernel
- [ ] Globus authentication completed
- [ ] Jupyter launches and shows project kernel
- [ ] DerivaML imports successfully in notebook

## Troubleshooting

### Kernel not showing in Jupyter
```bash
# List installed kernels
uv run jupyter kernelspec list

# Reinstall kernel
uv run deriva-ml-install-kernel
```

### nbstripout not stripping outputs
```bash
# Check if installed
uv run nbstripout --status

# Reinstall
uv run nbstripout --install
```

### Authentication errors
```bash
# Re-authenticate
uv run deriva-globus-auth-utils login --host <HOSTNAME>

# Check credentials
ls ~/.deriva/
```

### Missing dependencies
```bash
# Update lock file
uv lock --upgrade

# Reinstall
uv sync --group=jupyter
```
"""

    @mcp.prompt(
        name="run-notebook",
        description="Step-by-step guide to develop and run a DerivaML notebook with tracking",
    )
    def run_notebook_prompt() -> str:
        """Guide for developing and running notebooks with DerivaML."""
        return """# Developing and Running DerivaML Notebooks

Follow these steps to develop a notebook and run it with full DerivaML
execution tracking and provenance.

## Prerequisites
- Environment set up (see `setup-notebook-environment` prompt)
- nbstripout and kernel installed
- Authenticated to Deriva catalog

## Step 0: Verify Environment Setup

Before developing notebooks, ensure your environment is ready:

```bash
# Verify nbstripout is installed
uv run nbstripout --status

# Verify kernel is installed
uv run jupyter kernelspec list

# Should show your project kernel
```

**MCP Tools:**
```
list_jupyter_kernels()
```

If not set up, follow the `setup-notebook-environment` prompt first.

## Step 1: Create Notebook from Template

Start with the notebook template from deriva-ml-model-template:

```
notebooks/
└── notebook_template.ipynb
```

Copy and rename the template for your analysis:
```bash
cp notebooks/notebook_template.ipynb notebooks/my_analysis.ipynb
```

### Required Notebook Structure

Your notebook MUST include:

1. **Imports cell**: DerivaML and hydra-zen imports
```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration, Execution
from hydra_zen import builds, store, launch, zen
```

2. **Parameters cell**: Tagged with "parameters" metadata for papermill
```python
# Parameters cell - values can be overridden at runtime
dry_run: bool = False
host: str = "www.example.org"
catalog: str = "1"
learning_rate: float = 0.001
epochs: int = 10
```

3. **Configuration loading**: Load hydra-zen configs
```python
import configs.datasets
import configs.deriva
import configs.assets
```

4. **Execution context**: Create and use DerivaML execution
```python
ml_instance = DerivaML(hostname=host, catalog_id=catalog)

config = ExecutionConfiguration(
    workflow=ml_instance.create_workflow('my-workflow', 'Analysis'),
    datasets=config.datasets,
)

with ml_instance.create_execution(config, dry_run=dry_run) as exe:
    # Your analysis code here...
    pass

# Upload AFTER context manager
exe.upload_execution_outputs()
```

5. **Save execution info**: For the notebook runner
```python
import os
import json

rid_path = os.environ.get("DERIVA_ML_SAVE_EXECUTION_RID")
if rid_path:
    with open(rid_path, "w") as f:
        json.dump({
            "execution_rid": exe.rid,
            "hostname": host,
            "catalog_id": catalog,
            "workflow_rid": exe.workflow_rid,
        }, f)
```

## Step 2: Develop Your Notebook

Follow these guidelines while developing:

### Coding Guidelines
- **Single task focus**: Each notebook should do one thing well
- **Parameterize values**: Use the parameters cell for anything that might change
- **Clear outputs before commit**: Or rely on nbstripout
- **Run end-to-end**: Ensure notebook runs completely after Kernel > Restart & Run All
- **Use execution context**: Register all outputs with `exe.asset_file_path()`

### Testing Your Notebook
1. Clear all outputs: Kernel > Restart & Clear Output
2. Run all cells: Kernel > Restart & Run All
3. Verify no errors
4. Verify outputs are generated correctly

### Using dry_run for Development
Set `dry_run=True` in the parameters cell during development:
- Downloads input data normally
- Does NOT create Execution records
- Does NOT upload results
- Safe for iterative testing

## Step 3: Commit and Version

Once your notebook runs successfully end-to-end:

```bash
# Check status (outputs should be stripped)
git status

# Stage and commit
git add notebooks/my_analysis.ipynb
git commit -m "Add my analysis notebook"

# Bump version
uv run bump-version patch

# Push
git push && git push --tags
```

**CRITICAL**: Always commit before running with DerivaML tracking.
This ensures proper code provenance.

**MCP Tools:**
```
bump_version("patch")
get_current_version()
```

## Step 4: Run with DerivaML Tracking

Run your notebook with full execution tracking:

### Using Hydra Config Defaults (Recommended)
If your notebook config specifies host/catalog in Hydra configs, you don't need
to specify them on the command line:

```bash
# Use host/catalog from Hydra config defaults
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb

# With Hydra config overrides
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \\
    assets=my_assets \\
    datasets=my_dataset

# Show available Hydra configs
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --info
```

### With Explicit Host/Catalog
Override host/catalog from command line:

```bash
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \\
    --host <HOSTNAME> \\
    --catalog <CATALOG_ID> \\
    --kernel <your-project-kernel>
```

### With Parameter Overrides
```bash
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \\
    --host www.example.org \\
    --catalog 1 \\
    -p learning_rate 0.01 \\
    -p epochs 50 \\
    --kernel my-project
```

### From Parameter File
```bash
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \\
    --file parameters.yaml \\
    --kernel my-project
```

### Inspect Available Options
```bash
# Show notebook parameters
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --inspect

# Show Hydra config options
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --info
```

### MCP Tools
```
# Inspect available parameters
inspect_notebook("notebooks/my_analysis.ipynb")

# Run with tracking
run_notebook(
    "notebooks/my_analysis.ipynb",
    hostname="www.example.org",
    catalog_id="1",
    parameters={"learning_rate": 0.01, "epochs": 50},
    kernel="my-project"
)
```

## What Happens During Execution

1. **Environment setup**: Sets provenance environment variables
   - `DERIVA_ML_WORKFLOW_URL`: GitHub URL to notebook
   - `DERIVA_ML_WORKFLOW_CHECKSUM`: MD5 of notebook file
   - `DERIVA_ML_NOTEBOOK_PATH`: Local path

2. **Parameter injection**: Papermill injects parameters into notebook

3. **Execution**: Notebook runs with DerivaML tracking

4. **Output conversion**: Executed notebook converted to Markdown

5. **Upload**: Both `.ipynb` and `.md` uploaded as execution assets

6. **Citation**: Execution RID and citation info printed

## Complete Workflow Checklist

**Before developing:**
- [ ] Environment set up (kernel, nbstripout)
- [ ] Template notebook copied

**During development:**
- [ ] Parameters cell defined
- [ ] Execution context used
- [ ] Save execution info included
- [ ] Runs end-to-end without errors
- [ ] Tested with `dry_run=True`

**Before running:**
- [ ] All changes committed
- [ ] Version bumped
- [ ] Changes pushed

**Running:**
- [ ] Correct host and catalog specified
- [ ] Correct kernel specified
- [ ] Parameters set as needed

**After running:**
- [ ] Execution record created in catalog
- [ ] Notebook and Markdown uploaded
- [ ] Results viewable in Chaise

## Troubleshooting

### "Notebook did not save execution metadata"
- Ensure notebook has the execution info save block
- Check `DERIVA_ML_SAVE_EXECUTION_RID` environment variable handling

### "Kernel not found"
- Verify kernel name: `uv run jupyter kernelspec list`
- Reinstall: `uv run deriva-ml-install-kernel`

### "nbstripout warning"
- Install nbstripout: `uv run nbstripout --install`
- Clear outputs before commit: Kernel > Restart & Clear Output

### "Authentication failed"
- Re-authenticate: `uv run deriva-globus-auth-utils login --host <HOST>`
"""

    @mcp.prompt(
        name="query-catalog-data",
        description="Step-by-step guide for querying and exploring data in a Deriva catalog",
    )
    def query_catalog_data_prompt() -> str:
        """Guide for querying data from Deriva catalogs."""
        return """# Querying Data from Deriva Catalogs

This guide covers how to explore and query data in a DerivaML catalog,
from simple lookups to complex joins and aggregations.

## Prerequisites
- Connected to a DerivaML catalog (`connect_catalog`)

## Step 1: Understand the Schema

Before querying, explore what data is available:

```
# List all tables in the domain schema - use the resource
# deriva-ml://catalog/tables

# Get details about a specific table - use the resource
# deriva-ml://table/Image/schema

# See the full schema structure
get_schema_description()
```

## Step 2: Simple Queries

### Query All Records
```
query_table("Image")
```
Returns first 100 records with all columns.

### Select Specific Columns
```
query_table("Image", columns=["RID", "Filename", "URL"])
```

### Limit Results
```
query_table("Image", limit=10)
```

### Paginate Large Results
```
# First page
query_table("Image", limit=100, offset=0)

# Second page
query_table("Image", limit=100, offset=100)
```

## Step 3: Filter Queries

### Equality Filter
```
query_table("Image", filters={"Format": "PNG"})
```

### Multiple Filters (AND)
```
query_table("Image", filters={"Format": "PNG", "Width": 1024})
```

### Count Records
```
count_table("Image")
count_table("Image", filters={"Format": "PNG"})
```

## Step 4: Get Specific Records

### By RID
```
get_record("Image", "1-ABC")
```

### Resolve Unknown RID
```
resolve_rid("1-ABC")
```
Returns the table and schema for any RID.

## Step 5: Query Related Data

### Using Denormalization
Join related tables for ML-ready data:

```
# First, see what's in a dataset
list_dataset_members("<dataset-rid>")

# Then denormalize to join tables
denormalize_dataset(
    "<dataset-rid>",
    include_tables=["Image", "Subject", "Diagnosis"],
    limit=1000
)
```

Returns flat data with prefixed column names:
- `Image.RID`, `Image.Filename`
- `Subject.Name`, `Subject.Age`
- `Diagnosis.Label`

### Query Table Data
```
query_table("Image", limit=500)
```

## Step 6: Vocabulary Lookups

### List Vocabulary Terms
```
# Read vocabulary resources
deriva-ml://vocabulary/Dataset_Type
deriva-ml://vocabulary/Workflow_Type
```

### Find a Term
```
# Use the resource: deriva-ml://vocabulary/Dataset_Type/Training
# Also works with synonyms: deriva-ml://vocabulary/Dataset_Type/train
```

## Step 7: Feature Queries

### Find Features for a Table
```
# Read deriva-ml://table/Image/features resource
```

### Get Feature Structure
```
# Read deriva-ml://feature/Image/Diagnosis resource
```

### Get All Feature Values
```
# Read deriva-ml://feature/Image/Diagnosis/values resource
```
Returns target RIDs, values, and which execution created them.

## Common Query Patterns

### Find All Images for a Subject
```
# If you know the Subject RID
query_table("Image", filters={"Subject": "<subject-rid>"})
```

### Find Records by Date Range
Use the catalog's REST API for complex date queries:
```
# For records created after a date
query_table("Image", filters={"RCT": "2024-01-01"})  # Exact match only
```

For range queries, use the Python API or web interface.

### Export Data for ML
```
# Get dataset members
members = list_dataset_members("<dataset-rid>")

# Denormalize for training
data = denormalize_dataset(
    "<dataset-rid>",
    include_tables=["Image", "Label"],
    limit=10000
)

# Or download the full dataset
download_dataset("<dataset-rid>", materialize=True)
```

## Historical Queries (Snapshots)

Query data as it existed at a specific version:

```
# Get version history - use the resource
# deriva-ml://dataset/<dataset-rid>/versions

# Query specific version
list_dataset_members("<dataset-rid>", version="1.0.0")
denormalize_dataset("<dataset-rid>", ["Image"], version="1.0.0")
```

## View in Web Interface

Get URLs to browse data in Chaise using resources:

```
# URL for a table - use the resource
# deriva-ml://chaise-url/Image

# URL for a specific record - use the resource
# deriva-ml://chaise-url/1-ABC
```

## Complete Example: Explore a Dataset

```
# 1. Connect (if not already connected)
connect_catalog("example.org", "1")

# 2. Read deriva-ml://catalog/datasets resource to list available datasets

# 3. Get dataset details - use the resource
# deriva-ml://dataset/1-ABC

# 4. See what tables have data - use the resource
# deriva-ml://dataset/1-ABC/members
# Returns: {"Image": [...], "Subject": [...]}

# 5. Check table structure - use the resource
# deriva-ml://table/Image/schema

# 6. Query image data
query_table("Image", columns=["RID", "Filename", "Subject"], limit=10)

# 7. Get denormalized data for ML
denormalize_dataset("1-ABC", ["Image", "Subject", "Diagnosis"], limit=100)

# 8. View in browser - use the resource
# deriva-ml://chaise-url/1-ABC
```

## Tips

1. **Start small**: Use `limit=10` while exploring
2. **Check schema first**: Understand column names and types
3. **Use RIDs**: They're globally unique identifiers
4. **Pin versions**: For reproducible ML, always specify dataset version
5. **Use denormalize**: It handles joins automatically for ML workflows

## Troubleshooting

### "Table not found"
```
# Use the resource: deriva-ml://catalog/tables
```

### "Column not found"
```
# Use the resource: deriva-ml://table/<table>/schema
```

### "No results returned"
- Check filter column names match exactly (case-sensitive)
- Try without filters first to verify data exists
- Check you're connected to the correct catalog

### "Too many results"
- Add filters to narrow down
- Use pagination with `limit` and `offset`
- Consider using `count_table()` first
"""

    @mcp.prompt(
        name="work-with-assets",
        description="Step-by-step guide to working with assets (files) - lookup, provenance, and management",
    )
    def work_with_assets_prompt() -> str:
        """Guide for working with assets in DerivaML."""
        return """# Working with Assets in DerivaML

Assets are file-based records (images, models, data files) with automatic
URL/checksum tracking and provenance. Follow this guide to work with assets.

## Prerequisites
- Connected to a DerivaML catalog
- Understanding of asset tables vs regular tables

## Understanding Assets

**Asset Tables** automatically include:
- `URL`: File location (uploaded to object store)
- `Filename`: Original filename
- `Length`: File size in bytes
- `MD5`: Checksum for integrity
- `Description`: Human-readable description

**Asset Types** are vocabulary terms that categorize assets (e.g., "Model_File",
"Training_Data", "Segmentation_Mask").

## Step 1: Discover Asset Tables

Find what asset tables exist in the catalog:

```
# Read deriva-ml://catalog/asset-tables resource
```

Returns: `[{"name": "Image", "schema": "domain"}, {"name": "Model", "schema": "domain"}, ...]`

## Step 2: Find Assets

### List All Assets in a Table

```
# Use the resource: deriva-ml://table/Image/assets
# Or: deriva-ml://table/Model/assets
```

### Search with Filters

```
# Read deriva-ml://catalog/assets resource for all assets in catalog
```

### Look Up Specific Asset

```
# Read deriva-ml://asset/<asset-rid> resource
```

Returns detailed info: RID, table, filename, URL, types, execution that created it.

## Step 3: Check Asset Provenance

Find which execution created an asset:

```
list_asset_executions("<asset-rid>", asset_role="Output")
```

Find which executions used an asset as input:

```
list_asset_executions("<asset-rid>", asset_role="Input")
```

## Step 4: Manage Asset Types

### View Available Types

```
# Read deriva-ml://vocabulary/Asset_Type resource
```

### Add New Type

```
add_asset_type("Segmentation_Mask", "Binary mask images for segmentation")
```

## Step 5: Create Asset Tables

Create a new asset table for your domain:

```
create_asset_table(
    "Scan",
    columns=[
        {"name": "Resolution", "type": "float4"},
        {"name": "Modality", "type": "text"}
    ],
    referenced_tables=["Subject"],  # Foreign keys
    comment="Medical scan images"
)
```

## Step 6: Upload Assets via Execution

Assets are uploaded through executions for provenance tracking.

**Python API (recommended):**
```python
config = ExecutionConfiguration(
    workflow=Workflow(name="Process Images", workflow_type="Preprocessing", description="Generate masks")
)

with ml.create_execution(config) as exe:
    # Create output file
    mask_path = exe.asset_file_path(
        "Image",                    # Asset table
        "mask_001.png",             # Filename
        asset_types=["Segmentation_Mask"]  # Types
    )
    # Write file to mask_path...

# Upload AFTER exiting context manager
exe.upload_execution_outputs()
```

**MCP Tools:**
```
# 1. Create execution
create_execution("Process Images", "Preprocessing", "Generate masks")

# 2. Register output (copies existing file)
asset_file_path("Image", "/path/to/mask_001.png", ["Segmentation_Mask"])

# 3. Upload to catalog
upload_execution_outputs()
```

## Step 6: Use Assets as Execution Inputs

**Python API:**
```python
config = ExecutionConfiguration(
    workflow=Workflow(name="Train Model", workflow_type="Training", description="Train on images"),
    assets=["<asset-rid-1>", "<asset-rid-2>"]  # Input assets
)

with ml.create_execution(config) as exe:
    # Assets are automatically downloaded
    # Access via exe.asset_paths
    for asset in exe.asset_paths:
        print(f"Using: {asset}")
```

**MCP Tools:**
```
create_execution(
    "Train Model",
    "Training",
    "Train on images",
    asset_rids=["<asset-rid-1>", "<asset-rid-2>"]
)
# Assets downloaded automatically
```

## Complete Example: Image Processing Pipeline

```python
# 1. Create preprocessing execution
config = ExecutionConfiguration(
    workflow=Workflow(name="Preprocess", workflow_type="Preprocessing", description="Resize images"),
    assets=["3-RAW"]  # Input: raw images
)

with ml.create_execution(config) as exe:
    # 2. Process each input
    for input_asset in exe.asset_paths:
        # Load, process, save
        output_path = exe.asset_file_path(
            "Image",
            f"processed_{input_asset.name}",
            asset_types=["Preprocessed"]
        )
        # Write processed image to output_path...

# 3. Upload
exe.upload_execution_outputs()

# 4. Check provenance
uploaded = exe.uploaded_assets
for table, assets in uploaded.items():
    for asset in assets:
        print(f"Created: {asset.asset_rid} in {table}")
        # Can later query: list_asset_executions(asset.asset_rid)
```

## Tips

- Use `deriva-ml://asset/<rid>` resource to get full details about any asset
- Use `list_asset_executions` to trace provenance
- Asset types help organize and filter assets
- Always upload through executions for provenance
- Read `deriva-ml://catalog/assets` resource for bulk discovery operations

## Related Resources/Tools

- `deriva-ml://table/{table}/assets` resource - List assets in a table
- `deriva-ml://catalog/assets` resource - All assets summary
- `deriva-ml://asset/{rid}` resource - Get asset details
- `list_asset_executions()` tool - Asset provenance
- `asset_file_path()` tool - Register output files
- `upload_execution_outputs()` tool - Upload registered files
"""

    @mcp.prompt(
        name="troubleshoot-execution",
        description="Troubleshooting guide for common execution problems",
    )
    def troubleshoot_execution_prompt() -> str:
        """Guide for troubleshooting execution issues."""
        return """# Troubleshooting Execution Problems

Common issues and solutions when running DerivaML executions.

## Problem: "No active execution"

**Symptom**: Error when calling `asset_file_path()` or `upload_execution_outputs()`

**Cause**: Execution was not created or context manager exited

**Solution**:
```python
# Make sure execution is created
with ml.create_execution(config) as exe:
    # Call asset_file_path() INSIDE the with block
    path = exe.asset_file_path("Model", "output.pt")
    # Write file...

# Call upload AFTER the with block
exe.upload_execution_outputs()
```

## Problem: "Files not uploaded"

**Symptom**: Execution completes but no assets appear in catalog

**Causes**:
1. Forgot to call `upload_execution_outputs()`
2. Called upload inside the `with` block (before files were written)
3. Files weren't written to the paths from `asset_file_path()`

**Solution**:
```python
with ml.create_execution(config) as exe:
    path = exe.asset_file_path("Model", "model.pt")

    # MUST write to the exact path returned
    torch.save(model, path)  # Correct
    # torch.save(model, "model.pt")  # WRONG - different path

# MUST call upload AFTER with block
exe.upload_execution_outputs()
```

## Problem: "Dataset not found"

**Symptom**: Error resolving dataset RID

**Causes**:
1. Dataset RID is wrong
2. Dataset was deleted
3. Not connected to correct catalog

**Solution**:
```
# Verify connection
get_catalog_info()

# Read deriva-ml://catalog/datasets resource to list available datasets

# Check if dataset exists - use the resource
# deriva-ml://dataset/<rid>
```

## Problem: "Invalid RID"

**Symptom**: `DerivaMLException: Invalid RID`

**Causes**:
1. RID doesn't exist in catalog
2. RID is from different catalog
3. Typo in RID

**Solution**:
```
# Resolve RID to check what it is
resolve_rid("<rid>")

# Returns table name and schema if valid
```

## Problem: "Permission denied"

**Symptom**: 403 Forbidden error

**Causes**:
1. Not authenticated
2. Insufficient permissions
3. Session expired

**Solution**:
```bash
# Re-authenticate
deriva-globus-auth-utils login --host <hostname>

# Check current credentials
deriva-globus-auth-utils token validate
```

## Problem: "Version mismatch"

**Symptom**: Dataset version doesn't match expected

**Causes**:
1. Dataset was modified after your reference
2. Using wrong version specifier

**Solution**:
```
# Check version history - use the resource
# deriva-ml://dataset/<rid>/versions

# Pin to specific version
config = ExecutionConfiguration(
    datasets=[DatasetSpec(rid="<rid>", version="1.2.3")]
)
```

## Problem: "Feature not found"

**Symptom**: Can't find feature or feature values

**Causes**:
1. Feature name is misspelled
2. Feature is on different table
3. Feature was deleted

**Solution**:
```
# List all feature names - use the resource
# deriva-ml://catalog/features

# Read deriva-ml://table/Image/features resource to find features for Image table

# Get feature details
# Read deriva-ml://feature/Image/Diagnosis resource
```

## Problem: "Upload timeout"

**Symptom**: Upload hangs or times out on large files

**Causes**:
1. Network issues
2. Very large files
3. Server-side limits

**Solution**:
- Check network connectivity
- Upload in smaller batches
- Use `download_dataset` for large data transfers
- Contact server administrator for limits

## Problem: "Execution stuck in Running state"

**Symptom**: Execution status shows "Running" but code finished

**Causes**:
1. Exception occurred before `stop_execution()`
2. Code crashed
3. Didn't use context manager

**Solution**:
```python
# Always use context manager for automatic cleanup
with ml.create_execution(config) as exe:
    try:
        # Your code
        pass
    except Exception as e:
        # Context manager will still call stop_execution
        raise

# Or manually update status
ml._update_status(Status.failed, "Crashed: <error>", "<execution-rid>")
```

## Problem: "Vocabulary term not found"

**Symptom**: Error when using a term that doesn't exist

**Solution**:
```
# Read vocabulary terms from resource
deriva-ml://vocabulary/<vocabulary-name>

# Add missing term
add_term("<vocabulary>", "<term>", "<description>")
```

## Debugging Tips

1. **Enable verbose logging**:
```python
import logging
logging.getLogger("deriva_ml").setLevel(logging.DEBUG)
```

2. **Check execution details**:
```
get_execution_info()
```

3. **Verify catalog connection**:
```
get_catalog_info()
```

4. **List recent executions**:
```
# Read deriva-ml://catalog/executions resource
```

5. **Check working directory**:
```
get_execution_working_dir()
```

## Getting Help

- Check DerivaML documentation: `deriva-ml://docs/overview`
- Review execution logs in the execution metadata
- Contact support with execution RID for tracking
"""

    @mcp.prompt(
        name="api-naming-conventions",
        description="Reference guide for DerivaML API naming conventions and patterns",
    )
    def api_naming_conventions_prompt() -> str:
        """Reference for API naming conventions."""
        return """# DerivaML API Naming Conventions

This reference documents the naming patterns used throughout the DerivaML API
for consistency and discoverability.

## Method Prefixes

### `lookup_*` - Single Item Retrieval

Retrieves one item by identifier. Raises exception if not found.

```python
dataset = ml.lookup_dataset("4HM")        # Returns Dataset
asset = ml.lookup_asset("3JSE")           # Returns Asset
term = ml.lookup_term("Image_Type", "X-ray")  # Returns VocabularyTerm
workflow = ml.lookup_workflow("http://...")   # Returns Workflow RID
feature = ml.lookup_feature("Image", "Diagnosis")  # Returns Feature
```

**Pattern**: `lookup_<entity>(identifier) -> Entity`

### `find_*` - Search/Discovery

Returns iterable of matching items. Empty result is valid (not an error).

```python
datasets = ml.find_datasets()             # Returns Iterable[Dataset]
assets = ml.find_assets()                 # Returns Iterable[Asset]
features = ml.find_features("Image")      # Returns Iterable[Feature]
workflows = ml.list_workflows()           # Returns list[Workflow]
```

**Pattern**: `find_<entities>(filters) -> Iterable[Entity]`

### `list_*` - Enumerate Items

Lists all items of a type, often in a specific context.

```python
# Catalog-level
terms = ml.list_vocabulary_terms("Asset_Type")  # All terms in vocab
tables = ml.list_tables()                       # All domain tables
assets = ml.list_assets("Image")                # All assets in table

# Entity-level
members = dataset.list_dataset_members()        # Members of this dataset
parents = dataset.list_dataset_parents()        # Parent datasets
children = dataset.list_dataset_children()      # Child datasets
executions = asset.list_executions()            # Executions using asset
```

**Pattern**: `list_<items>(context) -> list[Item]`

### `get_*` - Data Retrieval with Transformation

Retrieves data with optional transformation (e.g., to DataFrame).

```python
df = ml.get_table_as_dataframe("Image")        # Returns DataFrame
metadata = asset.get_metadata()                 # Returns dict
url = dataset.get_chaise_url()                  # Returns URL string
info = ml.get_schema_description()              # Returns schema dict
```

**Pattern**: `get_<data>() -> TransformedData`

### `create_*` - Create New Entities

Creates new records in the catalog.

```python
dataset = exe.create_dataset(["Training"])     # New dataset
workflow = ml.create_workflow("Name", "Type")  # New workflow
feature = ml.create_feature("Image", "Label")  # New feature
table = ml.create_table("Subject", [...])      # New table
vocab = ml.create_vocabulary("Status", "...")  # New vocabulary
execution = ml.create_execution(config)        # New execution
```

**Pattern**: `create_<entity>(params) -> Entity`

### `add_*` - Add to Existing

Adds items to existing entities or creates vocabulary terms.

```python
# Add to collections
dataset.add_dataset_members({"Image": ["1-A", "1-B"]})
dataset.add_dataset_type("Training")
asset.add_asset_type("Model_File")
ml.add_dataset_child(parent_rid, child_rid)

# Add vocabulary terms
ml.add_term("Asset_Type", "New_Type", "Description")
term = ml.lookup_term("Asset_Type", "New_Type")
term.synonyms = term.synonyms + ("alias",)  # Add synonym via property
```

**Pattern**: `add_<item>(target, item_to_add)`

### `delete_*` / `remove_*` - Remove Items

Removes items from entities.

```python
# delete_* for batch removal
dataset.delete_dataset_members(["1-A", "1-B"])
ml.delete_feature("Image", "Old_Label")
ml.delete_term("Asset_Type", "Unused_Type")

# remove_* for single items
asset.remove_asset_type("Wrong_Type")
term = ml.lookup_term("Asset_Type", "Term")
term.synonyms = tuple(s for s in term.synonyms if s != "old_alias")  # Remove via property
```

**Pattern**: `delete_<items>(identifiers)` or `remove_<item>(identifier)`

## Parameter Naming

### RID Parameters

Use semantic names that indicate the entity type:

```python
dataset_rid: RID      # Dataset identifier
asset_rid: RID        # Asset identifier
execution_rid: RID    # Execution identifier
workflow_rid: RID     # Workflow identifier
member_rids: list[RID]  # Multiple member identifiers
```

### Table/Column Parameters

```python
table_name: str       # Name of a table (e.g., "Image")
column_name: str      # Name of a column (e.g., "Filename")
feature_name: str     # Name of a feature (e.g., "Diagnosis")
vocab_name: str       # Name of vocabulary (e.g., "Asset_Type")
```

### Boolean Parameters

Use positive names with `bool` type:

```python
include_deleted: bool = False   # Include deleted items
recurse: bool = True            # Recursive operation
materialize: bool = True        # Download files
validate: bool = True           # Validate input
```

## Return Types

### Consistent Patterns

| Method Type | Returns |
|-------------|---------|
| `lookup_*` | Single entity or raises |
| `find_*` | `Iterable[Entity]` |
| `list_*` | `list[Item]` or `dict[str, list]` |
| `get_*` | Specific type (DataFrame, dict, str) |
| `create_*` | Created entity |
| `add_*` | Usually `None` (modifies in place) |
| `delete_*` | Usually `None` |

## Class Conventions

### Entity Classes

- `Dataset` - Catalog-backed dataset operations
- `DatasetBag` - Downloaded/offline dataset
- `Asset` - Catalog-backed asset operations
- `Execution` - ML execution context
- `Feature` - Feature definition

### Protocol Classes (Interfaces)

- `DatasetLike` - Read-only dataset interface
- `WritableDataset` - Writable dataset interface
- `AssetLike` - Read-only asset interface
- `WritableAsset` - Writable asset interface
- `DerivaMLCatalog` - Full catalog interface
- `DerivaMLCatalogReader` - Read-only catalog interface

## MCP Tool Conventions

MCP tools follow the same naming but use underscores and sometimes
abbreviated or clearer parameter names:

| Python API | MCP Resource/Tool |
|------------|----------|
| `lookup_dataset(rid)` | `deriva-ml://dataset/<rid>` resource |
| `find_datasets(deleted=True)` | `deriva-ml://catalog/datasets` resource |
| `list_dataset_members()` | `list_dataset_members(dataset_rid)` tool |

## Tips for API Discovery

1. **Use `find_*` when you don't know if items exist**
2. **Use `lookup_*` when you expect the item to exist**
3. **Use `list_*` for enumeration within a context**
4. **Use `get_*` when you need transformed output**
5. **Check `deriva-ml://catalog/features` resource before creating features**
6. **Check `deriva-ml://vocabulary/{name}` resource before adding terms**
"""

    @mcp.prompt(
        name="maintain-experiment-notes",
        description="Maintain a running summary of experiment work in experiment-notes.md - use PROACTIVELY to track progress",
    )
    def maintain_experiment_notes_prompt() -> str:
        """Guide for maintaining experiment notes throughout a session."""
        return """# Maintaining Experiment Notes

**PROACTIVE USAGE**: Update `experiment-notes.md` throughout the session to maintain
a running summary of what the user is working on. This file serves as persistent
documentation that helps with:
- Continuity between sessions
- Onboarding other team members
- Tracking experiment history and decisions
- Documenting catalog configuration and structure

## File Location

Create/update `experiment-notes.md` in the project root directory.

## When to Update

Update the notes file:
- At the **start** of a session (add session header with date)
- After **significant actions** (creating datasets, running experiments, adding features)
- When **decisions are made** (why a particular approach was chosen)
- When **problems are encountered** and solved
- At the **end** of a session (summarize what was accomplished)

## File Structure

```markdown
# Experiment Notes

## Project Overview
Brief description of the ML project and its goals.

**Catalog**: `<hostname>` / `<catalog_id>`
**Repository**: `<repo_url>`

---

## Catalog Structure

### Vocabularies
- `<VocabularyName>`: <description of terms>

### Asset Tables
- `<TableName>`: <what files it stores>

### Features
- `<TableName>.<FeatureName>`: <what it labels>

### Datasets
| Name | RID | Type | Description |
|------|-----|------|-------------|
| ... | ... | ... | ... |

---

## Session Log

### YYYY-MM-DD: <Brief Title>

**Goal**: What the user wanted to accomplish

**Actions**:
- Created dataset `<name>` (RID: `<rid>`)
- Configured experiment `<name>` with parameters...
- Ran training with results...

**Decisions**:
- Chose X over Y because...

**Issues & Solutions**:
- Problem: ...
- Solution: ...

**Next Steps**:
- [ ] Task 1
- [ ] Task 2

---

### YYYY-MM-DD: <Previous Session>
...
```

## Update Guidelines

### Session Start
```markdown
### YYYY-MM-DD: <Infer title from user's first request>

**Goal**: <What user wants to accomplish today>
```

### After Creating Catalog Entities

When creating datasets, features, vocabularies, or assets, update the
appropriate section:

```markdown
**Actions**:
- Created dataset `training_split_v2` (RID: `4HM`)
  - Type: Training
  - 5,000 images from CIFAR-10
  - Version: 1.0.0
```

Also update the Catalog Structure section if this is a new entity type.

### After Running Experiments

```markdown
**Actions**:
- Ran experiment `cifar10_quick` (Execution RID: `5JK`)
  - Config: epochs=3, lr=0.001, batch=64
  - Dataset: `training_split_v2` (4HM)
  - Result: 72% accuracy on test set
```

### When Decisions Are Made

```markdown
**Decisions**:
- Using labeled split dataset instead of unlabeled because we need
  ground truth for ROC curve analysis
- Chose learning rate 0.001 based on sweep results showing best
  convergence
```

### When Problems Are Solved

```markdown
**Issues & Solutions**:
- Problem: "Dataset not found" error when running experiment
- Solution: Dataset version was not pinned in config; added explicit
  version="1.0.0" to DatasetSpecConfig
```

### Session End

```markdown
**Summary**: Completed initial training runs, identified best learning
rate (0.001), created ROC analysis notebook.

**Next Steps**:
- [ ] Run extended training with best hyperparameters
- [ ] Generate prediction probabilities for full test set
- [ ] Create comparative analysis across model architectures
```

## Example Session Update

After user asks "help me create a training dataset":

1. **Before action**: Add session header if not present
2. **After action**: Update notes

```markdown
### 2024-01-15: Create Training Dataset

**Goal**: Set up training data for CIFAR-10 experiments

**Actions**:
- Created dataset `cifar10_training` (RID: `4HM`)
  - Type: Training, Labeled_Training
  - Contains 45,000 images from CIFAR-10 training partition
  - Features: Image_Class (ground truth labels)
  - Version: 1.0.0

**Catalog Structure Update**:
- Added to Datasets section:
  | cifar10_training | 4HM | Training | 45k labeled training images |
```

## Reading Existing Notes

At session start, read the existing `experiment-notes.md` to understand:
- What catalog is being used
- What datasets/features already exist
- What was done in previous sessions
- Any pending tasks from last session

## Tips

- Keep entries concise but informative
- Include RIDs for easy reference
- Document the "why" not just the "what"
- Update incrementally, don't wait until session end
- Use checkboxes for actionable next steps
- Link to relevant executions in the catalog
"""

    @mcp.prompt(
        name="generate-descriptions",
        description="ALWAYS use: How to generate meaningful descriptions for catalog entities when user doesn't provide one",
    )
    def generate_descriptions_prompt() -> str:
        """Guide for generating descriptions for catalog entities."""
        return """# Generating Descriptions for Catalog Entities

**CRITICAL**: Every catalog entity that accepts a description MUST have one.
If the user doesn't provide a description, YOU MUST generate a meaningful one
based on context from the repository, conversation, and catalog state.

**Related prompt**: For hydra-zen configuration file syntax (`with_description()`,
`zen_meta`), see the `add-config-descriptions` prompt.

## Why Descriptions Matter

1. **Discoverability** - Users and AI assistants can find and understand entities
2. **Reproducibility** - Future users can understand what data/experiments contain
3. **Documentation** - Descriptions serve as inline documentation
4. **Provenance** - Descriptions capture intent and context at creation time

## Markdown Support

Descriptions support **GitHub-flavored Markdown** which renders in the Chaise web UI.
Use markdown to improve readability:

- **Headers** for sections: `## Overview`
- **Bold/italic** for emphasis: `**important**`, `*note*`
- **Lists** for multiple items: `- Item 1\\n- Item 2`
- **Tables** for structured data: `| Col1 | Col2 |`
- **Code** for technical terms: `` `model.pt` ``
- **Links** for references: `[link text](url)`

**Example with markdown**:
```markdown
## Training Dataset

Contains **45,000 labeled images** from CIFAR-10.

### Contents
- 10 object categories
- 32x32 RGB images
- Ground truth `Image_Class` labels

### Usage
Use for training CNN classifiers. Pair with `cifar10_test` for evaluation.
```

## Entities That Require Descriptions

### Datasets
- `create_dataset(description=...)`
- `DatasetSpecConfig` in hydra-zen configs (use `with_description()`)

### Executions/Workflows
- `create_execution(description=...)`
- `Workflow(description=...)`

### Features
- `create_feature(description=...)`

### Vocabulary Terms
- `add_term(description=...)`

### Tables and Columns
- `create_table(description=...)` for table
- Column definitions with `description` field
- `set_table_description()`, `set_column_description()`

### Assets (when uploaded)
- Asset metadata `Description` field

### Configurations (hydra-zen)
- `with_description()` for asset/dataset lists
- `zen_meta={"description": ...}` for model configs

## How to Generate Descriptions

### Step 1: Gather Context

Before creating any entity, gather context from:

1. **User's request** - What did they ask for?
2. **Repository structure** - What does the codebase do?
3. **Existing catalog entities** - What patterns exist?
4. **Configuration files** - What do configs reveal about the project?
5. **Conversation history** - What decisions were made?

### Step 2: Generate Description

Create a description that answers:
- **What** is this entity?
- **Why** was it created?
- **How** should it be used?
- **What** does it contain (for datasets/assets)?

### Step 3: Confirm with User

Present the generated description and ask for confirmation:

> I'll create the dataset with this description:
> "Training partition of CIFAR-10 with 45,000 labeled images. Contains
> ground truth Image_Class labels. Use for model training experiments."
>
> Is this description accurate, or would you like to modify it?

## Description Templates by Entity Type

### Datasets

**Template**:
```
<Purpose> of <source data> with <count> <items>. <Key characteristics>.
<Usage guidance>.
```

**Examples**:
- "Training partition of CIFAR-10 with 45,000 labeled images. Contains ground truth class labels for 10 categories. Use for model training."
- "Test split containing 5,000 images with held-out labels for evaluation. Derived from cifar10_full dataset."
- "Curated subset of fundus images showing diabetic retinopathy grades 2-4. Created for severity classification training."

### Executions/Workflows

Workflow descriptions benefit greatly from markdown formatting:

**Simple format**:
```
<Action> <target> using <method/model>. <Key parameters>. <Expected outputs>.
```

**Rich markdown format** (recommended for complex workflows):
```markdown
## ResNet-50 Training

Train image classifier on CIFAR-10 dataset.

### Parameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 50 |

### Outputs
- `model.pt` - Trained model weights
- `metrics.json` - Training/validation metrics
```

**Examples**:
- "Train ResNet-50 on CIFAR-10 training set. Using learning rate 0.001, batch size 64, 50 epochs. Outputs: model weights, training metrics."
- "Generate prediction probabilities for test images using trained model from execution 4HM. Outputs: probability CSV for ROC analysis."
- "Preprocess raw images: resize to 224x224, normalize to ImageNet stats. Outputs: processed image dataset."

### Features

**Template**:
```
<What it labels> for <target table>. Values from <vocabulary>. <Usage context>.
```

**Examples**:
- "Diagnostic classification for fundus images. Values: Normal, Mild_DR, Moderate_DR, Severe_DR, Proliferative_DR. Used for training diagnostic models."
- "Quality assessment labels for retinal images. Values: Good, Acceptable, Poor, Ungradable. Used to filter training data."
- "Ground truth segmentation mask availability. Values: Has_Mask, No_Mask. Indicates which images have expert annotations."

### Vocabulary Terms

**Template**:
```
<Definition of the term>. <When to use it>. <Relationship to other terms if relevant>.
```

**Examples**:
- "Images suitable for model training with clear features and correct labeling. Use for primary training datasets."
- "Preliminary model weights from early training. Use for transfer learning or continued training, not production inference."
- "Automated prediction without manual verification. May require expert review before clinical use."

### Tables

**Template**:
```
<What records represent>. <Key relationships>. <Primary use case>.
```

**Examples**:
- "Research subjects enrolled in the study. Links to Image table via Subject_ID. Used for demographic analysis and cohort selection."
- "Imaging devices used for data collection. Referenced by Image.Device_ID. Tracks device-specific calibration and quality metrics."

### Columns

**Template**:
```
<What the value represents>. <Format/units if applicable>. <Constraints or valid values>.
```

**Examples**:
- "Patient age at time of imaging in years. Integer, range 0-120."
- "Acquisition timestamp in UTC. ISO 8601 format."
- "Foreign key to Subject table. Required for all clinical images."

### Configuration Descriptions (hydra-zen)

**For asset lists** (`with_description()`):

Use triple quotes for multi-line markdown descriptions:
```python
description = (
    "## Learning Rate Sweep Results\\n\\n"
    "Model weights from hyperparameter sweep experiment.\\n\\n"
    "| Asset | Learning Rate | Notes |\\n"
    "|-------|---------------|-------|\\n"
    "| RID1 | 0.001 | Baseline, stable convergence |\\n"
    "| RID2 | 0.01 | Fast convergence, slight instability |\\n\\n"
    "Use with `roc_analysis` notebook to compare performance."
)
with_description(["RID1", "RID2"], description)
```

**For model configs** (`zen_meta`):
```python
model_store(
    MyConfig,
    name="quick",
    epochs=3,
    zen_meta={
        "description": (
            "**Quick validation config** for testing pipeline.\\n\\n"
            "- Epochs: 3\\n"
            "- Batch size: 64\\n\\n"
            "Use before full training runs to verify setup."
        )
    }
)
```

## Contextual Description Generation

### From Repository Analysis

If creating entities for a new project, analyze:
- `README.md` - Project purpose and goals
- `src/configs/` - Existing configuration patterns
- `src/models/` - Model architecture details
- `notebooks/` - Analysis workflows

**Example**: If README mentions "diabetic retinopathy classification",
generate dataset descriptions like:
"Fundus images for diabetic retinopathy classification. Contains [N] images
with severity grades from the [source] dataset."

### From Catalog State

Query existing entities to match patterns:
```
# Read deriva-ml://catalog/datasets resource to see existing dataset description styles
# Read deriva-ml://vocabulary/Dataset_Type resource to understand categories
# Read deriva-ml://experiment/<rid> resource to see how similar experiments are described
```

### From Conversation Context

Use information from the current session:
- User's stated goals
- Decisions made about data splits
- Parameter choices and rationale
- Problem context (what they're trying to solve)

## Prompting for Missing Descriptions

When calling tools that accept descriptions, ALWAYS include one:

**WRONG**:
```
create_dataset(
    dataset_types=["Training"],
    # No description!
)
```

**CORRECT**:
```
create_dataset(
    dataset_types=["Training"],
    description="Training partition with 45,000 labeled CIFAR-10 images for CNN classification experiments."
)
```

## Quality Checklist

Before finalizing a description, verify:

- [ ] **Specific**: Avoids vague terms like "data" or "stuff"
- [ ] **Informative**: Answers what, why, and how
- [ ] **Accurate**: Matches the actual content/purpose
- [ ] **Concise**: 1-3 sentences, no unnecessary words
- [ ] **Consistent**: Matches style of existing descriptions in catalog
- [ ] **Actionable**: Helps users know when to use this entity

## Example Workflow

User: "Create a dataset from the test images"

Assistant thinks:
1. What test images? Check catalog for image tables and existing datasets
2. What's the project? Check README/configs - it's CIFAR-10 classification
3. Why test dataset? Likely for model evaluation
4. Any special characteristics? Check if labeled or unlabeled split

Assistant responds:
> I'll create a test dataset. Based on the CIFAR-10 project structure and
> existing datasets, I'll use this description:
>
> "Test partition of CIFAR-10 with 10,000 labeled images for model evaluation.
> Contains ground truth class labels for computing accuracy and generating
> ROC curves. Complements the cifar10_training dataset."
>
> Does this accurately describe the dataset you want to create?

User: "Yes, create it"
"""

    @mcp.prompt(
        name="semantic-awareness",
        description="ALWAYS use: Check for existing entities and suggest alternatives before creating new tables, vocabularies, features, or datasets",
    )
    def semantic_awareness_prompt() -> str:
        """Guide for semantic lookup and duplicate prevention."""
        return """# Semantic Awareness for Catalog Operations

**CRITICAL**: Before creating ANY new catalog entity (table, vocabulary term, feature,
dataset, workflow), you MUST check for existing entities that might serve the same
purpose. This prevents duplicate/redundant entities and promotes reuse.

## Core Principle

The semantics of catalog entities are determined by:
- **Tables**: table name, description, column names, and column descriptions
- **Vocabularies**: term name, synonyms, and description
- **Features**: feature name, target table, and associated vocabulary
- **Datasets**: dataset types, description, and member composition
- **Workflows**: workflow name, type, and description

## Semantic Lookup Process

### Before Creating ANY Entity

1. **Gather semantic intent** from the user's request
2. **Search existing entities** using names, descriptions, and synonyms
3. **Present matches** to the user with similarity explanation
4. **Suggest reuse** if appropriate matches exist
5. **Only create new** if no suitable entity exists

---

## Step 1: Semantic Search by Entity Type

### For Tables

Before `create_table()`, check existing tables:

```
# Read schema resource for all tables with descriptions
deriva-ml://catalog/schema

# Get specific table details - use the resources
# deriva-ml://table/<table_name>/schema
get_table_columns("<table_name>")
```

**Search for semantic matches**:
- Table names similar to requested name
- Tables with similar column structures
- Tables with similar descriptions
- Tables in the same domain area

**Example check**:
```
User wants: "Create a table for patient demographics"

Search existing tables for:
- Names containing: patient, subject, person, individual, demographic
- Descriptions mentioning: demographic, personal, patient info
- Columns like: age, gender, birth_date, name

Found: "Subject" table with columns Age, Gender, Enrollment_Date
→ Suggest: "The 'Subject' table already exists with demographic columns.
   Should we add columns to it, or do you need a separate table?"
```

### For Vocabulary Terms

Before `add_term()`, check existing terms AND synonyms:

```
# Read vocabulary with all terms and synonyms
deriva-ml://vocabulary/<vocab_name>

# Check if term exists (searches names AND synonyms) - use the resource
# deriva-ml://vocabulary/<vocab_name>/<term_name>
```

**Search for semantic matches**:
- Exact term name match
- Term synonyms that match the requested name
- Terms with similar descriptions
- Terms in related vocabularies

**Example check**:
```
User wants: add_term("Image_Type", "Xray", "X-ray medical images")

Search existing terms:
- deriva-ml://vocabulary/Image_Type/Xray → Not found
- deriva-ml://vocabulary/Image_Type/X-ray → Found! Has synonym "Xray"
- Check description similarity

Found: "X-ray" term already exists with synonym "Xray"
→ Suggest: "The term 'X-ray' already exists and includes 'Xray' as a synonym.
   Did you mean to use the existing term, or add a different synonym?"
```

### For Features

Before `create_feature()`, check existing features:

```
# List all features for the target table
deriva-ml://table/<table_name>/features

# Get detailed feature info
deriva-ml://feature/<table_name>/<feature_name>

# List all registered feature names
deriva-ml://catalog/features
```

**Search for semantic matches**:
- Features on the same target table
- Features with similar names across tables
- Features using similar vocabularies

**Example check**:
```
User wants: create_feature("Image", "Diagnosis", ...)

Search existing features:
- deriva-ml://table/Image/features → Found "Disease_Classification"
- Check if "Diagnosis" is similar to existing feature names

Found: "Disease_Classification" feature on Image table
→ Suggest: "The Image table already has a 'Disease_Classification' feature.
   Is 'Diagnosis' intended to be different, or should we use the existing feature?"
```

### For Datasets

Before `create_dataset()`, check existing datasets:

```
# List all datasets with types and descriptions
deriva-ml://catalog/datasets

# Read deriva-ml://catalog/datasets resource to search datasets

# Get specific dataset details
deriva-ml://dataset/<rid>
```

**Search for semantic matches**:
- Datasets with similar types
- Datasets with similar descriptions
- Datasets with overlapping members

**Example check**:
```
User wants: Create training dataset from CIFAR-10 images

Search existing datasets:
- Read `deriva-ml://catalog/datasets` resource → Check types containing "Training"
- Check descriptions mentioning "CIFAR" or "training"

Found: "cifar10_training" dataset (RID: 4HM) - "Training partition with 45k images"
→ Suggest: "A CIFAR-10 training dataset already exists (RID: 4HM).
   Should we use it, create a new version, or create a different dataset?"
```

### For Workflows

Before `create_workflow()`, check existing workflows:

```
# List all workflows
deriva-ml://catalog/workflows

# Search by URL or checksum
lookup_workflow_by_url("<url>")
```

**Search for semantic matches**:
- Workflows with similar names
- Workflows with the same type
- Workflows with similar descriptions

---

## Step 2: Present Matches to User

When matches are found, present them clearly:

### Format for Presenting Matches

```markdown
## Existing Entities Found

I found existing entities that might match your request:

### 1. [Entity Name] (RID: XXX)
- **Type**: [entity type]
- **Description**: [description]
- **Why it matches**: [explanation of semantic similarity]

### 2. [Another Entity] (RID: YYY)
- **Type**: [entity type]
- **Description**: [description]
- **Why it matches**: [explanation]

## Options

1. **Use existing**: Use [Entity Name] as-is
2. **Extend existing**: Add to [Entity Name] (e.g., new columns, new members)
3. **Create new**: Create a new entity if these don't meet your needs

Which would you prefer?
```

---

## Step 3: Semantic Validation on Creation

If creating a new entity, validate semantic consistency:

### Table Creation Checklist

- [ ] Table name follows naming conventions (PascalCase, descriptive)
- [ ] Table name doesn't conflict with existing tables
- [ ] Description clearly explains the table's purpose
- [ ] Column names are meaningful and consistent with existing patterns
- [ ] Column descriptions explain what each column contains
- [ ] Foreign keys reference valid existing tables

### Vocabulary Term Checklist

- [ ] Term name doesn't conflict with existing term names
- [ ] Term name doesn't conflict with existing synonyms
- [ ] Synonyms don't conflict with other term names
- [ ] Description clearly defines the term's meaning
- [ ] Term fits semantically within the vocabulary's domain

### Feature Creation Checklist

- [ ] Feature name is unique for the target table
- [ ] Feature name is descriptive of what it labels
- [ ] Referenced vocabulary table exists
- [ ] Feature description explains its purpose
- [ ] Target table is appropriate for this feature

### Dataset Creation Checklist

- [ ] Dataset types are valid (exist in Dataset_Type vocabulary)
- [ ] Description explains what the dataset contains
- [ ] No existing dataset serves the same purpose
- [ ] Member composition is well-defined

---

## Semantic Search Patterns

### Search by Name Similarity

Look for entities with names that:
- Are exact matches (case-insensitive)
- Contain the search term as a substring
- Share common prefixes/suffixes
- Use synonymous terms (e.g., "patient" vs "subject")

### Search by Description Similarity

Look for descriptions that:
- Contain similar keywords
- Describe similar purposes
- Reference similar data types or domains

### Search by Structure Similarity

For tables, look for:
- Similar column names
- Similar column types
- Similar foreign key relationships

For features, look for:
- Same target table
- Similar vocabulary types
- Similar labeling purposes

---

## Example Workflow: Creating a New Feature

```
User: "I need to add quality labels to the Image table"

Assistant actions:
1. Check existing features on Image table:
   → Read deriva-ml://table/Image/features
   → Found: "Image_Quality" feature already exists

2. Check the existing feature details:
   → Read deriva-ml://feature/Image/Image_Quality
   → Uses vocabulary: Image_Quality_Type with terms: Good, Acceptable, Poor, Ungradable

3. Present to user:
   "I found an existing 'Image_Quality' feature on the Image table that labels
   images with quality assessments (Good, Acceptable, Poor, Ungradable).

   Does this match what you need, or do you need a different type of quality label?"

User: "That's exactly what I need, use that"

→ Reuse existing feature instead of creating duplicate
```

---

## Quick Reference: Resources for Semantic Search

| Entity Type | Resource for Discovery | Key Semantic Fields |
|-------------|------------------------|---------------------|
| Tables | `deriva-ml://catalog/tables` | name, description, columns |
| Table Details | `deriva-ml://table/{table}/schema`, `get_table_columns()` | column names, types, descriptions |
| Vocabularies | `deriva-ml://catalog/vocabularies` | all vocabs with terms |
| Vocab Terms | `deriva-ml://vocabulary/{name}` | term name, synonyms, description |
| Term Lookup | `deriva-ml://vocabulary/{vocab}/{term}` | find by name or synonym |
| Features | `deriva-ml://table/{table}/features` | feature name, vocabulary |
| Feature Details | `deriva-ml://feature/{table}/{feature}` | structure, columns |
| Datasets | `deriva-ml://catalog/datasets` | types, description |
| Dataset Details | `deriva-ml://dataset/{rid}` | members, hierarchy |
| Workflows | `deriva-ml://catalog/workflows` | name, type, description |
| Workflow Details | `deriva-ml://workflow/{rid}` | name, type, URL, checksum |
| Assets | `deriva-ml://catalog/assets` | tables, types |
| Table Assets | `deriva-ml://table/{table}/assets` | all assets in a table |
| Asset Details | `deriva-ml://asset/{rid}` | filename, types, provenance |

## Summary

**Always follow this pattern**:

1. **STOP** before any creation operation
2. **SEARCH** for existing entities with similar semantics
3. **PRESENT** matches to the user with clear explanations
4. **SUGGEST** reuse when appropriate
5. **CREATE** only when no suitable entity exists
6. **VALIDATE** semantic consistency of new entities

This prevents catalog pollution, promotes consistency, and helps users discover
existing resources they may not know about.
"""

    # =========================================================================
    # Hydra-zen Configuration Update Prompt
    # =========================================================================

    @mcp.prompt(
        name="update-hydra-configs",
        description="ALWAYS use after creating catalog entities: Guide for updating hydra-zen config files when datasets, assets, or executions are created",
    )
    def update_hydra_configs_prompt() -> str:
        """Guide for keeping hydra-zen configs in sync with catalog entities."""
        return """# Updating Hydra-zen Configuration Files

**Use this prompt PROACTIVELY**: After creating datasets, assets, workflows, or running
executions, offer to update the user's hydra-zen configuration files to include the
new entities. This keeps configurations in sync with the catalog.

## When to Trigger This Workflow

Update configs after ANY of these operations:

| Operation | Config File to Update | Config Type |
|-----------|----------------------|-------------|
| Create dataset | `configs/datasets.py` | DatasetSpecConfig |
| Create asset group | `configs/assets.py` | Asset RID list |
| Run execution with outputs | `configs/assets.py` | Output asset RIDs |
| Create workflow | `configs/workflow.py` | Workflow config |
| Define new model variant | `configs/<model>.py` | Model builds() |
| Create experiment preset | `configs/experiments.py` | Experiment store |

---

## Step 1: Identify What Was Created

After a catalog operation, gather the key information:

### For Datasets:
```
Dataset created:
  RID: 2-ABC
  Name: CIFAR10 Small Training Split
  Current Version: 1.0.0
  Types: [Training, Labeled_Training]
  Description: Small subset of CIFAR-10 for quick experiments
```

### For Assets (execution outputs):
```
Execution completed:
  Execution RID: 3-XYZ
  Output assets:
    - Model weights: RID 4-DEF (Model type)
    - Predictions: RID 4-GHI (Execution_Metadata type)
    - Metrics: RID 4-JKL (Execution_Metadata type)
```

### For Workflows:
```
Workflow created:
  RID: 1-MNO
  Name: ResNet50 Training
  Type: Training
  Description: Train ResNet50 on image classification tasks
```

---

## Step 2: Read Current Config Files

Check what config files exist and their current structure:

```python
# Common config file locations
configs/
  __init__.py           # Must import all config modules
  base.py              # Base configuration (deriva connection, workflow)
  datasets.py          # DatasetSpecConfig entries
  assets.py            # Asset RID configurations
  experiments.py       # Experiment presets
  <model_name>.py      # Model-specific configs (e.g., cifar10_cnn.py)
  workflow.py          # Workflow configurations
```

Read the relevant file to understand the existing pattern and naming conventions.

---

## Step 3: Generate Config Updates

### Dataset Configuration (`configs/datasets.py`)

**Template:**
```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

# Register dataset configs
datasets_store = store(group="datasets")

# [EXISTING CONFIGS...]

# NEW: {dataset_name}
# {description}
# Types: {types}
datasets_store(
    [
        DatasetSpecConfig(
            rid="{rid}",
            version="{version}",
            materialize=True,  # Download data locally
        ),
    ],
    name="{config_name}",  # Use snake_case, descriptive name
)
```

**Naming conventions for datasets:**
- Use snake_case: `cifar10_small_training`
- Include key characteristics: `{data_name}_{size}_{split_type}`
- Examples: `mnist_full_training`, `cifar10_small_labeled_test`

### Asset Configuration (`configs/assets.py`)

**Template for asset groups:**
```python
from hydra_zen import store
from deriva_ml.execution import asset_store, with_description

# [EXISTING CONFIGS...]

# NEW: {asset_group_name}
# Created by execution {execution_rid}
asset_store(
    with_description(
        ["{asset_rid_1}", "{asset_rid_2}"],
        '''{description}

Source: Execution {execution_rid}
- Model: {model_config_used}
- Dataset: {dataset_used}
''',
    ),
    name="{config_name}",
)
```

**Naming conventions for assets:**
- Include model or purpose: `resnet50_weights_v1`
- Include relevant context: `lr_sweep_predictions`
- For output groups: `{experiment_name}_outputs`

### Workflow Configuration (`configs/workflow.py`)

**Template:**
```python
from hydra_zen import builds, store
from deriva_ml.execution import Workflow

workflow_store = store(group="workflow")

# [EXISTING CONFIGS...]

# NEW: {workflow_name}
workflow_store(
    builds(
        Workflow,
        name="{name}",
        workflow_type="{type}",
        description="{description}",
    ),
    name="{config_name}",
)
```

### Experiment Configuration (`configs/experiments.py`)

**Template:**
```python
from hydra_zen import make_config, store

experiment_store = store(group="experiment", package="_global_")

# [EXISTING CONFIGS...]

# NEW: {experiment_name}
# {description}
experiment_store(
    make_config(
        defaults=[
            "_self_",
            {"model_config": "{model_config_name}"},
            {"datasets": "{dataset_config_name}"},
            {"assets": "{asset_config_name}"},  # Optional
        ],
        # Override specific settings if needed
        hydra_defaults=[
            "_self_",
            {"override /model_config": "{model_config_name}"},
        ],
    ),
    name="{experiment_name}",
)
```

---

## Step 4: Update __init__.py

Ensure the config module is imported in `configs/__init__.py`:

```python
# configs/__init__.py
from . import base
from . import datasets
from . import assets
from . import experiments
from . import workflow
# Add new config modules here
```

---

## Step 5: Verify Configuration

After updating, verify the configuration is valid:

```bash
# Check for syntax errors
uv run python -c "from configs import *"

# List available configs
uv run deriva-ml-run --info

# Dry run with new config
uv run deriva-ml-run datasets={new_config_name} dry_run=true
```

---

## Example: Complete Workflow

### Scenario: User creates a new labeled dataset

```
User: "Create a training dataset from images with the 'good' quality label"

→ Assistant creates dataset via MCP tools
→ Dataset created with RID: 2-XYZ, version 1.0.0

Assistant response:
"I've created the dataset. Would you like me to add it to your hydra-zen
configuration so you can use it in experiments?

I'll add this to `configs/datasets.py`:
```python
# Good quality training images
# 150 images with quality='good' label
datasets_store(
    [DatasetSpecConfig(rid='2-XYZ', version='1.0.0', materialize=True)],
    name='good_quality_training',
)
```

You can then use it with:
```bash
uv run deriva-ml-run datasets=good_quality_training
```
"
```

### Scenario: User runs an experiment and wants to save outputs

```
User: "Run the cifar10_quick experiment"

→ Assistant runs experiment via deriva-ml-run
→ Execution completes with output assets:
   - Model: 4-ABC
   - Predictions: 4-DEF

Assistant response:
"The experiment completed successfully. I can add the output assets to your
configuration for future use (e.g., for inference or analysis).

Add to `configs/assets.py`:
```python
# CIFAR-10 quick experiment outputs
# Execution: 3-XYZ, Model: cifar10_quick, Dataset: cifar10_small
asset_store(
    with_description(
        ['4-ABC', '4-DEF'],
        '''CIFAR-10 Quick Experiment Outputs

Model weights and predictions from quick training run.
- Epochs: 3
- Learning rate: 0.001
- Test accuracy: 72.5%
''',
    ),
    name='cifar10_quick_outputs',
)
```
"
```

---

## Quick Reference: Config Templates

Use these MCP resources to get full templates:

| Config Type | Resource |
|-------------|----------|
| Deriva connection | `deriva-ml://config/deriva-ml-template` |
| Dataset specs | `deriva-ml://config/dataset-spec-template` |
| Model configs | `deriva-ml://config/model-template` |
| Experiments | `deriva-ml://config/experiment-template` |
| Multiruns | `deriva-ml://config/multirun-template` |

---

## Important Notes

1. **Materialize flag**: Set `materialize=True` for datasets that need local files,
   `False` for metadata-only access.

2. **Description quality**: Include provenance information in asset descriptions:
   - Source execution RID
   - Model configuration used
   - Key hyperparameters
   - Performance metrics if available

3. **Naming consistency**: Follow existing naming patterns in the config files.
   Check what conventions are already in use.

4. **Git commit**: After updating configs, remind the user to commit changes
   before running experiments (DerivaML tracks code provenance).

---

## CRITICAL: Dataset Version Management

**See the `dataset-versioning` prompt for complete version management rules.**

### Key Version Rules for Config Files

#### Rule 1: ALWAYS Use Explicit Versions for Real Experiments

```python
# CORRECT: Explicit version for reproducibility
DatasetSpecConfig(rid="1-ABC", version="1.2.0")

# WRONG: No version - results may not be reproducible
DatasetSpecConfig(rid="1-ABC")  # AVOID for production runs
```

**Only omit version for:**
- Quick debugging (with `dry_run=true`)
- Exploratory data analysis
- Development testing

#### Rule 2: Update Configs After Version Increments

When the catalog is modified and a dataset version is incremented:

1. Increment the version with description:
   ```
   increment_dataset_version("2-XYZ", "Added quality labels to all images", "minor")
   → New version: 1.1.0
   ```

2. Update the config file:
   ```python
   # OLD
   DatasetSpecConfig(rid="2-XYZ", version="1.0.0")

   # NEW - updated to latest version
   DatasetSpecConfig(rid="2-XYZ", version="1.1.0")
   ```

3. Commit the config change before running experiments

#### Rule 3: Provide "Update to Current Version" Workflow

When user requests: "Update my dataset configs to use current versions"

1. Read the dataset config file
2. For each DatasetSpecConfig:
   - Read `deriva-ml://dataset/<rid>` resource to get `current_version`
   - Compare with configured version
3. Show proposed changes:
   ```python
   # Changes to configs/datasets.py:
   # - cifar10_training: 1.0.0 → 1.2.0
   # - cifar10_test: 1.1.0 → 1.2.0 (no change needed)
   ```
4. Apply changes if user approves
5. Remind user to commit before running experiments

### Version Update Example

```
User: "I added quality labels to the images. Update my configs to use the new data."

Assistant workflow:
1. Increment dataset version:
   increment_dataset_version(
       "2-ABC",
       "Added Image_Quality labels (Good/Fair/Poor) to all 5000 images",
       "minor"
   )
   → Version 1.0.0 → 1.1.0

2. Update configs/datasets.py:
   # Before
   DatasetSpecConfig(rid="2-ABC", version="1.0.0")

   # After
   DatasetSpecConfig(rid="2-ABC", version="1.1.0")

3. Response:
   "I've incremented the dataset version to 1.1.0 (added quality labels) and
   updated your config. Please commit these changes before running experiments:

   git add configs/datasets.py
   git commit -m 'Update dataset version to 1.1.0 with quality labels'
   "
```

### Why This Matters

- **Reproducibility**: Explicit versions ensure the same data is used each time
- **Provenance**: Version history shows what changed and when
- **Debugging**: If results differ, you can check if data versions match
- **Collaboration**: Team members use the same data snapshot
"""
