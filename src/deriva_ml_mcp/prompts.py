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
        description="Step-by-step guide to run an ML workflow with full provenance tracking",
    )
    def run_ml_execution_prompt() -> str:
        """Guide for running an ML execution with provenance."""
        return """# Running an ML Execution with Provenance

Follow these steps to run an ML workflow (training, inference, preprocessing, etc.)
with full provenance tracking in DerivaML.

## Prerequisites
- Connected to a DerivaML catalog (use `connect_catalog` if not connected)
- Input dataset(s) available (use `list_datasets` to find them)

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

Use `list_workflow_types()` to see available workflow types.

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
- Use `list_executions()` to see past workflow runs
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
- Know your dataset RID (use `list_datasets()` to find it)

## Step 1: Explore Your Dataset

First, understand what data is available:

```
# List all datasets
list_datasets()

# Get details about your dataset
get_dataset("<dataset-rid>")

# See which tables have data in this dataset
list_dataset_members("<dataset-rid>")
```

## Step 2: Understand Table Structure

For each table you want to use:

```
# See columns and their types
get_table_schema("<table-name>")
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

### Option B: Single Table Access

Get raw data from one table:

```
get_dataset_table("<dataset-rid>", "<table-name>", limit=1000)
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
# Check version history
get_dataset_version_history("<dataset-rid>")

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
        name="customize-table-display",
        description="Step-by-step guide to customize how a table appears in the Chaise web UI",
    )
    def customize_table_display_prompt() -> str:
        """Guide for customizing table annotations."""
        return """# Customizing Table Display in Chaise

Follow these steps to customize how a table appears in the Chaise web interface
using Deriva annotations.

## Prerequisites
- Connected to a DerivaML catalog
- Know which table you want to customize

## Step 1: Check Current Annotations

See what's already configured:

```
get_table_annotations("<table-name>")
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

Use `get_annotation_contexts()` for full documentation.

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
# First, find available foreign keys
list_foreign_keys("<table>")

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
# 1. Check current state
get_table_annotations("Image")

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
list_foreign_keys("Image")  # Find FK names
add_visible_foreign_key("Image", "detailed", ["domain", "Image_Subject_fkey"])

# 6. Commit changes
apply_annotations()
```

## Tips

- Always check current annotations before making changes
- Use `*` context for defaults, specific contexts for overrides
- Test changes in Chaise after applying
- Use `get_chaise_url("<table>")` to get the URL to view changes
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
# Find features for a table
find_features("<table-name>")

# Get details about a specific feature
lookup_feature("<table-name>", "<feature-name>")

# List all registered feature names
list_feature_names()
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

For features with multiple columns (check with `lookup_feature`):

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
list_feature_values("Image", "Diagnosis")
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
ml.list_feature_values("Image", "Classification")
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
list_feature_values("Image", "Classification")
```

## Tips

- Always add feature values within an execution for provenance
- Use `lookup_feature` to see the feature's column structure
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

Create a table with basic columns:

```
create_table(
    "Subject",
    columns=[
        {"name": "Name", "type": "text", "nullok": false},
        {"name": "Age", "type": "int4"},
        {"name": "Notes", "type": "markdown"}
    ],
    comment="Research subjects in the study"
)
```

**Column options:**
- `name` (required): Column name
- `type`: Data type (default: "text")
- `nullok`: Allow NULL values (default: true)
- `comment`: Description of the column

## Step 4: Create Tables with Foreign Keys

To link tables together, use foreign keys:

```
# First, create the parent table
create_table(
    "Subject",
    columns=[
        {"name": "Name", "type": "text", "nullok": false},
        {"name": "Species", "type": "text"}
    ],
    comment="Research subjects"
)

# Then create a child table that references it
create_table(
    "Sample",
    columns=[
        {"name": "Name", "type": "text", "nullok": false},
        {"name": "Subject", "type": "text", "nullok": false},
        {"name": "Collection_Date", "type": "date"},
        {"name": "Tissue_Type", "type": "text"}
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
# List all tables
list_tables()

# Get detailed schema
get_table_schema("Subject")

# Get Chaise URL to view in browser
get_chaise_url("Subject")
```

## Common Patterns

### Subject -> Sample -> Measurement Hierarchy

```
# Subject (top level)
create_table("Subject", columns=[
    {"name": "Name", "type": "text", "nullok": false},
    {"name": "Age", "type": "int4"}
])

# Sample references Subject
create_table("Sample", columns=[
    {"name": "Name", "type": "text", "nullok": false},
    {"name": "Subject", "type": "text", "nullok": false},
    {"name": "Collection_Date", "type": "date"}
], foreign_keys=[
    {"column": "Subject", "referenced_table": "Subject", "on_delete": "CASCADE"}
])

# Measurement references Sample
create_table("Measurement", columns=[
    {"name": "Sample", "type": "text", "nullok": false},
    {"name": "Value", "type": "float8", "nullok": false},
    {"name": "Unit", "type": "text"},
    {"name": "Measured_At", "type": "timestamptz"}
], foreign_keys=[
    {"column": "Sample", "referenced_table": "Sample", "on_delete": "CASCADE"}
])
```

### Protocol with Versioning

```
create_table("Protocol", columns=[
    {"name": "Name", "type": "text", "nullok": false},
    {"name": "Version", "type": "text", "nullok": false},
    {"name": "Description", "type": "markdown"},
    {"name": "Is_Active", "type": "boolean"}
], comment="Experimental protocols with version tracking")
```

## Tips

- **Naming conventions**: Use singular nouns (Subject, not Subjects)
- **RID column**: Every table automatically gets an RID (unique identifier)
- **Required fields**: Set `nullok: false` for required columns
- **Descriptions**: Add comments to tables and columns for documentation
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
- **Dataset Types**: Labels like "Training", "Testing", "Validation"
- **Nested Datasets**: Parent datasets can contain child datasets
- **Versioning**: Semantic versioning (major.minor.patch) for reproducibility

## Step 1: Create Dataset via Execution

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
create_execution_dataset(
    description="Training images for model v2",
    dataset_types=["Training"]
)
```

Note the returned `dataset_rid` for adding members.

## Step 2: Register Element Types (if needed)

Before adding records, their table must be registered as an element type:

```
# Check which tables are registered
list_dataset_element_types()

# Register a new table
add_dataset_element_type("Image")
add_dataset_element_type("Subject")
```

## Step 3: Add Dataset Members

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

## Step 4: Upload Outputs (REQUIRED)

**Python API:** The context manager handles timing automatically. Call upload after exiting:
```python
# After exiting the `with` block:
exe.upload_execution_outputs()
```

**MCP Tools:**
```
upload_execution_outputs()
```

## Step 5: Manage Dataset Types

Add or remove type labels:

```
# Add a type
add_dataset_type("<dataset-rid>", "Curated")

# Remove a type
remove_dataset_type("<dataset-rid>", "Draft")

# List available types
list_dataset_types()
```

## Step 6: Create Nested Datasets (Optional)

Create training/testing splits as child datasets:

**Python API (recommended):**
```python
config = ExecutionConfiguration(
    workflow=Workflow(name="Create Train/Test Split", workflow_type="Preprocessing", description="Split data")
)

with ml.create_execution(config) as exe:
    # Create training subset
    training_ds = exe.create_execution_dataset("Training subset (80%)", ["Training"])
    # Add training members...

    # Create testing subset
    testing_ds = exe.create_execution_dataset("Testing subset (20%)", ["Testing"])
    # Add testing members...

# Upload AFTER exiting context manager
exe.upload_execution_outputs()

# Link as nested datasets
ml.add_dataset_child("<parent-rid>", training_ds.rid)
ml.add_dataset_child("<parent-rid>", testing_ds.rid)
```

**MCP Tools:**
```
# Create parent "Complete" dataset first (via execution)
# Then create children
create_execution("Create Train/Test Split", "Preprocessing", "Split data")

# Create training subset
create_execution_dataset("Training subset (80%)", ["Training"])
# Add training members...

# Create testing subset
create_execution_dataset("Testing subset (20%)", ["Testing"])
# Add testing members...

# Upload outputs
upload_execution_outputs()

# Link as nested datasets
add_dataset_child("<parent-rid>", "<training-rid>")
add_dataset_child("<parent-rid>", "<testing-rid>")
```

## Step 7: Version Management

```
# View version history
get_dataset_version_history("<dataset-rid>")

# Manually increment version
increment_dataset_version("<dataset-rid>", "major", "Schema change")

# Query specific version
list_dataset_members("<dataset-rid>", version="1.0.0")
```

## Complete Example: Create Train/Test Split

**Python API (recommended):**
```python
config = ExecutionConfiguration(
    workflow=Workflow(name="Dataset Curation", workflow_type="Preprocessing", description="Create ML datasets")
)

with ml.create_execution(config) as exe:
    # Ensure element types registered
    ml.add_dataset_element_type("Image")

    # Create main dataset
    complete_ds = exe.create_execution_dataset("Complete Image Set", ["Complete"])

    # Add all images
    ml.add_dataset_members(complete_ds.rid, ["2-D01", "2-D02", "2-D03", ..., "2-D100"])

    # Create training split
    training_ds = exe.create_execution_dataset("Training Set (80%)", ["Training"])
    ml.add_dataset_members(training_ds.rid, ["2-D01", "2-D02", ..., "2-D80"])

    # Create test split
    testing_ds = exe.create_execution_dataset("Test Set (20%)", ["Testing"])
    ml.add_dataset_members(testing_ds.rid, ["2-D81", ..., "2-D100"])

# Upload AFTER exiting context manager
exe.upload_execution_outputs()

# Link as children
ml.add_dataset_child(complete_ds.rid, training_ds.rid)
ml.add_dataset_child(complete_ds.rid, testing_ds.rid)

# Verify
ml.get_dataset(complete_ds.rid)  # Shows children
ml.list_dataset_children(complete_ds.rid)
```

**MCP Tools:**
```
# 1. Create execution
create_execution("Dataset Curation", "Preprocessing", "Create ML datasets")

# 2. Ensure element types registered
add_dataset_element_type("Image")

# 3. Create main dataset
create_execution_dataset("Complete Image Set", ["Complete"])
# Returns dataset_rid: "1-ABC"

# 4. Add all images
add_dataset_members("1-ABC", ["2-D01", "2-D02", "2-D03", ..., "2-D100"])

# 5. Create training split
create_execution_dataset("Training Set (80%)", ["Training"])
# Returns: "1-DEF"
add_dataset_members("1-DEF", ["2-D01", "2-D02", ..., "2-D80"])

# 6. Create test split
create_execution_dataset("Test Set (20%)", ["Testing"])
# Returns: "1-GHI"
add_dataset_members("1-GHI", ["2-D81", ..., "2-D100"])

# 7. Upload and link as children
upload_execution_outputs()

add_dataset_child("1-ABC", "1-DEF")
add_dataset_child("1-ABC", "1-GHI")

# 8. Verify
get_dataset("1-ABC")  # Shows children
list_dataset_children("1-ABC")
```

## Tips

- Always create datasets within an execution for provenance
- Use `list_datasets()` to find existing datasets
- Use semantic versioning: patch=metadata, minor=elements, major=breaking
- Nested datasets share elements - good for train/test splits
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

    @mcp.prompt(
        name="configure-ml-experiment",
        description="Step-by-step guide to configure ML experiments using hydra-zen with DerivaML",
    )
    def configure_ml_experiment_prompt() -> str:
        """Guide for configuring ML experiments with hydra-zen."""
        return """# Configuring ML Experiments with Hydra-Zen and DerivaML

This guide explains how to set up reproducible ML experiments using hydra-zen
configuration management with DerivaML for data and provenance tracking.

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

Create `configs/datasets.py` to specify input data:

```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

datasets_store = store(group="datasets")

# Training dataset - pinned version for reproducibility
training_data = [
    DatasetSpecConfig(rid="1-ABC", version="1.0.0")
]
datasets_store(training_data, name="training")

# Full dataset - all available data
full_data = [
    DatasetSpecConfig(rid="1-ABC"),  # Latest version
    DatasetSpecConfig(rid="1-DEF"),  # Additional data
]
datasets_store(full_data, name="full")

# Default for quick testing
datasets_store(training_data, name="default_dataset")
```

## Step 3: Configure Assets (Optional)

Create `configs/assets.py` for pre-trained weights or other files:

```python
from hydra_zen import store

asset_store = store(group="assets")

# Pre-trained weights
pretrained = ["2-XYZ"]  # RID of model checkpoint
asset_store(pretrained, name="pretrained_weights")

# No assets needed
asset_store([], name="default_asset")
```

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

Create `configs/my_model.py` for model hyperparameters:

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

# Register variations
model_store(MyModelConfig, name="default_model")

# Quick test - fewer epochs
model_store(MyModelConfig, name="quick", epochs=2, batch_size=64)

# Extended training
model_store(
    MyModelConfig,
    name="extended",
    epochs=50,
    hidden_size=256,
    dropout=0.2,
)
```

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

## Best Practices

1. **Pin dataset versions** for reproducibility
2. **Use meaningful names** for configurations
3. **Document workflows** with markdown descriptions
4. **Create experiment presets** for common configurations
5. **Use `dry_run=True`** to test config before long runs
6. **Register all configs** with hydra-zen store for discoverability

## Tips

- Use `list_workflow_types()` to see available workflow types
- Use `list_datasets()` to find dataset RIDs
- Group related configs (e.g., all CIFAR-10 configs in one file)
- Use `zen_partial=True` for model configs that need runtime context
- Store experiments in `experiments.py` for easy discovery
"""

    @mcp.prompt(
        name="setup-catalog-display",
        description="Step-by-step guide to configure Chaise web UI navigation and display settings",
    )
    def setup_catalog_display_prompt() -> str:
        """Guide for setting up catalog-level display configuration."""
        return """# Setting Up Catalog Display in Chaise

Follow these steps to configure how your catalog appears in the Chaise web interface,
including navigation menus, branding, and default behaviors.

## Prerequisites
- Connected to a DerivaML catalog
- Admin access to modify catalog annotations

## Step 1: Apply Default Catalog Annotations

Set up the standard navigation bar and display settings:

```
apply_catalog_annotations(
    navbar_brand_text="My ML Project",  # Text in nav bar
    head_title="ML Catalog"              # Browser tab title
)
```

This automatically configures:
- Navigation bar with organized dropdown menus
- User info, Deriva-ML tables, domain tables, vocabularies
- Default landing page (Dataset table)
- Display settings (underscores as spaces, system columns visible)
- Bulk upload support for asset tables

## Step 2: Customize Table Display Names

Set friendly names for your tables:

```
# Domain tables
set_display_annotation("Image", {"name": "Images"})
set_display_annotation("Subject", {"name": "Subjects"})
set_display_annotation("Diagnosis", {"name": "Diagnoses"})

# Commit changes
apply_annotations()
```

## Step 3: Configure Key Tables

For your most important tables, set up complete display:

### Example: Image Table

```
# 1. Set display name
set_display_annotation("Image", {"name": "Images"})

# 2. Configure compact view (listings)
set_visible_columns("Image", {
    "compact": ["RID", "Filename", "Subject", "Created"],
    "detailed": ["RID", "Filename", "URL", "Subject", "Description", "Created", "Modified"]
})

# 3. Set row name for listings
set_table_display("Image", {
    "row_name": {
        "row_markdown_pattern": "{{{Filename}}}"
    }
})

# 4. Apply changes
apply_annotations()
```

## Step 4: Configure Column Display

Customize how individual columns appear:

```
# Set column label
set_display_annotation("Image", {"name": "File Name"}, column_name="Filename")

# Set column display pattern (e.g., format as link)
set_column_display("Image", "URL", {
    "*": {
        "markdown_pattern": "[Download]({{{URL}}})"
    }
})

apply_annotations()
```

## Step 5: Configure Related Tables

Control which foreign key relationships appear:

```
# List available foreign keys
list_foreign_keys("Image")

# Add to detailed view
add_visible_foreign_key("Image", "detailed", ["domain", "Image_Subject_fkey"])

apply_annotations()
```

## Step 6: Test Your Configuration

Get the URL to view your changes:

```
get_chaise_url("Image")
```

Open this URL in a browser to verify the display.

## Common Patterns

### Hide System Columns in Entry Forms

```
set_visible_columns("Image", {
    "entry": ["Filename", "Description", "Subject"],  # No RID, Created, etc.
    "entry/create": ["Filename", "Description", "Subject"]
})
```

### Show Thumbnails in Listings

```
set_visible_columns("Image", {
    "compact": [
        {"sourcekey": "thumbnail_url", "markdown_pattern": "![thumb]({{{URL}}}?w=50)"},
        "Filename",
        "Subject"
    ]
})
```

### Configure Sort Order

```
set_table_display("Image", {
    "row_order": [{"column": "Created", "descending": true}]
})
```

## Complete Setup Example

```
# 1. Apply catalog-level settings
apply_catalog_annotations("Medical Imaging ML", "Med-ML Catalog")

# 2. Configure Image table
set_display_annotation("Image", {"name": "Medical Images"})
set_visible_columns("Image", {
    "compact": ["RID", "Filename", "Subject", "Diagnosis"],
    "detailed": ["*"],
    "entry": ["Filename", "Description", "Subject"]
})
set_table_display("Image", {
    "row_name": {"row_markdown_pattern": "{{{Filename}}}"},
    "row_order": [{"column": "RID", "descending": true}]
})

# 3. Configure Subject table
set_display_annotation("Subject", {"name": "Patients"})
set_visible_columns("Subject", {
    "compact": ["RID", "Name", "Age", "Sex"]
})

# 4. Apply all changes
apply_annotations()

# 5. Verify
get_chaise_url("Image")
```

## Tips

- Run `apply_catalog_annotations()` first to set up navigation
- Use `get_table_annotations()` to see current configuration
- Use `get_chaise_url()` to quickly view changes in browser
- Context `*` provides defaults; specific contexts override
- Always call `apply_annotations()` after making changes
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
        description="Step-by-step checklist for running an ML experiment with DerivaML",
    )
    def run_experiment_prompt() -> str:
        """Guide for running an ML experiment."""
        return """# Running an ML Experiment with DerivaML

Follow this checklist to run a reproducible ML experiment with full provenance tracking.

## Prerequisites
- Repository set up following the DerivaML model template
- Connected to the target DerivaML catalog
- Model code implemented and tested

## Step 1: Verify Repository Structure

Ensure your repository follows the recommended configuration structure:

```
src/
├── configs/
│   ├── __init__.py         # Auto-loads all config modules
│   ├── deriva.py           # Catalog connection configs
│   ├── datasets.py         # Dataset specifications
│   ├── assets.py           # Input asset references
│   ├── workflow.py         # Workflow definitions
│   ├── <model_name>.py     # Model hyperparameters
│   └── experiments.py      # Experiment presets
├── models/
│   └── <model_name>.py     # Model implementation
└── deriva_run.py           # Main entry point
```

**Check:**
- [ ] All config modules are in `configs/` directory
- [ ] `configs/__init__.py` calls `load_all_configs()`
- [ ] Model implementation exists in `models/`
- [ ] Entry point `deriva_run.py` is present

## Step 2: Configure Datasets

Edit `configs/datasets.py` to specify the datasets for your experiment.

**Find dataset RIDs in the catalog:**
```
list_datasets()
get_dataset("<rid>")
get_dataset_version_history("<rid>")
```

**Update configuration:**
```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

datasets_store = store(group="datasets")

# Pin versions for reproducibility
training_data = [
    DatasetSpecConfig(rid="<DATASET_RID>", version="<VERSION>")
]
datasets_store(training_data, name="training")
```

**Check:**
- [ ] Dataset RIDs are correct for target catalog
- [ ] Versions are pinned for production runs
- [ ] Multiple datasets listed if needed

## Step 3: Configure Input Assets

Edit `configs/assets.py` to specify pre-trained weights or other input files.

**Find asset RIDs in the catalog:**
```
list_assets("<asset_table>")  # e.g., "Model", "Checkpoint"
query_table("<asset_table>", columns=["RID", "Filename", "Description"])
```

**Update configuration:**
```python
from hydra_zen import store

asset_store = store(group="assets")

# Pre-trained weights
pretrained = ["<MODEL_RID>"]
asset_store(pretrained, name="pretrained_weights")

# No assets needed
asset_store([], name="no_assets")
```

**Check:**
- [ ] Asset RIDs reference correct files in catalog
- [ ] All required input files are specified
- [ ] Asset table names match catalog schema

## Step 4: Configure Model Parameters

Edit `configs/<model_name>.py` to define model hyperparameters.

**Create configurations for different scenarios:**
```python
from hydra_zen import builds, store
from models.my_model import train_model

model_store = store(group="model_config")

# Base configuration
BaseConfig = builds(
    train_model,
    epochs=10,
    learning_rate=1e-3,
    batch_size=32,
    hidden_size=128,
    populate_full_signature=True,
    zen_partial=True,
)

# Register variants
model_store(BaseConfig, name="default_model")
model_store(BaseConfig, name="quick", epochs=2)
model_store(BaseConfig, name="full", epochs=50, hidden_size=256)
```

**Check:**
- [ ] All hyperparameters have sensible defaults
- [ ] Quick/test configuration exists for debugging
- [ ] Production configuration is defined
- [ ] `zen_partial=True` set for configs needing runtime context

## Step 5: Commit Changes and Bump Version

**CRITICAL: Commit all changes before running!** DerivaML requires a clean git state
for provenance tracking. The execution will warn if there are uncommitted changes.

**RECOMMENDED: Bump version for significant experiments.** While not required, incrementing
the version before important experiments makes it easier to identify and reproduce results
later. Consider bumping the version when running experiments you may want to reference.

```bash
# Check for uncommitted changes
git status

# Stage and commit all changes (REQUIRED)
git add .
git commit -m "Configure experiment: <description>"

# Bump version (RECOMMENDED for significant experiments)
uv run bump-version patch   # Bug fixes, small tweaks
uv run bump-version minor   # New features, config changes
uv run bump-version major   # Breaking changes

# Push changes and tags
git push && git push --tags

# Verify version
uv run python -m setuptools_scm
```

**Check:**
- [ ] All code changes committed (required)
- [ ] All config changes committed (required)
- [ ] `uv.lock` is current and committed (required)
- [ ] Version bumped appropriately (recommended)
- [ ] Tags pushed to remote (if version bumped)

## Step 6: Run the Experiment

### Test First (Dry Run)
```bash
# Test configuration without creating records
uv run python deriva_run.py dry_run=True

# Test with specific experiment preset
uv run python deriva_run.py +experiment=quick_test dry_run=True
```

### Production Run
```bash
# Run with defaults
uv run python deriva_run.py

# Run specific experiment
uv run python deriva_run.py +experiment=full_training

# Override parameters
uv run python deriva_run.py \\
    datasets=training \\
    model_config=full \\
    model_config.epochs=100

# Run parameter sweep
uv run python deriva_run.py -m \\
    model_config.learning_rate=1e-2,1e-3,1e-4
```

**Common Hydra overrides:**
| Override | Purpose |
|----------|---------|
| `+experiment=<name>` | Use experiment preset |
| `datasets=<name>` | Select dataset config |
| `model_config=<name>` | Select model config |
| `assets=<name>` | Select asset config |
| `dry_run=True` | Test without records |
| `-m param=a,b,c` | Parameter sweep |

## Step 7: Verify Results

After the run completes:

```
# List recent executions
list_executions(limit=5)

# Check execution details
get_execution_info()

# View in Chaise
get_chaise_url("Execution")
```

**Verify:**
- [ ] Execution record created in catalog
- [ ] Output assets uploaded
- [ ] Metrics/metadata captured
- [ ] Provenance links correct (datasets, code version)

## Complete Checklist

Before running:
- [ ] Repository structure matches template
- [ ] Dataset RIDs and versions configured
- [ ] Asset RIDs configured (if needed)
- [ ] Model parameters configured
- [ ] All changes committed to Git
- [ ] Version bumped with `bump-version`
- [ ] Dry run successful

After running:
- [ ] Execution record exists in catalog
- [ ] Outputs uploaded successfully
- [ ] Results reproducible with same config

## Troubleshooting

### "Dataset not found"
- Verify RID exists: `get_dataset("<rid>")`
- Check you're connected to correct catalog
- Ensure version exists: `get_dataset_version_history("<rid>")`

### "Asset not found"
- Verify RID exists: `resolve_rid("<rid>")`
- Check asset table name matches catalog

### "Execution failed"
- Check logs for error details
- Try `dry_run=True` to debug
- Verify all dependencies installed: `uv sync`

### "Config not found"
- Ensure `configs/__init__.py` loads all modules
- Check config name matches store registration
- Verify no syntax errors in config files
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

### Command Line
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

### Inspect Parameters First
```bash
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --inspect
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
        name="create-new-dataset-workflow",
        description="Complete workflow for creating a new dataset with workflow, execution, and proper cleanup",
    )
    def create_new_dataset_workflow_prompt() -> str:
        """Complete workflow for creating datasets from scratch."""
        return """# Complete Dataset Creation Workflow

This guide walks through the entire process of creating a new dataset in DerivaML,
including creating a workflow, execution context, datasets, and proper cleanup.

## Prerequisites
- Connected to a DerivaML catalog (`connect_catalog`)
- Data exists in domain tables (e.g., Images, Subjects)
- Know which records you want to include in the dataset

## Overview

Creating a dataset requires these components:
1. **Workflow**: Defines what type of operation this is (reusable)
2. **Execution**: A single run of the workflow (tracks this specific operation)
3. **Dataset**: The collection of data being created
4. **Upload**: Finalizes and persists everything

## Step 1: Check for Existing Workflow (Optional)

If you've created datasets before, you may already have a workflow:

```
find_workflows()
```

If a suitable workflow exists (e.g., "Dataset Curation" with type "Preprocessing"),
you can reuse it. Otherwise, create a new one in Step 2.

## Step 2: Create the Execution

The execution automatically creates or references a workflow:

```
create_execution(
    workflow_name="Dataset Curation",           # Descriptive name
    workflow_type="Preprocessing",              # From Workflow_Type vocabulary
    description="Create curated dataset for training"
)
```

**Available workflow types** (use `list_workflow_types()` to see all):
- `Preprocessing` - Data preparation and curation
- `Training` - Model training runs
- `Inference` - Running predictions
- `Annotation` - Adding labels/features

## Step 3: Ensure Element Types are Registered

Before adding records, their tables must be registered as dataset element types:

```
# Check what's already registered
list_dataset_element_types()

# Register tables if needed
add_dataset_element_type("Image")
add_dataset_element_type("Subject")
```

## Step 4: Create the Dataset

Create a new dataset within the execution context:

```
create_execution_dataset(
    description="Training images - batch 1",
    dataset_types=["Training"]                  # From Dataset_Type vocabulary
)
```

**Returns**: The new dataset's RID (e.g., "1-ABC"). Save this for the next step.

**Available dataset types** (use `list_vocabulary_terms("Dataset_Type")` to see all):
- `Training` - For model training
- `Testing` - For model evaluation
- `Validation` - For hyperparameter tuning
- `Complete` - Full dataset (often parent of train/test splits)

## Step 5: Add Members to the Dataset

Add records by their RIDs:

```
# Find records to add
query_table("Image", limit=100)

# Add them to the dataset
add_dataset_members("<dataset-rid>", [
    "<image-rid-1>",
    "<image-rid-2>",
    "<image-rid-3>"
])
```

Each call to `add_dataset_members` automatically increments the dataset's minor version.

## Step 6: Upload and Finalize (REQUIRED)

**Critical**: Nothing is persisted until you upload!

```
upload_execution_outputs()
```

This:
- Finalizes the execution record
- Persists all dataset relationships
- Records provenance (who created what, when)

## Step 7: Verify the Dataset

Confirm everything was created correctly:

```
# Check the dataset exists
get_dataset("<dataset-rid>")

# View members
list_dataset_members("<dataset-rid>")

# Check version
get_dataset_version_history("<dataset-rid>")

# Get URL to view in browser
get_chaise_url("<dataset-rid>")
```

## Complete Example: Single Dataset

```
# 1. Create execution (creates/references workflow automatically)
create_execution(
    "Image Dataset Creation",
    "Preprocessing",
    "Create initial training dataset from uploaded images"
)

# 2. Register element type if needed
add_dataset_element_type("Image")

# 3. Create the dataset
create_execution_dataset(
    "Training Images v1",
    ["Training"]
)
# Returns: "1-ABC"

# 4. Query for images to add
query_table("Image", columns=["RID", "Filename"], limit=50)
# Returns list of images with RIDs

# 5. Add images to dataset
add_dataset_members("1-ABC", [
    "2-D01", "2-D02", "2-D03", "2-D04", "2-D05"
])

# 6. Upload to finalize
upload_execution_outputs()

# 7. Verify
get_dataset("1-ABC")
list_dataset_members("1-ABC")
```

## Complete Example: Train/Test Split

```
# 1. Create execution
create_execution(
    "Train/Test Split",
    "Preprocessing",
    "Split images into 80/20 train/test sets"
)

# 2. Ensure element types registered
add_dataset_element_type("Image")

# 3. Create parent dataset
create_execution_dataset("Complete Image Set", ["Complete"])
# Returns: "1-PARENT"

# 4. Add all images to parent
add_dataset_members("1-PARENT", [
    "2-D01", "2-D02", "2-D03", "2-D04", "2-D05",
    "2-D06", "2-D07", "2-D08", "2-D09", "2-D10"
])

# 5. Create training dataset
create_execution_dataset("Training Set (80%)", ["Training"])
# Returns: "1-TRAIN"
add_dataset_members("1-TRAIN", [
    "2-D01", "2-D02", "2-D03", "2-D04",
    "2-D05", "2-D06", "2-D07", "2-D08"
])

# 6. Create test dataset
create_execution_dataset("Test Set (20%)", ["Testing"])
# Returns: "1-TEST"
add_dataset_members("1-TEST", ["2-D09", "2-D10"])

# 7. Upload to finalize
upload_execution_outputs()

# 8. Link children to parent
add_dataset_child("1-PARENT", "1-TRAIN")
add_dataset_child("1-PARENT", "1-TEST")

# 9. Verify
get_dataset("1-PARENT")  # Shows children
list_dataset_children("1-PARENT")
```

## Python API Equivalent

For reference, the same workflow in Python:

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration, Workflow

ml = DerivaML(hostname="example.org", catalog_id="1")

# Create execution with context manager
config = ExecutionConfiguration(
    workflow=Workflow(
        name="Dataset Curation",
        workflow_type="Preprocessing",
        description="Create curated dataset"
    )
)

with ml.create_execution(config) as exe:
    # Ensure element type registered
    ml.add_dataset_element_type("Image")

    # Create dataset
    dataset = exe.create_execution_dataset(
        description="Training Images v1",
        dataset_types=["Training"]
    )

    # Add members
    ml.add_dataset_members(dataset.rid, ["2-D01", "2-D02", "2-D03"])

# Upload AFTER context manager exits
exe.upload_execution_outputs()

# Verify
print(ml.get_dataset(dataset.rid))
```

## Common Issues

### "Element type not registered"
```
add_dataset_element_type("<table-name>")
```

### "Dataset not found after creation"
Did you call `upload_execution_outputs()`? Nothing persists without it.

### "Workflow type not found"
```
list_workflow_types()  # See available types
add_workflow_type("MyType", "Description")  # Add new type if needed
```

### "Dataset type not found"
```
list_vocabulary_terms("Dataset_Type")  # See available types
add_term("Dataset_Type", "MyType", "Description")  # Add new type
```

## Summary Checklist

- [ ] Connected to catalog
- [ ] Created execution with workflow name and type
- [ ] Registered element types for tables being used
- [ ] Created dataset with description and types
- [ ] Added members (records) to dataset
- [ ] Called `upload_execution_outputs()` to finalize
- [ ] Verified dataset exists and has correct members
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
# List all tables in the domain schema
list_tables()

# Get details about a specific table
get_table_schema("Image")

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

### Query Dataset Tables
```
get_dataset_table("<dataset-rid>", "Image", limit=500)
```

## Step 6: Vocabulary Lookups

### List Vocabulary Terms
```
list_vocabulary_terms("Dataset_Type")
list_vocabulary_terms("Workflow_Type")
```

### Find a Term
```
lookup_term("Dataset_Type", "Training")
lookup_term("Dataset_Type", "train")  # Works with synonyms
```

## Step 7: Feature Queries

### Find Features for a Table
```
find_features("Image")
```

### Get Feature Structure
```
lookup_feature("Image", "Diagnosis")
```

### Get All Feature Values
```
list_feature_values("Image", "Diagnosis")
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
# Get version history
get_dataset_version_history("<dataset-rid>")

# Query specific version
list_dataset_members("<dataset-rid>", version="1.0.0")
get_dataset_table("<dataset-rid>", "Image", version="1.0.0")
denormalize_dataset("<dataset-rid>", ["Image"], version="1.0.0")
```

## View in Web Interface

Get URLs to browse data in Chaise:

```
# URL for a table
get_chaise_url("Image")

# URL for a specific record
get_chaise_url("1-ABC")
```

## Complete Example: Explore a Dataset

```
# 1. Connect (if not already connected)
connect_catalog("example.org", "1")

# 2. List available datasets
list_datasets()

# 3. Get dataset details
get_dataset("1-ABC")

# 4. See what tables have data
list_dataset_members("1-ABC")
# Returns: {"Image": [...], "Subject": [...]}

# 5. Check table structure
get_table_schema("Image")

# 6. Query image data
query_table("Image", columns=["RID", "Filename", "Subject"], limit=10)

# 7. Get denormalized data for ML
denormalize_dataset("1-ABC", ["Image", "Subject", "Diagnosis"], limit=100)

# 8. View in browser
get_chaise_url("1-ABC")
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
list_tables()  # See available tables
```

### "Column not found"
```
get_table_schema("<table>")  # See column names
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
        name="customize-chaise-ui",
        description="Guide for customizing the Chaise web interface using annotations",
    )
    def customize_chaise_ui_prompt() -> str:
        """Guide for customizing Chaise UI with annotations."""
        return """# Customizing the Chaise Web Interface

This guide covers how to customize the Chaise web UI for your DerivaML catalog
using annotations. Annotations control display names, column visibility,
sorting, and more.

## Prerequisites
- Connected to a DerivaML catalog
- Understanding of your schema structure (`list_tables()`, `get_table_schema()`)

## Quick Start: Apply Default Annotations

DerivaML provides a convenience method for common customizations:

```
apply_catalog_annotations(
    navbar_brand_text="My ML Project",
    head_title="ML Catalog"
)
```

This configures:
- Navigation bar with organized menus
- Display settings (underscores as spaces)
- Default landing page
- Bulk upload configuration

## Step 1: Understand Display Contexts

Annotations are context-sensitive. Common contexts:

| Context | Where Used | Example |
|---------|-----------|---------|
| `compact` | Table listings | Search results |
| `detailed` | Record page | Single record view |
| `entry` | Forms | Create/edit forms |
| `filter` | Facet panel | Search filters |
| `*` | Default | Applies everywhere |

## Step 2: Customize Display Names

Make table and column names more readable:

### Table Display Names
Set via the `display` annotation on the table:
```json
{
  "tag:misd.isi.edu,2015:display": {
    "name": "Medical Images",
    "comment": "Collection of medical imaging data"
  }
}
```

### Column Display Names
Set via the `display` annotation on the column:
```json
{
  "tag:misd.isi.edu,2015:display": {
    "name": "File Name",
    "comment": "Original filename of the uploaded image"
  }
}
```

### Automatic Name Styling
Apply to catalog or schema for all nested elements:
```json
{
  "tag:misd.isi.edu,2015:display": {
    "name_style": {
      "underline_space": true,
      "title_case": true
    }
  }
}
```
This converts `image_file_name` to "Image File Name".

## Step 3: Configure Visible Columns

Control which columns appear in different contexts:

```json
{
  "tag:isrd.isi.edu,2016:visible-columns": {
    "compact": ["RID", "Name", "Status", "RCT"],
    "detailed": ["RID", "Name", "Description", "Status", "Subject", "RCT", "RMT"],
    "entry": ["Name", "Description", "Status", "Subject"]
  }
}
```

**Tips:**
- `compact`: Keep it short (4-6 columns) for listings
- `detailed`: Show all relevant columns
- `entry`: Only editable columns (exclude system columns)

## Step 4: Configure Facets (Search Filters)

Add facets to the filter panel:

```json
{
  "tag:isrd.isi.edu,2016:visible-columns": {
    "filter": {
      "and": [
        {"source": "Status", "open": true},
        {"source": "Subject", "entity": true},
        {"source": "RCT", "ux_mode": "ranges"}
      ]
    }
  }
}
```

**Facet options:**
- `open`: Expanded by default
- `entity`: Show as linked entity (for foreign keys)
- `ux_mode`: `choices` (checklist), `ranges` (slider), `check_presence`
- `choices`: Pre-selected values
- `markdown_name`: Custom facet title

## Step 5: Configure Related Tables

Control which related tables appear on the record page:

```json
{
  "tag:isrd.isi.edu,2016:visible-foreign-keys": {
    "detailed": [
      ["schema", "Image_Subject_fkey"],
      ["schema", "Diagnosis_Image_fkey"]
    ]
  }
}
```

This shows tables that reference the current record.

## Step 6: Configure Table Display

Set default sorting and row display:

```json
{
  "tag:isrd.isi.edu,2016:table-display": {
    "row_order": [
      {"column": "RCT", "descending": true}
    ],
    "page_size": 25
  }
}
```

## Step 7: Configure Asset Columns

Mark columns that contain file URLs:

```json
{
  "tag:isrd.isi.edu,2017:asset": {
    "url_pattern": "/hatrac/data/{{{MD5}}}/{{{Filename}}}",
    "filename_column": "Filename",
    "byte_count_column": "Length",
    "md5_column": "MD5",
    "browser_upload": true
  }
}
```

This enables:
- Download links
- File previews (for images)
- Drag-and-drop upload

## Common Customization Patterns

### Hide System Columns in Entry Forms
```json
{
  "tag:isrd.isi.edu,2016:visible-columns": {
    "entry": ["Name", "Description", "Value"]
  }
}
```
Excludes RID, RCT, RMT, RCB, RMB from forms.

### Show Friendly NULL Display
```json
{
  "tag:misd.isi.edu,2015:display": {
    "show_null": {
      "*": "N/A",
      "entry": ""
    }
  }
}
```

### Custom Column Formatting
```json
{
  "tag:isrd.isi.edu,2016:column-display": {
    "*": {
      "markdown_pattern": "**{{{_self}}}**"
    }
  }
}
```

### Foreign Key Display Name
```json
{
  "tag:isrd.isi.edu,2016:foreign-key": {
    "to_name": "Subject",
    "from_name": "Images"
  }
}
```

## Applying Annotations

### Option 1: Via DerivaML (Recommended for ML projects)

The `apply_catalog_annotations()` tool sets up sensible defaults:
```
apply_catalog_annotations("My Project", "Project Catalog")
```

### Option 2: Via DERIVA Workbench (GUI)

1. Launch `deriva-workbench`
2. Connect to your catalog
3. Browse to the table/column
4. Right-click annotations → Add
5. Edit using graphical or JSON editor
6. Click Update to save

### Option 3: Via Python API

```python
from deriva.core import ErmrestCatalog
import deriva.core.ermrest_model as em

catalog = ErmrestCatalog('https', 'example.org', '1')
model = catalog.getCatalogModel()

# Set table annotation
table = model.table('domain', 'Image')
table.annotations[em.tag.visible_columns] = {
    "compact": ["RID", "Name", "Subject"],
    "detailed": "*"
}

# Apply changes
model.apply()
```

## Verification

After applying annotations:

1. **Get Chaise URL**: `get_chaise_url("Image")`
2. **Open in browser** to verify changes
3. **Test different contexts**: listings, record page, forms

## Troubleshooting

### Changes not appearing
- Clear browser cache
- Verify annotation was applied (check in workbench)
- Check for typos in annotation keys

### Column not showing
- Check `visible-columns` annotation
- Verify column exists in table
- Check context (compact vs detailed)

### Facet not working
- Verify source column exists
- Check facet configuration syntax
- Try simpler configuration first

## Best Practices

1. **Start with defaults**: Use `apply_catalog_annotations()` first
2. **Customize incrementally**: Change one thing at a time
3. **Test in Chaise**: Verify each change visually
4. **Document changes**: Keep notes on customizations
5. **Use workbench**: For complex visual editing
6. **Backup annotations**: Use dump/restore in workbench

## Related Resources

- `deriva-ml://docs/chaise-annotations` - Annotation reference
- `deriva-ml://docs/ermrest-model-management` - Schema management
- DERIVA Workbench - GUI annotation editor
"""

    @mcp.prompt(
        name="pre-execution-checklist",
        description="REQUIRED checklist before running ML executions - verifies git state and version, requires user confirmation",
    )
    def pre_execution_checklist_prompt() -> str:
        """Pre-execution checklist requiring user confirmation."""
        return """# Pre-Execution Checklist (REQUIRED)

**IMPORTANT**: This checklist MUST be completed before running any ML execution.
You MUST get explicit user confirmation before proceeding.

## Why This Matters

For reproducibility, every execution must be traceable to:
1. A specific **git commit** (exact code version)
2. A **semantic version tag** (human-readable version)
3. A **clean working tree** (no uncommitted changes)

Without this, you cannot reproduce results or know what code produced them.

---

## Step 1: Check Git Status

Run this command and report the results to the user:

```bash
git status --porcelain
```

### If output is EMPTY (clean):
✅ Working tree is clean - proceed to Step 2

### If output shows changes:
⚠️ **STOP** - There are uncommitted changes!

Report to the user:
- Which files are modified/untracked
- Ask: "There are uncommitted changes. Should I commit these before proceeding?"

If user confirms, commit the changes:
```bash
git add .
git commit -m "<descriptive message>"
```

---

## Step 2: Check Current Version

Run this command:

```bash
git describe --tags --always 2>/dev/null || echo "No tags found"
```

Report the current version to the user.

### If version ends with `-g<hash>` (e.g., `v1.2.3-5-gabcd123`):
⚠️ There are commits since the last version tag.

Ask the user: "The current version is `<version>`. Should I bump the version before proceeding?"

If user confirms, bump the version:
```bash
# For bug fixes/small changes:
uv run bump-my-version bump patch

# For new features/config changes:
uv run bump-my-version bump minor

# For breaking changes:
uv run bump-my-version bump major
```

Then push the tag:
```bash
git push && git push --tags
```

### If version is a clean tag (e.g., `v1.2.3`):
✅ Version is current - proceed to Step 3

---

## Step 3: Verify Lock File

Check that `uv.lock` is committed and current:

```bash
# Regenerate lock file
uv lock

# Check if it changed
git status --porcelain uv.lock
```

### If `uv.lock` changed:
⚠️ Lock file was out of date. Commit it:
```bash
git add uv.lock
git commit -m "Update uv.lock"
```

---

## Step 4: Final Verification

Run these commands and report results:

```bash
# Show final commit
git log -1 --oneline

# Show version
git describe --tags --always

# Confirm clean state
git status --porcelain
```

---

## Step 5: USER CONFIRMATION REQUIRED

**You MUST ask the user explicitly:**

> I have verified the repository state:
> - **Commit**: `<commit hash and message>`
> - **Version**: `<version tag>`
> - **Status**: Clean working tree
>
> **Is it OK to proceed with the ML execution?**

**Wait for explicit user confirmation** (e.g., "yes", "proceed", "ok").

### If user says NO:
- Ask what changes they want to make first
- Do NOT proceed with execution

### If user says YES:
- ✅ Proceed with the execution
- The execution will be traceable to this exact code version

---

## Quick Reference

| Check | Command | Expected |
|-------|---------|----------|
| Clean tree | `git status --porcelain` | Empty output |
| Current version | `git describe --tags` | Clean tag (no `-g<hash>`) |
| Lock file | `git status uv.lock` | Not modified |

## After Execution

After the execution completes, remind the user:
- The execution is linked to version `<version>`
- Results can be reproduced by checking out this version
- Consider tagging successful experiments

---

## Example Interaction

```
Assistant: Let me verify the repository state before running the execution.

[Runs git status --porcelain]
Output: M src/model.py

There are uncommitted changes:
- Modified: src/model.py

Should I commit these changes before proceeding?

User: Yes, commit them with message "Fix learning rate bug"

[Commits changes]
[Runs git describe --tags]
Output: v1.2.3-1-gabcd123

The version is v1.2.3-1-gabcd123, which indicates 1 commit since v1.2.3.
Should I bump the version? (patch/minor/major)

User: Bump patch

[Runs bump-my-version bump patch]
[Pushes tags]

I have verified the repository state:
- Commit: abcd123 Fix learning rate bug
- Version: v1.2.4
- Status: Clean working tree

Is it OK to proceed with the ML execution?

User: Yes, proceed

[Proceeds with execution]
```
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

## Step 1: Find Assets

### List All Assets in a Table

```
list_assets("Image")       # All images
list_assets("Model")       # All models
```

### Search with Filters

```
find_assets()                              # All assets in catalog
find_assets(asset_table="Image")           # Images only
find_assets(asset_type="Training_Data")    # By type
```

### Look Up Specific Asset

```
lookup_asset("<asset-rid>")
```

Returns detailed info: RID, table, filename, URL, types, execution that created it.

## Step 2: Check Asset Provenance

Find which execution created an asset:

```
list_asset_executions("<asset-rid>", asset_role="Output")
```

Find which executions used an asset as input:

```
list_asset_executions("<asset-rid>", asset_role="Input")
```

## Step 3: Manage Asset Types

### View Available Types

```
list_asset_types()
```

### Add New Type

```
add_asset_type("Segmentation_Mask", "Binary mask images for segmentation")
```

## Step 4: Create Asset Tables

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

## Step 5: Upload Assets via Execution

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

- Use `lookup_asset` to get full details about any asset
- Use `list_asset_executions` to trace provenance
- Asset types help organize and filter assets
- Always upload through executions for provenance
- Use `find_assets` for bulk discovery operations

## Related Tools

- `list_assets()` - List assets in a table
- `find_assets()` - Search assets with filters
- `lookup_asset()` - Get asset details
- `list_asset_executions()` - Asset provenance
- `asset_file_path()` - Register output files
- `upload_execution_outputs()` - Upload registered files
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

# List available datasets
find_datasets()

# Check if dataset exists
lookup_dataset("<rid>")
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
# Check version history
get_dataset_version_history("<rid>")

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
# List all feature names
list_feature_names()

# Find features for a specific table
find_features("Image")

# Get feature details
lookup_feature("Image", "Diagnosis")
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
# List available terms
list_vocabulary_terms("<vocabulary-name>")

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
list_executions()
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
assets = ml.find_assets(asset_type="Model")  # Returns Iterable[Asset]
features = ml.find_features("Image")      # Returns Iterable[Feature]
workflows = ml.find_workflows()           # Returns list[Workflow]
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
ml.add_synonym("Asset_Type", "New_Type", "alias")
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
ml.remove_synonym("Asset_Type", "Term", "old_alias")
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

| Python API | MCP Tool |
|------------|----------|
| `lookup_dataset(rid)` | `lookup_dataset(dataset_rid)` |
| `find_datasets(deleted=True)` | `find_datasets(include_deleted=True)` |
| `list_dataset_members()` | `list_dataset_members(dataset_rid)` |

## Tips for API Discovery

1. **Use `find_*` when you don't know if items exist**
2. **Use `lookup_*` when you expect the item to exist**
3. **Use `list_*` for enumeration within a context**
4. **Use `get_*` when you need transformed output**
5. **Check `list_feature_names()` before creating features**
6. **Check `list_vocabulary_terms()` before adding terms**
"""
