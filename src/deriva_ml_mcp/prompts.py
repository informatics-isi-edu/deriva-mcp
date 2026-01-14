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

## Step 1: Create the Execution

Create an execution to track your workflow run:

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

## Step 2: Start Timing

Begin the execution timer:

```
start_execution()
```

## Step 3: Do Your ML Work

This is where you run your actual ML pipeline:

- **Download input data**: Use `download_execution_dataset(dataset_rid)` to get data
- **Process/train/infer**: Run your ML code
- **Register outputs**: For each output file, call `asset_file_path()`:

```
asset_file_path(
    asset_name="<asset-table>",  # e.g., "Model", "Execution_Metadata"
    file_name="<path-or-name>",  # Existing file path OR new filename
    asset_types=["<type>"]       # Optional: from Asset_Type vocabulary
)
```

The returned path is where to write/read the file.

## Step 4: Stop Timing

When your workflow completes:

```
stop_execution()
```

## Step 5: Upload Outputs (REQUIRED)

Upload all registered files to the catalog:

```
upload_execution_outputs()
```

**Important**: Files are NOT persisted until this is called!

## Example: Training Workflow

```
# 1. Create execution
create_execution("CIFAR-10 ResNet", "Training", "Train ResNet50 on CIFAR-10", ["1-ABC"])

# 2. Start timing
start_execution()

# 3. Get training data
download_execution_dataset("1-ABC")

# 4. [Run training code here...]

# 5. Register model output
asset_file_path("Model", "/tmp/model.pt", ["Trained Model"])

# 6. Register metrics
asset_file_path("Execution_Metadata", "metrics.json")

# 7. Stop timing
stop_execution()

# 8. Upload everything
upload_execution_outputs()
```

## Tips

- Use `get_execution_info()` to check current execution status
- Use `update_execution_status(status, message)` for progress updates
- Use `restore_execution(rid)` to resume a previous execution
- Use `list_executions()` to see past workflow runs
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
# List features for a table
list_features("<table-name>")

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

```
# First, create and start an execution
create_execution("Labeling Run", "Annotation", "Manual image labeling")
start_execution()

# Add labels
add_feature_value("Image", "Diagnosis", "<image-rid>", "Normal")
add_feature_value("Image", "Diagnosis", "<image-rid-2>", "Abnormal")

# Complete execution
stop_execution()
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

```
# 1. Create vocabulary
create_vocabulary("Image_Class", "Image classification categories")
add_term("Image_Class", "Cat", "Image contains a cat")
add_term("Image_Class", "Dog", "Image contains a dog")
add_term("Image_Class", "Other", "Other content")

# 2. Create feature
create_feature("Image", "Classification", "Image class label", terms=["Image_Class"])

# 3. Start labeling execution
create_execution("Manual Labeling", "Annotation", "Label training images")
start_execution()

# 4. Add labels (typically in a loop over images)
add_feature_value("Image", "Classification", "1-ABC", "Cat")
add_feature_value("Image", "Classification", "1-DEF", "Dog")
add_feature_value("Image", "Classification", "1-GHI", "Cat")

# 5. Complete
stop_execution()
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

Datasets must be created through an execution for provenance:

```
# Start an execution for dataset creation
create_execution("Create Training Set", "Preprocessing", "Curate training data")
start_execution()

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

## Step 4: Complete the Execution

```
stop_execution()
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

```
# Create parent "Complete" dataset first (via execution)
# Then create children
create_execution("Create Train/Test Split", "Preprocessing", "Split data")
start_execution()

# Create training subset
create_execution_dataset("Training subset (80%)", ["Training"])
# Add training members...

# Create testing subset
create_execution_dataset("Testing subset (20%)", ["Testing"])
# Add testing members...

stop_execution()
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

```
# 1. Start execution
create_execution("Dataset Curation", "Preprocessing", "Create ML datasets")
start_execution()

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

# 7. Link as children
stop_execution()
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

**CRITICAL: Commit all changes before running!**

```bash
# Check for uncommitted changes
git status

# Stage and commit all changes
git add .
git commit -m "Configure experiment: <description>"

# Bump version (choose appropriate level)
uv run bump-version patch   # Bug fixes, small tweaks
uv run bump-version minor   # New features, config changes
uv run bump-version major   # Breaking changes

# Push changes and tags
git push && git push --tags

# Verify version
uv run python -m setuptools_scm
```

**Check:**
- [ ] All code changes committed
- [ ] All config changes committed
- [ ] Version bumped appropriately
- [ ] Tags pushed to remote
- [ ] `uv.lock` is current and committed

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
