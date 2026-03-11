# Creating and Populating a Feature

Features in DerivaML link a target table (e.g., Image, Subject) to vocabulary terms, creating a structured labeling system for ML. A feature is essentially a many-to-many relationship between domain records and vocabulary terms, with provenance tracking through executions.

## Table of Contents

1. [Concepts](#concepts) — Feature, target table, vocabulary, feature value
2. [Step 1: Check Existing Features](#step-1-check-existing-features)
3. [Step 2: Create a Vocabulary](#step-2-create-a-vocabulary-if-needed)
4. [Step 3: Create the Feature](#step-3-create-the-feature) — With terms, assets, both, metadata columns
5. [Step 4: Add Feature Values](#step-4-add-feature-values) — MCP tools and Python API
6. [Step 5: Query Feature Values](#step-5-query-feature-values)
7. [Complete Example: Image Classification Labels](#complete-example-image-classification-labels)
8. [Managing Features](#managing-features) — Delete, list
9. [Tips](#tips)

---

## Concepts

- **Feature**: A named labeling dimension (e.g., "Tumor Classification", "Image Quality Score").
- **Target table**: The table whose records are being labeled (e.g., Image, Subject).
- **Vocabulary**: A controlled set of terms used as labels (e.g., "Benign", "Malignant", "Unknown").
- **Feature value**: An individual label assignment -- one record labeled with one term, within one execution.

## Step 1: Check Existing Features

```
# List existing features
query_table(table_name="Feature")

# Check existing vocabularies
query_table(table_name="Diagnosis")
query_table(table_name="Image_Quality")
```

## Step 2: Create a Vocabulary (if needed)

If your feature needs a new set of terms, create a vocabulary first.

```
# Create a new vocabulary table
create_vocabulary(
    vocabulary_name="Tumor_Grade",
    comment="Histological grading of tumor samples"
)
```

Then add terms to the vocabulary:

```
# Add individual terms with descriptions
add_term(
    vocabulary_name="Tumor_Grade",
    term_name="Grade I",
    description="Well-differentiated, low grade"
)

add_term(
    vocabulary_name="Tumor_Grade",
    term_name="Grade II",
    description="Moderately differentiated, intermediate grade"
)

add_term(
    vocabulary_name="Tumor_Grade",
    term_name="Grade III",
    description="Poorly differentiated, high grade"
)

add_term(
    vocabulary_name="Tumor_Grade",
    term_name="Grade IV",
    description="Undifferentiated, high grade"
)
```

**Always provide meaningful descriptions for terms.** They appear in the UI and help users understand what each label means.

You can also add synonyms for terms:

```
add_synonym(vocabulary_name="Tumor_Grade", term_name="Grade I", synonym="Low Grade")
add_synonym(vocabulary_name="Tumor_Grade", term_name="Grade III", synonym="High Grade")
```

## Step 3: Create the Feature

Create the feature, linking a target table to vocabulary terms.

### Feature with vocabulary terms

```
create_feature(
    table_name="Image",
    feature_name="Tumor_Classification",
    terms=["Tumor_Grade"],         # Vocabulary table(s) to use as labels
    comment="Classification of tumor grade from histology images"
)
```

This creates:
- A `Tumor_Classification` feature record
- A `Tumor_Classification_Feature_Value` association table linking Image records to Tumor_Grade terms

### Feature with asset terms

If the feature values are files (e.g., segmentation masks, annotation overlays):

```
create_feature(
    table_name="Image",
    feature_name="Segmentation_Mask",
    assets=["Mask_Image"],         # Asset table(s) to use as values
    comment="Pixel-level segmentation masks for images"
)
```

### Feature with both terms and assets

```
create_feature(
    table_name="Image",
    feature_name="Annotated_Region",
    terms=["Region_Label"],
    assets=["Region_Overlay"],
    comment="Labeled regions with overlay images"
)
```

### Feature with metadata columns

Features can include additional columns for structured metadata like confidence scores, reviewer references, or notes. Use the `metadata` parameter:

```
# Feature with a confidence score
create_feature(
    table_name="Image",
    feature_name="Diagnosis",
    terms=["Diagnosis_Type"],
    metadata=[{"name": "confidence", "type": {"typename": "float4"}}],
    comment="Diagnosis with confidence score"
)

# Feature referencing another table (e.g., a Reviewer table)
create_feature(
    table_name="Image",
    feature_name="Review",
    terms=["Review_Status"],
    metadata=["Reviewer"],
    comment="Review annotations with reviewer tracking"
)
```

Each metadata item can be:
- **A string**: Treated as a table name, adds a foreign key reference to that table
- **A dict**: Column definition with `name` and `type` keys. Valid type names: `text`, `int2`, `int4`, `int8`, `float4`, `float8`, `boolean`, `date`, `timestamp`, `timestamptz`, `json`, `jsonb`. Optional keys: `nullok`, `default`, `comment`.

## Step 4: Add Feature Values

Feature values require an active execution for provenance tracking. Every label assignment is tied to the execution that created it.

### MCP Tools

The MCP execution tools operate on the **active execution** -- you don't pass execution RIDs to start/stop/upload.

```
# Step 1: Create execution (sets it as active)
create_execution(
    workflow_name="Manual Tumor Grading",
    workflow_type="Annotation",
    description="Expert pathologist tumor grading session"
)

# Step 2: Start timing
start_execution()

# Step 3: Add feature values
# For simple features (single term or asset column), use add_feature_value:
# Pass a list of entries, each with target_rid and value.
add_feature_value(
    table_name="Image",
    feature_name="Tumor_Classification",
    entries=[
        {"target_rid": "2-IMG1", "value": "Grade II"},
        {"target_rid": "2-IMG2", "value": "Grade III"}
    ]
)

# For features with multiple columns (e.g., term + confidence), use add_feature_value_record:
# Pass a list of entries, each with target_rid plus column values.
add_feature_value_record(
    table_name="Image",
    feature_name="Diagnosis",
    entries=[
        {"target_rid": "2-IMG1", "Diagnosis_Type": "Normal", "confidence": 0.95}
    ]
)

# Step 4: Stop timing
stop_execution()

# Step 5: Upload outputs
upload_execution_outputs()
```

### Python API with Context Manager

```python
from deriva_ml import DerivaML, ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)

# Look up the workflow by URL or RID
workflow = ml.lookup_workflow_by_url("https://github.com/my-org/my-repo")

config = ExecutionConfiguration(
    workflow=workflow,
    datasets=[],
    assets=[],
    description="Expert pathologist tumor grading"
)

with ml.create_execution(config) as exe:
    # Look up the feature and get its record class
    feature = exe.catalog.lookup_feature("Image", "Tumor_Classification")
    RecordClass = feature.feature_record_class()

    # Create feature records
    records = [
        RecordClass(Image="2-IMG1", Tumor_Grade="Grade II"),
        RecordClass(Image="2-IMG2", Tumor_Grade="Grade III"),
    ]

    # Bulk feature values
    for image_rid, grade in labeling_results.items():
        records.append(RecordClass(Image=image_rid, Tumor_Grade=grade))

    # Add all records in batch (execution RID set automatically)
    exe.add_features(records)
```

## Step 5: Query Feature Values

After populating feature values, query them for analysis or training.

```
# Get all feature values for a feature
query_table(table_name="Tumor_Classification_Feature_Value")

# Get feature values for a specific image
query_table(
    table_name="Tumor_Classification_Feature_Value",
    filters={"Image": "2-IMG1"}
)

# Get all images with a specific grade
query_table(
    table_name="Tumor_Classification_Feature_Value",
    filters={"Tumor_Grade": "Grade III"}
)
```

## Complete Example: Image Classification Labels

```
# 1. Create vocabulary
create_vocabulary(
    vocabulary_name="Cell_Type",
    comment="Cell type classifications for microscopy images"
)

add_term(vocabulary_name="Cell_Type", term_name="Epithelial", description="Epithelial cells lining surfaces and cavities")
add_term(vocabulary_name="Cell_Type", term_name="Stromal", description="Connective tissue support cells")
add_term(vocabulary_name="Cell_Type", term_name="Immune", description="Immune system cells including lymphocytes and macrophages")
add_term(vocabulary_name="Cell_Type", term_name="Necrotic", description="Dead or dying cells")
add_term(vocabulary_name="Cell_Type", term_name="Artifact", description="Non-biological artifact in image")

# 2. Create the feature
create_feature(
    table_name="Image",
    feature_name="Cell_Classification",
    terms=["Cell_Type"],
    comment="Primary cell type visible in microscopy image"
)

# 3. Add values within an execution
create_execution(
    workflow_name="Expert Cell Annotation",
    workflow_type="Annotation",
    description="Expert cell type annotation - batch 1"
)
start_execution()

add_feature_value(
    table_name="Image",
    feature_name="Cell_Classification",
    entries=[
        {"target_rid": "2-IMG1", "value": "Epithelial"},
        {"target_rid": "2-IMG2", "value": "Immune"}
    ]
)

stop_execution()
upload_execution_outputs()
```

## Managing Features

### Delete a feature
```
delete_feature(table_name="Image", feature_name="Tumor_Classification")
```
This removes the feature and its feature value table. Existing data will be lost.

### List all features
```
query_table(table_name="Feature")
```

## Tips

- **Feature values require an active execution.** This is a hard requirement for provenance tracking.
- **Always provide descriptions** for vocabulary terms -- they appear in the UI and help annotators.
- Use separate executions for different labeling sessions or annotators to track who labeled what.
- Multiple features can target the same table (e.g., an Image can have both "Tumor_Classification" and "Image_Quality" features).
- A single record can have multiple values for the same feature (e.g., an image labeled by multiple annotators).
- Use `add_synonym` to make vocabulary terms discoverable under alternative names.
- Use `add_feature_value` for simple features (single term/asset) and `add_feature_value_record` for features with multiple columns. Both accept a batch `entries` list for efficient insertion.
- Feature values are queryable like any other table, making them easy to use for training data preparation.
