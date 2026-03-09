---
name: debug-bag-contents
description: "Diagnose missing data in DerivaML dataset bag (BDBag) exports — FK traversal issues, missing tables, materialization problems, export timeouts. Use when a downloaded dataset bag is missing expected records, images, or feature values."
---

# Debugging Dataset Bag Contents

When a dataset bag export is missing expected data, follow this step-by-step diagnostic process to identify and fix the issue.

---

## Step 1: Check Dataset Members

Dataset members are the explicit records that belong to a dataset. If data is missing from a bag, the first question is whether the right members are in the dataset.

- **Resource**: Check the dataset resource to see the dataset's summary and member counts.
- **Tool**: `list_dataset_members` with the dataset RID to get the full list of members, grouped by table.
- Verify that the records you expect are listed as members. If they are missing, add them with `add_dataset_members`.

---

## Step 2: Check Element Type Registration

Every table that contributes members to a dataset must be registered as a **dataset element type**. If a table is not registered, its members will be silently excluded from the bag.

- **Resource**: Check the dataset element types resource to see which tables are registered for this dataset type.
- **Tool**: `add_dataset_element_type` to register a table as an element type if it is missing. You need to specify the dataset type and the table name.
- Common tables that should be registered: `Subject`, `Observation`, `Image` (or other asset tables), and any custom tables whose records appear as dataset members.

---

## Step 3: Preview Bag Export Paths

Before downloading a full bag, preview what the export will contain.

- **Resource**: Check the dataset bag preview resource to see the projected file paths and record counts per table.
- This preview shows which tables will be included and how many rows each will have, without actually downloading anything.
- Compare the preview counts against your expectations to spot discrepancies early.

---

## Step 4: Understand FK Path Traversal

The bag export algorithm uses foreign key (FK) path traversal to determine which related records to include. Understanding this is critical for diagnosing missing data.

### Key rules:
1. **Starting points are dataset members only from registered element types.** Records in tables that are not registered as element types will not serve as starting points for traversal, even if they are dataset members.
2. **FK traversal follows both directions.** From each starting point record, the export follows foreign keys both outward (this table references another) and inward (another table references this one).
3. **Vocabulary table endpoints are exported separately.** Vocabulary/controlled-vocabulary tables encountered during traversal are collected and exported in their own section of the bag, not inline with the data tables.
4. **Traversal depth is bounded.** The export does not follow FK chains indefinitely. It follows direct FK relationships from the member records.

### How traversal works in practice:
- If `Subject` is a registered element type and you have Subject members, the export will:
  - Include those Subject records.
  - Follow FKs from Subject to related tables (e.g., Subject_Phenotype).
  - Follow FKs pointing back to Subject from other tables (e.g., Image.Subject_RID -> Subject).
  - Export vocabulary terms referenced by any included records.

---

## Step 5: Diagnose Common Scenarios

### Scenario: Images missing from a Subject-only dataset

**Problem**: Dataset has Subject members but the exported bag does not include the associated Image records.

**Diagnosis**:
- Images are in a separate asset table with an FK to Subject.
- The FK traversal should find Images that reference the Subject members.

**Fix checklist**:
1. Verify the Image table has a direct FK to Subject (not through an intermediate table).
2. If the FK path goes through an intermediate table (e.g., `Observation`), that intermediate table may need to be registered as an element type, or intermediate records need to be added as members.
3. Alternatively, add the Image records directly as dataset members and register the Image table as an element type.

### Scenario: Observation data missing

**Problem**: Observations associated with Subjects are not in the bag.

**Diagnosis**:
- Check whether Observation has a direct FK to Subject.
- If yes, the FK traversal from Subject members should pick up Observations.
- If not, the path may be indirect and not traversed.

**Fix**:
- Add Observation records as explicit dataset members and register `Observation` as an element type.
- Or ensure there is a direct FK link between the tables.

### Scenario: Vocabulary terms missing

**Problem**: Controlled vocabulary values referenced by data records are not in the bag.

**Diagnosis**:
- Vocabulary terms are exported separately from data tables.
- Check that the vocabulary table is properly configured as a vocabulary (not a regular table).

**Fix**:
- Vocabulary terms referenced by included records should be automatically exported. If they are missing, verify the FK relationship between the data table and the vocabulary table is intact.
- **Tool**: `get_table` on the vocabulary table to confirm its structure.

---

## Step 6: Download and Validate the Bag

Use the validation tool to get a detailed comparison of expected vs. actual bag contents.

- **Tool**: `validate_dataset_bag` with the dataset RID.
  - Returns a **per-table comparison** showing:
    - Expected RIDs (based on dataset members and FK traversal).
    - Actual RIDs present in the downloaded bag.
    - **Missing RIDs**: records that should be in the bag but are not.
    - **Extra RIDs**: records in the bag that were not expected (usually not a problem but worth investigating).
  - Use the missing RIDs to identify exactly which records are being dropped and from which tables.

---

## Step 7: Check FK Paths for All Element Types

For each registered element type, examine the FK paths that the export will follow.

- **Resource**: Check the FK path resource for each element type to see the full traversal graph.
- Look for:
  - **Missing links**: Tables you expect to be reachable but are not connected by FKs.
  - **Indirect paths**: FK chains that go through intermediate tables, which may not be traversed if those intermediates are not included.
  - **Circular references**: These are handled correctly but may cause confusion when reading the path graph.

---

## Step 8: Fix Common Issues

### Deep join timeouts
**Problem**: FK traversal through many intermediate tables causes slow exports or timeouts.

**Fix**: Add records from intermediate tables as direct dataset members rather than relying on deep FK traversal. This flattens the traversal and speeds up the export.

### Missing element type registration
**Problem**: Records from a table are added as members but the table is not a registered element type, so those records are ignored during export.

**Fix**:
- **Tool**: `add_dataset_element_type` to register the table.
- Then re-export the bag.

### Stale dataset version
**Problem**: The bag reflects an older version of the dataset, missing recently added members.

**Fix**:
- **Tool**: `increment_dataset_version` to create a new version that captures current membership.
- Re-export the bag after incrementing.

### Records exist but FK not established
**Problem**: Related records exist in the catalog but are not linked via FK to the member records.

**Fix**:
- Check the FK columns on the related records. Ensure they contain the correct RID values pointing to the dataset member records.
- **Tool**: `query_table` with filters to verify FK column values.

---

## Quick Diagnostic Checklist

Use this checklist when data is missing from a bag:

1. **Are the records dataset members?**
   - `list_dataset_members` -- check if expected records appear.
   - If not: `add_dataset_members`.

2. **Is the table a registered element type?**
   - Check element types resource.
   - If not: `add_dataset_element_type`.

3. **Is there a direct FK path?**
   - Check FK paths resource for the element type.
   - If not: add intermediate records as members, or restructure FKs.

4. **Does validation show the discrepancy?**
   - `validate_dataset_bag` -- look at missing RIDs per table.

5. **Is the version current?**
   - `increment_dataset_version` if members were recently changed.

6. **Preview before full download.**
   - Check the bag preview resource to confirm expected counts before downloading.

## Related Tools

| Tool | Purpose |
|------|---------|
| `list_dataset_members` | List all members of a dataset |
| `add_dataset_members` | Add records to a dataset |
| `delete_dataset_members` | Remove records from a dataset |
| `add_dataset_element_type` | Register a table as dataset element type |
| `validate_dataset_bag` | Validate bag contents against expectations |
| `increment_dataset_version` | Bump dataset version after changes |
| `get_dataset_spec` | View dataset specification |
| `download_dataset` | Download the dataset bag |
| `denormalize_dataset` | Flatten dataset for analysis |
| `query_table` | Inspect FK column values |
| `get_table` | Check table schema and FK relationships |
