---
name: troubleshoot-execution
description: "ALWAYS use when any DerivaML execution fails, errors, gets stuck, or produces unexpected results. Covers authentication errors, missing files, stuck 'Running' status, version mismatches, permission denied, upload timeouts, and dataset download failures."
user-invocable: false
disable-model-invocation: true
---

# Troubleshooting DerivaML Executions

This guide covers common problems encountered when running DerivaML executions and their solutions.

---

## Problem: "No Active Execution"

**Symptom**: Tools that require an execution context (like `asset_file_path`, `upload_execution_outputs`) fail with an error about no active execution.

**Cause**: The execution was not properly started, or you are outside the execution context.

**Solution**:
- In Python, always use the context manager pattern:
  ```python
  from deriva_ml import DerivaML, ExecutionConfiguration

  with ml.create_execution(config) as exe:
      # All execution work goes here
  ```
- With MCP tools, ensure you called `start_execution()` before attempting execution-scoped operations.
- If the execution was started but the error persists, the execution may have been stopped or may have failed. Check with `get_execution_info`.

---

## Problem: "Files Not Uploaded"

**Symptom**: Execution completes but asset files are not visible in the catalog.

**Cause**: `upload_execution_outputs` was not called, or files were written to the wrong path.

**Solution**:
1. Call `upload_execution_outputs()` **after** the `with` block exits in Python, not inside it. With MCP tools, call it after `stop_execution()`.
2. Ensure files are written to the **exact path** returned by `asset_file_path()`. Writing to any other directory will cause the upload to miss those files.
3. Verify the file actually exists at the path before uploading:
   ```python
   path = exe.asset_file_path("Execution_Asset", "output.csv")
   # Write file to `path`
   # Verify: os.path.exists(path) should be True
   ```
4. Check that the execution is still in `Running` status when you attempt the upload. If it was already stopped or failed, uploads will not work.

---

## Problem: "Dataset Not Found"

**Symptom**: Attempting to use a dataset RID returns an error or empty result.

**Cause**: Wrong catalog connection, dataset was deleted, or the RID is incorrect.

**Solution**:
- Verify you are connected to the correct catalog with `connect_catalog` or check the active catalog.
- Check the dataset resources to list available datasets.
- Use `validate_rids` to confirm the RID is valid and belongs to a dataset table.
- If the dataset was recently created, it should be visible immediately -- there is no propagation delay.

---

## Problem: "Invalid RID"

**Symptom**: A tool rejects a RID value or returns "not found".

**Cause**: The RID is malformed, belongs to a different table than expected, or refers to a deleted record.

**Solution**:
- **Tool**: `validate_rids` to check whether the RID exists and what table it belongs to.
- RIDs are case-sensitive alphanumeric strings (e.g., `1-A2B3`). Ensure there are no extra spaces or characters.
- If the RID comes from a different catalog, it will not resolve in the current catalog. Verify you are connected to the right catalog.

---

## Problem: "Permission Denied"

**Symptom**: Operations fail with authentication or authorization errors.

**Cause**: Your credentials have expired or you lack the required role.

**Solution**:
- Re-authenticate using `deriva-globus-auth-utils`:
  ```bash
  deriva-globus-auth-utils login --host <hostname>
  ```
- Check that your user account has the necessary group membership for the operation (read, write, or admin).
- Some operations (like creating tables or modifying schemas) require elevated permissions.

---

## Problem: "Version Mismatch"

**Symptom**: Dataset contents do not match expectations, or a workflow references an outdated dataset version.

**Cause**: The dataset was modified after the version was pinned, or version tracking was not used.

**Solution**:
- Check the dataset's version history through the dataset resources.
- Use `increment_dataset_version` after making changes to a dataset to create a new version snapshot.
- When referencing datasets in workflows, consider pinning to a specific version.
- Use `get_dataset_spec` to see the current dataset specification and version.

---

## Problem: "Feature Not Found"

**Symptom**: Attempting to add feature values fails because the feature does not exist.

**Cause**: The feature was not created, or the name does not match exactly.

**Solution**:
- Check the feature resources to list existing features.
- Feature names are case-sensitive. Verify exact spelling.
- **Tool**: `create_feature` to create the feature if it does not exist.
- Ensure the feature is associated with the correct table.

---

## Problem: "Upload Timeout"

**Symptom**: `upload_execution_outputs` hangs or times out.

**Cause**: Large files, network issues, or server limits.

**Solution**:
- Check your network connectivity.
- For large files, consider breaking them into smaller batches.
- The server may have upload size limits. Check with your catalog administrator.
- Retry the upload -- transient network issues are the most common cause.
- **Tool**: `get_execution_info` to check if partial uploads succeeded.

---

## Problem: "Execution Stuck in Running"

**Symptom**: An execution shows status `Running` but the process has ended or crashed.

**Cause**: The execution context was not properly closed (e.g., crash without cleanup, not using context manager).

**Solution**:
- **Best practice**: Always use the context manager (`with ml.create_execution(config) as exe:`) which automatically handles cleanup on both success and failure.
- To fix a stuck execution manually:
  - **Tool**: `update_execution_status` with `status="Failed"` and `message="Manually marked as failed"` (or `status="Completed"` if the work actually finished).
- **Tool**: `get_execution_info` to inspect the execution's current state and metadata.
- For future runs, always use the context manager to prevent this issue.

---

## Problem: "Vocabulary Term Not Found"

**Symptom**: An operation fails because a required vocabulary term does not exist.

**Cause**: The term was not added to the vocabulary, or the name does not match exactly.

**Solution**:
- Check the relevant vocabulary resource to list existing terms.
- Vocabulary term names are case-sensitive.
- **Tool**: `add_term` to add the missing term to the appropriate vocabulary.
- Common vocabularies: `Dataset_Type`, `Asset_Type`, `Workflow_Type`.

---

## Reference Resources

- `deriva://execution/{rid}` — Inspect execution state, status, and metadata
- `deriva://storage/execution-dirs` — Check execution working directories
- `deriva://catalog/vocabularies` — Verify vocabulary terms exist (e.g., status types, workflow types)

## General Debugging Tips

### Enable Verbose Logging
When using the Python API, enable verbose logging to see detailed request/response information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Execution State
- **Tool**: `get_execution_info` with the execution RID to see full execution metadata, status, inputs, and outputs.
- **Tool**: `get_execution_working_dir` to find the local working directory and inspect files directly. (No params -- operates on the active execution.)

### Check Catalog State
- Use the catalog resources to review the current catalog schema, tables, and vocabularies.
- **Tool**: `count_table` to quickly verify data exists in expected tables.

### Review Recent Executions
- Check the recent executions resource to see the latest execution activity, statuses, and any patterns of failure.
- **Tool**: `list_nested_executions` if the execution is part of a larger workflow to see the full execution tree.
- **Tool**: `list_parent_executions` to find the parent execution if this is a nested step.

### Verify Working Directory
- **Tool**: `get_execution_working_dir` returns the local filesystem path for the active execution.
- Inspect this directory to verify:
  - Input files were downloaded correctly.
  - Output files were written to the correct locations.
  - No unexpected files or directory structures.

### Clean Up
- **Resource**: Read `deriva://storage/execution-dirs` to list local execution working directories. Remove unneeded directories manually to free disk space.
