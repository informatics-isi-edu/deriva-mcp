# DerivaML MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that exposes [DerivaML](https://github.com/informatics-isi-edu/deriva-ml) operations as tools for LLM applications.

## Overview

This MCP server provides an interface to DerivaML, enabling AI assistants like Claude to:

- Connect to and manage Deriva catalogs
- Create and manage datasets with versioning
- Work with controlled vocabularies
- Define and execute ML workflows
- Create and manage features for ML experiments

## Installation

```bash
# Using uv (recommended)
uv pip install deriva-ml-mcp

# Or using pip
pip install deriva-ml-mcp
```

### From source

```bash
git clone https://github.com/your-org/deriva-ml-mcp.git
cd deriva-ml-mcp
uv sync
```

## Configuration

### Claude Desktop

Add the server to your Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "deriva-ml": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/deriva-ml-mcp",
        "run",
        "deriva-ml-mcp"
      ]
    }
  }
}
```

Or if installed globally:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "command": "deriva-ml-mcp"
    }
  }
}
```

### Claude Code

Add to your `.claude/settings.json`:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "command": "deriva-ml-mcp"
    }
  }
}
```

## Available Tools

### Catalog Management

| Tool | Description |
|------|-------------|
| `connect_catalog` | Connect to a DerivaML catalog |
| `disconnect_catalog` | Disconnect from the active catalog |
| `list_connections` | List all active connections |
| `set_active_catalog` | Set which connection is active |
| `get_catalog_info` | Get information about the active catalog |
| `list_users` | List users with catalog access |
| `get_chaise_url` | Get web interface URL for a table |

### Dataset Management

| Tool | Description |
|------|-------------|
| `list_datasets` | List all datasets in the catalog |
| `get_dataset` | Get detailed information about a dataset |
| `create_dataset` | Create a new dataset |
| `list_dataset_members` | List members of a dataset |
| `add_dataset_members` | Add members to a dataset |
| `get_dataset_version_history` | Get version history |
| `increment_dataset_version` | Update dataset version |
| `delete_dataset` | Delete a dataset |
| `list_dataset_element_types` | List valid element types |
| `add_dataset_element_type` | Enable a table as element type |

### Vocabulary Management

| Tool | Description |
|------|-------------|
| `list_vocabularies` | List all vocabulary tables |
| `list_vocabulary_terms` | List terms in a vocabulary |
| `lookup_term` | Find a term by name or synonym |
| `add_term` | Add a term to a vocabulary |
| `create_vocabulary` | Create a new vocabulary table |

### Workflow Management

| Tool | Description |
|------|-------------|
| `list_workflows` | List all workflows |
| `lookup_workflow` | Find a workflow by URL/checksum |
| `create_workflow` | Create and register a workflow |
| `list_workflow_types` | List available workflow types |
| `add_workflow_type` | Add a new workflow type |

### Feature Management

| Tool | Description |
|------|-------------|
| `list_features` | List features for a table |
| `lookup_feature` | Get feature details |
| `list_feature_values` | Get all values for a feature |
| `create_feature` | Create a feature definition |
| `delete_feature` | Delete a feature |
| `list_feature_names` | List all feature names |

### Schema Management

| Tool | Description |
|------|-------------|
| `create_table` | Create a new table in the domain schema |
| `create_asset_table` | Create an asset table for file management |
| `list_assets` | List all assets in an asset table |
| `list_tables` | List all tables in the domain schema |
| `get_table_schema` | Get column and key definitions for a table |
| `list_asset_types` | List available asset type terms |
| `add_asset_type` | Add a new asset type to the vocabulary |

### Execution Management

| Tool | Description |
|------|-------------|
| `create_execution` | Create a new execution for ML workflows |
| `start_execution` | Start the active execution |
| `stop_execution` | Stop and complete the active execution |
| `update_execution_status` | Update execution status and message |
| `get_execution_info` | Get details about the active execution |
| `restore_execution` | Restore a previous execution by RID |
| `register_asset_file` | Register a file for upload as an execution output |
| `upload_execution_outputs` | Upload all registered outputs to the catalog |
| `list_executions` | List recent executions |
| `create_execution_dataset` | Create a dataset within an execution |
| `download_execution_dataset` | Download a dataset for processing |
| `get_execution_working_dir` | Get the working directory path |

#### Execution Workflow

The typical execution workflow is:

1. `create_execution()` - Create the execution record with workflow info
2. `start_execution()` - Mark execution as running, begin timing
3. `register_asset_file()` - Register output files (repeat as needed)
4. `stop_execution()` - Mark execution as complete
5. `upload_execution_outputs()` - **Required**: Upload all registered files to catalog

**Important**: You must call `upload_execution_outputs()` after completing your work
to upload any registered assets to the catalog. This is not automatic.

## Usage Examples

Once configured, you can interact with DerivaML through your LLM application:

```
User: Connect to the deriva catalog at example.org with ID 123

Claude: I'll connect to that catalog for you.
[Uses connect_catalog tool]
Connected to example.org, catalog 123. The domain schema is 'my_project'.

User: What datasets are available?

Claude: Let me check what datasets exist.
[Uses list_datasets tool]
Found 5 datasets:
1. Training Images (v1.2.0) - 1500 images for model training
2. Validation Set (v1.0.0) - 300 images for validation
...
```

## Authentication

The server uses Deriva's credential system. Before using the MCP server, ensure you're authenticated:

```python
from deriva_ml import DerivaML
DerivaML.globus_login('example.org')
```

Or use the Deriva Auth Agent for browser-based authentication.

## Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
uv run ruff check src/
uv run ruff format src/
```

## Requirements

- Python 3.10+
- MCP SDK 1.2.0+
- DerivaML 0.1.0+

## License

Apache 2.0

## Related Projects

- [DerivaML](https://github.com/informatics-isi-edu/deriva-ml) - Core library for ML workflows on Deriva
- [Deriva](https://github.com/informatics-isi-edu/deriva-py) - Python SDK for Deriva scientific data management
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Official Python SDK for Model Context Protocol
