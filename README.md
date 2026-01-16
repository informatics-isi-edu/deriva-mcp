# DerivaML MCP Server

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that exposes [DerivaML](https://github.com/informatics-isi-edu/deriva-ml) operations as tools for LLM applications.

## Overview

This MCP server provides an interface to DerivaML, enabling AI assistants like Claude to:

- Connect to and manage Deriva catalogs
- Create and manage datasets with versioning
- Work with controlled vocabularies
- Define and execute ML workflows
- Create and manage features for ML experiments

For full ML workflow management, this server is designed to work alongside the [GitHub MCP Server](https://github.com/github/github-mcp-server) to enable:

- Storing and versioning hydra-zen configurations in GitHub repositories
- Managing workflow code and model implementations
- Collaborative development of ML experiments

## Prerequisites

### Deriva Authentication

DerivaML uses Globus for authentication. Before using the MCP server, you must authenticate with your Deriva server:

```bash
# Install deriva-ml if not already installed
pip install deriva-ml

# Authenticate with your Deriva server
python -c "from deriva_ml import DerivaML; DerivaML.globus_login('your-server.org')"
```

This opens a browser window for Globus authentication. Credentials are cached locally and persist across sessions.

Alternatively, use the **Deriva Auth Agent** for browser-based authentication:
1. Install the Deriva Auth Agent from [deriva-py](https://github.com/informatics-isi-edu/deriva-py)
2. Run `deriva-globus-auth-utils login --host your-server.org`

### GitHub Authentication (for configuration management)

Create a GitHub Personal Access Token (PAT) for the GitHub MCP Server:

1. Go to [GitHub Settings > Personal Access Tokens](https://github.com/settings/personal-access-tokens/new)
2. Create a fine-grained token with these permissions:
   - **Repository access**: Select repositories containing your ML configurations
   - **Permissions**:
     - Contents: Read and write (for pushing configs)
     - Pull requests: Read and write (optional, for PR workflows)
     - Issues: Read (optional, for tracking)
3. Copy the token securely - you'll need it for configuration

## Installation

### Using Docker (Recommended)

Docker provides the simplest setup with no Python environment management:

```bash
# Pull the image
docker pull ghcr.io/informatics-isi-edu/deriva-ml-mcp:latest

# Or build locally
git clone https://github.com/informatics-isi-edu/deriva-ml-mcp.git
cd deriva-ml-mcp
docker build -t deriva-ml-mcp .
```

### Using uv

```bash
uv pip install deriva-ml-mcp
```

### Using pip

```bash
pip install deriva-ml-mcp
```

### From source

```bash
git clone https://github.com/informatics-isi-edu/deriva-ml-mcp.git
cd deriva-ml-mcp
uv sync
```

## Configuration

### Claude Desktop - Full Setup with GitHub Integration

For the complete ML workflow experience, configure both DerivaML and GitHub MCP servers together.

**Configuration file locations:**
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

#### Option 1: Both Servers with Docker (Recommended)

Uses Docker for both MCP servers - most consistent setup:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --add-host localhost:host-gateway -v $HOME/.deriva:/home/mcpuser/.deriva:ro -v $HOME/.bdbag:/home/mcpuser/.bdbag -v $HOME/.deriva-ml:/home/mcpuser/.deriva-ml ghcr.io/informatics-isi-edu/deriva-ml-mcp:latest"
      ],
      "env": {}
    },
    "github": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

**Docker arguments explained:**
- `--add-host localhost:host-gateway` - Allows connecting to a Deriva server running on localhost

**For localhost with self-signed certificates**, the image defaults to using `~/.deriva/allCAbundle-with-local.pem` as the CA bundle. See [Troubleshooting](#docker-with-localhost-deriva-server) for how to create this file.

**Volume mounts explained:**
- `$HOME/.deriva:/home/mcpuser/.deriva:ro` - Mounts your Deriva credentials (read-only)
- `$HOME/.bdbag:/home/mcpuser/.bdbag` - Mounts bdbag keychain for dataset download authentication (writable)
- `$HOME/.deriva-ml:/home/mcpuser/.deriva-ml` - Working directory for execution outputs (writable)

**Note:** Create the workspace directory before first use:
```bash
mkdir -p ~/.deriva-ml
```
If the directory doesn't exist, Docker creates it as root, causing permission issues.

#### Option 2: Direct Install with GitHub Remote

Uses pip-installed DerivaML MCP with GitHub's hosted server:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "deriva-ml-mcp",
      "env": {}
    },
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@anthropic-ai/github-mcp-server"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

#### Option 3: From Source (Development)

For development or customization:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/deriva-ml-mcp",
        "run",
        "deriva-ml-mcp"
      ],
      "env": {}
    },
    "github": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

#### Option 4: DerivaML Only (No GitHub)

If you don't need GitHub integration:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --add-host localhost:host-gateway -v $HOME/.deriva:/home/mcpuser/.deriva:ro -v $HOME/.bdbag:/home/mcpuser/.bdbag -v $HOME/.deriva-ml:/home/mcpuser/.deriva-ml ghcr.io/informatics-isi-edu/deriva-ml-mcp:latest"
      ],
      "env": {}
    }
  }
}
```

Or with direct install:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "deriva-ml-mcp",
      "env": {}
    }
  }
}
```

### Claude Code

Add to `~/.mcp.json` (global) or your project's `.mcp.json` file:

**With Docker:**

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --add-host localhost:host-gateway -v $HOME/.deriva:/home/mcpuser/.deriva:ro -v $HOME/.bdbag:/home/mcpuser/.bdbag -v $HOME/.deriva-ml:/home/mcpuser/.deriva-ml ghcr.io/informatics-isi-edu/deriva-ml-mcp:latest"
      ],
      "env": {}
    },
    "github": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    }
  }
}
```

**With direct install:**

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "deriva-ml-mcp",
      "env": {}
    },
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@anthropic-ai/github-mcp-server"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

Then enable in `.claude/settings.local.json`:

```json
{
  "enableAllProjectMcpServers": true,
  "enabledMcpjsonServers": ["deriva-ml", "github"]
}
```

### VS Code with Continue or Cline

Add to your MCP configuration (typically `.vscode/mcp.json`):

```json
{
  "mcp": {
    "servers": {
      "deriva-ml": {
        "type": "stdio",
        "command": "/bin/sh",
        "args": [
          "-c",
          "docker run -i --rm --add-host localhost:host-gateway -v $HOME/.deriva:/home/mcpuser/.deriva:ro -v $HOME/.bdbag:/home/mcpuser/.bdbag -v $HOME/.deriva-ml:/home/mcpuser/.deriva-ml ghcr.io/informatics-isi-edu/deriva-ml-mcp:latest"
        ],
        "env": {}
      },
      "github": {
        "type": "stdio",
        "command": "docker",
        "args": [
          "run", "-i", "--rm",
          "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
          "ghcr.io/github/github-mcp-server"
        ],
        "env": {
          "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
        }
      }
    }
  }
}
```

### Environment Variables

For security, store tokens in environment variables instead of config files:

```bash
# Add to ~/.bashrc, ~/.zshrc, or equivalent
export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_your_token_here"
```

Then reference in config:

```json
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    }
  }
}
```

## Verifying Your Setup

After configuration, verify both servers are working:

```
User: What MCP servers are available?

Claude: I have access to two MCP servers:
1. deriva-ml - For managing ML workflows in Deriva catalogs
2. github - For managing GitHub repositories and configurations

User: Connect to the deriva catalog at example.org with ID 42

Claude: [Uses connect_catalog tool]
Connected to example.org, catalog 42. The domain schema is 'my_project'.

User: List the hydra-zen configs in the my-ml-project repo

Claude: [Uses GitHub get_file_contents tool]
Found configuration files in configs/:
- deriva.py - DerivaML connection settings
- datasets.py - Dataset specifications
- model.py - Model hyperparameters
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
| `find_workflows` | Find all workflows |
| `lookup_workflow` | Find a workflow by URL/checksum |
| `create_workflow` | Create and register a workflow |
| `list_workflow_types` | List available workflow types |
| `add_workflow_type` | Add a new workflow type |

### Feature Management

| Tool | Description |
|------|-------------|
| `find_features` | Find features for a table |
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
| `asset_file_path` | Register a file for upload as an execution output |
| `upload_execution_outputs` | Upload all registered outputs to the catalog |
| `list_executions` | List recent executions |
| `create_execution_dataset` | Create a dataset within an execution |
| `download_execution_dataset` | Download a dataset for processing |
| `get_execution_working_dir` | Get the working directory path |

#### Execution Workflow

The typical execution workflow using the context manager:

```python
with execution.execute() as exe:
    # Do your work here
    exe.asset_file_path(asset_name="Image", file_name="output.png")
    # ... more processing ...

# After context exits, upload outputs
execution.upload_execution_outputs()
```

Using MCP tools, the equivalent workflow is:

1. `create_execution()` - Create the execution record with workflow info
2. `start_execution()` - Mark execution as running, begin timing
3. `asset_file_path()` - Register output files (repeat as needed)
4. `stop_execution()` - Mark execution as complete
5. `upload_execution_outputs()` - **Required**: Upload all registered files to catalog

**Important**: You must call `upload_execution_outputs()` after completing your work
to upload any registered assets to the catalog. This is not automatic.

## Available Resources

MCP resources provide read-only access to catalog information and configuration templates.

### Static Resources - Configuration Templates

These resources provide code templates for configuring DerivaML with hydra-zen:

| Resource URI | Description |
|--------------|-------------|
| `deriva-ml://config/deriva-ml-template` | Hydra-zen configuration template for DerivaML connection |
| `deriva-ml://config/dataset-spec-template` | Configuration template for dataset specifications |
| `deriva-ml://config/execution-template` | Configuration template for ML executions |
| `deriva-ml://config/model-template` | Configuration template for ML models with zen_partial |

### Dynamic Resources - Catalog Information

These resources return current catalog state (requires active connection):

| Resource URI | Description |
|--------------|-------------|
| `deriva-ml://catalog/schema` | Current catalog schema structure in JSON |
| `deriva-ml://catalog/vocabularies` | All vocabulary tables and their terms |
| `deriva-ml://catalog/datasets` | All datasets in the current catalog |
| `deriva-ml://catalog/workflows` | All registered workflows |
| `deriva-ml://catalog/features` | All feature names defined in the catalog |

### Template Resources - Parameterized

These resources accept parameters to return specific information:

| Resource URI | Description |
|--------------|-------------|
| `deriva-ml://dataset/{dataset_rid}` | Detailed information about a specific dataset |
| `deriva-ml://table/{table_name}/features` | Features defined for a specific table |
| `deriva-ml://vocabulary/{vocab_name}` | Terms in a specific vocabulary table |

### Documentation Resources

Documentation is fetched dynamically from GitHub repositories with 1-hour caching:

| Resource URI | Description |
|--------------|-------------|
| `deriva-ml://docs/overview` | DerivaML overview and architecture |
| `deriva-ml://docs/datasets` | Guide to creating and managing datasets |
| `deriva-ml://docs/features` | Guide to defining and using features |
| `deriva-ml://docs/execution-configuration` | Guide to configuring ML executions |
| `deriva-ml://docs/hydra-zen` | Guide to hydra-zen configuration |
| `deriva-ml://docs/file-assets` | Guide to managing file assets |
| `deriva-ml://docs/notebooks` | Guide to Jupyter notebook integration |
| `deriva-ml://docs/identifiers` | Guide to RIDs, MINIDs, and identifiers |
| `deriva-ml://docs/install` | Installation instructions |
| `deriva-ml://docs/ermrest/*` | ERMrest API documentation |
| `deriva-ml://docs/chaise/*` | Chaise UI documentation |
| `deriva-ml://docs/deriva-py/*` | Deriva Python SDK documentation |

### Using Resources

Resources are accessed differently than tools - they provide static or semi-static data that can be read without side effects:

```
User: Show me the DerivaML configuration template

Claude: [Reads deriva-ml://config/deriva-ml-template resource]
Here's a hydra-zen configuration template for DerivaML...

User: What datasets are in the catalog?

Claude: [Reads deriva-ml://catalog/datasets resource]
Found the following datasets in your catalog...
```

## Usage Examples

### Basic Catalog Operations

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

### Configuration Workflow with GitHub

```
User: Create a new hydra-zen config for training on the cell-images dataset

Claude: I'll create a configuration file for you.
[Reads deriva-ml://config/model-template resource for template]
[Uses GitHub create_or_update_file to save configs/cell_training.py]

Created configs/cell_training.py with:
- DerivaML connection to example.org
- Dataset spec for cell-images v2.0.0
- Model config with learning_rate=1e-3, epochs=50

User: Commit and push that config

Claude: [Uses GitHub create_or_update_file with commit message]
Committed and pushed to main branch.
```

## Hydra-zen Configuration

DerivaML integrates with [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) for configuration management, enabling reproducible ML workflows with structured configuration.

### Basic Configuration

```python
from hydra_zen import builds, instantiate
from deriva_ml import DerivaML
from deriva_ml.core.config import DerivaMLConfig

# Create a structured config using hydra-zen
DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

# Configure for your environment
conf = DerivaMLConf(
    hostname='deriva.example.org',
    catalog_id='42',
    domain_schema='my_domain',
)

# Instantiate to get a DerivaMLConfig object, then create DerivaML
config = instantiate(conf)
ml = DerivaML.instantiate(config)
```

### Working Directory Configuration

DerivaML automatically configures Hydra's output directory based on your `working_dir` setting:

```python
conf = DerivaMLConf(
    hostname='deriva.example.org',
    working_dir='/shared/ml_workspace',  # Custom working directory
)
```

Hydra outputs will be organized under: `{working_dir}/{username}/deriva-ml/hydra/{timestamp}/`

### Configuration Composition

Create environment-specific configurations using hydra-zen's store:

```python
from hydra_zen import store

# Development configuration
store(DerivaMLConf(
    hostname='dev.example.org',
    catalog_id='1',
), name='dev')

# Production configuration
store(DerivaMLConf(
    hostname='prod.example.org',
    catalog_id='100',
), name='prod')
```

### Dataset Specification Configuration

Use `DatasetSpecConfig` for cleaner dataset specifications:

```python
from deriva_ml.dataset import DatasetSpecConfig

# Create dataset specs (hydra-zen compatible)
training_data = DatasetSpecConfig(
    rid="1ABC",
    version="1.0.0",
    materialize=True,       # Download asset files
    description="Training images"
)

metadata_only = DatasetSpecConfig(
    rid="2DEF",
    version="2.0.0",
    materialize=False,      # Only download table data
)

# Use in hydra-zen store
from hydra_zen import store
datasets_store = store(group="datasets")
datasets_store([training_data], name="training")
datasets_store([metadata_only], name="metadata_only")
```

### Asset Configuration

Use `AssetRIDConfig` for input assets (model weights, config files):

```python
from deriva_ml.execution import AssetRIDConfig

# Define input assets
model_weights = AssetRIDConfig(rid="WXYZ", description="Pretrained model")
config_file = AssetRIDConfig(rid="ABCD", description="Hyperparameters")

# Store asset collections
assets_store = store(group="assets")
assets_store([model_weights, config_file], name="default_assets")
```

### Execution Configuration

Configure ML executions with `ExecutionConfiguration`:

```python
from hydra_zen import builds, instantiate
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.dataset import DatasetSpecConfig

# Build execution config
ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)

# Configure execution with datasets and assets
conf = ExecConf(
    description="Training run",
    datasets=[
        DatasetSpecConfig(rid="1ABC", version="1.0.0", materialize=True),
    ],
    assets=["WXYZ", "ABCD"],  # Asset RIDs
)

exec_config = instantiate(conf)
```

### Configuration Summary

| Class | Module | Purpose |
|-------|--------|---------|
| `DerivaMLConfig` | `deriva_ml.core.config` | Main DerivaML connection config |
| `DatasetSpecConfig` | `deriva_ml.dataset` | Dataset specification for executions |
| `AssetRIDConfig` | `deriva_ml.execution` | Input asset specification |
| `ExecutionConfiguration` | `deriva_ml.execution` | Full execution configuration |
| `Workflow` | `deriva_ml.execution` | Workflow definition |

See the [DerivaML Hydra-zen Guide](https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/user-guide/hydra-zen-configuration.md) for complete documentation.

## Troubleshooting

### Docker with Localhost Deriva Server

When running the MCP server in Docker and connecting to a Deriva server on your local machine, you need additional configuration depending on how Deriva is running.

#### Option A: Deriva Running Directly on Host (not in Docker)

If your Deriva server is running directly on the host machine (not in Docker), use `host-gateway`:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --add-host localhost:host-gateway -v $HOME/.deriva:/home/mcpuser/.deriva:ro -v $HOME/.bdbag:/home/mcpuser/.bdbag -v $HOME/.deriva-ml:/home/mcpuser/.deriva-ml ghcr.io/informatics-isi-edu/deriva-ml-mcp:latest"
      ],
      "env": {}
    }
  }
}
```

#### Option B: Deriva Running in Docker (deriva-localhost)

If your Deriva server is running in Docker (e.g., using deriva-localhost), the MCP container must join the same Docker network and map `localhost` to the webserver container's IP:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --network deriva-localhost_internal_network --add-host localhost:172.28.3.15 -v $HOME/.deriva:/home/mcpuser/.deriva:ro -v $HOME/.bdbag:/home/mcpuser/.bdbag -v $HOME/.deriva-ml:/home/mcpuser/.deriva-ml deriva-ml-mcp:latest"
      ],
      "env": {}
    }
  }
}
```

**Finding the webserver IP:**
```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' deriva-webserver
```

**Why this is needed:** The MCP container needs to download dataset assets from the Deriva server. When Deriva runs in Docker, URLs in the dataset bags reference `localhost`, which must resolve to the Deriva webserver container. The entrypoint script in the MCP image automatically adjusts `/etc/hosts` so that the `--add-host` mapping takes effect.

#### SSL Certificate Configuration

If your localhost Deriva server uses a self-signed certificate (common for development), the container won't trust it. You need to:

1. Export your local CA certificate to a PEM file accessible to the container
2. Set the `REQUESTS_CA_BUNDLE` environment variable

**Creating the CA bundle with your local certificate:**

If your local Deriva CA is in the macOS System Keychain:

```bash
# Export the local CA certificate
security find-certificate -a -c "DERIVA Dev Local CA" -p /Library/Keychains/System.keychain > /tmp/deriva-local-ca.pem

# Combine with existing CA bundle (if you have one)
cat ~/.deriva/allCAbundle.pem /tmp/deriva-local-ca.pem > ~/.deriva/allCAbundle-with-local.pem

# Or just use the local CA alone
cp /tmp/deriva-local-ca.pem ~/.deriva/allCAbundle-with-local.pem
```

### Deriva Authentication Issues

**Error: "No credentials found"**
```bash
# Re-authenticate with Deriva
python -c "from deriva_ml import DerivaML; DerivaML.globus_login('your-server.org')"
```

**Error: "Token expired"**
```bash
# Force re-authentication
python -c "from deriva_ml import DerivaML; DerivaML.globus_login('your-server.org', force=True)"
```

### GitHub MCP Issues

**Error: "Bad credentials"**
- Verify your PAT hasn't expired
- Check the token has required permissions (Contents: Read/Write)
- Ensure the token is correctly set in your config

**Docker not found**
- Install Docker Desktop or use the npx method instead
- On Linux, ensure your user is in the docker group

### MCP Server Connection Issues

**Server not responding**
1. Check the server is installed: `which deriva-ml-mcp`
2. Test manually: `deriva-ml-mcp` (should start without errors)
3. Check Claude Desktop logs for errors

**Multiple server conflicts**
- Ensure each server has a unique name in the config
- Restart Claude Desktop after config changes

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
- Docker (optional, for GitHub MCP local server)

## License

Apache 2.0

## Related Projects

- [DerivaML](https://github.com/informatics-isi-edu/deriva-ml) - Core library for ML workflows on Deriva
- [Deriva](https://github.com/informatics-isi-edu/deriva-py) - Python SDK for Deriva scientific data management
- [GitHub MCP Server](https://github.com/github/github-mcp-server) - Official GitHub MCP server
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Official Python SDK for Model Context Protocol
