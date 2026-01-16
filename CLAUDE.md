# CLAUDE.md

This file provides guidance to Claude Code when working with the deriva-ml-mcp codebase.

## Project Overview

DerivaML MCP Server is a Model Context Protocol (MCP) server that exposes DerivaML operations as tools and resources for LLM applications like Claude. It enables AI assistants to manage ML workflows, datasets, features, and executions in Deriva catalogs.

## Build and Development Commands

```bash
# Install dependencies
uv sync

# Run the MCP server directly (for testing)
uv run deriva-ml-mcp

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/
uv run ruff format src/
```

## Architecture

### Project Structure

```
src/deriva_ml_mcp/
├── server.py          # Main MCP server entry point
├── connection.py      # ConnectionManager for catalog connections
├── resources.py       # MCP resources (config templates, catalog info, docs)
├── github_docs.py     # Fetches documentation from GitHub with caching
└── tools/             # MCP tools organized by domain
    ├── __init__.py    # Exports all register_*_tools functions
    ├── catalog.py     # Connection and catalog management tools
    ├── dataset.py     # Dataset CRUD and versioning tools
    ├── vocabulary.py  # Controlled vocabulary tools
    ├── workflow.py    # Workflow management tools
    ├── feature.py     # Feature definition and value tools
    ├── schema.py      # Table and asset creation tools
    ├── execution.py   # ML execution lifecycle tools
    └── data.py        # Data query and manipulation tools
```

### Key Components

**server.py**: FastMCP server initialization and tool/resource registration. Entry point via `deriva-ml-mcp` command.

**connection.py**: `ConnectionManager` class that maintains multiple DerivaML connections with one active connection. All tools access the catalog through `conn_manager.get_active_connection()`.

**resources.py**: MCP resources providing read-only access to:
- Static config templates (hydra-zen configurations)
- Dynamic catalog info (schema, vocabularies, datasets, workflows)
- Parameterized resources (specific dataset/table/vocabulary details)
- Documentation fetched dynamically from GitHub repositories

**github_docs.py**: Fetches documentation from GitHub with 1-hour caching. Supports deriva-ml, deriva-py, ermrest, and chaise repositories.

**tools/**: Each module registers tools for a specific domain. Tools follow the pattern:
```python
def register_*_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    @mcp.tool()
    def tool_name(...) -> str:
        ml = conn_manager.get_active_connection()
        # ... implementation
        return json.dumps(result)
```

### Tool Categories

| Module | Purpose | Key Tools |
|--------|---------|-----------|
| catalog.py | Connection management | connect_catalog, disconnect_catalog, get_catalog_info |
| dataset.py | Dataset CRUD | create_dataset, list_datasets, add_dataset_members |
| vocabulary.py | Controlled vocabularies | list_vocabularies, add_term, lookup_term |
| workflow.py | Workflow registration | create_workflow, find_workflows |
| feature.py | Feature definitions | create_feature, find_features, list_feature_values |
| schema.py | Schema management | create_table, create_asset_table |
| execution.py | ML execution lifecycle | create_execution, start_execution, upload_execution_outputs |
| data.py | Data queries | query_table, insert_records |

### Resource Categories

| URI Pattern | Type | Description |
|-------------|------|-------------|
| `deriva-ml://config/*` | Static | Hydra-zen configuration templates |
| `deriva-ml://catalog/*` | Dynamic | Current catalog state (requires connection) |
| `deriva-ml://dataset/{rid}` | Template | Specific dataset details |
| `deriva-ml://vocabulary/{name}` | Template | Specific vocabulary terms |
| `deriva-ml://docs/*` | Dynamic | Documentation fetched from GitHub repos |

## Installation

> **Full installation guide:** See [README.md](README.md) for complete installation options including GitHub MCP integration.

Choose **one** of the following options to run the MCP server.

### Claude Desktop vs Claude Code

| Feature | Claude Desktop | Claude Code |
|---------|---------------|-------------|
| **What it is** | GUI app for chatting with Claude | CLI tool for coding with Claude in your terminal |
| **Config location** | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) | `.mcp.json` in project root, or `~/.claude/settings.json` globally |
| **Use case** | General conversations, document analysis | Software development, code editing |
| **MCP scope** | Global (all conversations) | Per-project or global |

### Option 1: Docker (Recommended)

Uses the published Docker image. No local setup required.

<!-- copy-button -->
**For Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --add-host localhost:host-gateway -e HOME=$HOME -v $HOME/.deriva:$HOME/.deriva:ro -v $HOME/.bdbag:$HOME/.bdbag -v $HOME/.deriva-ml:$HOME/.deriva-ml ghcr.io/informatics-isi-edu/deriva-ml-mcp:latest"
      ],
      "env": {}
    }
  }
}
```

<!-- copy-button -->
**For Claude Code** (`~/.mcp.json` or `.mcp.json` in project root):
```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --add-host localhost:host-gateway -e HOME=$HOME -v $HOME/.deriva:$HOME/.deriva:ro -v $HOME/.bdbag:$HOME/.bdbag -v $HOME/.deriva-ml:$HOME/.deriva-ml ghcr.io/informatics-isi-edu/deriva-ml-mcp:latest"
      ],
      "env": {}
    }
  }
}
```

**Docker arguments:**
- `--add-host localhost:host-gateway` - Allows connecting to localhost Deriva server
- `-e HOME=$HOME` - Passes your home directory path into the container so mounted paths are found correctly

**For localhost with self-signed certificates**, the image defaults to using `~/.deriva/allCAbundle-with-local.pem` as the CA bundle. See [Connecting to Localhost from Docker](#connecting-to-localhost-from-docker) for how to create this file.

**Volume mounts:**
- `$HOME/.deriva:$HOME/.deriva:ro` - Deriva credentials (read-only)
- `$HOME/.bdbag:$HOME/.bdbag` - bdbag keychain for dataset download authentication (writable)
- `$HOME/.deriva-ml:$HOME/.deriva-ml` - Working directory for execution outputs (writable)

**Note:** If using the workspace volume, create the directory first:
```bash
mkdir -p ~/.deriva-ml
```
If the directory doesn't exist, Docker creates it as root, causing permission issues.

### Connecting to Localhost from Docker

When connecting to a Deriva server running on localhost, the Docker container needs additional configuration depending on how Deriva is running.

#### Deriva Running Directly on Host

If Deriva runs directly on the host (not in Docker), use `host-gateway`:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --add-host localhost:host-gateway -e HOME=$HOME -v $HOME/.deriva:$HOME/.deriva:ro -v $HOME/.bdbag:$HOME/.bdbag -v $HOME/.deriva-ml:$HOME/.deriva-ml deriva-ml-mcp:latest"
      ],
      "env": {}
    }
  }
}
```

#### Deriva Running in Docker (deriva-localhost)

If Deriva runs in Docker (e.g., deriva-localhost), join the same network and map localhost to the webserver IP:

```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --network deriva-localhost_internal_network --add-host localhost:172.28.3.15 -e HOME=$HOME -v $HOME/.deriva:$HOME/.deriva:ro -v $HOME/.bdbag:$HOME/.bdbag -v $HOME/.deriva-ml:$HOME/.deriva-ml deriva-ml-mcp:latest"
      ],
      "env": {}
    }
  }
}
```

**Find the webserver IP:**
```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' deriva-webserver
```

The entrypoint script automatically adjusts `/etc/hosts` so that `--add-host` takes effect for localhost resolution. It also sets `REQUESTS_CA_BUNDLE` to `$HOME/.deriva/allCAbundle-with-local.pem` by default.

**Creating the CA bundle (macOS):**
```bash
# Export local CA from System Keychain
security find-certificate -a -c "DERIVA Dev Local CA" -p /Library/Keychains/System.keychain > /tmp/deriva-local-ca.pem

# Combine with existing bundle
cat ~/.deriva/allCAbundle.pem /tmp/deriva-local-ca.pem > ~/.deriva/allCAbundle-with-local.pem
```

To use a different CA bundle, override with `-e REQUESTS_CA_BUNDLE=/path/to/bundle.pem`.

### Option 2: From Source (Development)

Run directly using `uv`. Use this when developing or modifying the MCP server.

<!-- copy-button -->
**For Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/path/to/deriva-ml-mcp", "run", "deriva-ml-mcp"],
      "env": {}
    }
  }
}
```

<!-- copy-button -->
**For Claude Code** (`~/.mcp.json` or `.mcp.json` in project root):
```json
{
  "mcpServers": {
    "deriva-ml": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "deriva-ml-mcp"],
      "cwd": "/path/to/deriva-ml-mcp",
      "env": {}
    }
  }
}
```

### Docker Build (for development)

Build and test locally:

```bash
# Build the image
docker build -t deriva-ml-mcp .

# Test the image
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | docker run -i --rm deriva-ml-mcp
```

The Dockerfile uses a multi-stage build:
1. **Builder stage**: Installs uv and dependencies into /opt/venv
2. **Runtime stage**: Copies venv, creates non-root user, sets entrypoint

## Adding New Tools

1. Create or edit the appropriate module in `src/deriva_ml_mcp/tools/`
2. Add the tool function with `@mcp.tool()` decorator
3. Return JSON-serialized results for structured data
4. If creating a new module, add the register function to `tools/__init__.py`
5. Call the register function in `server.py`

Example:
```python
@mcp.tool()
def my_new_tool(param: str) -> str:
    """Tool description shown to LLM."""
    ml = conn_manager.get_active_connection()
    if ml is None:
        return "Error: Not connected to a catalog"

    result = ml.some_operation(param)
    return json.dumps({"status": "success", "data": result})
```

## Adding New Resources

Add to `resources.py` within `register_resources()`:

```python
@mcp.resource(
    "deriva-ml://my-resource/{param}",
    name="My Resource",
    description="Description for LLM",
    mime_type="application/json",
)
def get_my_resource(param: str) -> str:
    ml = conn_manager.get_active_connection()
    # ... implementation
    return json.dumps(data)
```

## Dependencies

Key dependencies from `pyproject.toml`:
- `mcp>=1.2.0` - MCP Python SDK
- `deriva-ml` - Core DerivaML library (from git)
- `pydantic>=2.0` - Data validation

## Common Patterns

### Error Handling
Tools should return error messages as strings rather than raising exceptions:
```python
if ml is None:
    return "Error: Not connected to a catalog. Use connect_catalog first."
```

### JSON Serialization
All structured data should be returned as JSON strings:
```python
return json.dumps({"datasets": [...], "count": len(datasets)})
```

### Connection Check
Always verify active connection before catalog operations:
```python
ml = conn_manager.get_active_connection()
if ml is None:
    return "Error: No active catalog connection"
```
