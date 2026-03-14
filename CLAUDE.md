# CLAUDE.md

This file provides guidance to Claude Code when working with the deriva-mcp codebase.

## Project Overview

DerivaML MCP Server is a Model Context Protocol (MCP) server that exposes DerivaML operations as tools and resources for LLM applications like Claude. It enables AI assistants to manage ML workflows, datasets, features, and executions in Deriva catalogs.

## Build and Development Commands

```bash
# Install dependencies
uv sync

# Run the MCP server directly (for testing)
uv run deriva-mcp

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Bump version (requires clean working tree — commit first)
uv run bump-version patch   # Bug fix (0.5.1 → 0.5.2)
uv run bump-version minor   # New feature (0.5.1 → 0.6.0)
uv run bump-version major   # Breaking change (0.5.1 → 1.0.0)
```

**Version bumping:** Use `bump-version` from deriva-ml (not `bump-my-version` directly). It finds the latest git tag, increments the version, creates a new tag, and pushes both the tag and commits to remote. The working tree must be clean (all changes committed) before bumping. Version is derived dynamically from git tags via `setuptools_scm`.

## Architecture

### Project Structure

```
src/deriva_mcp/
├── server.py          # Main MCP server entry point
├── connection.py      # ConnectionManager for catalog connections
├── resources.py       # MCP resources (config templates, catalog info, docs)
├── github_docs.py     # Fetches documentation from GitHub with caching
├── proxy.py           # Reverse proxy for serving web apps locally
└── tools/             # MCP tools organized by domain
    ├── __init__.py    # Exports all register_*_tools functions
    ├── catalog.py     # Connection and catalog management tools
    ├── dataset.py     # Dataset CRUD and versioning tools
    ├── vocabulary.py  # Controlled vocabulary tools
    ├── workflow.py    # Workflow management tools
    ├── feature.py     # Feature definition and value tools
    ├── schema.py      # Table and asset creation tools
    ├── execution.py   # ML execution lifecycle tools
    ├── data.py        # Data query and manipulation tools
    └── devtools.py    # Version management, app launcher, notebook execution
```

### Key Components

**server.py**: FastMCP server initialization and tool/resource registration. Entry point via `deriva-mcp` command.

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
| devtools.py | Dev utilities | bump_version, start_app, list_apps, stop_app, run_notebook |

**proxy.py**: Reverse proxy that serves web app static files and proxies `/ermrest`, `/authn`, `/chaise` to a remote Deriva server. Also provides `/api/storage` endpoints for local filesystem operations (listing and deleting cached datasets/executions in `~/.deriva-ml/`). Uses only Python stdlib — no external dependencies.

**devtools.py**: Development utility tools including semantic version management (`bump_version`), generalized app launcher (`list_apps`, `start_app`, `stop_app`), and notebook execution (`inspect_notebook`, `run_notebook`). The app launcher discovers apps from an `apps.json` catalog in the `deriva-ml-apps` repo.

### Resource Categories

| URI Pattern | Type | Description |
|-------------|------|-------------|
| `deriva://config/*` | Static | Hydra-zen configuration templates |
| `deriva://catalog/*` | Dynamic | Current catalog state (requires connection) |
| `deriva://dataset/{rid}` | Template | Specific dataset details |
| `deriva://vocabulary/{name}` | Template | Specific vocabulary terms |
| `deriva://docs/*` | Dynamic | Documentation fetched from GitHub repos |

## Installation

See [README.md](README.md) for complete installation options (Docker, source, Claude Desktop vs Claude Code configs, localhost setup).

## Adding New Tools

1. Create or edit the appropriate module in `src/deriva_mcp/tools/`
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
    "deriva://my-resource/{param}",
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

## Testing

```bash
uv run pytest                    # Run all tests
uv run pytest tests/test_devtools.py  # Test devtools registration
```

Tests use a mock `ConnectionManager` — no live catalog needed. When adding new tools to a module, update the corresponding test file's tool registration assertions.

## Gotchas

- **Version from git tags** — version is derived dynamically via `setuptools_scm`. No hardcoded version. Working tree must be clean before `bump-version`.
- **Proxy SSL verification disabled** — `proxy.py` disables TLS cert verification for backend connections (common for self-signed dev certs). Never use in production.
- **Proxy is single-instance** — only one app can run at a time. `start_app()` auto-stops any previously running app.
- **App discovery** — `start_app()` looks for `deriva-ml-apps` repo as a sibling directory, or set `DERIVA_ML_APPS_PATH` env var.
- **`deriva-ml` import is lazy** — `proxy.py` imports `deriva_ml.cache_tui` only when the storage API is hit. If `deriva-ml` isn't installed, storage endpoints return a descriptive error.
