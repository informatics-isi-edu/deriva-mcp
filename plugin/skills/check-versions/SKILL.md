---
name: check-versions
description: "Check if the DerivaML ecosystem (deriva-ml, skills plugin, MCP server) is up to date and offer to update outdated components. Use when the user asks about versions, updates, or whether their environment is current. Triggers on: 'check versions', 'am I up to date', 'update deriva', 'what version', 'upgrade packages'."
disable-model-invocation: true
---

# Check and Update DerivaML Ecosystem

Check whether the user's DerivaML components are up to date and offer to update outdated ones.

## How to Check

```bash
python <skill-dir>/scripts/check_versions.py
```

To automatically update outdated components:

```bash
python <skill-dir>/scripts/check_versions.py --update
```

To check/update a single component:

```bash
python <skill-dir>/scripts/check_versions.py --component deriva-ml --update
```

## What It Checks

| Component | What it checks | How it updates |
|-----------|---------------|----------------|
| **deriva-ml** | Installed package version vs latest GitHub release tag | `uv lock --upgrade-package deriva-ml && uv sync` |
| **skills** | Cached plugin version vs latest GitHub release tag | `/plugin install deriva` (must be run by user in Claude Code) |
| **mcp-server** | Docker container age vs latest repo commit | Rebuild via `docker compose up -d --build` |

The MCP server check adapts to the deployment mode:
- **Local dev Docker** (e.g., `deriva-mcp:dev`): Compares container creation time against latest repo commit, rebuilds via `docker compose up -d --build`
- **Registry Docker** (e.g., `ghcr.io/.../deriva-mcp:latest`): Checks remote registry for newer image, updates via `docker pull` + restart. New images are built automatically by GitHub Actions when the version is bumped.
- **Native/direct**: Compares installed package version against latest GitHub release tag, updates via `uv lock && uv sync`

## Interpreting Results

- **UP TO DATE** — No action needed
- **OUTDATED** — Component has a newer version available
- **UPDATED** — Component was successfully updated (when using `--update`)
- **UNKNOWN** — Could not determine status (network issue, not installed, etc.)
- **Dev version** — Installed from git HEAD, at or ahead of latest release. Normal for library developers.

## Behavior

- Run the check when the user invokes `/deriva:check-versions`
- Report which components are up to date and which are outdated
- For **deriva-ml**: can be updated automatically with `--update`
- For **skills**: tell the user to run `/plugin install deriva` to update
- For **mcp-server**: ask before rebuilding — it takes time and restarts the server
