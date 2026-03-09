---
name: check-versions
description: "Run this skill proactively at the START of every new conversation to check if the DerivaML ecosystem is up to date. Also use when the user asks about versions, updates, or whether their environment is current. Triggers on: session start, 'check versions', 'am I up to date', 'update deriva', 'what version', 'upgrade packages'."
---

# Check and Update DerivaML Ecosystem

Run the version checker at the start of every session to ensure the user's environment is current. The script checks three components and can automatically update them.

## When to Run

- **Every new session** — Run automatically before other work
- When the user asks about versions or updates
- After pulling code changes or setting up a new project

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
| **skills** | Local repo commits vs remote branch | `git pull --ff-only` in the deriva-mcp repo |
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

## Behavior on Session Start

- Run the check automatically
- If everything is up to date, say nothing (unless the user asked)
- If something is outdated, briefly report which component and offer to update
- For the MCP server, always ask before rebuilding — it takes time and restarts the server
- Don't block the user's actual request — report concisely and move on
