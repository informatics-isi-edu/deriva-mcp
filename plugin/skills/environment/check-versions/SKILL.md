---
name: check-versions
description: "Run this skill proactively at the START of every new conversation to check if the DerivaML ecosystem is up to date. Also use when the user asks about versions, updates, or whether their environment is current. Triggers on: session start, 'check versions', 'am I up to date', 'update deriva', 'what version', 'upgrade packages'."
---

# Check DerivaML Ecosystem Versions

Run the version checker script at the start of every session to ensure the user's environment is current.

## When to Run

- **Every new session** — Run this automatically before any other work
- When the user asks about versions or updates
- After the user clones or sets up a new project

## How to Check

Run the bundled script:

```bash
python <skill-dir>/scripts/check_versions.py
```

This checks:
1. **deriva-ml** — Compares installed version against latest GitHub release tag
2. **deriva-mcp** — Compares installed/running server version against latest GitHub release tag

## Interpreting Results

- **UP TO DATE** — No action needed
- **OUTDATED** — Report to the user with the update command from the script output
- **Dev version** — Installed from git HEAD, ahead of or at the latest release. This is normal for developers working on the library itself.
- **UNKNOWN** — Could not determine (network issue, not installed, etc.)

## Update Commands

If components are outdated, suggest:

- **deriva-ml**: `uv lock --upgrade-package deriva-ml && uv sync`
- **deriva-mcp**: Pull latest from the repository and rebuild (or `uv lock --upgrade-package deriva-mcp && uv sync` if installed as a package)

## Behavior

- Run silently at session start — only report if something is outdated or an error occurs
- If everything is up to date, don't mention it unless the user specifically asked
- If outdated, briefly mention which component and the update command
- Don't block the user's actual request — report version status concisely and move on
