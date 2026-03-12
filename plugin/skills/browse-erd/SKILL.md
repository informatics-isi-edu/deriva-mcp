---
name: browse-erd
description: "Use this skill to launch an interactive ERD (Entity-Relationship Diagram) browser for the currently connected Deriva catalog. Triggers on: 'browse erd', 'show erd', 'schema diagram', 'entity relationship diagram', 'visualize schema', 'catalog diagram', 'show relationships', 'explore schema visually'. Requires an active catalog connection."
---

# Interactive ERD Browser

Launch a local React application that provides an interactive visual ERD for the connected catalog.

## Prerequisites

- An active catalog connection (call `connect_catalog` first if needed)
- Node.js 18+ installed locally
- The `deriva-mcp` repository cloned locally

## Steps

### 1. Get catalog connection info

Read the catalog connection info. You need the hostname and catalog ID.

Use the MCP tool `get_record` or read `deriva://catalog/schema` to confirm the connection is active and get `hostname` and `catalog_id`.

### 2. Locate the ERD browser app

The app lives in the `deriva-mcp` repository at `apps/erd-browser/`. Find it relative to the plugin installation:

```bash
# The app is at: <deriva-mcp-repo>/apps/erd-browser/
# Find it relative to this skill's location
SKILL_DIR="$(dirname "$(find ~/.claude -path '*/skills/browse-erd/SKILL.md' -print -quit 2>/dev/null)")"
APP_DIR="$(cd "$SKILL_DIR/../../../apps/erd-browser" && pwd)"
```

If the app directory doesn't exist, tell the user they need to update their `deriva-mcp` repo:
```
git -C <repo> pull
```

### 3. Install dependencies (first run only)

```bash
cd "$APP_DIR"
if [ ! -d node_modules ]; then
  pnpm install
fi
```

### 4. Launch the dev server

Set environment variables for the catalog connection and start the Vite dev server:

```bash
cd "$APP_DIR"
VITE_CATALOG_HOST=<hostname> VITE_CATALOG_ID=<catalog_id> pnpm dev
```

This will:
- Start a local dev server (typically at http://localhost:5173)
- Open the browser automatically
- Connect to the catalog using the browser's existing authentication cookies

### 5. Tell the user

Let the user know:
- The ERD browser is running at the URL shown in the terminal
- They must be logged into the Deriva server in their browser for authentication to work
- Press Ctrl+C in the terminal to stop the server
- They can search, filter by table type, click tables for details, and link out to Chaise

## Features

- **Interactive graph**: Tables as nodes, foreign keys as directed edges, dagre-based hierarchical layout
- **Color-coded nodes**: Domain (slate), ML (amber), Vocabulary (emerald), Asset (sky), Association (zinc)
- **Detail panel**: Click any table to see columns, foreign keys, sample data, and features
- **Search**: Filter tables by name or schema
- **Type filter**: Show all / domain only / ML only / vocabulary only / asset only
- **Hide associations**: Toggle to reduce clutter from junction tables
- **Chaise links**: "Open in Chaise" button for each table
- **Zoom/pan**: Mouse wheel zoom, drag to pan, minimap for navigation

## Troubleshooting

- **CORS errors**: The browser must have valid cookies for the Deriva server. Log in via Chaise first.
- **Empty graph**: Check that the catalog has tables. Try `count_table` on a known table.
- **Auth failure**: For `localhost` catalogs, ensure the local server is running and you're logged in.
