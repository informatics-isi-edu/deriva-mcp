"""Reverse proxy for serving Deriva web apps locally.

Serves static files from an app's build directory and proxies API requests
(/ermrest, /authn, /chaise) to a remote Deriva server, forwarding cookies
to handle authentication. This avoids CORS issues during development.

Also provides local API endpoints for app-specific functionality:
- /api/storage — browse and delete cached datasets and execution directories

Can be used standalone (python -m deriva_mcp.proxy) or started
programmatically via start_proxy() for use as an MCP tool.

Requirements: Python 3.10+ (stdlib only, no dependencies).
"""

from __future__ import annotations

import http.server
import json
import logging
import mimetypes
import re
import shutil
import socket
import ssl
import threading
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_REQUEST_BODY = 1_048_576  # 1 MB

_RID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Paths that get proxied to the Deriva backend
PROXY_PREFIXES = ("/ermrest", "/authn", "/chaise")

# Local API paths handled by the proxy itself
API_PREFIXES = ("/api/",)


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    """Serves static files and proxies Deriva API requests."""

    backend: str  # e.g., "https://dev.example.org"
    static_dir: Path
    ssl_context: ssl.SSLContext

    def translate_path(self, path: str) -> str:
        """Resolve static file paths relative to the app directory."""
        path = urllib.parse.urlparse(path).path
        parts = path.strip("/").split("/")
        result = self.static_dir
        for part in parts:
            if part and part != "..":
                result = result / part
        return str(result)

    def do_GET(self) -> None:
        if self._is_api_path():
            self._handle_api("GET")
        elif self._is_proxy_path():
            self._proxy_request("GET")
        else:
            self._serve_static()

    def do_POST(self) -> None:
        if self._is_api_path():
            self._handle_api("POST")
        elif self._is_proxy_path():
            self._proxy_request("POST")
        else:
            self.send_error(405, "POST only supported for proxied or API paths")

    def do_PUT(self) -> None:
        if self._is_proxy_path():
            self._proxy_request("PUT")
        else:
            self.send_error(405, "PUT only supported for proxied paths")

    def do_DELETE(self) -> None:
        if self._is_proxy_path():
            self._proxy_request("DELETE")
        else:
            self.send_error(405, "DELETE only supported for proxied paths")

    def _is_proxy_path(self) -> bool:
        path = urllib.parse.urlparse(self.path).path
        return any(path.startswith(p) for p in PROXY_PREFIXES)

    def _is_api_path(self) -> bool:
        path = urllib.parse.urlparse(self.path).path
        return any(path.startswith(p) for p in API_PREFIXES)

    def _handle_api(self, method: str) -> None:
        """Route local API requests to the appropriate handler."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/storage" and method == "GET":
            self._api_storage_list(parsed.query)
        elif path == "/api/storage/delete" and method == "POST":
            self._api_storage_delete()
        else:
            self._send_json(404, {"status": "error", "error": f"Unknown API endpoint: {path}"})

    def _send_json(self, status: int, data: dict | list) -> None:
        """Send a JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _api_storage_list(self, query_string: str) -> None:
        """List storage entries from ~/.deriva-ml/."""
        params = urllib.parse.parse_qs(query_string)
        filter_type = params.get("filter", ["all"])[0]

        try:
            entries = _discover_storage_entries()
        except Exception as e:
            self._send_json(500, {"status": "error", "error": str(e)})
            return

        if filter_type == "cache":
            entries = [e for e in entries if e["category"] == "dataset"]
        elif filter_type == "executions":
            entries = [e for e in entries if e["category"] == "execution"]

        entries.sort(key=lambda e: e["size_bytes"], reverse=True)

        total_bytes = sum(e["size_bytes"] for e in entries)
        self._send_json(200, {
            "status": "success",
            "filter": filter_type,
            "entries": entries,
            "total_entries": len(entries),
            "total_size_bytes": total_bytes,
            "total_size": _human_size(total_bytes),
        })

    def _api_storage_delete(self) -> None:
        """Delete storage entries by RID."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"status": "error", "error": "Missing request body"})
            return

        if content_length > MAX_REQUEST_BODY:
            self._send_json(413, {"status": "error", "error": f"Request body too large (max {MAX_REQUEST_BODY} bytes)"})
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "error": "Invalid JSON"})
            return

        rids = body.get("rids", [])
        confirm = body.get("confirm", False)

        if not rids:
            self._send_json(400, {"status": "error", "error": "No RIDs provided"})
            return

        invalid_rids = [r for r in rids if not _RID_PATTERN.match(r)]
        if invalid_rids:
            self._send_json(400, {"status": "error", "error": f"Invalid RID format: {', '.join(invalid_rids)}"})
            return

        try:
            entries = _discover_storage_entries()
        except Exception as e:
            self._send_json(500, {"status": "error", "error": str(e)})
            return

        rid_set = set(rids)
        matches = [e for e in entries if e.get("rid") in rid_set]

        if not matches:
            self._send_json(200, {
                "status": "success",
                "message": f"No entries found matching: {', '.join(rids)}",
                "entries_found": 0,
            })
            return

        total_bytes = sum(e["size_bytes"] for e in matches)

        if not confirm:
            self._send_json(200, {
                "status": "dry_run",
                "message": f"Would delete {len(matches)} entries ({_human_size(total_bytes)})",
                "entries": matches,
                "total_bytes": total_bytes,
                "total_size": _human_size(total_bytes),
            })
            return

        deleted = []
        errors = []
        bytes_freed = 0
        for entry in matches:
            entry_path = Path(entry["path"])
            try:
                shutil.rmtree(entry_path)
                bytes_freed += entry["size_bytes"]
                deleted.append(entry)
            except Exception as e:
                errors.append({"path": entry["path"], "error": str(e)})

        # Clean up empty parent directories
        deriva_root = Path.home() / ".deriva-ml"
        for entry in deleted:
            entry_path = Path(entry["path"])
            if not entry_path.exists():
                parent = entry_path.parent
                while parent != deriva_root and parent.exists():
                    try:
                        if not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                        else:
                            break
                    except OSError:
                        break

        self._send_json(200, {
            "status": "success",
            "deleted": deleted,
            "entries_deleted": len(deleted),
            "bytes_freed": bytes_freed,
            "size_freed": _human_size(bytes_freed),
            "errors": errors,
        })

    def _serve_static(self) -> None:
        """Serve static files, falling back to index.html for SPA routing."""
        translated = self.translate_path(self.path)
        file_path = Path(translated)

        if file_path.is_file():
            super().do_GET()
            return

        # SPA fallback
        index = self.static_dir / "index.html"
        if index.is_file():
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            content = index.read_bytes()
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "Not found (and no index.html for SPA fallback)")

    def _proxy_request(self, method: str) -> None:
        """Forward request to the Deriva backend, passing cookies through."""
        target_url = self.backend + self.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        req = urllib.request.Request(target_url, data=body, method=method)

        for header in ("Cookie", "Content-Type", "Accept", "Authorization"):
            value = self.headers.get(header)
            if value:
                req.add_header(header, value)

        req.add_header("X-Forwarded-For", self.client_address[0])

        try:
            with urllib.request.urlopen(req, context=self.ssl_context) as resp:
                self.send_response(resp.status)
                for key, value in resp.getheaders():
                    lower = key.lower()
                    if lower in ("transfer-encoding", "connection", "keep-alive"):
                        continue
                    if lower == "set-cookie":
                        value = _rewrite_cookie_domain(value)
                    self.send_header(key, value)
                self.end_headers()
                while chunk := resp.read(65536):
                    self.wfile.write(chunk)

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, value in e.headers.items():
                lower = key.lower()
                if lower in ("transfer-encoding", "connection", "keep-alive"):
                    continue
                self.send_header(key, value)
            self.end_headers()
            body_bytes = e.read()
            if body_bytes:
                self.wfile.write(body_bytes)

        except urllib.error.URLError as e:
            self.send_error(502, f"Backend unreachable: {e.reason}")

    def log_message(self, format: str, *args) -> None:
        """Route HTTP access logs to the module logger."""
        logger.debug(format, *args)


def _rewrite_cookie_domain(cookie: str) -> str:
    """Remove Domain and Secure attributes so cookies work on localhost."""
    parts = []
    for part in cookie.split(";"):
        stripped = part.strip().lower()
        if stripped.startswith("domain=") or stripped == "secure":
            continue
        parts.append(part)
    return ";".join(parts)


def _find_free_port(start: int = 8080, end: int = 8180) -> int:
    """Find a free port in the given range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


# ---------------------------------------------------------------------------
# Storage API helpers
# ---------------------------------------------------------------------------

def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.1f} {units[i]}"


def _discover_storage_entries() -> list[dict]:
    """Import and call cache_tui.discover_entries(), returning dicts."""
    try:
        from deriva_ml.cache_tui import discover_entries
    except ImportError:
        raise RuntimeError(
            "deriva-ml is not installed or does not include cache_tui. "
            "Install with: pip install deriva-ml"
        )
    return [entry.to_dict() for entry in discover_entries()]


# ---------------------------------------------------------------------------
# Server lifecycle (for use as MCP tool)
# ---------------------------------------------------------------------------

_server_lock = threading.Lock()
_active_server: http.server.HTTPServer | None = None
_active_thread: threading.Thread | None = None


def start_proxy(
    backend: str,
    static_dir: Path,
    port: int = 0,
    bind: str = "127.0.0.1",
) -> tuple[str, int]:
    """Start the proxy server in a background thread.

    Args:
        backend: Deriva server hostname (e.g., "dev.example.org") or full URL
            (e.g., "https://dev.example.org"). A bare hostname will be prefixed
            with ``https://`` automatically.
        static_dir: Path to directory with built static files (index.html).
        port: Port to bind to. 0 = auto-select a free port.
        bind: Address to bind to.

    Returns:
        Tuple of (url, port) where the server is listening.

    Raises:
        RuntimeError: If a proxy is already running or static_dir is invalid.
    """
    global _active_server, _active_thread

    with _server_lock:
        if _active_server is not None:
            raise RuntimeError("Proxy already running. Call stop_proxy() first.")

        static_dir = static_dir.resolve()
        if not (static_dir / "index.html").exists():
            raise FileNotFoundError(
                f"No index.html found in {static_dir}. "
                "Build the app first (e.g., cd schema-workbench && pnpm build)."
            )

        if not backend.startswith("http"):
            backend = f"https://{backend}"

        if port == 0:
            port = _find_free_port()

        # SSL context for backend connections (accept self-signed certs)
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        # Configure handler class
        ProxyHandler.backend = backend
        ProxyHandler.static_dir = static_dir
        ProxyHandler.ssl_context = ssl_ctx

        # Ensure common MIME types
        mimetypes.add_type("application/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        mimetypes.add_type("image/svg+xml", ".svg")

        _active_server = http.server.HTTPServer((bind, port), ProxyHandler)
        _active_thread = threading.Thread(
            target=_active_server.serve_forever,
            daemon=True,
            name="deriva-proxy",
        )
        _active_thread.start()

        url = f"http://{bind}:{port}"
        logger.info(f"Proxy started: {url} -> {backend}")
        return url, port


def stop_proxy() -> None:
    """Stop the running proxy server."""
    global _active_server, _active_thread

    with _server_lock:
        if _active_server is not None:
            _active_server.shutdown()
            _active_server.server_close()
            _active_server = None
            logger.info("Proxy stopped")

        if _active_thread is not None:
            _active_thread.join(timeout=5)
            _active_thread = None


def is_proxy_running() -> bool:
    """Check if the proxy server is currently running."""
    with _server_lock:
        return _active_server is not None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Reverse proxy for Deriva web apps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m deriva_mcp.proxy --backend dev.example.org --app /path/to/schema-workbench/dist
  python -m deriva_mcp.proxy --backend dev.example.org --app /path/to/schema-workbench/dist --port 9000
        """,
    )
    parser.add_argument(
        "--backend", required=True,
        help="Deriva server hostname (e.g., dev.example.org)",
    )
    parser.add_argument(
        "--app", required=True,
        help="Path to directory with built static files (must contain index.html)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Local port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--bind", default="127.0.0.1",
        help="Address to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    app_path = Path(args.app)
    if app_path.is_dir() and (app_path / "index.html").exists():
        static_dir = app_path
    elif (app_path / "dist").is_dir():
        static_dir = app_path / "dist"
    else:
        print(f"Error: Cannot find static files at {app_path} or {app_path / 'dist'}")
        sys.exit(1)

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    url, port = start_proxy(args.backend, static_dir, args.port, args.bind)
    print(f"Serving {static_dir} at {url}")
    print(f"Proxying {', '.join(PROXY_PREFIXES)} -> https://{args.backend}")
    print("Press Ctrl+C to stop\n")

    try:
        _active_thread.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_proxy()


if __name__ == "__main__":
    main()
