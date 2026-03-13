"""Reverse proxy for serving Deriva web apps locally.

Serves static files from an app's build directory and proxies API requests
(/ermrest, /authn, /chaise) to a remote Deriva server, forwarding cookies
to handle authentication. This avoids CORS issues during development.

Can be used standalone (python -m deriva_mcp.proxy) or started
programmatically via start_proxy() for use as an MCP tool.

Requirements: Python 3.10+ (stdlib only, no dependencies).
"""

from __future__ import annotations

import http.server
import logging
import mimetypes
import socket
import ssl
import threading
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths that get proxied to the Deriva backend
PROXY_PREFIXES = ("/ermrest", "/authn", "/chaise")


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
        if self._is_proxy_path():
            self._proxy_request("GET")
        else:
            self._serve_static()

    def do_POST(self) -> None:
        if self._is_proxy_path():
            self._proxy_request("POST")
        else:
            self.send_error(405, "POST only supported for proxied paths")

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
# Server lifecycle (for use as MCP tool)
# ---------------------------------------------------------------------------

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
        backend: Deriva server hostname (e.g., "dev.example.org").
        static_dir: Path to directory with built static files (index.html).
        port: Port to bind to. 0 = auto-select a free port.
        bind: Address to bind to.

    Returns:
        Tuple of (url, port) where the server is listening.

    Raises:
        RuntimeError: If a proxy is already running or static_dir is invalid.
    """
    global _active_server, _active_thread

    if _active_server is not None:
        raise RuntimeError("Proxy already running. Call stop_proxy() first.")

    static_dir = static_dir.resolve()
    if not (static_dir / "index.html").exists():
        raise FileNotFoundError(
            f"No index.html found in {static_dir}. "
            "Build the app first (e.g., cd erd-browser && pnpm build)."
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
  python -m deriva_mcp.proxy --backend dev.example.org --app /path/to/erd-browser/dist
  python -m deriva_mcp.proxy --backend dev.example.org --app /path/to/erd-browser/dist --port 9000
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
