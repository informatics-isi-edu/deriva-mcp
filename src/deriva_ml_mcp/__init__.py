"""DerivaML MCP Server.

Model Context Protocol server for DerivaML, providing tools for
managing ML workflows, datasets, vocabularies, and features in
a Deriva catalog.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("deriva-mcp")
except PackageNotFoundError:
    # Package is not installed (running from source without install)
    __version__ = "0.0.0.dev0"
