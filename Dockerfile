# DerivaML MCP Server
# Multi-stage build for smaller final image

# Build arguments for version injection
# These should be set by the build script using setuptools_scm:
#   VERSION=$(python -c "from setuptools_scm import get_version; print(get_version())")
#   GIT_COMMIT=$(git rev-parse --short HEAD)
#   docker build --build-arg VERSION=$VERSION --build-arg GIT_COMMIT=$GIT_COMMIT ...
ARG VERSION="0.0.0.dev0"
ARG GIT_COMMIT=""

# Build stage
FROM python:3.12-slim AS builder

# Re-declare build arg in this stage
ARG VERSION

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Use package-specific SETUPTOOLS_SCM_PRETEND_VERSION since .git is not available in Docker build
# The env var name uses normalized package name (deriva_ml_mcp)
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DERIVA_ML_MCP=${VERSION}
RUN uv pip install --no-cache .

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies (git needed for some deriva operations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create entrypoint wrapper script that fixes localhost resolution for Docker networking.
# When running in Docker with --add-host localhost:<webserver-ip>, Docker adds the custom
# entry AFTER the default 127.0.0.1 localhost entry. Since /etc/hosts is resolved first-match,
# the custom entry is ignored. This script comments out the default 127.0.0.1 localhost line
# so that the custom --add-host entry takes effect, allowing the container to reach the
# Deriva webserver when URLs use "localhost".
RUN printf '#!/bin/sh\n\
# Fix localhost resolution: comment out default localhost entries so --add-host takes effect\n\
# Cannot use sed -i on /etc/hosts (mounted file), so use cp to modify in place\n\
cp /etc/hosts /tmp/hosts.tmp\n\
sed -e "s/^127\\.0\\.0\\.1[[:space:]]\\+localhost/#&/" -e "s/^::1[[:space:]]\\+localhost/#&/" /tmp/hosts.tmp > /etc/hosts 2>/dev/null || true\n\
rm -f /tmp/hosts.tmp\n\
# Set default CA bundle for localhost with self-signed certificates if not already set\n\
: "${REQUESTS_CA_BUNDLE:=$HOME/.deriva/allCAbundle-with-local.pem}"\n\
export REQUESTS_CA_BUNDLE\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Re-declare build args in runtime stage (they don't persist across FROM)
ARG VERSION
ARG GIT_COMMIT

# Workflow metadata environment variables
# These are used to create the MCP workflow and execution on connection
ENV DERIVAML_MCP_WORKFLOW_NAME="DerivaML MCP Server"
ENV DERIVAML_MCP_WORKFLOW_TYPE="DerivaML MCP"
ENV DERIVAML_MCP_VERSION=${VERSION}
ENV DERIVAML_MCP_GIT_COMMIT=${GIT_COMMIT}
ENV DERIVAML_MCP_IMAGE_NAME="ghcr.io/informatics-isi-edu/deriva-ml-mcp"

# Flag to indicate running in Docker container
ENV DERIVAML_MCP_IN_DOCKER="true"

# MCP servers communicate via stdio
ENTRYPOINT ["/entrypoint.sh", "deriva-ml-mcp"]

# Labels for container metadata
LABEL org.opencontainers.image.title="DerivaML MCP Server"
LABEL org.opencontainers.image.description="Model Context Protocol server for DerivaML ML workflows"
LABEL org.opencontainers.image.source="https://github.com/informatics-isi-edu/deriva-ml-mcp"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version=${VERSION}
LABEL org.opencontainers.image.revision=${GIT_COMMIT}
