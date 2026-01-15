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

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mcpuser
USER mcpuser

# Re-declare build args in runtime stage (they don't persist across FROM)
ARG VERSION
ARG GIT_COMMIT

# Set home directory for the non-root user
# Deriva credentials should be mounted at ~/.deriva at runtime
ENV HOME=/home/mcpuser

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
ENTRYPOINT ["deriva-ml-mcp"]

# Labels for container metadata
LABEL org.opencontainers.image.title="DerivaML MCP Server"
LABEL org.opencontainers.image.description="Model Context Protocol server for DerivaML ML workflows"
LABEL org.opencontainers.image.source="https://github.com/informatics-isi-edu/deriva-ml-mcp"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version=${VERSION}
LABEL org.opencontainers.image.revision=${GIT_COMMIT}
