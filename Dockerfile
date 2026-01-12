# DerivaML MCP Server
# Multi-stage build for smaller final image

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

# Set home directory for the non-root user
# Deriva credentials should be mounted at ~/.deriva at runtime
ENV HOME=/home/mcpuser

# MCP servers communicate via stdio
ENTRYPOINT ["deriva-ml-mcp"]

# Labels for container metadata
LABEL org.opencontainers.image.title="DerivaML MCP Server"
LABEL org.opencontainers.image.description="Model Context Protocol server for DerivaML ML workflows"
LABEL org.opencontainers.image.source="https://github.com/informatics-isi-edu/deriva-ml-mcp"
LABEL org.opencontainers.image.licenses="Apache-2.0"
