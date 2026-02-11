# Deriva MCP Server
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
# The env var name uses normalized package name (deriva_mcp)
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DERIVA_MCP=${VERSION}
RUN uv pip install --no-cache .

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies (git needed for some deriva operations, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create entrypoint wrapper script that fixes localhost resolution for Docker networking.
#
# Two mechanisms for remapping localhost to the Deriva webserver:
#
# 1. DERIVA_MCP_LOCALHOST_ALIAS (preferred for compose): Set to a Docker DNS name like
#    "deriva-webserver". The entrypoint resolves it to an IP at startup and adds it to
#    /etc/hosts as "localhost". No hardcoded IPs needed in compose files.
#
# 2. --add-host localhost:<ip> (for docker run): Docker adds the custom entry AFTER the
#    default 127.0.0.1 localhost entry. Since /etc/hosts is resolved first-match, the
#    entrypoint comments out the default entry so the custom one takes effect.
#
# Both approaches ensure that URLs referencing "localhost" reach the Deriva webserver.
RUN printf '#!/bin/sh\n\
# Remap localhost to Deriva webserver for Docker networking\n\
if [ -n "$DERIVA_MCP_LOCALHOST_ALIAS" ]; then\n\
    # Resolve DNS name to IP and add as localhost\n\
    ALIAS_IP=$(getent hosts "$DERIVA_MCP_LOCALHOST_ALIAS" 2>/dev/null | awk "{print \\$1}")\n\
    if [ -n "$ALIAS_IP" ]; then\n\
        cp /etc/hosts /tmp/hosts.tmp\n\
        sed -e "s/^127\\.0\\.0\\.1[[:space:]]\\+localhost/#&/" -e "s/^::1[[:space:]]\\+localhost/#&/" /tmp/hosts.tmp > /etc/hosts 2>/dev/null || true\n\
        echo "$ALIAS_IP localhost" >> /etc/hosts\n\
        rm -f /tmp/hosts.tmp\n\
    fi\n\
else\n\
    # Fix localhost resolution for --add-host: comment out default entries so custom one takes effect\n\
    cp /etc/hosts /tmp/hosts.tmp\n\
    sed -e "s/^127\\.0\\.0\\.1[[:space:]]\\+localhost/#&/" -e "s/^::1[[:space:]]\\+localhost/#&/" /tmp/hosts.tmp > /etc/hosts 2>/dev/null || true\n\
    rm -f /tmp/hosts.tmp\n\
fi\n\
# Set default CA bundle for localhost with self-signed certificates if not already set\n\
: "${REQUESTS_CA_BUNDLE:=$HOME/.deriva/allCAbundle-with-local.pem}"\n\
export REQUESTS_CA_BUNDLE\n\
# pip-system-certs/truststore ignores REQUESTS_CA_BUNDLE but respects SSL_CERT_FILE\n\
: "${SSL_CERT_FILE:=$REQUESTS_CA_BUNDLE}"\n\
export SSL_CERT_FILE\n\
# Warn if credentials directory is not found (likely missing -e HOME=$HOME)\n\
if [ ! -d "$HOME/.deriva" ]; then\n\
    echo "WARNING: $HOME/.deriva not found. Credentials may not be mounted correctly." >&2\n\
    echo "  Ensure you pass -e HOME=\\$HOME and -v \\$HOME/.deriva:\\$HOME/.deriva:ro" >&2\n\
fi\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Re-declare build args in runtime stage (they don't persist across FROM)
ARG VERSION
ARG GIT_COMMIT

# Workflow metadata environment variables
# These are used to create the MCP workflow and execution on connection
ENV DERIVA_MCP_WORKFLOW_NAME="Deriva MCP Server"
ENV DERIVA_MCP_WORKFLOW_TYPE="Deriva MCP"
ENV DERIVA_MCP_VERSION=${VERSION}
ENV DERIVA_MCP_GIT_COMMIT=${GIT_COMMIT}
ENV DERIVA_MCP_IMAGE_NAME="ghcr.io/informatics-isi-edu/deriva-mcp"

# Flag to indicate running in Docker container
ENV DERIVA_MCP_IN_DOCKER="true"

# Task persistence and SSE keepalive configuration
# These can be overridden at runtime via docker run -e or docker-compose
ENV DERIVA_MCP_TASK_STATE_PATH="/app/data/task_state.json"
ENV DERIVA_MCP_TASK_RETENTION_HOURS="168"
ENV DERIVA_MCP_TASK_SYNC_INTERVAL="5"
ENV DERIVA_MCP_SSE_KEEPALIVE="30"

# Create data directory for task persistence
RUN mkdir -p /app/data && chmod 755 /app/data

# Expose HTTP port for streamable-http transport
EXPOSE 8000

# Default entrypoint runs deriva-mcp with STDIO transport
# Override with: docker run ... deriva-mcp --transport streamable-http --port 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["deriva-mcp"]

# Labels for container metadata
LABEL org.opencontainers.image.title="Deriva MCP Server"
LABEL org.opencontainers.image.description="Model Context Protocol server for Deriva catalogs and ML workflows"
LABEL org.opencontainers.image.source="https://github.com/informatics-isi-edu/deriva-mcp"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version=${VERSION}
LABEL org.opencontainers.image.revision=${GIT_COMMIT}
