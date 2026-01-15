#!/bin/bash
# Build Docker image with version from setuptools_scm
#
# Usage:
#   ./scripts/docker-build.sh [tag]
#
# Examples:
#   ./scripts/docker-build.sh                    # Build with :latest tag
#   ./scripts/docker-build.sh v1.0.0             # Build with :v1.0.0 tag
#   ./scripts/docker-build.sh ghcr.io/org/repo   # Build with full image name

set -e

# Get version from setuptools_scm (try uv run first, then plain python)
VERSION=$(uv run python -c "from setuptools_scm import get_version; print(get_version())" 2>/dev/null || \
          python -c "from setuptools_scm import get_version; print(get_version())" 2>/dev/null || \
          echo "0.0.0.dev0")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Default image name and tag
IMAGE_NAME="${1:-deriva-ml-mcp:latest}"

echo "Building Docker image: $IMAGE_NAME"
echo "  Version: $VERSION"
echo "  Git commit: $GIT_COMMIT"
echo ""

docker build \
    --build-arg VERSION="$VERSION" \
    --build-arg GIT_COMMIT="$GIT_COMMIT" \
    -t "$IMAGE_NAME" \
    .

echo ""
echo "Build complete: $IMAGE_NAME"
echo ""
echo "To verify version:"
echo "  docker run --rm --entrypoint /bin/bash $IMAGE_NAME -c 'env | grep DERIVAML'"
