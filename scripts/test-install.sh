#!/bin/bash
# Test script for verifying both STDIO and HTTP Docker install modes.
# Tests that authenticated connections to a Deriva catalog work.
#
# Usage:
#   ./scripts/test-install.sh [stdio|http|both]
#
# Requires:
#   - Docker running with deriva-localhost stack
#   - Valid credentials in ~/.deriva/credential.json
#   - Built image: deriva-mcp:test

set -e

IMAGE="deriva-mcp:test"
NETWORK="deriva-localhost_internal_network"
WEBSERVER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' deriva-webserver 2>/dev/null || echo "")
HOME_DIR="$HOME"

if [ -z "$WEBSERVER_IP" ]; then
    echo "ERROR: Could not find deriva-webserver container IP"
    echo "Is the deriva-localhost stack running?"
    exit 1
fi

echo "Using webserver IP: $WEBSERVER_IP"
echo "Home dir: $HOME_DIR"
echo ""

# MCP protocol messages
INIT_MSG='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
INIT_NOTIFY='{"jsonrpc":"2.0","method":"notifications/initialized"}'
CONNECT_MSG='{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"connect_catalog","arguments":{"hostname":"localhost","catalog_id":"1"}}}'

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASS${NC}: $1"; }
fail() { echo -e "${RED}FAIL${NC}: $1"; }
info() { echo -e "${YELLOW}INFO${NC}: $1"; }

#############################
# Test 1: STDIO mode
#############################
test_stdio() {
    echo "=========================================="
    echo "TEST: STDIO Docker Install"
    echo "=========================================="

    info "Sending MCP initialize + connect_catalog via STDIO (--add-host)..."

    # Send init, notification, then connect, with small delays
    # STDIO mode uses --add-host for localhost remapping
    RESULT=$(printf '%s\n%s\n%s\n' "$INIT_MSG" "$INIT_NOTIFY" "$CONNECT_MSG" | \
        docker run -i --rm \
            --network "$NETWORK" \
            --add-host "localhost:$WEBSERVER_IP" \
            -e "HOME=$HOME_DIR" \
            -v "$HOME_DIR/.deriva:$HOME_DIR/.deriva:ro" \
            -v "$HOME_DIR/.bdbag:$HOME_DIR/.bdbag" \
            -v "$HOME_DIR/.deriva-ml:$HOME_DIR/.deriva-ml" \
            "$IMAGE" 2>/dev/null)

    echo ""
    echo "--- Raw output (last 3 lines) ---"
    echo "$RESULT" | tail -3
    echo "--- End output ---"
    echo ""

    # Check for successful connection (handle escaped JSON in JSON)
    if echo "$RESULT" | grep -q 'connected'; then
        pass "STDIO mode - catalog connection succeeded"
    elif echo "$RESULT" | grep -q 'status'; then
        info "Got a status response but not 'connected':"
        echo "$RESULT" | grep 'status' | tail -1
        fail "STDIO mode - connection did not return 'connected'"
        return 1
    else
        fail "STDIO mode - no connection response found"
        return 1
    fi

    # Check for domain schema detection
    if echo "$RESULT" | grep -q 'domain_schema'; then
        pass "STDIO mode - domain schema detected"
    else
        info "STDIO mode - domain schema not found in output (may still be ok)"
    fi

    echo ""
}

#############################
# Test 2: HTTP mode
#############################
test_http() {
    echo "=========================================="
    echo "TEST: HTTP Docker Install (streamable-http)"
    echo "=========================================="

    CONTAINER_NAME="deriva-mcp-http-test"

    # Clean up any previous test container
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    info "Starting HTTP server container (DERIVA_MCP_LOCALHOST_ALIAS=deriva-webserver)..."
    # HTTP mode uses DERIVA_MCP_LOCALHOST_ALIAS for DNS-based localhost remapping
    # (no hardcoded IPs needed - the entrypoint resolves the DNS name at startup)
    docker run -d --rm \
        --name "$CONTAINER_NAME" \
        --network "$NETWORK" \
        -p 18000:8000 \
        -e "HOME=$HOME_DIR" \
        -e "DERIVA_MCP_LOCALHOST_ALIAS=deriva-webserver" \
        -v "$HOME_DIR/.deriva:$HOME_DIR/.deriva:ro" \
        -v "$HOME_DIR/.bdbag:$HOME_DIR/.bdbag" \
        -v "$HOME_DIR/.deriva-ml:$HOME_DIR/.deriva-ml" \
        "$IMAGE" \
        deriva-mcp --transport streamable-http --host 0.0.0.0 --port 8000 \
        > /dev/null

    # Wait for server to be ready
    info "Waiting for HTTP server to start..."
    for i in $(seq 1 30); do
        if curl -sf --max-time 2 http://localhost:18000/health > /dev/null 2>&1; then
            break
        fi
        if [ "$i" -eq 30 ]; then
            fail "HTTP server did not start within 30 seconds"
            docker logs "$CONTAINER_NAME" 2>&1 | tail -20
            docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
            return 1
        fi
        sleep 1
    done
    pass "HTTP server health check passed"

    # Test health endpoint
    HEALTH=$(curl -sf http://localhost:18000/health 2>/dev/null)
    echo "Health response: $HEALTH"
    if echo "$HEALTH" | grep -q '"ok"'; then
        pass "HTTP mode - /health endpoint working"
    else
        fail "HTTP mode - /health endpoint not working"
    fi

    # Test MCP initialization via HTTP
    info "Sending MCP initialize via HTTP POST to /mcp..."
    INIT_RESPONSE=$(curl -sf -X POST http://localhost:18000/mcp \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d "$INIT_MSG" 2>/dev/null)

    echo ""
    echo "--- Init response (first 500 chars) ---"
    echo "$INIT_RESPONSE" | head -c 500
    echo ""
    echo "--- End init response ---"
    echo ""

    # Extract session ID from response (if using SSE)
    SESSION_ID=""
    if echo "$INIT_RESPONSE" | grep -q "Mcp-Session-Id"; then
        SESSION_ID=$(echo "$INIT_RESPONSE" | grep -o 'Mcp-Session-Id: [^ ]*' | head -1 | cut -d' ' -f2)
        info "Got session ID: $SESSION_ID"
    fi

    # For streamable-http, try to get session from headers
    if [ -z "$SESSION_ID" ]; then
        info "Trying to get session from response headers..."
        HEADERS=$(curl -sf -D - -X POST http://localhost:18000/mcp \
            -H "Content-Type: application/json" \
            -H "Accept: application/json, text/event-stream" \
            -d "$INIT_MSG" -o /dev/null 2>/dev/null)
        SESSION_ID=$(echo "$HEADERS" | grep -i "mcp-session-id" | tr -d '\r' | awk '{print $2}')
        if [ -n "$SESSION_ID" ]; then
            info "Got session ID from headers: $SESSION_ID"
        fi
    fi

    if [ -n "$SESSION_ID" ]; then
        pass "HTTP mode - MCP session established"

        # Send initialized notification
        curl -sf --max-time 5 -X POST http://localhost:18000/mcp \
            -H "Content-Type: application/json" \
            -H "Mcp-Session-Id: $SESSION_ID" \
            -d "$INIT_NOTIFY" > /dev/null 2>&1 || true

        sleep 1

        # Test connect_catalog using --max-time to avoid hanging on SSE stream
        info "Sending connect_catalog via HTTP..."
        CONNECT_RESPONSE=$(curl -s --max-time 60 -X POST http://localhost:18000/mcp \
            -H "Content-Type: application/json" \
            -H "Accept: application/json, text/event-stream" \
            -H "Mcp-Session-Id: $SESSION_ID" \
            -d "$CONNECT_MSG" 2>/dev/null || true)

        echo ""
        echo "--- Connect response (first 1000 chars) ---"
        echo "$CONNECT_RESPONSE" | head -c 1000
        echo ""
        echo "--- End connect response ---"
        echo ""

        if echo "$CONNECT_RESPONSE" | grep -q "connected"; then
            pass "HTTP mode - catalog connection succeeded"
        else
            info "Connection response may be in SSE format, checking..."
            if echo "$CONNECT_RESPONSE" | grep -q "domain_schema\|catalog_id"; then
                pass "HTTP mode - catalog connection succeeded (SSE format)"
            else
                fail "HTTP mode - catalog connection did not succeed"
            fi
        fi
    else
        info "Could not extract session ID - server may use different session management"
        # Still check if we got a valid MCP response
        if echo "$INIT_RESPONSE" | grep -q "serverInfo\|protocolVersion\|capabilities"; then
            pass "HTTP mode - MCP protocol response received"
        else
            fail "HTTP mode - no valid MCP response"
        fi
    fi

    # Cleanup
    info "Stopping HTTP test container..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    echo ""
}

#############################
# Main
#############################
MODE="${1:-both}"

case "$MODE" in
    stdio)
        test_stdio
        ;;
    http)
        test_http
        ;;
    both|*)
        test_stdio
        test_http
        ;;
esac

echo "=========================================="
echo "Tests complete"
echo "=========================================="
