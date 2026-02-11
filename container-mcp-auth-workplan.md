# MCP OAuth Integration & Docker Compose — Implementation Plan

## Summary

This plan covers two workstreams for adding authenticated HTTP transport to the `deriva-mcp` DerivaML MCP server, and running it in the `deriva-docker` Docker Compose stack:

**Workstream 1 — Additional Docker Integration:** Add deriva-mcp as a service in the `deriva-docker` compose stack, routed via Traefik at `/mcp/*`. Can be implemented before auth is ready using mounted `~/.deriva` credentials (same approach as the standalone Docker setup), but this is a temporary scaffold for internal testing -- not an end state.

**Workstream 2 — OIDC Authentication:** Secure the MCP server's streamable-http transport using OAuth 2.1, with Credenza as a narrow OAuth AS and the upstream IDP (Keycloak, Okta, Cognito, Globus) handling identity. The MCP server (deriva-mcp) acts as an OAuth Resource Server using FastMCP's built-in `IntrospectionTokenVerifier` to validate opaque bearer tokens against Credenza. Credenza adds a narrow slice of AS functionality (authorization code + PKCE issuance, token introspection, token exchange) on top of its existing session brokering. MCP sessions use a medium TTL (4-8h) with no refresh tokens or offline_access. Derived DERIVA tokens are short-lived (30 min) and re-obtained via RFC 8693 token exchange as needed.

**Key constraints driving the design:**
- Every major MCP client (Claude Desktop, Claude Code, VS Code) only supports `authorization_code` with PKCE -- no device_code flow support exists in the MCP ecosystem
- The MCP spec requires RFC 9728 (Protected Resource Metadata) on the resource server and RFC 8414 (OAuth Metadata) on the authorization server
- Tokens must be audience-bound (RFC 8707) to prevent cross-service replay between MCP and DERIVA services
- Credenza's existing resource-scoped opaque token pattern (`session.py:65-126`) is the foundation for the token scoping model

**Repos involved:** `deriva-docker`, `deriva-mcp`, `credenza`.

---

## Architecture Overview

Credenza already supports multiple OIDC identity providers (Keycloak, Okta, Cognito, Globus) via `oidc_idp_profiles.json`. The design preserves this -- no new endpoints are IDP-specific. The diagram below shows Keycloak because that is the IDP in the deriva-docker stack, but any configured IDP works identically.

```
+--------------+     +--------------+     +--------------+     +--------------+
|  MCP Client  |---->|   Traefik    |---->|  deriva-mcp  |---->|   DERIVA     |
|  (Claude)    |     |   /mcp/*     |     |  (FastMCP)   |     |  (ermrest,   |
|              |     |              |     | RS: RFC 9728 |     |   hatrac)    |
+------+-------+     +--------------+     +------+-------+     +------+-------+
       |                                    |    ^    |              ^
       |                                    |    |    | uses deriva  |
       |                                    |    |    |    token     |
       |             +--------------+       |    |    +--------------+
       +------------>|  Credenza    |<------+    |
        OAuth 2.1    |   Narrow     |  introspect (RFC 7662)
      AuthCode+PKCE  |     AS       |  token exchange (RFC 8693)
                     |              |------------+
                     +------+-------+  returns deriva-scoped token
                            |
                     +------v-------+
                     |  OIDC IDP    |
                     |  (Keycloak,  |
                     |  Okta, etc.) |
                     +--------------+
```

**Key roles:**
- **MCP Client** (Claude Code, Claude Desktop): Discovers auth via RFC 9728, obtains tokens via OAuth 2.1 through Credenza
- **deriva-mcp** (Resource Server): Validates opaque bearer tokens via RFC 7662 introspection against Credenza
- **Credenza** (Narrow OAuth AS / Session Broker): Credenza adds a narrow slice of Authorization Server functionality: authorization code issuance (with PKCE) and token exchange. This is required because every major MCP client (Claude Desktop, Claude Code, VS Code) only supports `authorization_code` with PKCE -- no MCP client implements the device_code flow, and the MCP spec doesn't mention it. Credenza delegates all identity work to the upstream IDP and continues to do what it already does (manage opaque sessions, validate resource bindings). The new AS surface is intentionally minimal: just `authorization_code` + PKCE, `token-exchange`, and introspection. No JWT issuance, no JWKS, no `client_credentials` grant.
- **OIDC Identity Provider** (Keycloak in deriva-docker, but also Okta, Cognito, Globus, others): The upstream OIDC provider that authenticates users and issues identity tokens. Credenza proxies the OAuth flow to the configured IDP and translates the result into its own opaque session tokens. The facade design is IDP-agnostic -- it works with any provider in `oidc_idp_profiles.json`.

**Token families:**
- `mcp:deriva-ml` — audience-bound to the MCP server (what the client presents)
- `deriva:ermrest`, `deriva:hatrac`, etc. — audience-bound to DERIVA services (obtained by MCP server via token exchange)

---

## Workstream 1: Docker Integration

This is the simpler workstream with no open design questions.

### 1.1 Create `deriva-docker/deriva/mcp/docker-compose.yml`

Follow the existing service pattern (credenza, groups, jupyter):

```yaml
services:
  deriva-mcp:
    image: ghcr.io/informatics-isi-edu/deriva-mcp:latest
    container_name: deriva-mcp
    hostname: deriva-mcp
    environment:
      - CONTAINER_HOSTNAME=${CONTAINER_HOSTNAME}
      - DERIVA_MCP_TRANSPORT=streamable-http
      - DERIVA_MCP_HOST=0.0.0.0
      - DERIVA_MCP_PORT=8000
      - DERIVA_MCP_LOCALHOST_ALIAS=deriva-webserver  # DNS alias for DERIVA services
      - CREDENZA_INTROSPECTION_URL=http://credenza:8999/introspect
      - CREDENZA_AS_METADATA_URL=https://${CONTAINER_HOSTNAME}/authn/.well-known/oauth-authorization-server
      - MCP_RESOURCE_IDENTIFIER=mcp:deriva-ml
    networks:
      - internal_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 10s
      retries: 4
      start_period: 10s
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.mcp.rule=(Host(`${CONTAINER_HOSTNAME}`) || Host(`${CONTAINER_HOSTNAME_INTERNAL}`)) && PathPrefix(`/mcp`)"
      - "traefik.http.routers.mcp.entrypoints=websecure"
      - "traefik.http.routers.mcp.tls=true"
      - "traefik.http.routers.mcp.tls.options=mintls13@file"
      - "traefik.http.routers.mcp.priority=50"
      - "traefik.http.routers.mcp.middlewares=mcp-stripprefix"
      - "traefik.http.services.mcp.loadbalancer.server.port=8000"
      - "traefik.http.middlewares.mcp-stripprefix.stripPrefix.prefixes=/mcp"
```

Notes:
- Priority 50 (below Credenza's 60, above Apache catch-all's 1)
- Same Traefik label pattern as Credenza
- `stripPrefix` so the MCP server sees requests at `/` not `/mcp/`
- Internal-only HTTP between Traefik and the container (TLS terminates at Traefik)
- Introspection call is internal HTTP to `credenza:8999` (never goes through Traefik)

### 1.2 Add to main `docker-compose.yml`

Add to `deriva-docker/deriva/docker-compose.yml`:

```yaml
    # DerivaML MCP Server
    deriva-mcp:
      profiles:
        - deriva-mcp
      extends:
        file: mcp/docker-compose.yml
        service: deriva-mcp
      depends_on:
        rproxy:
          condition: service_started
```

### 1.3 Update `generate-env.sh`

Add `--enable-mcp` flag:
- New variable: `ENABLE_MCP="false"` (line ~14)
- New parse case: `--enable-mcp|-m) ENABLE_MCP="true"; shift ;;`
- Profile addition: `[[ "$ENABLE_MCP" == "true" ]] && COMPOSE_PROFILES+=",deriva-mcp"`
- Auto-enable in test: `ENABLE_MCP="true"` (alongside `ENABLE_KEYCLOAK` and `ENABLE_JUPYTER`)
- Add MCP-specific env vars to the generated `.env` file

### 1.4 Dockerfile considerations

The existing `ghcr.io/informatics-isi-edu/deriva-mcp:latest` image should work as-is for the Docker integration. Auth configuration will be via environment variables, requiring no image changes for the Docker side.

---

## Workstream 2: OIDC Authentication

This is the complex workstream. It has two halves: the MCP server side (Resource Server) and the Credenza side (narrow OAuth AS).

### 2A: MCP Server — Resource Server (deriva-mcp changes)

**FastMCP provides almost everything out of the box.** Key findings:

| Requirement                            | FastMCP Support               | What we need                    |
|----------------------------------------|-------------------------------|---------------------------------|
| RFC 9728 (Protected Resource Metadata) | Automatic when `auth=` is set | Just configure it               |
| Bearer token validation                | `IntrospectionTokenVerifier`  | Point at Credenza `/introspect` |
| `WWW-Authenticate` on 401              | Built-in                      | Nothing                         |
| Resource identifier in metadata        | Configurable                  | Set `mcp:deriva-ml`             |

**Implementation in `server.py`:**

```python
from fastmcp.server.auth.providers.introspection import IntrospectionTokenVerifier

auth = IntrospectionTokenVerifier(
    introspection_url=os.environ["CREDENZA_INTROSPECTION_URL"],
    resource_id=os.environ.get("MCP_RESOURCE_IDENTIFIER", "mcp:deriva-ml"),
    authorization_server_url=os.environ.get("CREDENZA_AS_METADATA_URL"),
)

mcp = FastMCP(
    "deriva-ml",
    host=host,
    port=port,
    instructions=SERVER_INSTRUCTIONS,
    auth=auth,  # <-- this is all that's needed
)
```

FastMCP will then:
1. Serve `/.well-known/oauth-protected-resource` automatically (RFC 9728)
2. Validate every request's `Authorization: Bearer <token>` by calling Credenza's `/introspect`
3. Return 401 with `WWW-Authenticate` header pointing to Credenza's AS metadata URL
4. Pass `resource=mcp:deriva-ml` in introspection requests

**Additional MCP server work:**
- Add token exchange call to Credenza when making DERIVA API calls (RFC 8693). The MCP server receives an `mcp:*`-scoped token but needs `deriva:*`-scoped tokens for ermrest/hatrac. This belongs in `connection.py`'s `ConnectionManager`.
- Make auth conditional: only activate when env var is set (so stdio mode and standalone Docker still work without auth)

### 2B: Credenza — Narrow OAuth AS (credenza changes)

Credenza's core role remains the same: session broker that delegates identity to the upstream IDP. However, to support MCP clients, it must add a narrow slice of OAuth AS functionality. Research confirms that every major MCP client (Claude Desktop, Claude Code, VS Code, MCP Python/TypeScript SDKs) only implements `authorization_code` with PKCE. The device_code flow is not mentioned in the MCP spec and no MCP client supports it. This means Credenza must issue authorization codes -- there is no facade-only path.

**What's new (AS behavior):**
- Authorization code issuance (`/authorize` + code generation)
- Code-for-token exchange (`/token` with `grant_type=authorization_code`)

**What's a thin wrapper over existing logic:**
- OAuth metadata (RFC 8414) -- static discovery document
- Token introspection (RFC 7662) -- reformatting of existing `GET /session`
- Resource indicators (RFC 8707) -- extending existing `resource` param to all flows
- Token exchange (RFC 8693) -- new entry point into existing session minting

**What Credenza does NOT become:**
- No JWT issuance or JWKS endpoint
- No `client_credentials` grant
- No full-featured AS (no consent screens, no scope negotiation UI, no token revocation endpoint beyond existing session deletion)

Here's the gap analysis mapped to specific implementation tasks:

#### 2B.1 RFC 8414 — OAuth Metadata (NEW)

**File:** New blueprint `credenza/rest/oauth_metadata.py`

Endpoint: `GET /.well-known/oauth-authorization-server`

Static discovery document that tells MCP clients where Credenza's OAuth endpoints are. The MCP spec requires the resource server to point clients at a URL serving this metadata.

Must return JSON with at minimum:
```json
{
  "issuer": "https://{hostname}/authn",
  "authorization_endpoint": "https://{hostname}/authn/authorize",
  "token_endpoint": "https://{hostname}/authn/token",
  "introspection_endpoint": "https://{hostname}/authn/introspect",
  "response_types_supported": ["code"],
  "grant_types_supported": [
    "authorization_code",
    "urn:ietf:params:oauth:grant-type:device_code",
    "urn:ietf:params:oauth:grant-type:token-exchange"
  ],
  "code_challenge_methods_supported": ["S256"],
  "token_endpoint_auth_methods_supported": ["none"],
  "introspection_endpoint_auth_methods_supported": ["bearer"],
  "scopes_supported": ["openid", "profile", "email"],
  "resource_indicators_supported": true
}
```

Note: `"issuer"` here refers to who issued the opaque session token -- which IS Credenza (that's its existing job). The upstream IDP issues identity tokens; Credenza issues session tokens. This is unchanged from today.

Static metadata derived from Credenza's configuration. Low complexity. Can optionally include `"registration_endpoint"` if DCR (2B.6) is implemented.

#### 2B.2 RFC 8707 — Resource Indicators (EXTEND existing)

**Current state:** `resource` parameter is accepted in `service_flow.py` only. Device flow and login flow have no `resource` parameter.

**Changes needed:**

1. **`rest/login_flow.py`** — Accept `resource` query parameter on `/login`:
   - Store `resource` in `authn_request_ctx`
   - Pass `resource` through to the upstream IDP's authorization request (if the IDP supports it, otherwise store for later binding at callback time)
   - On callback, bind the resource to the created session's userinfo (same pattern as `service_flow.py:290-298`)

2. **`rest/device_flow.py`** — Accept `resource` parameter on `/device/start`:
   - Store `resource` in the device flow data
   - On token poll response, ensure the returned session is resource-bound

3. **`rest/session.py`** — Generalize resource binding validation:
   - Currently only enforced for `SessionType.service` (lines 65-126)
   - Extend to validate `resource` on ALL session types when the `resource` query param is present on `GET /session`
   - This is the introspection path — the MCP server will call `GET /session?resource=mcp:deriva-ml` to validate tokens

**Existing foundation:** The intersection-based resource validation at `session.py:65-126` is exactly the right pattern. It just needs to be applied more broadly.

#### 2B.3 RFC 7662 — Token Introspection (NEW endpoint, existing logic)

**File:** New blueprint or extend `rest/session.py`

**Endpoint:** `POST /introspect`

Credenza already does token introspection -- `GET /session` with a bearer token looks up the session, validates resource binding (for service tokens), and returns identity info. This is exactly what RFC 7662 specifies, just at a non-standard URL with non-standard field names. The new endpoint is a reformatting layer over the same logic.

**New standard endpoint needed:**

```
POST /introspect
Content-Type: application/x-www-form-urlencoded

token=<opaque_token>&resource=mcp:deriva-ml
```

Response (RFC 7662 format):
```json
{
  "active": true,
  "sub": "idp-issuer/user-uuid",
  "iss": "https://hostname/authn",
  "aud": ["mcp:deriva-ml"],
  "scope": "openid profile email",
  "exp": 1707600000,
  "iat": 1707596400,
  "groups": ["admin", "curator"],
  "email": "user@example.com"
}
```

Implementation approach:
- Reuse `session_store.get_session_by_session_key()` to look up the token (same as `get_current_session()`)
- Reuse the resource intersection logic from `session.py:65-126` (same validation, different entry point)
- Reformat output per RFC 7662 field names (same data as `make_session_response()`, different shape)
- Return `{"active": false}` for invalid/expired/resource-mismatched tokens (never 401/403 on introspection -- that's the RFC spec)

#### 2B.4 RFC 8693 — Token Exchange (NEW, but pattern exists)

**File:** New blueprint `credenza/rest/token_exchange.py`

**Endpoint:** `POST /token` with `grant_type=urn:ietf:params:oauth:grant-type:token-exchange`

Credenza already brokers sessions between upstream IDPs and downstream DERIVA services -- that is its core job. Token exchange is a new entry point into that same brokering: the MCP server presents an existing session token and asks Credenza to mint a new session scoped to a different resource. This is analogous to how `service_flow.py` already mints resource-scoped sessions from adapter-verified credentials, just with a different input (an existing Credenza session token instead of a client_secret or AWS presigned URL).

Request:
```
POST /token
Content-Type: application/x-www-form-urlencoded

grant_type=urn:ietf:params:oauth:grant-type:token-exchange
&subject_token=<mcp_opaque_token>
&subject_token_type=urn:ietf:params:oauth:token-type:access_token
&resource=deriva:ermrest
&scope=openid
```

Response:
```json
{
  "access_token": "<new_opaque_token>",
  "token_type": "Bearer",
  "expires_in": 1800,
  "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
  "scope": "openid"
}
```

Implementation:
1. Validate the `subject_token` (look up in session store via existing `get_session_by_session_key()`, verify active)
2. Verify the subject token is `mcp:*`-scoped (prevent lateral movement)
3. Verify the requested `resource` is an allowed target (e.g., `deriva:*`)
4. Check authorization: does this user/principal have access to the requested resource?
5. Mint a new opaque session token with `resource=[deriva:ermrest]` (reuse existing `service_flow.issue()`)
6. The DERIVA service validates this new token via `GET /session?resource=deriva:ermrest` (existing flow, unchanged)

#### 2B.5 OAuth Authorization Code Flow (NEW — genuine new logic)

This is the one piece of the facade that is NOT a thin wrapper over existing logic. Desktop MCP clients like Claude Desktop need the standard OAuth authorization code flow (open browser, authenticate, receive code at redirect_uri, exchange code for token). CLI clients like Claude Code can use the existing device flow instead (2B.2 adds `resource` support), but desktop clients cannot.

**What's genuinely new:**
- **Authorization code issuance**: `/authorize` must generate a short-lived authorization code, store it, and redirect to the client's `redirect_uri?code=X&state=Y`
- **Code-for-token exchange**: `/token` with `grant_type=authorization_code` must validate the code, look up the associated session, and return the opaque bearer token
- **PKCE validation**: The code exchange must verify `code_verifier` against the `code_challenge` sent at authorization time

**What's reused from existing logic:**
- The IDP redirect (same as `/login` -- redirect browser to upstream IDP)
- The IDP callback handling (same as `/callback` -- exchange IDP code for tokens, create session)
- Session creation and opaque token minting (same as existing `session_store.create_session()`)

**Flow:**
1. MCP client opens browser to `/authorize?response_type=code&client_id=X&redirect_uri=Y&scope=Z&resource=R&code_challenge=C&state=S`
2. Credenza stores request context (same as `authn_request_ctx` in `/login`), redirects to IDP
3. IDP authenticates user, redirects back to Credenza's internal callback
4. Credenza creates session (same as `/callback`), generates authorization code, stores code-to-session mapping
5. Credenza redirects to client's `redirect_uri?code=CODE&state=S`
6. Client exchanges code at `/token` -- Credenza validates code + PKCE, returns opaque bearer token

**Implementation approach:**

**Option A: New endpoints alongside existing flows** -- `/authorize` and `/token` are new routes that share the core IDP-redirect and session-creation logic with `/login` and `/callback` but differ in the entry point (accepts OAuth params, stores client redirect_uri) and exit point (redirects to client with code instead of setting cookie). The `/login` + `/callback` flow for browser sessions remains unchanged.

**Option B: Refactor login_flow.py** -- Generalize the existing flow to handle both browser-cookie and authorization-code modes. Cleaner long-term but higher risk to existing functionality.

Recommendation: **Option A** -- additive, doesn't touch existing flows, and the shared logic (IDP redirect, token exchange, session creation) can be extracted into common functions called by both the existing `/login` flow and the new `/authorize` flow.

#### 2B.6 RFC 7591 — Dynamic Client Registration (OPTIONAL, SHOULD)

**Endpoint:** `POST /register`

The MCP spec says this SHOULD be supported. It allows MCP clients to self-register without pre-configuration. Credenza already manages client configurations statically (via `oidc_idp_profiles.json` for IDP clients, `service_auth.json` for M2M bindings). DCR is a dynamic version of the same concept -- storing a client record so the facade knows which `redirect_uris` to accept.

For initial implementation, this can be minimal:
- Accept registration requests, store client metadata in the session store
- Return a `client_id` (no `client_secret` -- public clients only, per OAuth 2.1)
- Validate `redirect_uris` against allowed patterns

Lower priority. Can be deferred to a follow-up phase.

---

## Implementation Phases

### Phase 1: Docker Integration (Workstream 1) — No auth dependency

1. Create `deriva-docker/deriva/mcp/docker-compose.yml`
2. Add service to main `docker-compose.yml`
3. Update `generate-env.sh` with `--enable-mcp`
4. Test: MCP server starts, health check passes, Traefik routes `/mcp/*` correctly
5. Test: MCP client can connect via `https://{hostname}/mcp/mcp` (streamable-http)

Can be done immediately; no Credenza changes needed. Auth is disabled (env var not set).

### Phase 2: Credenza OAuth AS Surface

1. RFC 8414 metadata endpoint (`/.well-known/oauth-authorization-server`)
2. RFC 7662 introspection endpoint (`POST /introspect`)
3. Extend `resource` parameter to login and device flows (RFC 8707)
4. `/authorize` and `/token` endpoints (authorization code issuance + exchange)

### Phase 3: MCP Server Auth Activation

1. Add `IntrospectionTokenVerifier` to `server.py` (conditional on env var)
2. Test: MCP client discovers auth via `/.well-known/oauth-protected-resource`
3. Test: MCP client redirects to Credenza for OAuth
4. Test: Bearer token validation via introspection works end-to-end

### Phase 4: Token Exchange & DERIVA Access

1. RFC 8693 token exchange endpoint in Credenza
2. `ConnectionManager` in deriva-mcp exchanges MCP token for DERIVA token
3. Test: MCP tool calls flow through with proper DERIVA auth
4. Test: Token scoping prevents cross-service replay

### Phase 5: Polish

1. Dynamic Client Registration (RFC 7591) if needed
2. Error handling and edge cases
3. Documentation

---

## MCP Session Lifecycle

This is a critical design area. Credenza currently has three distinct session models, and we need to decide which model MCP sessions follow -- or whether they need a new one.

**Current Credenza session models:**

| Flow                           | Session TTL                                | Refresh Tokens                                | offline_access             | Max Lifetime                           | Refresh Behavior                            |
|--------------------------------|--------------------------------------------|-----------------------------------------------|----------------------------|----------------------------------------|---------------------------------------------|
| Browser (`/login`)             | 2100s (35 min), extended on `PUT /session` | Explicitly stripped (`login_flow.py:136-137`) | Not requested              | Until idle timeout                     | None -- session just expires                |
| Device (`/device/*`)           | Up to refresh expiry                       | Yes                                           | Yes (requested by default) | 14 days (`MAX_REFRESH_TOKEN_LIFETIME`) | Background worker refreshes upstream tokens |
| Service/M2M (`/service/token`) | Default 1800s (30 min)                     | No                                            | N/A                        | Clamped by `max_ttl_seconds` policy    | None -- fixed TTL, no extension             |

**Considerations for MCP sessions:**

- MCP connections over streamable-http can be long-running (hours of interactive use)
- The device flow's `offline_access` + 14-day lifetime + background refresh is almost certainly too permissive for a tool execution server
- The browser model's 35-minute TTL with idle extension is designed for interactive browser use where `PUT /session` is called on page loads -- MCP clients won't do this
- The service/M2M model's fixed 30-minute TTL with no extension is too short for interactive use
- When an MCP session expires, the client gets 401 and must re-authenticate (open browser again) -- too-short TTLs mean frequent re-auth disruptions

**Design decisions needed:**

| # | Question                                | Options                                                                          |
|---|-----------------------------------------|----------------------------------------------------------------------------------|
| 1 | MCP session TTL                         | Short (1-2h), medium (4-8h), long (24h), or configurable?                        |
| 2 | Refresh tokens for MCP sessions         | Yes (allow extending sessions) vs No (fixed lifetime, re-auth on expiry)         |
| 3 | offline_access scope                    | Request from IDP (enables upstream refresh) vs skip (no upstream refresh)        |
| 4 | Session extension mechanism             | MCP server periodically calls `PUT /session` vs refresh_token grant vs fixed TTL |
| 5 | Derived token (token exchange) lifetime | Shorter than MCP session? Same? Independent TTL?                                 |

**Decided approach:**

MCP sessions follow a model closer to browser sessions than device sessions:
- **No `offline_access`** -- don't request it from the IDP
- **No refresh tokens** -- like browser sessions, strip them
- **Medium TTL** (4-8 hours) -- long enough for a work session, short enough to limit exposure
- **No background refresh** -- session expires, client re-authenticates
- **Derived tokens (token exchange)** get shorter TTLs (30 min) matching the service/M2M model, since they're scoped to specific DERIVA resources and can be re-obtained via token exchange as long as the MCP session is active

This keeps the security posture tight while avoiding constant re-auth interruptions. The MCP server re-does token exchange for DERIVA tokens as needed (they're short-lived), but the user's MCP session stays alive for the work session.

---

## Open Design Decisions

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | Resource identifier format | `mcp:deriva-ml` vs `urn:deriva:mcp:ml` vs URI-based | `mcp:deriva-ml` (simple, matches `deriva:*` family) |
| 2 | Authorization code flow strategy | Option A (new endpoints alongside `/login`) vs Option B (refactor `/login`) | Option A (additive, non-breaking) |
| 3 | How MCP server obtains DERIVA tokens | Token exchange per-request vs cached with refresh | Cached with TTL, re-exchange when expired |
| 4 | Auth toggle mechanism | Env var (`DERIVA_MCP_AUTH_ENABLED`) vs separate entrypoint | Env var (simpler, Docker-friendly) |
| 5 | DCR scope | Full RFC 7591 vs minimal registration | Minimal first, expand if MCP clients need it |
| 6 | MCP session TTL | ~~Short / medium / long / configurable~~ | **DECIDED**: Medium (4-8h) |
| 7 | Refresh tokens for MCP sessions | ~~Yes vs No~~ | **DECIDED**: No -- like browser sessions |
| 8 | offline_access scope for MCP | ~~Request from IDP vs skip~~ | **DECIDED**: Skip |
| 9 | Session extension mechanism | ~~PUT /session vs refresh_token vs fixed TTL~~ | **DECIDED**: Fixed TTL |
| 10 | Derived token (token exchange) lifetime | ~~Same as MCP session vs shorter~~ | **DECIDED**: Shorter (30 min), re-exchange as needed |

---

## Credenza Gap Analysis (Reference)

Summary of what Credenza already does vs. what needs to be added:

| RFC | Status | What Credenza Already Does | Work Needed |
|-----|--------|----------------------------|-------------|
| RFC 8414 (OAuth Metadata) | NOT IMPLEMENTED | Legacy `discovery.py` (webauthn2 only) | New static metadata endpoint (thin) |
| RFC 8707 (Resource Indicators) | PARTIAL | `service_flow.py:110-155` validates resources at issuance; `session.py:65-126` validates at introspection | Extend resource param to login + device flows (thin) |
| RFC 7662 (Introspection) | EXISTS (non-standard URL/format) | `GET /session` with bearer token + resource binding check | Reformat as standard `POST /introspect` (thin) |
| RFC 8693 (Token Exchange) | NOT IMPLEMENTED | `service_flow.issue()` already mints resource-scoped sessions | New entry point into existing session minting (thin) |
| Authorization Code + PKCE | NOT IMPLEMENTED | `/login` + `/callback` do IDP redirect + session creation | **New AS behavior**: code issuance, code storage, code-for-token exchange |
| RFC 7591 (Dynamic Client Reg) | NOT IMPLEMENTED | Static client config in JSON files | Dynamic version of existing config, deferrable |

**Key insight:** Most of the new endpoints are thin wrappers over existing Credenza logic. The exception is authorization code issuance (2B.5), which is genuinely new AS functionality -- but it reuses the existing IDP redirect and session creation logic, so the novel code is limited to code generation, storage, and exchange.

---

## FastMCP Auth Capabilities (Reference)

FastMCP (>=2.11.0) and the `mcp` Python SDK (>=1.17.0) provide:

| Feature | Support Level |
|---|---|
| RFC 9728 (Protected Resource Metadata) | Automatic — served at `/.well-known/oauth-protected-resource` |
| Bearer token validation | Built-in — `IntrospectionTokenVerifier` for opaque tokens |
| `WWW-Authenticate` on 401 | Built-in |
| Starlette middleware injection | Supported since FastMCP 2.3.2 |
| `OAuthProvider` (self-hosted AS) | Available but not needed (Credenza serves the AS role) |
| `JWTVerifier` | Available but not needed (using opaque tokens) |

Provider types available:

| Provider | Use Case |
|---|---|
| `IntrospectionTokenVerifier` | Validate opaque tokens via RFC 7662 (our choice) |
| `JWTVerifier` | Validate JWTs against JWKS |
| `RemoteAuthProvider` | Delegate to DCR-capable IdP |
| `OAuthProxy` | Bridge non-DCR IdPs (GitHub, Google, Azure) |
| `OAuthProvider` | Self-hosted full OAuth AS |

---

## MCP Client Grant Type Support (Reference)

Research confirms that `authorization_code` with PKCE is the only universally supported OAuth flow. The `device_code` flow (RFC 8628) is not mentioned in the MCP spec and is not implemented by any major MCP client.

| Client | authorization_code + PKCE | device_code (RFC 8628) |
|--------|---------------------------|------------------------|
| MCP Spec (2025-06-18) | Only flow depicted; effectively required | Not mentioned |
| Claude Desktop | Yes | No |
| Claude Code (CLI) | Yes (opens browser) | No |
| MCP Python SDK | Yes | No (accepts in DCR metadata, does not implement) |
| MCP TypeScript SDK | Yes | No |
| VS Code | Yes | Declares in DCR metadata, unclear if implemented |
| FastMCP client | Yes | No |

This is why Credenza must support authorization code issuance (2B.5) -- there is no device-flow-only path.

---

## Files Changed Summary

**deriva-docker (Workstream 1):**
- `deriva/mcp/docker-compose.yml` — NEW
- `deriva/docker-compose.yml` — ADD service block
- `utils/generate-env.sh` — ADD `--enable-mcp` flag

**credenza (Workstream 2B):**
- `rest/oauth_metadata.py` — NEW (static RFC 8414 metadata)
- `rest/introspect.py` — NEW (RFC 7662 reformatting of existing `GET /session` logic)
- `rest/token_exchange.py` — NEW (RFC 8693 entry point into existing `service_flow.issue()`)
- `rest/oauth_endpoints.py` — NEW (authorization code flow: `/authorize` + `/token` with code issuance and exchange)
- `rest/login_flow.py` — MODIFY (accept `resource` parameter, store in session)
- `rest/device_flow.py` — MODIFY (accept `resource` parameter, store in session)
- `rest/session.py` — MODIFY (generalize resource binding validation to all session types)
- `app.py` — MODIFY (register new blueprints)

**deriva-mcp (Workstream 2A):**
- `server.py` — MODIFY (add conditional `auth=IntrospectionTokenVerifier(...)`)
- `connection.py` — MODIFY (add token exchange for DERIVA calls)
- `pyproject.toml` — MODIFY (bump `fastmcp` minimum version to >=2.11.0)