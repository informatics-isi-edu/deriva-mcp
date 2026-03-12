// Catalog connection configuration.
// Priority: URL hash params > env vars > defaults.
//
// URL format: https://host/erd-browser/#host=dev.facebase.org&catalog=10
//
// When running via Vite dev proxy, env vars (VITE_CATALOG_HOST, VITE_CATALOG_ID)
// are used and requests go through the proxy to avoid CORS.
//
// When deployed to GitHub Pages (or any other origin), the URL params point at
// the Deriva server directly. Cross-origin requests skip credentials unless the
// server's CORS config supports them.

export interface CatalogConfig {
  hostname: string;
  catalogId: string;
  /** True when the app is served from the same origin as the Deriva server */
  isSameOrigin: boolean;
}

/** Parse hash params like #host=example.org&catalog=42 */
function parseHashParams(): Record<string, string> {
  const hash = window.location.hash.replace(/^#/, "");
  const params: Record<string, string> = {};
  for (const part of hash.split("&")) {
    const [k, v] = part.split("=");
    if (k && v) params[decodeURIComponent(k)] = decodeURIComponent(v);
  }
  return params;
}

let _config: CatalogConfig | null = null;

export function getCatalogConfig(): CatalogConfig {
  if (_config) return _config;

  const hashParams = parseHashParams();

  const hostname =
    hashParams.host ||
    import.meta.env.VITE_CATALOG_HOST ||
    "";
  const catalogId =
    hashParams.catalog ||
    import.meta.env.VITE_CATALOG_ID ||
    "";

  // Same-origin check: if hostname is empty or matches current origin
  const isSameOrigin =
    !hostname ||
    hostname === "localhost" ||
    hostname === window.location.hostname;

  _config = { hostname, catalogId, isSameOrigin };
  return _config;
}

/** Update the URL hash to reflect current catalog selection */
export function setCatalogInUrl(hostname: string, catalogId: string): void {
  window.location.hash = `host=${encodeURIComponent(hostname)}&catalog=${encodeURIComponent(catalogId)}`;
  _config = null; // force re-parse on next getCatalogConfig()
}

/** Check if we have enough config to connect */
export function hasCatalogConfig(): boolean {
  const { hostname, catalogId } = getCatalogConfig();
  return Boolean(hostname && catalogId);
}

/**
 * Build the base URL for ERMrest API requests.
 * - Same-origin: use relative paths (routed through Vite proxy in dev)
 * - Cross-origin: use absolute URL to the Deriva server
 */
export function ermrestBaseUrl(): string {
  const { hostname, catalogId, isSameOrigin } = getCatalogConfig();
  if (isSameOrigin) {
    return `/ermrest/catalog/${catalogId}`;
  }
  return `https://${hostname}/ermrest/catalog/${catalogId}`;
}

/**
 * Fetch options for ERMrest requests.
 * Only includes credentials for same-origin requests.
 */
export function ermrestFetchOptions(): RequestInit {
  const { isSameOrigin } = getCatalogConfig();
  return isSameOrigin ? { credentials: "include" } : {};
}

/** Build an absolute Chaise recordset URL */
export function chaiseRecordsetUrl(schema: string, table: string): string {
  const { hostname, catalogId } = getCatalogConfig();
  return `https://${hostname}/chaise/recordset/#${catalogId}/${schema}:${table}`;
}

/** Build an absolute Chaise record URL */
export function chaiseRecordUrl(
  schema: string,
  table: string,
  rid: string
): string {
  const { hostname, catalogId } = getCatalogConfig();
  return `https://${hostname}/chaise/record/#${catalogId}/${schema}:${table}/RID=${rid}`;
}
