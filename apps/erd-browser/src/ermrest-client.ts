// ERMrest client for fetching catalog schema and record counts
// In dev mode, requests go through Vite's proxy to avoid CORS issues.
// Chaise links are absolute URLs pointing at the real server.

import type { CatalogSchema, EnrichedTable } from "./types";
import { classifyTable } from "./types";

function getConfig() {
  const hostname = import.meta.env.VITE_CATALOG_HOST || "localhost";
  const catalogId = import.meta.env.VITE_CATALOG_ID || "1";
  const protocol = hostname === "localhost" ? "https" : "https";
  return { hostname, catalogId, protocol };
}

// API requests use relative paths → routed through Vite proxy in dev
function ermrestBase() {
  const { catalogId } = getConfig();
  return `/ermrest/catalog/${catalogId}`;
}

// Chaise links are absolute — they open in the browser directly
export function chaiseRecordsetUrl(schema: string, table: string): string {
  const { hostname, catalogId, protocol } = getConfig();
  return `${protocol}://${hostname}/chaise/recordset/#${catalogId}/${schema}:${table}`;
}

export function chaiseRecordUrl(
  schema: string,
  table: string,
  rid: string
): string {
  const { hostname, catalogId, protocol } = getConfig();
  return `${protocol}://${hostname}/chaise/record/#${catalogId}/${schema}:${table}/RID=${rid}`;
}

export function getCatalogInfo() {
  return getConfig();
}

export async function fetchSchema(): Promise<CatalogSchema> {
  const base = ermrestBase();
  const resp = await fetch(`${base}/schema`, {
    credentials: "include",
  });
  if (!resp.ok) {
    throw new Error(`Failed to fetch schema: ${resp.status} ${resp.statusText}`);
  }
  const raw = await resp.json();
  return parseErmrestSchema(raw);
}

export async function fetchTableCount(
  schema: string,
  table: string
): Promise<number> {
  const base = ermrestBase();
  const resp = await fetch(
    `${base}/aggregate/${schema}:${table}/cnt:=cnt(*)`,
    { credentials: "include" }
  );
  if (!resp.ok) return -1;
  const data = await resp.json();
  return data[0]?.cnt ?? -1;
}

export async function fetchSampleRows(
  schema: string,
  table: string,
  limit = 5
): Promise<Record<string, unknown>[]> {
  const base = ermrestBase();
  const resp = await fetch(
    `${base}/entity/${schema}:${table}?limit=${limit}`,
    { credentials: "include" }
  );
  if (!resp.ok) return [];
  return resp.json();
}

// Paged data fetching using ERMrest's @sort and @after for keyset pagination
export interface PagedResult {
  rows: Record<string, unknown>[];
  hasMore: boolean;
}

export async function fetchPagedData(
  schema: string,
  table: string,
  sortColumn: string,
  limit: number,
  afterValue?: string | null,
  searchTerm?: string,
  searchColumns?: string[]
): Promise<PagedResult> {
  const base = ermrestBase();

  // Build filter path for search
  let filterPath = "";
  if (searchTerm && searchColumns && searchColumns.length > 0) {
    // Use ERMrest's ciregexp for case-insensitive regex search across columns
    // Multiple columns are OR'd together using ;
    const escaped = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const filters = searchColumns.map(
      (col) => `${encodeURIComponent(col)}::ciregexp::${encodeURIComponent(escaped)}`
    );
    filterPath = `/${filters.join(";")}`;
  }

  let url = `${base}/entity/${schema}:${table}${filterPath}@sort(${encodeURIComponent(sortColumn)})`;
  if (afterValue !== undefined && afterValue !== null) {
    url += `@after(${encodeURIComponent(afterValue)})`;
  }
  url += `?limit=${limit + 1}`; // fetch one extra to detect hasMore

  const resp = await fetch(url, { credentials: "include" });
  if (!resp.ok) return { rows: [], hasMore: false };
  const data: Record<string, unknown>[] = await resp.json();

  const hasMore = data.length > limit;
  const rows = hasMore ? data.slice(0, limit) : data;
  return { rows, hasMore };
}

// Parse raw ERMrest /schema response into our CatalogSchema shape
function parseErmrestSchema(raw: any): CatalogSchema {
  const { hostname, catalogId } = getConfig();

  // ERMrest returns { schemas: { schemaName: { tables: { tableName: { ... } } } } }
  const schemas: CatalogSchema["schemas"] = {};
  const domainSchemas: string[] = [];
  let mlSchema = "deriva-ml";
  let defaultSchema = "";

  for (const [schemaName, schemaObj] of Object.entries(raw.schemas as Record<string, any>)) {
    // Skip system schemas
    if (schemaName === "" || schemaName === "public" || schemaName === "ERMrest") continue;

    if (schemaName === "deriva-ml") {
      mlSchema = schemaName;
    } else {
      domainSchemas.push(schemaName);
      if (!defaultSchema) defaultSchema = schemaName;
    }

    const tables: Record<string, any> = {};

    for (const [tableName, tableObj] of Object.entries(
      (schemaObj as any).tables as Record<string, any>
    )) {
      const columns = Object.values(
        (tableObj.column_definitions || []) as any[]
      ).map((col: any) => ({
        name: col.name,
        type: col.type?.typename || "unknown",
        nullok: col.nullok ?? true,
        comment: col.comment || "",
      }));

      // Filter system columns
      const systemCols = new Set(["RID", "RCT", "RMT", "RCB", "RMB"]);
      const userColumns = columns.filter((c) => !systemCols.has(c.name));

      const foreignKeys = (tableObj.foreign_keys || []).map((fk: any) => {
        const fkCols = fk.foreign_key_columns?.map((c: any) => c.column_name) || [];
        const refCols = fk.referenced_columns?.map((c: any) => c.column_name) || [];
        const refSchema = fk.referenced_columns?.[0]?.schema_name || "";
        const refTable = fk.referenced_columns?.[0]?.table_name || "";
        return {
          columns: fkCols,
          referenced_table: `${refSchema}.${refTable}`,
          referenced_columns: refCols,
        };
      });

      // Determine table type from annotations and column patterns
      const annotations = tableObj.annotations || {};
      const colNames = new Set(columns.map((c: any) => c.name));

      // Vocabulary: annotation tag or standard vocabulary column pattern
      const isVocabulary = Boolean(
        annotations["tag:misd.isi.edu,2015:vocabulary"] ||
        annotations["tag:isrd.isi.edu,2016:vocabulary"] ||
        (colNames.has("Name") && colNames.has("Description") &&
         colNames.has("Synonyms") && colNames.has("ID") && colNames.has("URI"))
      );

      // Asset: annotation tag or Filename+URL column pattern
      const isAsset = Boolean(
        annotations["tag:isrd.isi.edu,2017:asset"] ||
        Object.keys(annotations).some(k => k.includes("asset")) ||
        (colNames.has("Filename") && colNames.has("URL"))
      );
      // Association tables typically have only FKs and system columns
      const isAssociation =
        !isVocabulary &&
        !isAsset &&
        userColumns.length > 0 &&
        userColumns.every(
          (c) =>
            foreignKeys.some((fk: any) => fk.columns.includes(c.name))
        );

      // Extract visible-columns annotation
      // The annotation can have contexts: "compact", "detailed", "entry", "*"
      // We prefer "compact" for table browsing, fall back to "*"
      const visColsAnno = annotations["tag:isrd.isi.edu,2016:visible-columns"];
      let visibleColumns: string[] | undefined;
      if (visColsAnno) {
        const ctx = visColsAnno.compact || visColsAnno["*"] || visColsAnno.detailed;
        if (Array.isArray(ctx)) {
          // Each entry can be a string (column name) or an array/object (FK path)
          // We only extract simple column name strings
          visibleColumns = ctx
            .filter((entry: any) => typeof entry === "string")
            .filter((name: string) => !systemCols.has(name));
        }
      }

      // Extract display name
      const displayAnno =
        annotations["tag:isrd.isi.edu,2015:display"] ||
        annotations["tag:misd.isi.edu,2015:display"];
      const displayName = displayAnno?.name || undefined;

      // Extract row name pattern
      const tableDisplayAnno = annotations["tag:isrd.isi.edu,2016:table-display"];
      const rowNamePattern = tableDisplayAnno?.row_name?.row_markdown_pattern || undefined;

      tables[tableName] = {
        comment: tableObj.comment || "",
        is_vocabulary: isVocabulary,
        is_asset: isAsset,
        is_association: isAssociation,
        columns: userColumns,
        foreign_keys: foreignKeys,
        visible_columns: visibleColumns,
        display_name: displayName,
        row_name_pattern: rowNamePattern,
      };
    }

    schemas[schemaName] = { comment: (schemaObj as any).comment || "", tables };
  }

  return {
    domain_schemas: domainSchemas.sort(),
    default_schema: defaultSchema,
    ml_schema: mlSchema,
    hostname,
    catalog_id: catalogId,
    schemas,
  };
}

// Build enriched table list with counts
export async function buildEnrichedTables(
  schema: CatalogSchema,
  fetchCounts = true
): Promise<EnrichedTable[]> {
  const tables: EnrichedTable[] = [];

  for (const [schemaName, schemaInfo] of Object.entries(schema.schemas)) {
    for (const [tableName, tableInfo] of Object.entries(schemaInfo.tables)) {
      tables.push({
        name: tableName,
        schema: schemaName,
        qualifiedName: `${schemaName}.${tableName}`,
        info: tableInfo,
        recordCount: null,
        tableType: classifyTable(tableInfo, schemaName, schema.ml_schema),
      });
    }
  }

  if (fetchCounts) {
    // Fetch counts in parallel batches
    const batchSize = 10;
    for (let i = 0; i < tables.length; i += batchSize) {
      const batch = tables.slice(i, i + batchSize);
      const counts = await Promise.all(
        batch.map((t) => fetchTableCount(t.schema, t.name))
      );
      batch.forEach((t, j) => {
        t.recordCount = counts[j];
      });
    }
  }

  return tables;
}
