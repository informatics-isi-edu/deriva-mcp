// ERMrest client for fetching catalog schema and record counts.
// Uses catalog-config.ts for connection settings (URL hash params or env vars).
// Same-origin requests use credentials; cross-origin requests are anonymous.

import type { CatalogSchema, EnrichedTable } from "./types";
import { classifyTable } from "./types";
import {
  getCatalogConfig,
  ermrestBaseUrl,
  ermrestFetchOptions,
} from "./catalog-config";

// Re-export URL builders from catalog-config for backward compatibility
export { chaiseRecordsetUrl, chaiseRecordUrl } from "./catalog-config";

// Annotation tag constants (matching ERMRestJS and Chaise conventions)
const TAGS = {
  ASSET: "tag:isrd.isi.edu,2017:asset",
  DISPLAY: "tag:misd.isi.edu,2015:display",
  IGNORE: "tag:isrd.isi.edu,2016:ignore",
  TABLE_DISPLAY: "tag:isrd.isi.edu,2016:table-display",
  VISIBLE_COLUMNS: "tag:isrd.isi.edu,2016:visible-columns",
  VISIBLE_FOREIGN_KEYS: "tag:isrd.isi.edu,2016:visible-foreign-keys",
  VOCABULARY_MISD: "tag:misd.isi.edu,2015:vocabulary",
  VOCABULARY_ISRD: "tag:isrd.isi.edu,2016:vocabulary",
} as const;

export function getCatalogInfo() {
  const { hostname, catalogId } = getCatalogConfig();
  return { hostname, catalogId };
}

export async function fetchSchema(): Promise<CatalogSchema> {
  const base = ermrestBaseUrl();
  const opts = ermrestFetchOptions();
  const resp = await fetch(`${base}/schema`, opts);
  if (!resp.ok) {
    if (resp.status === 401) {
      const { isSameOrigin } = getCatalogConfig();
      throw new Error(
        isSameOrigin
          ? "Unauthorized — please log in to the Deriva server."
          : "Unauthorized — this catalog requires authentication. " +
            "Either the catalog is private, or the server needs CORS " +
            "configured to allow credentialed cross-origin requests."
      );
    }
    throw new Error(`Failed to fetch schema: ${resp.status} ${resp.statusText}`);
  }
  const raw = await resp.json();
  return parseErmrestSchema(raw);
}

export async function fetchTableCount(
  schema: string,
  table: string
): Promise<number> {
  const base = ermrestBaseUrl();
  const opts = ermrestFetchOptions();
  const resp = await fetch(
    `${base}/aggregate/${schema}:${table}/cnt:=cnt(*)`,
    opts
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
  const base = ermrestBaseUrl();
  const opts = ermrestFetchOptions();
  const resp = await fetch(
    `${base}/entity/${schema}:${table}?limit=${limit}`,
    opts
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
  const base = ermrestBaseUrl();
  const opts = ermrestFetchOptions();

  // Build filter path for search
  let filterPath = "";
  if (searchTerm && searchColumns && searchColumns.length > 0) {
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
  url += `?limit=${limit + 1}`;

  const resp = await fetch(url, opts);
  if (!resp.ok) return { rows: [], hasMore: false };
  const data: Record<string, unknown>[] = await resp.json();

  const hasMore = data.length > limit;
  const rows = hasMore ? data.slice(0, limit) : data;
  return { rows, hasMore };
}

// ── Schema parsing ────────────────────────────────────────────────

const SYSTEM_COLS = new Set(["RID", "RCT", "RMT", "RCB", "RMB"]);
const SKIP_SCHEMAS = new Set(["", "public", "ERMrest"]);

// Standard vocabulary table columns (DerivaML convention)
const VOCAB_COLUMNS = ["Name", "Description", "Synonyms", "ID", "URI"];

function hasAssetAnnotation(tableObj: any): boolean {
  const annotations = tableObj.annotations || {};
  // Table-level asset annotation
  if (annotations[TAGS.ASSET]) return true;
  // Column-level asset annotations
  const cols = tableObj.column_definitions || [];
  for (const col of cols) {
    if (col.annotations?.[TAGS.ASSET]) return true;
  }
  // Heuristic: Filename + URL columns (common Deriva asset pattern)
  const colNames = new Set(cols.map((c: any) => c.name));
  return colNames.has("Filename") && colNames.has("URL");
}

function hasVocabularyAnnotation(tableObj: any): boolean {
  const annotations = tableObj.annotations || {};
  if (annotations[TAGS.VOCABULARY_MISD] || annotations[TAGS.VOCABULARY_ISRD]) return true;
  // Heuristic: has all standard vocabulary columns
  const colNames = new Set((tableObj.column_definitions || []).map((c: any) => c.name));
  return VOCAB_COLUMNS.every((c) => colNames.has(c));
}

function isIgnored(tableObj: any): boolean {
  return Boolean(tableObj.annotations?.[TAGS.IGNORE]);
}

function parseErmrestSchema(raw: any): CatalogSchema {
  const { hostname, catalogId } = getCatalogConfig();

  const schemas: CatalogSchema["schemas"] = {};
  const domainSchemas: string[] = [];
  let mlSchema = "deriva-ml";
  let defaultSchema = "";

  for (const [schemaName, schemaObj] of Object.entries(raw.schemas as Record<string, any>)) {
    if (SKIP_SCHEMAS.has(schemaName)) continue;

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
      // Skip tables marked as ignored
      if (isIgnored(tableObj)) continue;

      const columns = (tableObj.column_definitions || []).map((col: any) => ({
        name: col.name,
        type: col.type?.typename || "unknown",
        nullok: col.nullok ?? true,
        comment: col.comment || "",
        annotations: col.annotations || {},
      }));

      const userColumns = columns.filter((c: any) => !SYSTEM_COLS.has(c.name));

      const foreignKeys = (tableObj.foreign_keys || []).map((fk: any) => {
        const fkCols = fk.foreign_key_columns?.map((c: any) => c.column_name) || [];
        const refCols = fk.referenced_columns?.map((c: any) => c.column_name) || [];
        const refSchema = fk.referenced_columns?.[0]?.schema_name || "";
        const refTable = fk.referenced_columns?.[0]?.table_name || "";
        return {
          columns: fkCols,
          referenced_table: refSchema && refTable ? `${refSchema}.${refTable}` : "",
          referenced_columns: refCols,
        };
      }).filter((fk: any) => fk.referenced_table); // drop malformed FKs

      const isVocabulary = hasVocabularyAnnotation(tableObj);
      const isAsset = !isVocabulary && hasAssetAnnotation(tableObj);

      // Association: non-vocab, non-asset table whose user columns are all FK columns
      const isAssociation =
        !isVocabulary &&
        !isAsset &&
        userColumns.length > 0 &&
        userColumns.every((c: any) =>
          foreignKeys.some((fk: any) => fk.columns.includes(c.name))
        );

      // Extract visible-columns annotation (prefer compact context)
      const annotations = tableObj.annotations || {};
      const visColsAnno = annotations[TAGS.VISIBLE_COLUMNS];
      let visibleColumns: string[] | undefined;
      if (visColsAnno) {
        const ctx = visColsAnno.compact || visColsAnno["*"] || visColsAnno.detailed;
        if (Array.isArray(ctx)) {
          visibleColumns = ctx
            .filter((entry: any) => typeof entry === "string")
            .filter((name: string) => !SYSTEM_COLS.has(name));
        }
      }

      // Display name from annotation
      const displayAnno = annotations[TAGS.DISPLAY];
      const displayName = displayAnno?.name || undefined;

      // Row name pattern from table-display annotation
      const tableDisplayAnno = annotations[TAGS.TABLE_DISPLAY];
      const rowNamePattern = tableDisplayAnno?.row_name?.row_markdown_pattern || undefined;

      // Extract features: look for DerivaML-style feature associations
      // Feature tables follow the pattern Execution_<TargetTable>_<FeatureName>
      // and have FKs to the target table, ML schema Execution, and Feature_Name
      const features = detectFeatures(tableName, schemaName, mlSchema, raw);

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
        features,
        annotations,
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

// Detect DerivaML feature tables that reference a given target table.
// Feature tables follow the naming pattern: Execution_<TargetTable>_<FeatureName>
// and must have FKs to all three: the target table, the ML schema's Execution table,
// and the ML schema's Feature_Name vocabulary.
function detectFeatures(
  targetTable: string,
  schemaName: string,
  mlSchema: string,
  rawSchema: any
): { name: string; feature_table: string }[] {
  const features: { name: string; feature_table: string }[] = [];
  const schemaObj = rawSchema.schemas?.[schemaName];
  if (!schemaObj?.tables) return features;

  const prefix = `Execution_${targetTable}_`;

  for (const [tblName, tblObj] of Object.entries(schemaObj.tables as Record<string, any>)) {
    if (!tblName.startsWith(prefix)) continue;

    const fks: any[] = (tblObj as any).foreign_keys || [];

    // Must have FK to target table in same schema
    const refsTarget = fks.some((fk: any) =>
      fk.referenced_columns?.some(
        (rc: any) => rc.table_name === targetTable && rc.schema_name === schemaName
      )
    );

    // Must have FK to Execution table in ML schema
    const refsExecution = fks.some((fk: any) =>
      fk.referenced_columns?.some(
        (rc: any) => rc.table_name === "Execution" && rc.schema_name === mlSchema
      )
    );

    // Must have FK to Feature_Name vocabulary in ML schema
    const refsFeatureName = fks.some((fk: any) =>
      fk.referenced_columns?.some(
        (rc: any) => rc.table_name === "Feature_Name" && rc.schema_name === mlSchema
      )
    );

    if (!refsTarget || !refsExecution || !refsFeatureName) continue;

    const featureName = tblName.slice(prefix.length);
    features.push({
      name: featureName,
      feature_table: `${schemaName}.${tblName}`,
    });
  }

  return features;
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
