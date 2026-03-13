// Schema types matching deriva://catalog/schema JSON structure

export interface ColumnInfo {
  name: string;
  type: string;
  nullok: boolean;
  comment: string;
  annotations: Record<string, any>;
}

export interface ForeignKeyInfo {
  columns: string[];
  referenced_table: string; // "schema.TableName"
  referenced_columns: string[];
}

export interface FeatureInfo {
  name: string;
  feature_table: string;
}

export interface TableInfo {
  comment: string;
  is_vocabulary: boolean;
  is_asset: boolean;
  is_association: boolean;
  columns: ColumnInfo[];
  foreign_keys: ForeignKeyInfo[];
  features?: FeatureInfo[];
  visible_columns?: string[]; // from tag:isrd.isi.edu,2016:visible-columns "compact" context
  display_name?: string; // from tag:isrd.isi.edu,2015:display
  row_name_pattern?: string; // from tag:isrd.isi.edu,2016:table-display
  annotations: Record<string, any>; // raw annotation map keyed by tag URI
}

export interface SchemaInfo {
  comment: string;
  tables: Record<string, TableInfo>;
  annotations: Record<string, any>;
}

export interface CatalogSchema {
  domain_schemas: string[];
  default_schema: string;
  ml_schema: string;
  hostname: string;
  catalog_id: string;
  schemas: Record<string, SchemaInfo>;
}

// Enriched table with schema context + computed data
export interface EnrichedTable {
  name: string;
  schema: string;
  qualifiedName: string; // "schema.TableName"
  info: TableInfo;
  recordCount: number | null;
  tableType: TableType;
}

export type TableType = "domain" | "ml" | "vocabulary" | "asset" | "association";

export function classifyTable(
  table: TableInfo,
  schemaName: string,
  mlSchema: string
): TableType {
  if (table.is_vocabulary) return "vocabulary";
  if (table.is_asset) return "asset";
  if (table.is_association) return "association";
  if (schemaName === mlSchema) return "ml";
  return "domain";
}

export type SchemaFilter = "all" | "domain" | "ml" | "vocabulary" | "asset";
