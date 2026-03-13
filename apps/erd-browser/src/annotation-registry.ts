/**
 * Annotation tag registry — mirrors the tag constants from deriva.core.tag
 * and the type constraints from the workbench editors (raise_on_invalid).
 *
 * Provides human-readable names, descriptions, valid object types, and
 * property schemas for each annotation tag.
 */

export type ModelObjectType = "catalog" | "schema" | "table" | "column" | "key" | "foreignkey";

export interface AnnotationTagInfo {
  /** Full URI tag */
  tag: string;
  /** Short human-readable name */
  name: string;
  /** Which object types this annotation applies to */
  appliesTo: ModelObjectType[];
  /** One-line description */
  description: string;
  /** Whether this annotation uses context-based values */
  contextualized: boolean;
  /** Key properties in the annotation body */
  properties?: PropertyInfo[];
}

export interface PropertyInfo {
  name: string;
  type: "string" | "boolean" | "number" | "object" | "array" | "enum";
  description: string;
  enumValues?: string[];
  optional?: boolean;
}

// ── Tag URIs ────────────────────────────────────────────────────

export const TAG = {
  display: "tag:misd.isi.edu,2015:display",
  visible_columns: "tag:isrd.isi.edu,2016:visible-columns",
  visible_foreign_keys: "tag:isrd.isi.edu,2016:visible-foreign-keys",
  table_display: "tag:isrd.isi.edu,2016:table-display",
  column_display: "tag:isrd.isi.edu,2016:column-display",
  table_alternatives: "tag:isrd.isi.edu,2016:table-alternatives",
  foreign_key: "tag:isrd.isi.edu,2016:foreign-key",
  key_display: "tag:isrd.isi.edu,2017:key-display",
  asset: "tag:isrd.isi.edu,2017:asset",
  generated: "tag:isrd.isi.edu,2016:generated",
  immutable: "tag:isrd.isi.edu,2016:immutable",
  non_deletable: "tag:isrd.isi.edu,2016:non-deletable",
  required: "tag:isrd.isi.edu,2018:required",
  citation: "tag:isrd.isi.edu,2018:citation",
  source_definitions: "tag:isrd.isi.edu,2019:source-definitions",
  app_links: "tag:isrd.isi.edu,2016:app-links",
  export: "tag:isrd.isi.edu,2016:export",
  export_2019: "tag:isrd.isi.edu,2019:export",
  export_fragment_definitions: "tag:isrd.isi.edu,2021:export-fragment-definitions",
  chaise_config: "tag:isrd.isi.edu,2019:chaise-config",
  google_dataset: "tag:isrd.isi.edu,2021:google-dataset",
  bulk_upload: "tag:isrd.isi.edu,2017:bulk-upload",
  indexing_preferences: "tag:isrd.isi.edu,2018:indexing-preferences",
  column_defaults: "tag:isrd.isi.edu,2023:column-defaults",
  viz_3d_display: "tag:isrd.isi.edu,2021:viz-3d-display",
} as const;

// ── Registry ────────────────────────────────────────────────────

export const ANNOTATION_REGISTRY: AnnotationTagInfo[] = [
  // ── Display & Presentation ──
  {
    tag: TAG.display,
    name: "Display",
    appliesTo: ["catalog", "schema", "table", "column", "key"],
    description: "Control display name, styling, comment text, and null value presentation.",
    contextualized: true,
    properties: [
      { name: "name", type: "string", description: "Display name (plain text)", optional: true },
      { name: "markdown_name", type: "string", description: "Display name (markdown format, mutually exclusive with name)", optional: true },
      { name: "name_style", type: "object", description: "Automatic name formatting: underline_space, title_case, markdown", optional: true },
      { name: "comment", type: "object", description: "Tooltip text, contextualized (e.g., {\"*\": \"...\", \"compact\": false})", optional: true },
      { name: "comment_display", type: "object", description: "How comments render: table_comment_display, column_comment_display (inline or tooltip)", optional: true },
      { name: "show_null", type: "object", description: "How NULL values display: true (show), false (hide), or custom string", optional: true },
      { name: "show_foreign_key_link", type: "object", description: "Whether FK values render as links (contextualized boolean)", optional: true },
    ],
  },
  {
    tag: TAG.visible_columns,
    name: "Visible Columns",
    appliesTo: ["table"],
    description: "Control which columns appear and their order in each UI context.",
    contextualized: true,
    properties: [
      { name: "(context)", type: "array", description: "Array of column directives: column names (string), constraint refs ([schema, name]), or pseudo-columns (object with source/sourcekey)" },
    ],
  },
  {
    tag: TAG.visible_foreign_keys,
    name: "Visible Foreign Keys",
    appliesTo: ["table"],
    description: "Control which related tables appear as sections on the detail page.",
    contextualized: true,
    properties: [
      { name: "(context)", type: "array", description: "Array of FK directives: constraint refs ([schema, name]) or pseudo-columns (object)" },
    ],
  },
  {
    tag: TAG.table_display,
    name: "Table Display",
    appliesTo: ["table"],
    description: "Row naming patterns, sort order, page size, and markdown templates.",
    contextualized: true,
    properties: [
      { name: "row_order", type: "array", description: "Sort keys: column name strings or {column, descending} objects", optional: true },
      { name: "row_markdown_pattern", type: "string", description: "Handlebars template for rendering each row", optional: true },
      { name: "page_markdown_pattern", type: "string", description: "Handlebars template for the entire page", optional: true },
      { name: "separator_pattern", type: "string", description: "Separator between rows", optional: true },
      { name: "prefix_pattern", type: "string", description: "Content before rows", optional: true },
      { name: "suffix_pattern", type: "string", description: "Content after rows", optional: true },
      { name: "page_size", type: "number", description: "Number of rows per page", optional: true },
      { name: "collapse_toc_panel", type: "boolean", description: "Collapse the TOC panel in detailed view", optional: true },
      { name: "hide_column_headers", type: "boolean", description: "Hide column headers in table view", optional: true },
    ],
  },
  {
    tag: TAG.column_display,
    name: "Column Display",
    appliesTo: ["column"],
    description: "Column value formatting: markdown patterns, pre-formatting, and sort behavior.",
    contextualized: true,
    properties: [
      { name: "markdown_pattern", type: "string", description: "Handlebars template for rendering the column value", optional: true },
      { name: "template_engine", type: "enum", description: "Template language", enumValues: ["handlebars", "mustache"], optional: true },
      { name: "pre_format", type: "object", description: "Value pre-formatting: format (printf string), bool_true_value, bool_false_value", optional: true },
      { name: "column_order", type: "array", description: "Sort keys when sorting by this column, or false to disable sorting", optional: true },
    ],
  },
  {
    tag: TAG.key_display,
    name: "Key Display",
    appliesTo: ["key"],
    description: "How unique key values are presented (markdown pattern, column order).",
    contextualized: true,
    properties: [
      { name: "markdown_pattern", type: "string", description: "Handlebars template for rendering the key", optional: true },
      { name: "column_order", type: "array", description: "Sort keys or false to disable", optional: true },
    ],
  },
  {
    tag: TAG.foreign_key,
    name: "Foreign Key",
    appliesTo: ["foreignkey"],
    description: "FK display augmentation: domain filter, default values, and presentation.",
    contextualized: true,
    properties: [
      { name: "to_name", type: "string", description: "Display name for this FK relationship", optional: true },
      { name: "from_name", type: "string", description: "Display name for the reverse relationship", optional: true },
      { name: "display", type: "object", description: "Presentation options", optional: true },
      { name: "domain_filter", type: "object", description: "Restrict FK choices by filter", optional: true },
    ],
  },

  // ── Asset & Data ──
  {
    tag: TAG.asset,
    name: "Asset",
    appliesTo: ["column"],
    description: "Mark a column as a file/asset with upload, checksum, and display settings.",
    contextualized: false,
    properties: [
      { name: "url_pattern", type: "string", description: "Hatrac URL pattern for file storage", optional: true },
      { name: "filename_column", type: "string", description: "Column storing the filename", optional: true },
      { name: "byte_count_column", type: "string", description: "Column storing file size", optional: true },
      { name: "md5", type: "string", description: "Column storing MD5 checksum", optional: true },
      { name: "sha256", type: "string", description: "Column storing SHA-256 checksum", optional: true },
      { name: "filename_ext_filter", type: "array", description: "Allowed file extensions", optional: true },
    ],
  },
  {
    tag: TAG.citation,
    name: "Citation",
    appliesTo: ["table"],
    description: "Generate citation metadata (journal style) for records.",
    contextualized: true,
    properties: [
      { name: "journal_pattern", type: "string", description: "Handlebars template for journal name" },
      { name: "author_pattern", type: "string", description: "Handlebars template for author(s)" },
      { name: "title_pattern", type: "string", description: "Handlebars template for title" },
      { name: "year_pattern", type: "string", description: "Handlebars template for year" },
      { name: "url_pattern", type: "string", description: "Handlebars template for URL" },
      { name: "id_pattern", type: "string", description: "Handlebars template for identifier (e.g., DOI)" },
    ],
  },
  {
    tag: TAG.source_definitions,
    name: "Source Definitions",
    appliesTo: ["table"],
    description: "Define reusable named column/FK source paths for visible-columns and visible-foreign-keys.",
    contextualized: false,
    properties: [
      { name: "fkeys", type: "object", description: "Named foreign key path definitions", optional: true },
      { name: "sources", type: "object", description: "Named source path definitions", optional: true },
    ],
  },

  // ── Behavioral ──
  {
    tag: TAG.generated,
    name: "Generated",
    appliesTo: ["catalog", "schema", "table", "column"],
    description: "Mark as system-generated (not user-editable). Hides from entry forms.",
    contextualized: false,
  },
  {
    tag: TAG.immutable,
    name: "Immutable",
    appliesTo: ["catalog", "schema", "table", "column"],
    description: "Mark as read-only after creation. Shows in entry/create but not entry/edit.",
    contextualized: false,
  },
  {
    tag: TAG.non_deletable,
    name: "Non-Deletable",
    appliesTo: ["catalog", "schema", "table"],
    description: "Prevent users from deleting records or tables.",
    contextualized: false,
  },
  {
    tag: TAG.required,
    name: "Required",
    appliesTo: ["column"],
    description: "Mark a column as required in entry forms (override nullok setting).",
    contextualized: false,
  },

  // ── Table structure ──
  {
    tag: TAG.table_alternatives,
    name: "Table Alternatives",
    appliesTo: ["table"],
    description: "Point to alternative tables for different contexts (compact vs detailed).",
    contextualized: true,
  },

  // ── Export & Integration ──
  {
    tag: TAG.app_links,
    name: "App Links",
    appliesTo: ["schema", "table"],
    description: "Define links to external applications from Chaise record pages.",
    contextualized: true,
  },
  {
    tag: TAG.export,
    name: "Export (legacy)",
    appliesTo: ["table"],
    description: "Legacy export templates (superseded by 2019 version).",
    contextualized: false,
  },
  {
    tag: TAG.export_2019,
    name: "Export",
    appliesTo: ["catalog", "schema", "table"],
    description: "Export templates defining download formats (CSV, BDBag, etc.).",
    contextualized: true,
  },
  {
    tag: TAG.export_fragment_definitions,
    name: "Export Fragments",
    appliesTo: ["catalog", "schema", "table"],
    description: "Reusable export template components.",
    contextualized: false,
  },
  {
    tag: TAG.chaise_config,
    name: "Chaise Config",
    appliesTo: ["catalog"],
    description: "Chaise UI configuration (navbar, default table, features).",
    contextualized: false,
  },
  {
    tag: TAG.google_dataset,
    name: "Google Dataset",
    appliesTo: ["table"],
    description: "Embed JSON-LD metadata for Google Dataset Search indexing.",
    contextualized: false,
  },
  {
    tag: TAG.bulk_upload,
    name: "Bulk Upload",
    appliesTo: ["table"],
    description: "Configure bulk upload behavior for asset tables.",
    contextualized: false,
  },
  {
    tag: TAG.indexing_preferences,
    name: "Indexing Preferences",
    appliesTo: ["table", "column"],
    description: "PostgreSQL indexing hints for search optimization.",
    contextualized: false,
  },
  {
    tag: TAG.column_defaults,
    name: "Column Defaults",
    appliesTo: ["catalog", "schema", "table"],
    description: "Apply default annotations to columns matching name or type patterns.",
    contextualized: false,
  },
  {
    tag: TAG.viz_3d_display,
    name: "3D Visualization",
    appliesTo: ["table"],
    description: "Configure 3D model visualization display settings.",
    contextualized: false,
  },
];

// ── Lookup helpers ──────────────────────────────────────────────

const _byTag = new Map(ANNOTATION_REGISTRY.map((a) => [a.tag, a]));

/** Look up tag info by URI */
export function getTagInfo(tag: string): AnnotationTagInfo | undefined {
  return _byTag.get(tag);
}

/** Get tags valid for a given object type */
export function getTagsForType(objectType: ModelObjectType): AnnotationTagInfo[] {
  return ANNOTATION_REGISTRY.filter((a) => a.appliesTo.includes(objectType));
}

/** Get tags valid for a type that are NOT in the given annotations map */
export function getMissingTags(
  objectType: ModelObjectType,
  annotations: Record<string, any>
): AnnotationTagInfo[] {
  return getTagsForType(objectType).filter((a) => !(a.tag in annotations));
}

/** Short display name from a tag URI (e.g., "2016:visible-columns") */
export function shortTagName(tag: string): string {
  const info = _byTag.get(tag);
  if (info) return info.name;
  return tag.split(",").pop() || tag;
}
