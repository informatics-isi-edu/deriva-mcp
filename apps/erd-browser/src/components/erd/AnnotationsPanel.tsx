/**
 * Annotation viewer panel for table detail, modeled after the
 * deriva-workbench Qt annotation editors.
 *
 * Shows parsed annotation data in structured sections:
 * - Display (name, name_style, comment)
 * - Visible Columns (per-context column lists)
 * - Visible Foreign Keys (per-context FK lists)
 * - Table Display (row_name, row_order, page_size, markdown patterns)
 */

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import type { EnrichedTable, ColumnInfo } from "@/types";
import {
  TAG,
  getTagInfo,
  getMissingTags,
  shortTagName,
  type ModelObjectType,
  type AnnotationTagInfo,
} from "@/annotation-registry";

// Short aliases
const TAGS = {
  DISPLAY: TAG.display,
  VISIBLE_COLUMNS: TAG.visible_columns,
  VISIBLE_FOREIGN_KEYS: TAG.visible_foreign_keys,
  TABLE_DISPLAY: TAG.table_display,
  COLUMN_DISPLAY: TAG.column_display,
  ASSET: TAG.asset,
} as const;

// Known context names
const CONTEXT_LABELS: Record<string, string> = {
  "*": "Default (*)",
  compact: "Compact",
  "compact/brief": "Compact/Brief",
  "compact/brief/inline": "Compact/Brief/Inline",
  "compact/select": "Compact/Select",
  detailed: "Detailed",
  entry: "Entry",
  "entry/create": "Entry/Create",
  "entry/edit": "Entry/Edit",
  export: "Export",
  filter: "Filter",
  row_name: "Row Name",
  "row_name/compact": "Row Name/Compact",
  "row_name/detailed": "Row Name/Detailed",
};

const SYSTEM_COLS = new Set(["RID", "RCT", "RMT", "RCB", "RMB"]);

function contextLabel(ctx: string): string {
  return CONTEXT_LABELS[ctx] || ctx;
}

function hasAnnotations(annotations: Record<string, any>): boolean {
  return Object.keys(annotations).length > 0;
}

interface AnnotationsPanelProps {
  table: EnrichedTable;
}

export default function AnnotationsPanel({ table }: AnnotationsPanelProps) {
  // "table" or a column name
  const [target, setTarget] = useState<string>("table");

  // Columns that have annotations (non-system only)
  const annotatedColumns = table.info.columns.filter(
    (c) => !SYSTEM_COLS.has(c.name) && hasAnnotations(c.annotations)
  );

  return (
    <div className="space-y-3 pt-2">
      {/* Target selector */}
      <div>
        <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
          Annotations for
        </div>
        <div className="flex flex-wrap gap-1">
          <button
            onClick={() => setTarget("table")}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
              target === "table"
                ? "bg-slate-800 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            Table
          </button>
          {table.info.columns
            .filter((c) => !SYSTEM_COLS.has(c.name))
            .map((col) => {
              const hasAnno = hasAnnotations(col.annotations);
              return (
                <button
                  key={col.name}
                  onClick={() => setTarget(col.name)}
                  className={`px-2 py-1 rounded text-[11px] font-mono transition-colors ${
                    target === col.name
                      ? "bg-slate-800 text-white"
                      : hasAnno
                      ? "bg-slate-100 text-slate-700 hover:bg-slate-200"
                      : "bg-white text-slate-300 border border-slate-100 hover:bg-slate-50 hover:text-slate-500"
                  }`}
                >
                  {col.name}
                  {hasAnno && target !== col.name && (
                    <span className="ml-1 inline-block w-1.5 h-1.5 rounded-full bg-blue-400" />
                  )}
                </button>
              );
            })}
        </div>
      </div>

      {/* Annotation content */}
      {target === "table" ? (
        <TableAnnotations annotations={table.info.annotations} />
      ) : (
        <ColumnAnnotations
          column={table.info.columns.find((c) => c.name === target)!}
        />
      )}
    </div>
  );
}

// ── Table-level annotations ─────────────────────────────────────

function TableAnnotations({ annotations }: { annotations: Record<string, any> }) {
  const hasDisplay = !!annotations[TAGS.DISPLAY];
  const hasVisibleCols = !!annotations[TAGS.VISIBLE_COLUMNS];
  const hasVisibleFKs = !!annotations[TAGS.VISIBLE_FOREIGN_KEYS];
  const hasTableDisplay = !!annotations[TAGS.TABLE_DISPLAY];

  // Other present annotations not handled by specialized sections
  const specializedTags = new Set([TAGS.DISPLAY, TAGS.VISIBLE_COLUMNS, TAGS.VISIBLE_FOREIGN_KEYS, TAGS.TABLE_DISPLAY]);
  const otherPresent = Object.keys(annotations).filter((t) => !specializedTags.has(t));

  const missing = getMissingTags("table", annotations);

  return (
    <div className="space-y-3">
      {hasDisplay && <DisplaySection data={annotations[TAGS.DISPLAY]} />}
      {hasVisibleCols && <VisibleColumnsSection data={annotations[TAGS.VISIBLE_COLUMNS]} />}
      {hasVisibleFKs && <VisibleForeignKeysSection data={annotations[TAGS.VISIBLE_FOREIGN_KEYS]} />}
      {hasTableDisplay && <TableDisplaySection data={annotations[TAGS.TABLE_DISPLAY]} />}
      {otherPresent.map((tag) => (
        <RawAnnotationSection key={tag} tag={tag} data={annotations[tag]} />
      ))}
      <AvailableAnnotations missing={missing} />
    </div>
  );
}

// ── Column-level annotations ────────────────────────────────────

function ColumnAnnotations({ column }: { column: ColumnInfo }) {
  const anno = column.annotations;
  const hasDisplay = !!anno[TAGS.DISPLAY];
  const hasColumnDisplay = !!anno[TAGS.COLUMN_DISPLAY];
  const hasAsset = !!anno[TAGS.ASSET];

  // Show any other annotations as raw JSON
  const knownTags = new Set([TAGS.DISPLAY, TAGS.COLUMN_DISPLAY, TAGS.ASSET]);
  const otherTags = Object.keys(anno).filter((t) => !knownTags.has(t));

  const missing = getMissingTags("column", anno);

  return (
    <div className="space-y-3">
      <div className="text-[11px] text-slate-500">
        <span className="font-mono font-medium text-slate-700">{column.name}</span>
        {" "}<span className="text-slate-400">({column.type}{column.nullok ? ", nullable" : ""})</span>
      </div>
      {hasDisplay && <DisplaySection data={anno[TAGS.DISPLAY]} />}
      {hasColumnDisplay && <ColumnDisplaySection data={anno[TAGS.COLUMN_DISPLAY]} />}
      {hasAsset && <AssetSection data={anno[TAGS.ASSET]} />}
      {otherTags.map((tag) => (
        <RawAnnotationSection key={tag} tag={tag} data={anno[tag]} />
      ))}
      <AvailableAnnotations missing={missing} />
    </div>
  );
}

// ── Schema-level annotations ────────────────────────────────────

export function SchemaAnnotationsPanel({
  annotations,
}: {
  annotations: Record<string, any>;
}) {
  const hasDisplay = !!annotations[TAGS.DISPLAY];

  // Show any other annotations as raw JSON
  const knownTags = new Set([TAGS.DISPLAY]);
  const otherTags = Object.keys(annotations).filter((t) => !knownTags.has(t));

  const missing = getMissingTags("schema", annotations);

  return (
    <div className="space-y-3">
      {hasDisplay && <DisplaySection data={annotations[TAGS.DISPLAY]} />}
      {otherTags.map((tag) => (
        <RawAnnotationSection key={tag} tag={tag} data={annotations[tag]} />
      ))}
      <AvailableAnnotations missing={missing} />
    </div>
  );
}

// ── Section wrapper ─────────────────────────────────────────────

function Section({
  title,
  tag,
  children,
  defaultOpen = true,
}: {
  title: string;
  tag: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const info = getTagInfo(tag);
  return (
    <div className="border border-slate-200 rounded-md overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full text-left px-3 py-2 bg-slate-50 hover:bg-slate-100 transition-colors"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-700">{title}</span>
          <span className="text-[10px] font-mono text-slate-400 truncate ml-2 max-w-[160px]">
            {tag.split(",").pop()}
          </span>
        </div>
        {info && (
          <div className="text-[10px] text-slate-400 mt-0.5 leading-snug">
            {info.description}
          </div>
        )}
      </button>
      {open && <div className="px-3 py-2 space-y-2">{children}</div>}
    </div>
  );
}

/**
 * Shows annotations that are valid for this object type but not yet present.
 * Mirrors the workbench's "Add missing annotations" context menu.
 */
function AvailableAnnotations({ missing }: { missing: AnnotationTagInfo[] }) {
  const [open, setOpen] = useState(false);

  if (missing.length === 0) return null;

  return (
    <div className="border border-dashed border-slate-200 rounded-md overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full text-left px-3 py-2 hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-400">
            Available annotations
          </span>
          <Badge variant="secondary" className="text-[9px]">
            {missing.length}
          </Badge>
        </div>
        <div className="text-[10px] text-slate-300 mt-0.5">
          Annotations that can be added to this object
        </div>
      </button>
      {open && (
        <div className="px-3 pb-2 space-y-1.5">
          {missing.map((info) => (
            <div
              key={info.tag}
              className="flex items-start gap-2 py-1 border-b border-slate-50 last:border-b-0"
            >
              <div className="min-w-0 flex-1">
                <div className="text-[11px] font-medium text-slate-500">
                  {info.name}
                  {info.contextualized && (
                    <span className="ml-1 text-[9px] text-slate-300 font-normal">
                      contextualized
                    </span>
                  )}
                </div>
                <div className="text-[10px] text-slate-400 leading-snug">
                  {info.description}
                </div>
                {info.properties && info.properties.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {info.properties.slice(0, 5).map((p) => (
                      <span
                        key={p.name}
                        className="text-[9px] font-mono text-slate-400 bg-slate-50 rounded px-1 py-0.5"
                        title={p.description}
                      >
                        {p.name}
                      </span>
                    ))}
                    {info.properties.length > 5 && (
                      <span className="text-[9px] text-slate-300">
                        +{info.properties.length - 5} more
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Context tabs ────────────────────────────────────────────────

function ContextTabs({
  contexts,
  children,
}: {
  contexts: string[];
  children: (ctx: string) => React.ReactNode;
}) {
  const [active, setActive] = useState(contexts[0] || "*");

  if (contexts.length === 0) return null;

  return (
    <div>
      <div className="flex flex-wrap gap-1 mb-2">
        {contexts.map((ctx) => (
          <button
            key={ctx}
            onClick={() => setActive(ctx)}
            className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
              active === ctx
                ? "bg-slate-700 text-white"
                : "bg-slate-100 text-slate-500 hover:bg-slate-200"
            }`}
          >
            {contextLabel(ctx)}
          </button>
        ))}
      </div>
      {children(active)}
    </div>
  );
}

// ── Display annotation ──────────────────────────────────────────

function DisplaySection({ data }: { data: any }) {
  return (
    <Section title="Display" tag={TAGS.DISPLAY}>
      <div className="space-y-1.5">
        {data.name && (
          <PropRow label="Name" value={data.name} />
        )}
        {data.markdown_name && (
          <PropRow label="Markdown Name" value={data.markdown_name} mono />
        )}
        {data.name_style && (
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-slate-500 w-24 flex-shrink-0">Name Style</span>
            <div className="flex gap-1">
              {data.name_style.underline_space && (
                <Badge variant="secondary" className="text-[9px]">underline→space</Badge>
              )}
              {data.name_style.title_case && (
                <Badge variant="secondary" className="text-[9px]">title case</Badge>
              )}
              {data.name_style.markdown && (
                <Badge variant="secondary" className="text-[9px]">markdown</Badge>
              )}
            </div>
          </div>
        )}
        {data.comment && typeof data.comment === "object" && (
          <ContextualProp label="Comment" data={data.comment} />
        )}
        {data.comment && typeof data.comment === "string" && (
          <PropRow label="Comment" value={data.comment} />
        )}
        {data.show_null && typeof data.show_null === "object" && (
          <ContextualProp label="Show Null" data={data.show_null} />
        )}
        {data.show_foreign_key_link && typeof data.show_foreign_key_link === "object" && (
          <ContextualProp label="Show FK Link" data={data.show_foreign_key_link} />
        )}
      </div>
    </Section>
  );
}

// ── Visible Columns ─────────────────────────────────────────────

function VisibleColumnsSection({ data }: { data: any }) {
  const contexts = Object.keys(data);
  if (contexts.length === 0) return null;

  return (
    <Section title="Visible Columns" tag={TAGS.VISIBLE_COLUMNS}>
      <ContextTabs contexts={contexts}>
        {(ctx) => {
          const value = data[ctx];
          // Context reference (string pointing to another context)
          if (typeof value === "string") {
            return (
              <div className="text-[11px] text-slate-500 italic py-1">
                References context: <span className="font-mono">{value}</span>
              </div>
            );
          }
          // Filter context uses { and: [...] }
          const entries = ctx === "filter" && value?.and
            ? value.and
            : Array.isArray(value) ? value : [];
          return <SourceEntryList entries={entries} isFilter={ctx === "filter"} />;
        }}
      </ContextTabs>
    </Section>
  );
}

// ── Visible Foreign Keys ────────────────────────────────────────

function VisibleForeignKeysSection({ data }: { data: any }) {
  const contexts = Object.keys(data);
  if (contexts.length === 0) return null;

  return (
    <Section title="Visible Foreign Keys" tag={TAGS.VISIBLE_FOREIGN_KEYS}>
      <ContextTabs contexts={contexts}>
        {(ctx) => {
          const value = data[ctx];
          if (typeof value === "string") {
            return (
              <div className="text-[11px] text-slate-500 italic py-1">
                References context: <span className="font-mono">{value}</span>
              </div>
            );
          }
          const entries = Array.isArray(value) ? value : [];
          return <SourceEntryList entries={entries} />;
        }}
      </ContextTabs>
    </Section>
  );
}

// ── Table Display ───────────────────────────────────────────────

function TableDisplaySection({ data }: { data: any }) {
  const contexts = Object.keys(data);
  if (contexts.length === 0) return null;

  return (
    <Section title="Table Display" tag={TAGS.TABLE_DISPLAY}>
      <ContextTabs contexts={contexts}>
        {(ctx) => {
          const value = data[ctx];
          if (typeof value === "string") {
            return (
              <div className="text-[11px] text-slate-500 italic py-1">
                References context: <span className="font-mono">{value}</span>
              </div>
            );
          }
          if (!value || typeof value !== "object") return null;
          return <TableDisplayContext data={value} />;
        }}
      </ContextTabs>
    </Section>
  );
}

function TableDisplayContext({ data }: { data: any }) {
  return (
    <div className="space-y-1.5">
      {data.row_order && Array.isArray(data.row_order) && (
        <div>
          <span className="text-[11px] text-slate-500 font-medium">Row Order</span>
          <div className="mt-0.5 space-y-0.5">
            {data.row_order.map((key: any, i: number) => {
              const col = typeof key === "string" ? key : key.column;
              const desc = typeof key === "object" && key.descending;
              return (
                <div key={i} className="text-[11px] font-mono text-slate-600 flex items-center gap-1">
                  <span className="text-slate-400 w-4 text-right">{i + 1}.</span>
                  {col}
                  {desc && <Badge variant="outline" className="text-[8px] px-1 py-0">DESC</Badge>}
                </div>
              );
            })}
          </div>
        </div>
      )}
      {data.row_markdown_pattern && (
        <PropRow label="Row Pattern" value={data.row_markdown_pattern} mono />
      )}
      {data.page_markdown_pattern && (
        <PropRow label="Page Pattern" value={data.page_markdown_pattern} mono />
      )}
      {data.separator_pattern && (
        <PropRow label="Separator" value={data.separator_pattern} mono />
      )}
      {data.page_size !== undefined && (
        <PropRow label="Page Size" value={String(data.page_size)} />
      )}
      {data.collapse_toc_panel !== undefined && (
        <PropRow label="Collapse TOC" value={data.collapse_toc_panel ? "Yes" : "No"} />
      )}
      {data.hide_column_headers !== undefined && (
        <PropRow label="Hide Headers" value={data.hide_column_headers ? "Yes" : "No"} />
      )}
    </div>
  );
}

// ── Shared components ───────────────────────────────────────────

function PropRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-[11px] text-slate-500 w-24 flex-shrink-0">{label}</span>
      <span
        className={`text-[11px] text-slate-700 break-all ${mono ? "font-mono" : ""}`}
      >
        {value}
      </span>
    </div>
  );
}

function ContextualProp({ label, data }: { label: string; data: Record<string, any> }) {
  const entries = Object.entries(data);
  if (entries.length === 0) return null;
  return (
    <div>
      <span className="text-[11px] text-slate-500">{label}</span>
      <div className="pl-2 mt-0.5 space-y-0.5">
        {entries.map(([ctx, val]) => (
          <div key={ctx} className="flex items-start gap-2 text-[11px]">
            <span className="text-slate-400 font-mono w-20 flex-shrink-0 truncate">
              {ctx}
            </span>
            <span className="text-slate-600 break-all">
              {typeof val === "boolean" ? (val ? "true" : "false") : String(val)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Renders a list of visible-source entries (columns, constraints, pseudo-columns).
 * Mirrors the workbench's CommonTableWidget with Type + Source columns.
 */
function SourceEntryList({
  entries,
  isFilter,
}: {
  entries: any[];
  isFilter?: boolean;
}) {
  if (entries.length === 0) {
    return (
      <div className="text-[11px] text-slate-400 py-1 italic">No entries</div>
    );
  }

  return (
    <div className="border border-slate-100 rounded overflow-hidden">
      {/* Header */}
      <div className="flex bg-slate-50 border-b border-slate-100 px-2 py-1">
        <span className="text-[10px] font-semibold text-slate-400 uppercase w-20 flex-shrink-0">
          Type
        </span>
        <span className="text-[10px] font-semibold text-slate-400 uppercase flex-1">
          Source
        </span>
      </div>
      {/* Rows */}
      {entries.map((entry, i) => {
        const { type, source } = describeSourceEntry(entry, isFilter);
        return (
          <div
            key={i}
            className="flex items-start px-2 py-1 border-b border-slate-50 last:border-b-0 hover:bg-slate-50"
          >
            <span className="text-[10px] w-20 flex-shrink-0">
              <Badge
                variant="outline"
                className={`text-[9px] px-1 py-0 ${
                  type === "Column"
                    ? "border-sky-200 text-sky-600"
                    : type === "Constraint"
                    ? "border-amber-200 text-amber-600"
                    : type === "Facet"
                    ? "border-violet-200 text-violet-600"
                    : "border-slate-200 text-slate-500"
                }`}
              >
                {type}
              </Badge>
            </span>
            <span className="text-[11px] font-mono text-slate-600 break-all flex-1">
              {source}
            </span>
          </div>
        );
      })}
      <div className="bg-slate-50 px-2 py-0.5 text-[10px] text-slate-400">
        {entries.length} {entries.length === 1 ? "entry" : "entries"}
      </div>
    </div>
  );
}

/**
 * Classify a source entry following the same logic as the workbench's
 * _source_entry_to_row function.
 */
function describeSourceEntry(
  entry: any,
  isFilter?: boolean
): { type: string; source: string } {
  if (typeof entry === "string") {
    return { type: "Column", source: entry };
  }
  if (Array.isArray(entry)) {
    // [schema_name, constraint_name]
    return {
      type: "Constraint",
      source: entry.length === 2 ? entry[1] : JSON.stringify(entry),
    };
  }
  if (typeof entry === "object" && entry !== null) {
    if (isFilter) {
      // Facet object
      const src = entry.source
        ? sourcePathToStr(entry.source)
        : entry.sourcekey || "virtual";
      const name = entry.markdown_name || "";
      return {
        type: "Facet",
        source: name ? `${name} (${src})` : src,
      };
    }
    // Pseudo-column
    const src = entry.source
      ? sourcePathToStr(entry.source)
      : entry.sourcekey || "virtual";
    const parts = [src];
    if (entry.aggregate) parts.push(`agg:${entry.aggregate}`);
    if (entry.markdown_name) parts.push(`"${entry.markdown_name}"`);
    return { type: "Pseudo", source: parts.join(" ") };
  }
  return { type: "?", source: JSON.stringify(entry) };
}

/**
 * Convert a source path (string, array of FK steps) to a readable string.
 * Mirrors the workbench's source_path_to_str helper.
 */
function sourcePathToStr(source: any): string {
  if (typeof source === "string") return source;
  if (Array.isArray(source)) {
    return source
      .map((step: any) => {
        if (typeof step === "string") return step;
        if (typeof step === "object") {
          // FK step: { "inbound": [...] } or { "outbound": [...] }
          if (step.inbound)
            return `← ${Array.isArray(step.inbound) ? step.inbound[1] || step.inbound.join(".") : step.inbound}`;
          if (step.outbound)
            return `→ ${Array.isArray(step.outbound) ? step.outbound[1] || step.outbound.join(".") : step.outbound}`;
          return JSON.stringify(step);
        }
        return String(step);
      })
      .join(" / ");
  }
  return JSON.stringify(source);
}

// ── Column Display annotation ───────────────────────────────────

function ColumnDisplaySection({ data }: { data: any }) {
  const contexts = Object.keys(data);
  if (contexts.length === 0) return null;

  return (
    <Section title="Column Display" tag={TAGS.COLUMN_DISPLAY}>
      <ContextTabs contexts={contexts}>
        {(ctx) => {
          const value = data[ctx];
          if (typeof value === "string") {
            return (
              <div className="text-[11px] text-slate-500 italic py-1">
                References context: <span className="font-mono">{value}</span>
              </div>
            );
          }
          if (!value || typeof value !== "object") return null;
          return <ColumnDisplayContext data={value} />;
        }}
      </ContextTabs>
    </Section>
  );
}

function ColumnDisplayContext({ data }: { data: any }) {
  return (
    <div className="space-y-1.5">
      {data.markdown_pattern && (
        <PropRow label="Markdown Pattern" value={data.markdown_pattern} mono />
      )}
      {data.template_engine && (
        <PropRow label="Template Engine" value={data.template_engine} />
      )}
      {data.pre_format && typeof data.pre_format === "object" && (
        <div>
          <span className="text-[11px] text-slate-500 font-medium">Pre-Format</span>
          <div className="pl-2 mt-0.5 space-y-0.5">
            {data.pre_format.format && (
              <PropRow label="Format" value={data.pre_format.format} mono />
            )}
            {data.pre_format.bool_true_value && (
              <PropRow label="True Value" value={data.pre_format.bool_true_value} />
            )}
            {data.pre_format.bool_false_value && (
              <PropRow label="False Value" value={data.pre_format.bool_false_value} />
            )}
          </div>
        </div>
      )}
      {data.column_order !== undefined && (
        <div>
          <span className="text-[11px] text-slate-500 font-medium">Column Order</span>
          <div className="pl-2 mt-0.5">
            {data.column_order === false ? (
              <span className="text-[11px] text-slate-400 italic">Sorting disabled</span>
            ) : data.column_order === null || data.column_order === undefined ? (
              <span className="text-[11px] text-slate-400 italic">Default</span>
            ) : Array.isArray(data.column_order) ? (
              data.column_order.map((key: any, i: number) => {
                const col = typeof key === "string" ? key : key.column;
                const desc = typeof key === "object" && key.descending;
                return (
                  <div key={i} className="text-[11px] font-mono text-slate-600 flex items-center gap-1">
                    <span className="text-slate-400 w-4 text-right">{i + 1}.</span>
                    {col}
                    {desc && <Badge variant="outline" className="text-[8px] px-1 py-0">DESC</Badge>}
                  </div>
                );
              })
            ) : (
              <span className="text-[11px] text-slate-600">{JSON.stringify(data.column_order)}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Asset annotation ────────────────────────────────────────────

function AssetSection({ data }: { data: any }) {
  return (
    <Section title="Asset" tag={TAGS.ASSET}>
      <div className="space-y-1.5">
        {data.url_pattern && (
          <PropRow label="URL Pattern" value={data.url_pattern} mono />
        )}
        {data.filename_column && (
          <PropRow label="Filename Col" value={data.filename_column} mono />
        )}
        {data.byte_count_column && (
          <PropRow label="Byte Count Col" value={data.byte_count_column} mono />
        )}
        {data.md5 && (
          <PropRow label="MD5 Col" value={typeof data.md5 === "string" ? data.md5 : data.md5.column || JSON.stringify(data.md5)} mono />
        )}
        {data.sha256 && (
          <PropRow label="SHA256 Col" value={typeof data.sha256 === "string" ? data.sha256 : data.sha256.column || JSON.stringify(data.sha256)} mono />
        )}
        {data.filename_ext_filter && Array.isArray(data.filename_ext_filter) && (
          <div className="flex items-start gap-2">
            <span className="text-[11px] text-slate-500 w-24 flex-shrink-0">Extensions</span>
            <div className="flex flex-wrap gap-1">
              {data.filename_ext_filter.map((ext: string, i: number) => (
                <Badge key={i} variant="secondary" className="text-[9px] font-mono">{ext}</Badge>
              ))}
            </div>
          </div>
        )}
        {data.display && typeof data.display === "object" && (
          <div>
            <span className="text-[11px] text-slate-500 font-medium">Display Options</span>
            <div className="pl-2 mt-0.5 space-y-0.5">
              {Object.entries(data.display).map(([k, v]) => (
                <PropRow key={k} label={k} value={String(v)} />
              ))}
            </div>
          </div>
        )}
      </div>
    </Section>
  );
}

// ── Raw/unknown annotation fallback ─────────────────────────────

function RawAnnotationSection({ tag, data }: { tag: string; data: any }) {
  return (
    <Section title={shortTagName(tag)} tag={tag} defaultOpen={false}>
      <pre className="text-[10px] font-mono text-slate-600 whitespace-pre-wrap break-all bg-slate-50 rounded p-2 max-h-40 overflow-auto">
        {JSON.stringify(data, null, 2)}
      </pre>
    </Section>
  );
}
