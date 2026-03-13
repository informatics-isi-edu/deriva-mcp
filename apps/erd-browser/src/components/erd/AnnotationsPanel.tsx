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
import { Separator } from "@/components/ui/separator";
import type { EnrichedTable } from "@/types";

// Annotation tag constants
const TAGS = {
  DISPLAY: "tag:misd.isi.edu,2015:display",
  VISIBLE_COLUMNS: "tag:isrd.isi.edu,2016:visible-columns",
  VISIBLE_FOREIGN_KEYS: "tag:isrd.isi.edu,2016:visible-foreign-keys",
  TABLE_DISPLAY: "tag:isrd.isi.edu,2016:table-display",
  COLUMN_DISPLAY: "tag:isrd.isi.edu,2016:column-display",
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

function contextLabel(ctx: string): string {
  return CONTEXT_LABELS[ctx] || ctx;
}

interface AnnotationsPanelProps {
  table: EnrichedTable;
}

export default function AnnotationsPanel({ table }: AnnotationsPanelProps) {
  const anno = table.info.annotations;
  const hasDisplay = !!anno[TAGS.DISPLAY];
  const hasVisibleCols = !!anno[TAGS.VISIBLE_COLUMNS];
  const hasVisibleFKs = !!anno[TAGS.VISIBLE_FOREIGN_KEYS];
  const hasTableDisplay = !!anno[TAGS.TABLE_DISPLAY];

  const hasAny = hasDisplay || hasVisibleCols || hasVisibleFKs || hasTableDisplay;

  if (!hasAny) {
    return (
      <div className="py-6 text-center text-xs text-slate-400">
        No display annotations on this table
      </div>
    );
  }

  return (
    <div className="space-y-4 pt-2">
      {hasDisplay && <DisplaySection data={anno[TAGS.DISPLAY]} />}
      {hasVisibleCols && <VisibleColumnsSection data={anno[TAGS.VISIBLE_COLUMNS]} />}
      {hasVisibleFKs && <VisibleForeignKeysSection data={anno[TAGS.VISIBLE_FOREIGN_KEYS]} />}
      {hasTableDisplay && <TableDisplaySection data={anno[TAGS.TABLE_DISPLAY]} />}
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
  return (
    <div className="border border-slate-200 rounded-md overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 bg-slate-50 hover:bg-slate-100 transition-colors"
      >
        <span className="text-xs font-semibold text-slate-700">{title}</span>
        <span className="text-[10px] font-mono text-slate-400 truncate ml-2 max-w-[160px]">
          {tag.split(",").pop()}
        </span>
      </button>
      {open && <div className="px-3 py-2 space-y-2">{children}</div>}
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
