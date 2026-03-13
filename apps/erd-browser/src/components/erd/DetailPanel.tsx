import { useState } from "react";
import { ExternalLink, X, Table2, KeyRound, Database, MousePointerClick, List, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { EnrichedTable, CatalogSchema } from "@/types";
import { chaiseRecordsetUrl } from "@/ermrest-client";
import DataBrowser from "./DataBrowser";
import AnnotationsPanel, { SchemaAnnotationsPanel, CatalogAnnotationsPanel } from "./AnnotationsPanel";

interface DetailPanelProps {
  // Table-level detail
  table: EnrichedTable | null;
  onClose: () => void;
  // Schema-level detail
  viewMode: "schemas" | "tables";
  activeSchema: string | null;
  schema: CatalogSchema | null;
  tables: EnrichedTable[];
  onDrillIntoSchema: (schemaName: string) => void;
  onJumpToTable?: (table: EnrichedTable) => void;
  // Catalog-level detail
  showCatalog?: boolean;
}

export default function DetailPanel({
  table,
  onClose,
  viewMode,
  activeSchema,
  schema,
  tables,
  onDrillIntoSchema,
  onJumpToTable,
  showCatalog,
}: DetailPanelProps) {
  if (showCatalog && schema) {
    return <CatalogDetail schema={schema} onClose={onClose} />;
  }

  if (viewMode === "schemas" && activeSchema && schema) {
    return (
      <SchemaDetail
        schemaName={activeSchema}
        tables={tables}
        schema={schema}
        onDrillIn={() => onDrillIntoSchema(activeSchema)}
        onClose={onClose}
        onJumpToTable={onJumpToTable}
      />
    );
  }

  if (viewMode === "tables" && table) {
    return <TableDetail table={table} onClose={onClose} />;
  }

  return (
    <div className="h-full flex items-center justify-center text-slate-400 text-sm px-6 text-center">
      <div>
        <MousePointerClick className="h-8 w-8 mx-auto mb-3 text-slate-300" />
        <p className="font-medium">
          {viewMode === "schemas"
            ? "Click a schema to see details"
            : "Click a table to see details"}
        </p>
        {viewMode === "schemas" && (
          <p className="text-xs mt-1 text-slate-300">
            Double-click to drill into tables
          </p>
        )}
      </div>
    </div>
  );
}

// ── Schema detail ──────────────────────────────────────────────────

function SchemaDetail({
  schemaName,
  tables,
  schema,
  onDrillIn,
  onClose,
  onJumpToTable,
}: {
  schemaName: string;
  tables: EnrichedTable[];
  schema: CatalogSchema;
  onDrillIn: () => void;
  onClose: () => void;
  onJumpToTable?: (table: EnrichedTable) => void;
}) {
  const schemaTables = tables.filter((t) => t.schema === schemaName);
  const domainTables = schemaTables.filter((t) => t.tableType === "domain");
  const vocabTables = schemaTables.filter((t) => t.tableType === "vocabulary");
  const assetTables = schemaTables.filter((t) => t.tableType === "asset");
  const assocTables = schemaTables.filter((t) => t.tableType === "association");
  const mlTables = schemaTables.filter((t) => t.tableType === "ml");

  // Cross-schema relationships
  const outgoing = new Set<string>();
  const incoming = new Set<string>();
  for (const t of tables) {
    for (const fk of t.info.foreign_keys) {
      const refSchema = fk.referenced_table.split(".")[0];
      if (t.schema === schemaName && refSchema !== schemaName) {
        outgoing.add(refSchema);
      }
      if (t.schema !== schemaName && refSchema === schemaName) {
        incoming.add(t.schema);
      }
    }
  }

  const schemaComment = schema.schemas[schemaName]?.comment;
  const schemaAnnotations = schema.schemas[schemaName]?.annotations || {};
  const annotationCount = Object.keys(schemaAnnotations).length;
  const [annotationsDirty, setAnnotationsDirty] = useState(false);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chaise-header bg-chaise-header/30 flex-shrink-0">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-base font-bold text-chaise-header-text">{schemaName}</h2>
            <p className="text-xs text-chaise-header-text/60 mt-0.5">
              {schemaTables.length} tables
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-7 w-7 -mr-1 -mt-1"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        {schemaComment && (
          <p className="text-xs text-chaise-header-text/70 mt-2 leading-relaxed">
            {schemaComment}
          </p>
        )}
        <Button
          size="sm"
          className="mt-3 w-full text-xs bg-brand hover:bg-brand/90"
          onClick={onDrillIn}
        >
          View tables in {schemaName}
        </Button>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="overview" className="flex-1 flex flex-col min-h-0">
        <TabsList className="mx-4 mt-2 h-8 shrink-0">
          <TabsTrigger value="overview" className="text-xs h-7 gap-1">
            <Database className="h-3 w-3" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="annotations" className="text-xs h-7 gap-1 relative">
            <Settings2 className="h-3 w-3" />
            Annotations
            {annotationCount > 0 && (
              <Badge variant="secondary" className="text-[10px] ml-0.5 px-1 py-0 h-4">
                {annotationCount}
              </Badge>
            )}
            {annotationsDirty && (
              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-amber-500 ring-2 ring-white" />
            )}
          </TabsTrigger>
        </TabsList>

        <ScrollArea className="flex-1 min-h-0">
          <TabsContent value="overview" className="mt-0 px-4 py-3 space-y-4">
            {/* Table breakdown */}
            <Section title="Table breakdown">
              {domainTables.length > 0 && (
                <TypeList label="Domain" tables={domainTables} color="bg-slate-700" onSelect={onJumpToTable} />
              )}
              {mlTables.length > 0 && (
                <TypeList label="ML" tables={mlTables} color="bg-amber-700" onSelect={onJumpToTable} />
              )}
              {vocabTables.length > 0 && (
                <TypeList label="Vocabulary" tables={vocabTables} color="bg-emerald-700" onSelect={onJumpToTable} />
              )}
              {assetTables.length > 0 && (
                <TypeList label="Asset" tables={assetTables} color="bg-sky-700" onSelect={onJumpToTable} />
              )}
              {assocTables.length > 0 && (
                <TypeList label="Association" tables={assocTables} color="bg-zinc-500" onSelect={onJumpToTable} />
              )}
            </Section>

            {/* Cross-schema relationships */}
            {(outgoing.size > 0 || incoming.size > 0) && (
              <Section title="Cross-schema relationships">
                {outgoing.size > 0 && (
                  <div className="text-xs text-slate-600">
                    <span className="font-medium">References →</span>{" "}
                    {[...outgoing].join(", ")}
                  </div>
                )}
                {incoming.size > 0 && (
                  <div className="text-xs text-slate-600 mt-1">
                    <span className="font-medium">Referenced by ←</span>{" "}
                    {[...incoming].join(", ")}
                  </div>
                )}
              </Section>
            )}
          </TabsContent>

          <TabsContent value="annotations" className="mt-0 px-4 pb-4">
            <SchemaAnnotationsPanel annotations={schemaAnnotations} schemaName={schemaName} onDirtyChange={setAnnotationsDirty} />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

// ── Catalog detail ─────────────────────────────────────────────────

function CatalogDetail({
  schema,
  onClose,
}: {
  schema: CatalogSchema;
  onClose: () => void;
}) {
  const annotationCount = Object.keys(schema.annotations).length;
  const totalTables = Object.values(schema.schemas).reduce(
    (sum, s) => sum + Object.keys(s.tables).length,
    0
  );
  const [annotationsDirty, setAnnotationsDirty] = useState(false);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chaise-header bg-chaise-header/30 flex-shrink-0">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-base font-bold text-chaise-header-text">
              {schema.hostname}
            </h2>
            <p className="text-xs text-chaise-header-text/60 mt-0.5 font-mono">
              Catalog #{schema.catalog_id}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-7 w-7 -mr-1 -mt-1"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex items-center gap-1.5 mt-2">
          <Badge variant="secondary" className="text-[10px]">
            {schema.domain_schemas.length} schemas
          </Badge>
          <Badge variant="outline" className="text-[10px]">
            {totalTables} tables
          </Badge>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="annotations" className="flex-1 flex flex-col min-h-0">
        <TabsList className="mx-4 mt-2 h-8 shrink-0">
          <TabsTrigger value="annotations" className="text-xs h-7 gap-1 relative">
            <Settings2 className="h-3 w-3" />
            Annotations
            {annotationCount > 0 && (
              <Badge variant="secondary" className="text-[10px] ml-0.5 px-1 py-0 h-4">
                {annotationCount}
              </Badge>
            )}
            {annotationsDirty && (
              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-amber-500 ring-2 ring-white" />
            )}
          </TabsTrigger>
        </TabsList>

        <ScrollArea className="flex-1 min-h-0">
          <TabsContent value="annotations" className="mt-0 px-4 pb-4">
            <CatalogAnnotationsPanel annotations={schema.annotations} onDirtyChange={setAnnotationsDirty} />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h3 className="text-[11px] font-semibold text-chaise-header-text uppercase tracking-wider mb-2 bg-chaise-header/50 -mx-4 px-4 py-1.5">
        {title}
      </h3>
      {children}
    </div>
  );
}

function TypeList({
  label,
  tables,
  color,
  onSelect,
}: {
  label: string;
  tables: EnrichedTable[];
  color: string;
  onSelect?: (table: EnrichedTable) => void;
}) {
  return (
    <div className="mb-2">
      <div className="flex items-center gap-1.5 mb-1">
        <div className={`w-2 h-2 rounded-sm ${color}`} />
        <span className="text-xs font-medium text-slate-700">
          {label} ({tables.length})
        </span>
      </div>
      <div className="pl-3.5 space-y-0.5">
        {tables.map((t) => (
          <button
            key={t.qualifiedName}
            onClick={() => onSelect?.(t)}
            className={`w-full text-left text-[11px] text-slate-500 flex items-center justify-between py-0.5 px-1 -mx-1 rounded transition-colors ${
              onSelect ? "hover:bg-chaise-hover hover:text-slate-800 cursor-pointer" : ""
            }`}
            title={t.info.comment || t.qualifiedName}
          >
            <span className="truncate">{t.name}</span>
            {t.recordCount !== null && t.recordCount >= 0 && (
              <span className="text-[11px] text-slate-400 ml-2 whitespace-nowrap">
                {t.recordCount.toLocaleString()}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Table detail ───────────────────────────────────────────────────

function TableDetail({
  table,
  onClose,
}: {
  table: EnrichedTable;
  onClose: () => void;
}) {
  const { info } = table;
  const chaiseUrl = chaiseRecordsetUrl(table.schema, table.name);
  const systemCols = new Set(["RID", "RCT", "RMT", "RCB", "RMB"]);
  const [annotationsDirty, setAnnotationsDirty] = useState(false);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chaise-header bg-chaise-header/30 flex-shrink-0">
        <div className="flex items-start justify-between">
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-bold text-chaise-header-text truncate">
              {table.name}
            </h2>
            <p className="text-xs text-chaise-header-text/60 mt-0.5">{table.schema}</p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-7 w-7 -mr-1 -mt-1"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-1.5 mt-2 flex-wrap">
          <Badge variant="secondary" className="text-[10px] capitalize">
            {table.tableType}
          </Badge>
          {table.recordCount !== null && table.recordCount >= 0 && (
            <Badge variant="outline" className="text-[10px]">
              {table.recordCount.toLocaleString()} rows
            </Badge>
          )}
        </div>

        {info.comment && (
          <p className="text-xs text-chaise-header-text/70 mt-2 leading-relaxed">
            {info.comment}
          </p>
        )}

        <div className="mt-2">
          <a
            href={chaiseUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-brand hover:text-brand/80 hover:underline"
          >
            Open in Chaise
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="columns" className="flex-1 flex flex-col min-h-0">
        <TabsList className="mx-4 mt-2 h-8 shrink-0">
          <TabsTrigger value="columns" className="text-xs h-7 gap-1">
            <Table2 className="h-3 w-3" />
            Columns
          </TabsTrigger>
          <TabsTrigger value="fks" className="text-xs h-7 gap-1">
            <KeyRound className="h-3 w-3" />
            FKs
          </TabsTrigger>
          <TabsTrigger value="browse" className="text-xs h-7 gap-1">
            <List className="h-3 w-3" />
            Browse
          </TabsTrigger>
          {info.features && info.features.length > 0 && (
            <TabsTrigger value="features" className="text-xs h-7 gap-1">
              <Database className="h-3 w-3" />
              Features
            </TabsTrigger>
          )}
          <TabsTrigger value="annotations" className="text-xs h-7 gap-1 relative">
            <Settings2 className="h-3 w-3" />
            Annotations
            {annotationsDirty && (
              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-amber-500 ring-2 ring-white" />
            )}
          </TabsTrigger>
        </TabsList>

        {/* Browse tab is NOT inside ScrollArea — it manages its own scrolling */}
        <TabsContent value="browse" className="mt-0 flex-1 min-h-0">
          <DataBrowser table={table} />
        </TabsContent>

        <ScrollArea className="flex-1 min-h-0">
          <TabsContent value="columns" className="mt-0 px-4 pb-4">
            <Table>
              <TableHeader>
                <TableRow className="bg-chaise-header/40">
                  <TableHead className="text-[11px] h-8 text-chaise-header-text font-semibold">Name</TableHead>
                  <TableHead className="text-[11px] h-8 text-chaise-header-text font-semibold">Type</TableHead>
                  <TableHead className="text-[11px] h-8 w-12 text-chaise-header-text font-semibold">Null</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {info.columns.map((col) => (
                  <TableRow
                    key={col.name}
                    className={`hover:bg-chaise-hover ${
                      systemCols.has(col.name) ? "opacity-50" : ""
                    }`}
                    title={col.comment || ""}
                  >
                    <TableCell className="text-xs py-1.5">
                      <span className="font-mono">{col.name}</span>
                      {col.comment && (
                        <p className="text-[11px] text-slate-400 mt-0.5 font-sans leading-snug truncate max-w-[200px]">
                          {col.comment}
                        </p>
                      )}
                    </TableCell>
                    <TableCell className="text-xs py-1.5 text-slate-500 align-top">
                      {col.type}
                    </TableCell>
                    <TableCell className="text-xs py-1.5 text-slate-400 align-top">
                      {col.nullok ? "yes" : "no"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TabsContent>

          <TabsContent value="fks" className="mt-0 px-4 pb-4">
            {info.foreign_keys.length === 0 ? (
              <p className="text-xs text-slate-400 py-4">No foreign keys</p>
            ) : (
              <div className="space-y-2 pt-2">
                {info.foreign_keys.map((fk, i) => (
                  <div
                    key={i}
                    className="border border-slate-200 rounded p-2 text-xs"
                  >
                    <div className="font-mono text-slate-700">
                      {fk.columns.join(", ")}
                    </div>
                    <Separator className="my-1.5" />
                    <div className="text-slate-500 flex items-center gap-1">
                      <span className="text-xs">→</span>
                      <span className="font-mono">
                        {fk.referenced_table}
                      </span>
                      <span className="text-slate-400">
                        ({fk.referenced_columns.join(", ")})
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Browse tab is rendered outside ScrollArea above */}

          {info.features && info.features.length > 0 && (
            <TabsContent value="features" className="mt-0 px-4 pb-4">
              <div className="space-y-2 pt-2">
                {info.features.map((feat) => (
                  <div
                    key={feat.name}
                    className="border border-slate-200 rounded p-2 text-xs"
                  >
                    <div className="font-semibold text-slate-700">
                      {feat.name}
                    </div>
                    <div className="text-slate-500 font-mono text-[11px] mt-0.5">
                      {feat.feature_table}
                    </div>
                  </div>
                ))}
              </div>
            </TabsContent>
          )}

          <TabsContent value="annotations" className="mt-0 px-4 pb-4">
            <AnnotationsPanel table={table} onDirtyChange={setAnnotationsDirty} />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}
