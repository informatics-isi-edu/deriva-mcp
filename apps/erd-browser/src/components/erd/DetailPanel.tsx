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
import AnnotationsPanel, { SchemaAnnotationsPanel } from "./AnnotationsPanel";

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
}

export default function DetailPanel({
  table,
  onClose,
  viewMode,
  activeSchema,
  schema,
  tables,
  onDrillIntoSchema,
}: DetailPanelProps) {
  if (viewMode === "schemas" && activeSchema && schema) {
    return (
      <SchemaDetail
        schemaName={activeSchema}
        tables={tables}
        schema={schema}
        onDrillIn={() => onDrillIntoSchema(activeSchema)}
        onClose={onClose}
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
}: {
  schemaName: string;
  tables: EnrichedTable[];
  schema: CatalogSchema;
  onDrillIn: () => void;
  onClose: () => void;
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

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 py-3 border-b border-slate-200 flex-shrink-0">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-900">{schemaName}</h2>
            <p className="text-xs text-slate-500 mt-0.5">
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
          <p className="text-xs text-slate-600 mt-2 leading-relaxed">
            {schemaComment}
          </p>
        )}
        <Button
          size="sm"
          className="mt-3 w-full text-xs"
          onClick={onDrillIn}
        >
          View tables in {schemaName}
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="px-4 py-3 space-y-4">
          {/* Table breakdown */}
          <Section title="Table breakdown">
            {domainTables.length > 0 && (
              <TypeList label="Domain" tables={domainTables} color="bg-slate-600" />
            )}
            {mlTables.length > 0 && (
              <TypeList label="ML" tables={mlTables} color="bg-amber-600" />
            )}
            {vocabTables.length > 0 && (
              <TypeList label="Vocabulary" tables={vocabTables} color="bg-emerald-600" />
            )}
            {assetTables.length > 0 && (
              <TypeList label="Asset" tables={assetTables} color="bg-sky-600" />
            )}
            {assocTables.length > 0 && (
              <TypeList label="Association" tables={assocTables} color="bg-zinc-400" />
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

          {/* Schema annotations */}
          <Section title="Annotations">
            <SchemaAnnotationsPanel
              annotations={schema.schemas[schemaName]?.annotations || {}}
            />
          </Section>
        </div>
      </ScrollArea>
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
      <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
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
}: {
  label: string;
  tables: EnrichedTable[];
  color: string;
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
          <div key={t.qualifiedName} className="text-[11px] text-slate-500 flex items-center justify-between">
            <span className="truncate">{t.name}</span>
            {t.recordCount !== null && t.recordCount >= 0 && (
              <span className="text-[10px] text-slate-400 ml-2 whitespace-nowrap">
                {t.recordCount.toLocaleString()}
              </span>
            )}
          </div>
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

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-200 flex-shrink-0">
        <div className="flex items-start justify-between">
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-bold text-slate-900 truncate">
              {table.name}
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">{table.schema}</p>
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
          <p className="text-xs text-slate-600 mt-2 leading-relaxed">
            {info.comment}
          </p>
        )}

        <div className="mt-2">
          <a
            href={chaiseUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 hover:underline"
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
          <TabsTrigger value="annotations" className="text-xs h-7 gap-1">
            <Settings2 className="h-3 w-3" />
            Annotations
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
                <TableRow>
                  <TableHead className="text-[11px] h-8">Name</TableHead>
                  <TableHead className="text-[11px] h-8">Type</TableHead>
                  <TableHead className="text-[11px] h-8 w-12">Null</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {info.columns.map((col) => (
                  <TableRow
                    key={col.name}
                    className={
                      systemCols.has(col.name) ? "opacity-50" : ""
                    }
                    title={col.comment || ""}
                  >
                    <TableCell className="text-xs py-1.5">
                      <span className="font-mono">{col.name}</span>
                      {col.comment && (
                        <p className="text-[10px] text-slate-400 mt-0.5 font-sans leading-snug truncate max-w-[200px]">
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
                      <span className="text-[10px]">→</span>
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
                    <div className="text-slate-500 font-mono text-[10px] mt-0.5">
                      {feat.feature_table}
                    </div>
                  </div>
                ))}
              </div>
            </TabsContent>
          )}

          <TabsContent value="annotations" className="mt-0 px-4 pb-4">
            <AnnotationsPanel table={table} />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}
