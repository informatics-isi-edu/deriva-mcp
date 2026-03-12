import { useCallback, useEffect, useMemo, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";

import ERDCanvas from "@/components/erd/ERDCanvas";
import DetailPanel from "@/components/erd/DetailPanel";
import Toolbar from "@/components/erd/Toolbar";
import SplitLayout from "@/components/erd/SplitLayout";
import CatalogPicker from "@/components/erd/CatalogPicker";
import type { CatalogSchema, EnrichedTable, SchemaFilter } from "@/types";
import {
  fetchSchema,
  buildEnrichedTables,
  getCatalogInfo,
} from "@/ermrest-client";
import { hasCatalogConfig, getCatalogConfig } from "@/catalog-config";

export default function App() {
  const [schema, setSchema] = useState<CatalogSchema | null>(null);
  const [tables, setTables] = useState<EnrichedTable[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasConfig, setHasConfig] = useState(hasCatalogConfig());

  // Navigation state
  const [viewMode, setViewMode] = useState<"schemas" | "tables">("schemas");
  const [activeSchema, setActiveSchema] = useState<string | null>(null);

  // UI state
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filter, setFilter] = useState<SchemaFilter>("all");
  const [hideAssociations, setHideAssociations] = useState(true);

  const { hostname, catalogId } = getCatalogInfo();

  // Listen for hash changes (user navigating to a new catalog)
  useEffect(() => {
    const onHashChange = () => {
      setHasConfig(hasCatalogConfig());
      // Reset state for new catalog
      setSchema(null);
      setTables([]);
      setError(null);
      setLoading(true);
      setViewMode("schemas");
      setActiveSchema(null);
      setSelectedTable(null);
      setSearchQuery("");
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (!hasConfig) {
      setLoading(false);
      return;
    }
    async function load() {
      try {
        setLoading(true);
        const s = await fetchSchema();
        setSchema(s);
        const enriched = await buildEnrichedTables(s);
        setTables(enriched);
      } catch (e: any) {
        setError(e.message || "Failed to load catalog schema");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [hasConfig]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (selectedTable) {
          setSelectedTable(null);
        } else if (viewMode === "tables") {
          setViewMode("schemas");
          setActiveSchema(null);
          setSelectedTable(null);
          setSearchQuery("");
          setFilter("all");
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedTable, viewMode]);

  const onDrillIntoSchema = useCallback((schemaName: string) => {
    setViewMode("tables");
    setActiveSchema(schemaName);
    setSelectedTable(null);
    setSearchQuery("");
    setFilter("all");
  }, []);

  const onBackToSchemas = useCallback(() => {
    setViewMode("schemas");
    setSelectedTable(null);
    setSearchQuery("");
    setFilter("all");
  }, []);

  const handleJumpToTable = useCallback(
    (table: EnrichedTable) => {
      // Drill into the table's schema if not already there
      if (viewMode !== "tables" || activeSchema !== table.schema) {
        setViewMode("tables");
        setActiveSchema(table.schema);
        setFilter("all");
        setHideAssociations(false); // unhide in case target is an association
      }
      setSelectedTable(table.qualifiedName);
      setSearchQuery("");
    },
    [viewMode, activeSchema]
  );

  const onSelectTable = useCallback(
    (qn: string | null) => {
      if (viewMode === "schemas") {
        setActiveSchema(qn);
        setSelectedTable(null);
      } else {
        setSelectedTable(qn);
      }
    },
    [viewMode]
  );

  const selectedTableData = useMemo(
    () => tables.find((t) => t.qualifiedName === selectedTable) ?? null,
    [tables, selectedTable]
  );

  const visibleCount = useMemo(() => {
    if (viewMode === "schemas") return tables.length;
    let filtered = tables.filter((t) => t.schema === activeSchema);
    if (filter !== "all")
      filtered = filtered.filter((t) => t.tableType === filter);
    if (hideAssociations)
      filtered = filtered.filter((t) => !t.info.is_association);
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (t) =>
          t.name.toLowerCase().includes(q) ||
          t.schema.toLowerCase().includes(q)
      );
    }
    return filtered.length;
  }, [tables, viewMode, activeSchema, filter, hideAssociations, searchQuery]);

  const activeSchemaTableCount = useMemo(() => {
    if (!activeSchema) return tables.length;
    return tables.filter((t) => t.schema === activeSchema).length;
  }, [tables, activeSchema]);

  // No catalog configured — show picker
  if (!hasConfig) {
    return <CatalogPicker />;
  }

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <div className="text-sm font-medium text-slate-600 mb-2">
            Loading catalog schema...
          </div>
          <div className="text-xs text-slate-400">
            {hostname} / #{catalogId}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center max-w-md">
          <div className="text-sm font-medium text-red-700 mb-2">
            Failed to load schema
          </div>
          <div className="text-xs text-slate-500 mb-4">{error}</div>
          <div className="text-xs text-slate-400 space-y-1">
            <p>
              {getCatalogConfig().isSameOrigin
                ? `Make sure you are logged into ${hostname} in this browser.`
                : "This catalog may require authentication, or the server may need CORS configured for credentialed requests."}
            </p>
            <p className="font-mono text-[10px]">
              {hostname} / catalog {catalogId}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-white">
      <Toolbar
        hostname={hostname}
        catalogId={catalogId}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        filter={filter}
        onFilterChange={setFilter}
        hideAssociations={hideAssociations}
        onToggleAssociations={setHideAssociations}
        tableCount={
          viewMode === "schemas" ? tables.length : activeSchemaTableCount
        }
        visibleCount={visibleCount}
        viewMode={viewMode}
        activeSchema={activeSchema}
        onBackToSchemas={onBackToSchemas}
        allTables={tables}
        onJumpToTable={handleJumpToTable}
      />

      <SplitLayout
        defaultRightWidth={380}
        minRightWidth={280}
        maxRightWidth={700}
        left={
          <ReactFlowProvider>
            <ERDCanvas
              tables={tables}
              schema={schema!}
              filter={filter}
              hideAssociations={hideAssociations}
              searchQuery={searchQuery}
              selectedTable={selectedTable}
              onSelectTable={onSelectTable}
              viewMode={viewMode}
              activeSchema={activeSchema}
              onDrillIntoSchema={onDrillIntoSchema}
            />
          </ReactFlowProvider>
        }
        right={
          <DetailPanel
            table={selectedTableData}
            onClose={() => {
              setSelectedTable(null);
              if (viewMode === "schemas") setActiveSchema(null);
            }}
            viewMode={viewMode}
            activeSchema={activeSchema}
            schema={schema}
            tables={tables}
            onDrillIntoSchema={onDrillIntoSchema}
          />
        }
      />
    </div>
  );
}
