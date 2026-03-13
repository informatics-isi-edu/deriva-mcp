import { useCallback, useMemo, useRef, useState } from "react";
import { Search, Filter, ChevronDown, ChevronRight, LayoutGrid, ZoomIn, ZoomOut, Maximize2, Map } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import type { EnrichedTable, SchemaFilter } from "@/types";
import type { CanvasControls } from "./ERDCanvas";

interface ToolbarProps {
  hostname: string;
  catalogId: string;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  filter: SchemaFilter;
  onFilterChange: (filter: SchemaFilter) => void;
  hideAssociations: boolean;
  onToggleAssociations: (hide: boolean) => void;
  tableCount: number;
  visibleCount: number;
  viewMode: "schemas" | "tables";
  activeSchema: string | null;
  onBackToSchemas: () => void;
  onDrillIntoSchema: (schemaName: string) => void;
  // Autocomplete
  allTables: EnrichedTable[];
  onJumpToTable: (table: EnrichedTable) => void;
  canvasControls: CanvasControls | null;
  onCatalogClick?: () => void;
}

const TYPE_DOT_COLORS: Record<string, string> = {
  domain: "bg-slate-700",
  ml: "bg-amber-700",
  vocabulary: "bg-emerald-700",
  asset: "bg-sky-700",
  association: "bg-zinc-500",
};

export default function Toolbar({
  hostname,
  catalogId,
  searchQuery,
  onSearchChange,
  filter,
  onFilterChange,
  hideAssociations,
  onToggleAssociations,
  tableCount,
  visibleCount,
  viewMode,
  activeSchema,
  onBackToSchemas,
  onDrillIntoSchema,
  allTables,
  onJumpToTable,
  canvasControls,
  onCatalogClick,
}: ToolbarProps) {
  const [open, setOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Derive unique schema names from tables for the schema dropdown
  const schemaNames = useMemo(() => {
    const names = new Set(allTables.map((t) => t.schema));
    return [...names].sort();
  }, [allTables]);

  // Compute filtered suggestions
  const suggestions = (() => {
    if (!searchQuery || searchQuery.length < 1) return [];
    const q = searchQuery.toLowerCase();
    return allTables
      .filter(
        (t) =>
          t.name.toLowerCase().includes(q) ||
          t.schema.toLowerCase().includes(q) ||
          t.qualifiedName.toLowerCase().includes(q) ||
          (t.info.comment && t.info.comment.toLowerCase().includes(q))
      )
      .slice(0, 12);
  })();

  const showDropdown = open && suggestions.length > 0;

  const handleSelect = useCallback(
    (table: EnrichedTable) => {
      onJumpToTable(table);
      onSearchChange("");
      setOpen(false);
      setHighlightIndex(0);
      inputRef.current?.blur();
    },
    [onJumpToTable, onSearchChange]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!showDropdown) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHighlightIndex((i) => Math.min(i + 1, suggestions.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setHighlightIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (suggestions[highlightIndex]) {
          handleSelect(suggestions[highlightIndex]);
        }
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    },
    [showDropdown, suggestions, highlightIndex, handleSelect]
  );

  return (
    <div className="h-11 border-b border-chaise-navbar bg-chaise-navbar px-4 flex items-center gap-3">
      {/* Breadcrumb navigation */}
      <div className="flex items-center gap-1.5">
        <button
          onClick={viewMode === "schemas" ? onCatalogClick : onBackToSchemas}
          className={`text-sm font-semibold tracking-tight transition-colors ${
            viewMode === "schemas"
              ? "text-chaise-navbar-text hover:text-white cursor-pointer"
              : "text-sky-300 hover:text-white hover:underline cursor-pointer"
          }`}
          title={viewMode === "schemas" ? "View catalog annotations" : "Back to schemas"}
        >
          {hostname}
          <span className="font-mono text-[11px] text-chaise-navbar-text/60 ml-1">
            #{catalogId}
          </span>
        </button>

        {viewMode === "tables" && activeSchema && (
          <>
            <ChevronRight className="h-3.5 w-3.5 text-chaise-navbar-text/50" />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="flex items-center gap-1 text-sm font-semibold text-chaise-navbar-text hover:text-white transition-colors">
                  {activeSchema}
                  <ChevronDown className="h-3 w-3 text-chaise-navbar-text/50" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="min-w-[160px]">
                <DropdownMenuLabel className="text-[11px]">
                  Switch schema
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                {schemaNames.map((name) => (
                  <DropdownMenuItem
                    key={name}
                    onClick={() => onDrillIntoSchema(name)}
                    className={`text-xs ${name === activeSchema ? "font-semibold text-brand" : ""}`}
                  >
                    {name}
                    {name === activeSchema && (
                      <span className="ml-auto text-[10px] text-slate-400">current</span>
                    )}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        )}
      </div>

      {viewMode === "tables" && (
        <>
          <div className="h-5 w-px bg-white/20" />
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs gap-1 bg-transparent border-white/30 text-chaise-navbar-text hover:bg-white/10 hover:text-white"
            onClick={onBackToSchemas}
          >
            <LayoutGrid className="h-3 w-3" />
            All schemas
          </Button>
        </>
      )}

      <div className="h-5 w-px bg-white/20" />

      {/* Search with autocomplete */}
      <div className="relative flex-1 max-w-sm">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-white/50 z-10" />
        <Input
          ref={inputRef}
          placeholder="Jump to table..."
          value={searchQuery}
          onChange={(e) => {
            onSearchChange(e.target.value);
            setOpen(true);
            setHighlightIndex(0);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => {
            // Delay to allow click on suggestion
            setTimeout(() => setOpen(false), 200);
          }}
          onKeyDown={handleKeyDown}
          className="pl-8 h-8 text-xs bg-white/10 border-white/20 text-white placeholder:text-white/40 focus:bg-white/15 focus:border-white/40"
        />

        {/* Autocomplete dropdown */}
        {showDropdown && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-slate-200 rounded-md shadow-lg z-50 overflow-hidden">
            {suggestions.map((t, i) => (
              <button
                key={t.qualifiedName}
                onMouseDown={(e) => {
                  e.preventDefault(); // prevent blur
                  handleSelect(t);
                }}
                onMouseEnter={() => setHighlightIndex(i)}
                className={`w-full text-left px-3 py-2 flex items-center gap-2 transition-colors ${
                  i === highlightIndex
                    ? "bg-chaise-hover"
                    : "hover:bg-chaise-hover/50"
                }`}
              >
                <div
                  className={`w-2 h-2 rounded-sm flex-shrink-0 ${
                    TYPE_DOT_COLORS[t.tableType] || "bg-slate-400"
                  }`}
                />
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium text-slate-800 truncate">
                    {t.name}
                  </div>
                  <div className="text-[10px] text-slate-400 truncate">
                    {t.schema}
                    {t.info.comment && ` · ${t.info.comment}`}
                  </div>
                </div>
                {t.recordCount !== null && t.recordCount >= 0 && (
                  <Badge
                    variant="secondary"
                    className="text-[10px] px-1.5 py-0 flex-shrink-0"
                  >
                    {t.recordCount.toLocaleString()}
                  </Badge>
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Filter dropdown (table view only) */}
      {viewMode === "tables" && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1.5 bg-transparent border-white/30 text-chaise-navbar-text hover:bg-white/10 hover:text-white"
            >
              <Filter className="h-3.5 w-3.5" />
              {filter === "all" ? "All types" : filter}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel className="text-[11px]">
              Table type
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            {(
              ["all", "domain", "ml", "vocabulary", "asset"] as SchemaFilter[]
            ).map((f) => (
              <DropdownMenuCheckboxItem
                key={f}
                checked={filter === f}
                onCheckedChange={() => onFilterChange(f)}
                className="text-xs capitalize"
              >
                {f === "all" ? "All types" : f}
              </DropdownMenuCheckboxItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuCheckboxItem
              checked={hideAssociations}
              onCheckedChange={(checked) =>
                onToggleAssociations(checked as boolean)
              }
              className="text-xs"
            >
              Hide associations
            </DropdownMenuCheckboxItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}

      {/* Count */}
      <span className="text-xs text-chaise-navbar-text/60 ml-auto">
        {viewMode === "schemas" ? (
          <>{tableCount} tables across schemas</>
        ) : (
          <>
            {visibleCount} / {tableCount} tables
          </>
        )}
      </span>

      {/* Canvas controls */}
      {canvasControls && (
        <>
          <div className="h-5 w-px bg-white/20" />
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-chaise-navbar-text/70 hover:text-white hover:bg-white/10"
              onClick={canvasControls.zoomIn}
              title="Zoom in"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-chaise-navbar-text/70 hover:text-white hover:bg-white/10"
              onClick={canvasControls.zoomOut}
              title="Zoom out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-chaise-navbar-text/70 hover:text-white hover:bg-white/10"
              onClick={canvasControls.fitView}
              title="Fit view"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className={`h-7 w-7 p-0 hover:bg-white/10 ${canvasControls.showMiniMap ? "text-white" : "text-chaise-navbar-text/50"}`}
              onClick={() => canvasControls.setShowMiniMap(!canvasControls.showMiniMap)}
              title={canvasControls.showMiniMap ? "Hide mini map" : "Show mini map"}
            >
              <Map className="h-3.5 w-3.5" />
            </Button>
            {canvasControls.showMiniMap && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-1.5 text-[10px] font-medium text-chaise-navbar-text/60 hover:text-white hover:bg-white/10"
                onClick={canvasControls.cycleMapSize}
                title={`Map size: ${canvasControls.mapSize}`}
              >
                {canvasControls.mapSize}
              </Button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
