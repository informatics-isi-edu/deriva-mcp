import { useCallback, useRef, useState } from "react";
import { Search, Filter, ChevronRight, LayoutGrid, ZoomIn, ZoomOut, Maximize2, Map } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
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
  // Autocomplete
  allTables: EnrichedTable[];
  onJumpToTable: (table: EnrichedTable) => void;
  canvasControls: CanvasControls | null;
}

const TYPE_DOT_COLORS: Record<string, string> = {
  domain: "bg-slate-600",
  ml: "bg-amber-600",
  vocabulary: "bg-emerald-600",
  asset: "bg-sky-600",
  association: "bg-zinc-400",
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
  allTables,
  onJumpToTable,
  canvasControls,
}: ToolbarProps) {
  const [open, setOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

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
    <div className="h-12 border-b border-slate-200 bg-white px-4 flex items-center gap-3">
      {/* Breadcrumb navigation */}
      <div className="flex items-center gap-1.5">
        <button
          onClick={onBackToSchemas}
          className={`text-sm font-semibold tracking-tight transition-colors ${
            viewMode === "schemas"
              ? "text-slate-800"
              : "text-blue-600 hover:text-blue-800 hover:underline cursor-pointer"
          }`}
        >
          {hostname}
          <span className="font-mono text-xs text-slate-400 ml-1">
            #{catalogId}
          </span>
        </button>

        {viewMode === "tables" && activeSchema && (
          <>
            <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-sm font-semibold text-slate-800">
              {activeSchema}
            </span>
          </>
        )}
      </div>

      {viewMode === "tables" && (
        <>
          <div className="h-5 w-px bg-slate-200" />
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs gap-1"
            onClick={onBackToSchemas}
          >
            <LayoutGrid className="h-3 w-3" />
            All schemas
          </Button>
        </>
      )}

      <div className="h-5 w-px bg-slate-200" />

      {/* Search with autocomplete */}
      <div className="relative flex-1 max-w-sm">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400 z-10" />
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
          className="pl-8 h-8 text-xs"
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
                    ? "bg-slate-100"
                    : "hover:bg-slate-50"
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
                    className="text-[9px] px-1.5 py-0 flex-shrink-0"
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
              className="h-8 text-xs gap-1.5"
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
      <span className="text-[11px] text-slate-400 ml-auto">
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
          <div className="h-5 w-px bg-slate-200" />
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
              onClick={canvasControls.zoomIn}
              title="Zoom in"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
              onClick={canvasControls.zoomOut}
              title="Zoom out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
              onClick={canvasControls.fitView}
              title="Fit view"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className={`h-7 w-7 p-0 ${canvasControls.showMiniMap ? "text-slate-700" : "text-slate-400"}`}
              onClick={() => canvasControls.setShowMiniMap(!canvasControls.showMiniMap)}
              title={canvasControls.showMiniMap ? "Hide mini map" : "Show mini map"}
            >
              <Map className="h-3.5 w-3.5" />
            </Button>
            {canvasControls.showMiniMap && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-1.5 text-[10px] font-medium text-slate-500"
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
