import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Columns3, Eye, EyeOff, Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import type { EnrichedTable } from "@/types";
import { fetchPagedData, type PagedResult } from "@/ermrest-client";

const PAGE_SIZE = 25;
const SYSTEM_COLS = new Set(["RID", "RCT", "RMT", "RCB", "RMB"]);

interface DataBrowserProps {
  table: EnrichedTable;
}

export default function DataBrowser({ table }: DataBrowserProps) {
  const [data, setData] = useState<PagedResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [pageNum, setPageNum] = useState(1);
  const [pageStack, setPageStack] = useState<(string | null)[]>([null]);
  const [useVisibleCols, setUseVisibleCols] = useState(true);
  const [hiddenCols, setHiddenCols] = useState<Set<string>>(new Set());
  const [searchInput, setSearchInput] = useState("");
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  // Ref to track the active search term for fetches (avoids effect loops)
  const activeSearchRef = useRef("");

  const sortCol = "RID";

  // All non-system columns
  const allColumns = useMemo(
    () => table.info.columns.map((c) => c.name).filter((n) => !SYSTEM_COLS.has(n)),
    [table.qualifiedName] // eslint-disable-line react-hooks/exhaustive-deps
  );

  // Visible columns from annotation
  const annotatedCols = table.info.visible_columns;
  const hasAnnotation = annotatedCols && annotatedCols.length > 0;

  // Active columns to display
  const activeColumns = (() => {
    let cols: string[];
    if (useVisibleCols && hasAnnotation) {
      const colSet = new Set(allColumns);
      cols = annotatedCols!.filter((c) => colSet.has(c));
    } else {
      cols = allColumns;
    }
    return cols.filter((c) => !hiddenCols.has(c));
  })();

  // Searchable columns: text-like types for ERMrest ciregexp
  const searchCols = useMemo(() => {
    const TEXT_TYPES = new Set(["text", "markdown", "longtext", "shorttext", "json", "jsonb"]);
    const textCols = table.info.columns
      .filter((c) => !SYSTEM_COLS.has(c.name) && TEXT_TYPES.has(c.type))
      .map((c) => c.name);
    if (textCols.length > 0) return textCols;
    return allColumns.filter((c) => c === "Name" || c === "name");
  }, [table.qualifiedName, allColumns]); // eslint-disable-line react-hooks/exhaustive-deps

  // Imperative fetch — no effects watching search state
  const doFetch = useCallback(
    async (afterValue: string | null, search: string) => {
      setLoading(true);
      try {
        const result = await fetchPagedData(
          table.schema,
          table.name,
          sortCol,
          PAGE_SIZE,
          afterValue,
          search || undefined,
          search ? searchCols : undefined
        );
        setData(result);
      } catch {
        setData({ rows: [], hasMore: false });
      } finally {
        setLoading(false);
      }
    },
    [table.schema, table.name, searchCols]
  );

  // Reset everything on table change
  useEffect(() => {
    setPageNum(1);
    setPageStack([null]);
    setHiddenCols(new Set());
    setSearchInput("");
    activeSearchRef.current = "";
    doFetch(null, "");
  }, [table.qualifiedName, doFetch]);

  // Search handlers — debounce input, then fetch imperatively
  const handleSearchChange = (value: string) => {
    setSearchInput(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      activeSearchRef.current = value;
      setPageNum(1);
      setPageStack([null]);
      doFetch(null, value);
    }, 400);
  };

  const clearSearch = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setSearchInput("");
    activeSearchRef.current = "";
    setPageNum(1);
    setPageStack([null]);
    doFetch(null, "");
  };

  const goNext = useCallback(() => {
    if (!data || !data.hasMore || data.rows.length === 0) return;
    const lastRow = data.rows[data.rows.length - 1];
    const afterVal = String(lastRow[sortCol] ?? "");
    const newStack = [...pageStack, afterVal];
    setPageStack(newStack);
    setPageNum(pageNum + 1);
    doFetch(afterVal, activeSearchRef.current);
  }, [data, pageStack, pageNum, doFetch]);

  const goPrev = useCallback(() => {
    if (pageNum <= 1) return;
    const newStack = pageStack.slice(0, -1);
    setPageStack(newStack);
    setPageNum(pageNum - 1);
    doFetch(newStack[newStack.length - 1], activeSearchRef.current);
  }, [pageNum, pageStack, doFetch]);

  const toggleCol = (col: string) => {
    setHiddenCols((prev) => {
      const next = new Set(prev);
      if (next.has(col)) next.delete(col);
      else next.add(col);
      return next;
    });
  };

  const formatCellValue = (val: unknown): string => {
    if (val === null || val === undefined) return "—";
    if (typeof val === "object") return JSON.stringify(val);
    const s = String(val);
    return s.length > 80 ? s.slice(0, 77) + "..." : s;
  };

  return (
    <div className="flex flex-col h-full">
      {/* Controls bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-100 flex-shrink-0">
        {/* Search */}
        <div className="relative flex-shrink-0 w-48">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-slate-400" />
          <Input
            placeholder="Search rows..."
            value={searchInput}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="pl-7 pr-7 h-7 text-[11px]"
          />
          {searchInput && (
            <button
              onClick={clearSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>

        {/* Visible columns toggle */}
        {hasAnnotation && (
          <Button
            variant={useVisibleCols ? "secondary" : "outline"}
            size="sm"
            className="h-7 text-[11px] gap-1"
            onClick={() => setUseVisibleCols(!useVisibleCols)}
          >
            {useVisibleCols ? (
              <Eye className="h-3 w-3" />
            ) : (
              <EyeOff className="h-3 w-3" />
            )}
            {useVisibleCols ? "Visible columns" : "All columns"}
          </Button>
        )}

        {/* Column picker */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm" className="h-7 text-[11px] gap-1">
              <Columns3 className="h-3 w-3" />
              Columns
              <Badge variant="secondary" className="text-[10px] px-1 py-0 ml-1">
                {activeColumns.length}
              </Badge>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="max-h-[300px] overflow-y-auto">
            <DropdownMenuLabel className="text-[10px]">
              Toggle columns
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            {(useVisibleCols && hasAnnotation ? annotatedCols! : allColumns).map(
              (col) => (
                <DropdownMenuCheckboxItem
                  key={col}
                  checked={!hiddenCols.has(col)}
                  onCheckedChange={() => toggleCol(col)}
                  className="text-[11px] font-mono"
                >
                  {col}
                </DropdownMenuCheckboxItem>
              )
            )}
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Page controls */}
        <div className="flex items-center gap-1 ml-auto">
          <Button
            variant="outline"
            size="icon"
            className="h-7 w-7"
            onClick={goPrev}
            disabled={pageNum <= 1 || loading}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="text-[11px] text-slate-500 px-1 min-w-[50px] text-center">
            Page {pageNum}
          </span>
          <Button
            variant="outline"
            size="icon"
            className="h-7 w-7"
            onClick={goNext}
            disabled={!data?.hasMore || loading}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Data table */}
      <ScrollArea className="flex-1 min-h-0">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs text-slate-400">Loading...</span>
          </div>
        ) : !data || data.rows.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs text-slate-400">
              {activeSearchRef.current ? "No matching rows" : "No data"}
            </span>
          </div>
        ) : (
          <div className="px-2">
            <Table>
              <TableHeader>
                <TableRow className="bg-chaise-header/40">
                  {activeColumns.map((col) => (
                    <TableHead
                      key={col}
                      className="text-[10px] h-7 whitespace-nowrap font-semibold text-chaise-header-text"
                    >
                      {col}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.rows.map((row, i) => (
                  <TableRow key={i} className="hover:bg-chaise-hover">
                    {activeColumns.map((col) => (
                      <TableCell
                        key={col}
                        className="text-[11px] py-1 max-w-[160px] truncate"
                        title={String(row[col] ?? "")}
                      >
                        {formatCellValue(row[col])}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
