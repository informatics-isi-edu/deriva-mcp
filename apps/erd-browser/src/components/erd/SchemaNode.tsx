import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { ChevronRight } from "lucide-react";
import NodeTooltip from "./NodeTooltip";

export interface SchemaNodeData extends Record<string, unknown> {
  label: string;
  tableCount: number;
  domainCount: number;
  vocabCount: number;
  assetCount: number;
  assocCount: number;
  isSelected: boolean;
  isMlSchema: boolean;
  comment: string;
  onDrillIn?: () => void;
}

function SchemaNode({ data }: NodeProps) {
  const d = data as unknown as SchemaNodeData;
  const selected = d.isSelected;

  const bg = d.isMlSchema ? "bg-amber-50" : "bg-white";
  const border = d.isMlSchema ? "border-amber-400" : "border-slate-400";
  const headerBg = d.isMlSchema ? "bg-amber-600" : "bg-chaise-navbar";

  return (
    <NodeTooltip
      text={d.comment || `${d.label} — ${d.tableCount} tables`}
      className={`
        rounded-lg border-2 shadow-md min-w-[220px] overflow-hidden
        transition-all duration-150 cursor-pointer
        ${bg} ${border}
        ${selected ? "ring-2 ring-offset-2 ring-brand shadow-lg scale-[1.02]" : ""}
        hover:shadow-lg hover:scale-[1.01]
      `}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !bg-slate-500 !border-2 !border-white"
      />

      {/* Header */}
      <div className={`${headerBg} px-4 py-2.5 flex items-center justify-between gap-2`}>
        <span className="text-sm font-bold text-white tracking-wide">
          {d.label}
        </span>
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-semibold text-white/80 bg-white/20 px-2 py-0.5 rounded-full">
            {d.tableCount}
          </span>
          {d.onDrillIn && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                d.onDrillIn!();
              }}
              className="flex items-center justify-center w-6 h-6 rounded bg-white/20 hover:bg-white/40 transition-colors"
              title="Expand schema tables"
            >
              <ChevronRight className="h-4 w-4 text-white" />
            </button>
          )}
        </div>
      </div>

      {/* Breakdown */}
      <div className="px-4 py-2.5 grid grid-cols-2 gap-x-4 gap-y-1">
        {d.domainCount > 0 && (
          <TypeRow label="Domain" count={d.domainCount} color="bg-slate-700" />
        )}
        {d.vocabCount > 0 && (
          <TypeRow label="Vocab" count={d.vocabCount} color="bg-emerald-700" />
        )}
        {d.assetCount > 0 && (
          <TypeRow label="Asset" count={d.assetCount} color="bg-sky-700" />
        )}
        {d.assocCount > 0 && (
          <TypeRow label="Assoc" count={d.assocCount} color="bg-zinc-500" />
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !bg-slate-500 !border-2 !border-white"
      />
    </NodeTooltip>
  );
}

function TypeRow({ label, count, color }: { label: string; count: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className={`w-2 h-2 rounded-sm ${color}`} />
      <span className="text-xs text-slate-600">
        {label}
      </span>
      <span className="text-xs font-semibold text-slate-800 ml-auto">
        {count}
      </span>
    </div>
  );
}

export default memo(SchemaNode);
