import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { TableType } from "@/types";
import NodeTooltip from "./NodeTooltip";

export interface TableNodeData extends Record<string, unknown> {
  label: string;
  schema: string;
  tableType: TableType;
  recordCount: number | null;
  isSelected: boolean;
  comment: string;
}

const TYPE_COLORS: Record<TableType, { bg: string; border: string; badge: string; header: string }> = {
  domain: {
    bg: "bg-white",
    border: "border-slate-400",
    badge: "bg-slate-700 text-white",
    header: "text-slate-900",
  },
  ml: {
    bg: "bg-amber-50",
    border: "border-amber-400",
    badge: "bg-amber-700 text-white",
    header: "text-amber-900",
  },
  vocabulary: {
    bg: "bg-emerald-50",
    border: "border-emerald-400",
    badge: "bg-emerald-700 text-white",
    header: "text-emerald-900",
  },
  asset: {
    bg: "bg-sky-50",
    border: "border-sky-400",
    badge: "bg-sky-700 text-white",
    header: "text-sky-900",
  },
  association: {
    bg: "bg-zinc-100",
    border: "border-zinc-400",
    badge: "bg-zinc-500 text-white",
    header: "text-zinc-700",
  },
};

function TableNode({ data }: NodeProps) {
  const d = data as unknown as TableNodeData;
  const colors = TYPE_COLORS[d.tableType];
  const selected = d.isSelected;

  return (
    <NodeTooltip
      text={d.comment || ""}
      className={`
        px-5 py-4 rounded-lg border-2 shadow-sm min-w-[240px] max-w-[320px]
        transition-all duration-150 cursor-pointer
        ${colors.bg} ${colors.border}
        ${selected ? "ring-2 ring-offset-2 ring-blue-500 shadow-lg scale-[1.02]" : ""}
        hover:shadow-md hover:scale-[1.01]
      `}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !bg-slate-500 !border-2 !border-white"
      />
      <div className="flex items-center justify-between gap-3">
        <span className={`text-base font-bold ${colors.header} truncate leading-tight`}>
          {d.label}
        </span>
        {d.recordCount !== null && d.recordCount >= 0 && (
          <span
            className={`
              text-xs font-semibold px-2.5 py-1 rounded-full leading-none whitespace-nowrap
              ${colors.badge}
            `}
          >
            {d.recordCount.toLocaleString()}
          </span>
        )}
      </div>
      {d.comment && (
        <p className="text-xs text-slate-500 mt-1.5 truncate leading-snug">
          {d.comment}
        </p>
      )}
      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !bg-slate-500 !border-2 !border-white"
      />
    </NodeTooltip>
  );
}

export default memo(TableNode);
