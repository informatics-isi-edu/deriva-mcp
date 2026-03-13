import type { TableType } from "@/types";

const LEGEND_ITEMS: { type: TableType; label: string; color: string }[] = [
  { type: "domain", label: "Domain", color: "bg-slate-700" },
  { type: "ml", label: "ML", color: "bg-amber-700" },
  { type: "vocabulary", label: "Vocabulary", color: "bg-emerald-700" },
  { type: "asset", label: "Asset", color: "bg-sky-700" },
  { type: "association", label: "Association", color: "bg-zinc-500" },
];

export default function Legend() {
  return (
    <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-md px-3 py-2 flex gap-3 z-10">
      {LEGEND_ITEMS.map((item) => (
        <div key={item.type} className="flex items-center gap-1.5">
          <div className={`w-2.5 h-2.5 rounded-sm ${item.color}`} />
          <span className="text-[11px] text-slate-600">{item.label}</span>
        </div>
      ))}
    </div>
  );
}
