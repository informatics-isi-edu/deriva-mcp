import { useCallback, useRef, useState } from "react";

interface SplitLayoutProps {
  left: React.ReactNode;
  right: React.ReactNode;
  defaultRightWidth?: number; // pixels
  minRightWidth?: number;
  maxRightWidth?: number;
}

export default function SplitLayout({
  left,
  right,
  defaultRightWidth = 360,
  minRightWidth = 280,
  maxRightWidth = 600,
}: SplitLayoutProps) {
  const [rightWidth, setRightWidth] = useState(defaultRightWidth);
  const dragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = true;

      const onMouseMove = (e: MouseEvent) => {
        if (!dragging.current || !containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const newRight = rect.right - e.clientX;
        setRightWidth(
          Math.max(minRightWidth, Math.min(maxRightWidth, newRight))
        );
      };

      const onMouseUp = () => {
        dragging.current = false;
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };

      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [minRightWidth, maxRightWidth]
  );

  return (
    <div ref={containerRef} className="flex h-full w-full overflow-hidden">
      {/* Left panel (canvas) */}
      <div className="flex-1 min-w-0 h-full overflow-hidden">
        {left}
      </div>

      {/* Drag handle */}
      <div
        onMouseDown={onMouseDown}
        className="w-[8px] flex-shrink-0 bg-slate-100 hover:bg-slate-300 active:bg-slate-400 cursor-col-resize flex items-center justify-center transition-colors border-x border-slate-200"
      >
        <div className="flex flex-col gap-[3px]">
          <div className="w-[3px] h-[3px] rounded-full bg-slate-400" />
          <div className="w-[3px] h-[3px] rounded-full bg-slate-400" />
          <div className="w-[3px] h-[3px] rounded-full bg-slate-400" />
        </div>
      </div>

      {/* Right panel (detail) */}
      <div
        className="flex-shrink-0 h-full overflow-hidden border-l border-slate-200"
        style={{ width: rightWidth }}
      >
        {right}
      </div>
    </div>
  );
}
