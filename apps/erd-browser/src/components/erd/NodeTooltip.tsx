import { useCallback, useRef, useState } from "react";
import { createPortal } from "react-dom";

interface NodeTooltipProps {
  text: string;
  children: React.ReactNode;
  className?: string;
}

export default function NodeTooltip({ text, children, className }: NodeTooltipProps) {
  const [visible, setVisible] = useState(false);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  const onEnter = useCallback((e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setPos({ x: rect.left + rect.width / 2, y: rect.top });
    timerRef.current = setTimeout(() => setVisible(true), 400);
  }, []);

  const onLeave = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setVisible(false);
  }, []);

  return (
    <div className={className} onMouseEnter={onEnter} onMouseLeave={onLeave}>
      {children}
      {visible && text && createPortal(
        <div
          style={{
            position: "fixed",
            left: pos.x,
            top: pos.y - 8,
            transform: "translate(-50%, -100%)",
            zIndex: 9999,
          }}
          className="bg-slate-800 text-slate-100 text-xs leading-relaxed px-3 py-2 rounded-md shadow-lg max-w-[300px] pointer-events-none"
        >
          {text}
        </div>,
        document.body
      )}
    </div>
  );
}
