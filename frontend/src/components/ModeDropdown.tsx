import { Check, ChevronDown } from "lucide-react";
import type { RefObject } from "react";
import { useEffect, useRef, useState } from "react";
import type { ModeId } from "../types";

const MODE_OPTIONS: Array<{ id: ModeId; label: string; description: string }> = [
  { id: "instant", label: "Instant", description: "适用于快速问答和短定义解释" },
  { id: "auto", label: "Auto", description: "自动判断检索策略和回答路径" },
  { id: "deepsearch", label: "DeepSearch", description: "适用于复杂问题、深度检索和多跳推理" }
];

export function ModeDropdown({ value, onChange }: { value: ModeId; onChange: (mode: ModeId) => void }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const current = MODE_OPTIONS.find((item) => item.id === value) || MODE_OPTIONS[1];

  useDropdownDismiss(rootRef, () => setOpen(false));

  return (
    <div className="picker" ref={rootRef}>
      <button className="picker-trigger" type="button" onClick={() => setOpen((state) => !state)}>
        <span>{current.label}</span>
        <ChevronDown size={16} />
      </button>
      {open ? (
        <div className="picker-menu">
          {MODE_OPTIONS.map((item) => (
            <button
              key={item.id}
              type="button"
              className="picker-option"
              onClick={() => {
                onChange(item.id);
                setOpen(false);
              }}
            >
              <span>
                <span className="picker-option-title">{item.label}</span>
                <span className="picker-option-desc">{item.description}</span>
              </span>
              {item.id === value ? <Check size={16} /> : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function useDropdownDismiss(ref: RefObject<HTMLElement>, onDismiss: () => void) {
  useEffect(() => {
    function onPointerDown(event: PointerEvent) {
      if (!ref.current?.contains(event.target as Node)) onDismiss();
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onDismiss();
    }

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [onDismiss, ref]);
}
