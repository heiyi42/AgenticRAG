import { Check, ChevronDown } from "lucide-react";
import type { RefObject } from "react";
import { useEffect, useRef, useState } from "react";
import type { SubjectId } from "../types";

const SUBJECT_OPTIONS: Array<{ id: SubjectId; label: string; description: string }> = [
  { id: "auto", label: "自动学科", description: "自动判断问题属于哪门课" },
  { id: "C_program", label: "C语言", description: "语法、指针、函数、结构体、程序题" },
  { id: "operating_systems", label: "操作系统", description: "进程、线程、内存、文件系统、中断" },
  { id: "cybersec_lab", label: "网络安全", description: "漏洞、攻击实验、协议安全、SEED Lab" }
];

export function SubjectDropdown({
  value,
  onChange
}: {
  value: SubjectId;
  onChange: (subject: SubjectId) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const current = SUBJECT_OPTIONS.find((item) => item.id === value) || SUBJECT_OPTIONS[0];

  useDropdownDismiss(rootRef, () => setOpen(false));

  return (
    <div className="picker" ref={rootRef}>
      <button className="picker-trigger" type="button" onClick={() => setOpen((state) => !state)}>
        <span>{current.label}</span>
        <ChevronDown size={16} />
      </button>
      {open ? (
        <div className="picker-menu subject-picker">
          {SUBJECT_OPTIONS.map((item) => (
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
