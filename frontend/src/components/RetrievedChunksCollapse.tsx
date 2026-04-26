import { ChevronDown } from "lucide-react";
import { useState } from "react";
import type { RetrievedChunk } from "../types";

export function RetrievedChunksCollapse({ chunks }: { chunks: RetrievedChunk[] }) {
  const [sectionOpen, setSectionOpen] = useState(false);
  const [openIds, setOpenIds] = useState<Set<string>>(new Set());

  function toggle(id: string) {
    setOpenIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className="trace-block">
      <button className="trace-section-toggle" type="button" onClick={() => setSectionOpen((value) => !value)}>
        <span>
          <span className="trace-heading">引用证据</span>
          <span className="trace-section-summary">{chunks.length ? `${chunks.length} 条证据，点击展开` : "暂无引用 chunk"}</span>
        </span>
        <ChevronDown size={16} className={sectionOpen ? "rotate-180 transition-transform" : "transition-transform"} />
      </button>
      {sectionOpen ? (
        chunks.length ? (
        <div className="chunk-list">
          {chunks.map((chunk) => {
            const id = chunk.id || chunk.chunkId;
            const open = openIds.has(id);
            return (
              <article className="chunk-card" key={id}>
                <button type="button" className="chunk-toggle" onClick={() => toggle(id)}>
                  <span>
                    <span className="font-mono text-[11px] text-stone-500">{chunk.chunkId || id}</span>
                    <span className="mt-1 line-clamp-2 text-sm text-stone-700">
                      {chunk.preview || chunk.content || "暂无摘要"}
                    </span>
                  </span>
                  <ChevronDown size={16} className={open ? "rotate-180 transition-transform" : "transition-transform"} />
                </button>
                {open ? (
                  <div className="chunk-content">
                    <div className="mb-2 text-xs text-stone-500">
                      {chunk.subjectId || "unknown"} {chunk.filePath ? `· ${chunk.filePath}` : ""}
                    </div>
                    <p>{chunk.content || chunk.preview}</p>
                  </div>
                ) : null}
              </article>
            );
          })}
        </div>
        ) : (
          <div className="empty-inline">暂无引用 chunk。</div>
        )
      ) : null}
    </div>
  );
}
