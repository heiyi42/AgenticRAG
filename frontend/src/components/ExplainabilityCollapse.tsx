import { Check, ChevronDown } from "lucide-react";
import { useState } from "react";
import type { ChatMessage } from "../types";
import { LocalKnowledgeGraphBlock } from "./LocalKnowledgeGraphBlock";
import { RetrievedChunksCollapse } from "./RetrievedChunksCollapse";
import { WorkflowTraceBlock } from "./WorkflowTraceBlock";

interface ExplainabilityCollapseProps {
  message: ChatMessage;
}

export function ExplainabilityCollapse({ message }: ExplainabilityCollapseProps) {
  const [open, setOpen] = useState(false);
  const specialStatus = getSpecialModuleStatus(message);
  const details = message.details?.explainability;

  if (specialStatus) {
    return (
      <section className={`special-explainability-status ${specialStatus.kind} ${specialStatus.done ? "done" : ""}`}>
        {specialStatus.done ? (
          <span className="special-status-check" aria-hidden="true">
            <Check size={14} strokeWidth={3} />
          </span>
        ) : (
          <span className="special-status-dot" aria-hidden="true" />
        )}
        <span>{specialStatus.label}</span>
      </section>
    );
  }

  const stepCount = details?.workflowSteps?.length || 0;
  const graphCount = details?.localSubgraphs?.reduce((sum, graph) => sum + graph.nodes.length, 0) || 0;
  const chunkCount = details?.chunks?.length || 0;
  const hasRuntimeData = stepCount > 0 || graphCount > 0 || chunkCount > 0 || details?.graphError;

  return (
    <section className="explainability-box">
      <button className="explainability-toggle" type="button" onClick={() => setOpen((value) => !value)}>
        <span>
          <span className="explainability-title">检索链路 · 知识图谱 · 引用证据</span>
          <span className="explainability-summary">
            {hasRuntimeData
              ? `${stepCount} 个节点 · ${graphCount} 个实体 · ${chunkCount} 条证据`
              : "本次回答暂无可解释信息"}
          </span>
        </span>
        <ChevronDown size={18} className={open ? "rotate-180 transition-transform" : "transition-transform"} />
      </button>
      {open ? (
        <div className="explainability-content">
          <WorkflowTraceBlock details={details} />
          <LocalKnowledgeGraphBlock
            subgraphs={details?.localSubgraphs || []}
            graphError={details?.graphError || ""}
          />
          <RetrievedChunksCollapse chunks={details?.chunks || []} />
        </div>
      ) : null}
    </section>
  );
}

function getSpecialModuleStatus(
  message: ChatMessage
): { kind: "code-analysis" | "problem-tutoring"; label: string; done: boolean } | null {
  const kind = stringFromUnknown(message.details?.kind);
  const route = recordFromUnknown(message.details?.route);
  const routeChain = stringFromUnknown(route?.chain);
  const meta = message.meta || "";
  const status = message.details?.explainability?.status;
  const isDone = status === "done";

  if (kind === "code_analysis" || routeChain === "code_analysis" || meta.includes("代码分析")) {
    return { kind: "code-analysis", label: isDone ? "分析完成" : "正在分析代码", done: isDone };
  }

  if (
    kind === "problem_tutoring" ||
    routeChain === "problem_tutoring" ||
    meta.includes("题目辅导")
  ) {
    return { kind: "problem-tutoring", label: isDone ? "解题完成" : "正在分步解题", done: isDone };
  }

  return null;
}

function recordFromUnknown(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function stringFromUnknown(value: unknown): string {
  return typeof value === "string" ? value : "";
}
