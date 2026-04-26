import { create } from "zustand";
import type {
  AgentExecutionStep,
  ChatMessage,
  ChatSession,
  DeepSearchTrace,
  ExplainabilityDetails,
  AutoRouteTrace,
  GraphPayload,
  LocalSubgraph,
  ModeId,
  RetrievedChunk,
  SubjectId
} from "./types";

export const WORKFLOW_NODES: AgentExecutionStep[] = [
  { nodeId: "query_understanding", nodeName: "问题理解", status: "pending" },
  { nodeId: "retrieval_gate", nodeName: "检索判断", status: "pending" },
  { nodeId: "subject_route", nodeName: "学科路由", status: "pending" },
  { nodeId: "lightrag_retrieve", nodeName: "LightRAG 检索", status: "pending" },
  { nodeId: "deepsearch_plan", nodeName: "DeepSearch 规划", status: "pending" },
  { nodeId: "deepsearch_subject_route", nodeName: "子问题学科路由", status: "pending" },
  { nodeId: "deepsearch_retrieve", nodeName: "多路 LightRAG 检索", status: "pending" },
  { nodeId: "deepsearch_review", nodeName: "证据评审", status: "pending" },
  { nodeId: "deepsearch_retry", nodeName: "改写/补充检索", status: "pending" },
  { nodeId: "neo4j_subgraph", nodeName: "Neo4j 子图", status: "pending" },
  { nodeId: "answer_generate", nodeName: "答案生成", status: "pending" },
  { nodeId: "final_response", nodeName: "最终输出", status: "pending" }
];

const SUBJECT_LABELS: Record<string, string> = {
  auto: "自动学科",
  C_program: "C语言",
  operating_systems: "操作系统",
  cybersec_lab: "网络安全"
};

interface WorkbenchState {
  chats: ChatSession[];
  activeChatId: string | null;
  activeMessages: ChatMessage[];
  preferredMode: ModeId;
  preferredSubject: SubjectId;
  sending: boolean;
  setChats: (chats: ChatSession[]) => void;
  setActiveChat: (chat: ChatSession | null) => void;
  setPreferredMode: (mode: ModeId) => void;
  setPreferredSubject: (subject: SubjectId) => void;
  setSending: (sending: boolean) => void;
  appendUserMessage: (content: string) => void;
  startAssistantMessage: (mode: ModeId, subject: SubjectId) => void;
  appendAssistantDelta: (text: string) => void;
  finishAssistantMessage: (content: string, meta?: string, details?: Record<string, unknown>) => void;
  updateCurrentWorkflowStep: (step: AgentExecutionStep) => void;
  updateCurrentGraph: (graph: GraphPayload) => void;
  updateCurrentChunks: (chunks: RetrievedChunk[]) => void;
  updateCurrentMeta: (meta: Record<string, unknown>) => void;
  markCurrentAssistantError: (error: string) => void;
}

export const useWorkbenchStore = create<WorkbenchState>((set) => ({
  chats: [],
  activeChatId: null,
  activeMessages: [],
  preferredMode: "auto",
  preferredSubject: "auto",
  sending: false,
  setChats: (chats) => set({ chats }),
  setActiveChat: (chat) =>
    set({
      activeChatId: chat?.chat_id || null,
      activeMessages: chat?.messages || [],
      preferredMode: chat?.mode || "auto"
    }),
  setPreferredMode: (preferredMode) => set({ preferredMode }),
  setPreferredSubject: (preferredSubject) => set({ preferredSubject }),
  setSending: (sending) => set({ sending }),
  appendUserMessage: (content) =>
    set((state) => ({
      activeMessages: [...state.activeMessages, { role: "user", content }]
    })),
  startAssistantMessage: (mode, subject) =>
    set((state) => ({
      activeMessages: [
        ...state.activeMessages,
        {
          role: "assistant",
          content: "",
          details: {
            explainability: createExplainability(mode, subject, "streaming")
          }
        }
      ]
    })),
  appendAssistantDelta: (text) =>
    set((state) => updateLastAssistant(state, (message) => ({
      ...message,
      content: `${message.content}${text}`
    }))),
  finishAssistantMessage: (content, meta, details) =>
    set((state) =>
      updateLastAssistant(state, (message) => {
        const incomingExplainability =
          isRecord(details) && isRecord(details.explainability)
            ? normalizeExplainability(details.explainability)
            : null;
        const currentExplainability = getExplainability(message);
        const nextExplainability = {
          ...(currentExplainability || createExplainability("auto", "auto", "done")),
          ...(incomingExplainability || {}),
          status: "done" as const
        };
        return {
          ...message,
          content,
          meta,
          details: {
            ...(message.details || {}),
            ...(details || {}),
            explainability: nextExplainability
          }
        };
      })
    ),
  updateCurrentWorkflowStep: (step) =>
    set((state) =>
      updateLastAssistantExplainability(state, (explainability) => ({
        ...explainability,
        workflowSteps: mergeWorkflowStep(explainability.workflowSteps, step)
      }))
    ),
  updateCurrentGraph: (graph) =>
    set((state) =>
      updateLastAssistantExplainability(state, (explainability) => ({
        ...explainability,
        localSubgraphs: graph.ok ? graphPayloadToSubgraphs(graph) : [],
        chunks: graph.chunks || [],
        graphError: graph.ok ? "" : graph.error || "Neo4j 未连接"
      }))
    ),
  updateCurrentChunks: (chunks) =>
    set((state) =>
      updateLastAssistantExplainability(state, (explainability) => ({
        ...explainability,
        chunks
      }))
    ),
  updateCurrentMeta: (meta) =>
    set((state) =>
      updateLastAssistant(state, (message) => {
        const explainability =
          getExplainability(message) ||
          createExplainability(state.preferredMode, state.preferredSubject, "streaming");
        const subjectRoute = isRecord(meta.subject_route) ? meta.subject_route : {};
        const detectedSubject = String(subjectRoute.primary_subject || "") || explainability.detectedSubject;
        const requestKind =
          meta.request_kind === "code_analysis" || meta.request_kind === "problem_tutoring"
            ? meta.request_kind
            : undefined;
        return {
          ...message,
          details: {
            ...(message.details || {}),
            ...(requestKind ? { kind: requestKind } : {}),
            explainability: {
              ...explainability,
              mode: String(meta.requested_mode || explainability.mode || "auto"),
              modeUsed:
                typeof meta.mode_used === "string"
                  ? meta.mode_used
                  : explainability.modeUsed,
              detectedSubject,
              retrievalUsed:
                typeof meta.retrieval_used === "boolean"
                  ? meta.retrieval_used
                  : explainability.retrievalUsed,
              retrievalGateReason:
                typeof meta.retrieval_gate_reason === "string"
                  ? meta.retrieval_gate_reason
                  : explainability.retrievalGateReason
            }
          }
        };
      })
    ),
  markCurrentAssistantError: (error) =>
    set((state) =>
      updateLastAssistantExplainability(state, (explainability) => ({
        ...explainability,
        status: "error",
        graphError: explainability.graphError || error
      }))
    )
}));

function createExplainability(
  mode: ModeId | string,
  subject: SubjectId | string,
  status: ExplainabilityDetails["status"]
): ExplainabilityDetails {
  return {
    mode,
    subject,
    workflowSteps: WORKFLOW_NODES.map((step) => ({ ...step })),
    localSubgraphs: [],
    chunks: [],
    graphError: "",
    status,
    createdAt: new Date().toISOString()
  };
}

function updateLastAssistant(
  state: Pick<WorkbenchState, "activeMessages">,
  updater: (message: ChatMessage) => ChatMessage
) {
  const messages = [...state.activeMessages];
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "assistant") {
      messages[index] = updater(messages[index]);
      break;
    }
  }
  return { activeMessages: messages };
}

function updateLastAssistantExplainability(
  state: Pick<WorkbenchState, "activeMessages" | "preferredMode" | "preferredSubject">,
  updater: (explainability: ExplainabilityDetails) => ExplainabilityDetails
) {
  return updateLastAssistant(state, (message) => {
    const explainability =
      getExplainability(message) ||
      createExplainability(state.preferredMode, state.preferredSubject, "streaming");
    return {
      ...message,
      details: {
        ...(message.details || {}),
        explainability: updater(explainability)
      }
    };
  });
}

function getExplainability(message: ChatMessage): ExplainabilityDetails | null {
  const raw = message.details?.explainability;
  return isRecord(raw) ? normalizeExplainability(raw) : null;
}

function normalizeExplainability(raw: Record<string, unknown>): ExplainabilityDetails {
  return {
    mode: typeof raw.mode === "string" ? raw.mode : "auto",
    modeUsed: typeof raw.modeUsed === "string" ? raw.modeUsed : undefined,
    subject: typeof raw.subject === "string" ? raw.subject : "auto",
    detectedSubject: typeof raw.detectedSubject === "string" ? raw.detectedSubject : undefined,
    subjectRoute: isRecord(raw.subjectRoute)
      ? (raw.subjectRoute as Record<string, unknown>)
      : undefined,
    workflowSteps: Array.isArray(raw.workflowSteps)
      ? raw.workflowSteps.map((step) => ({ ...(step as AgentExecutionStep) }))
      : WORKFLOW_NODES.map((step) => ({ ...step })),
    localSubgraphs: Array.isArray(raw.localSubgraphs)
      ? (raw.localSubgraphs as LocalSubgraph[])
      : [],
    chunks: Array.isArray(raw.chunks) ? (raw.chunks as RetrievedChunk[]) : [],
    graphError: typeof raw.graphError === "string" ? raw.graphError : "",
    status:
      raw.status === "streaming" || raw.status === "done" || raw.status === "error"
        ? raw.status
        : "done",
    createdAt: typeof raw.createdAt === "string" ? raw.createdAt : undefined,
    retrievalUsed: typeof raw.retrievalUsed === "boolean" ? raw.retrievalUsed : undefined,
    retrievalGateConfidence:
      typeof raw.retrievalGateConfidence === "number" ||
      typeof raw.retrievalGateConfidence === "string"
        ? raw.retrievalGateConfidence
        : undefined,
    retrievalGateReason:
      typeof raw.retrievalGateReason === "string" ? raw.retrievalGateReason : undefined,
    autoRoute: isRecord(raw.autoRoute)
      ? (raw.autoRoute as unknown as AutoRouteTrace)
      : undefined,
    autoTimings: isRecord(raw.autoTimings)
      ? (raw.autoTimings as unknown as ExplainabilityDetails["autoTimings"])
      : undefined,
    autoUpgraded: typeof raw.autoUpgraded === "boolean" ? raw.autoUpgraded : undefined,
    autoUpgradeReason:
      typeof raw.autoUpgradeReason === "string" ? raw.autoUpgradeReason : undefined,
    instantReview: isRecord(raw.instantReview)
      ? (raw.instantReview as Record<string, unknown>)
      : undefined,
    deepsearchTrace: isRecord(raw.deepsearchTrace)
      ? (raw.deepsearchTrace as unknown as DeepSearchTrace)
      : undefined
  };
}

function mergeWorkflowStep(
  currentSteps: AgentExecutionStep[],
  nextStep: AgentExecutionStep
): AgentExecutionStep[] {
  const existing = currentSteps.length ? currentSteps : WORKFLOW_NODES;
  let found = false;
  const merged = existing.map((step) => {
    if (step.nodeId !== nextStep.nodeId) return step;
    found = true;
    return { ...step, ...nextStep };
  });
  return found ? merged : [...merged, nextStep];
}

function graphPayloadToSubgraphs(graph: GraphPayload): LocalSubgraph[] {
  if (!graph.nodes?.length) return [];
  const groups = new Map<string, typeof graph.nodes>();
  for (const node of graph.nodes) {
    const subjectId = node.subjectId || graph.subjectIds?.[0] || "unknown";
    groups.set(subjectId, [...(groups.get(subjectId) || []), node]);
  }

  return Array.from(groups.entries()).map(([subjectId, nodes]) => {
    const nodeIds = new Set(nodes.map((node) => node.id));
    const edges = (graph.edges || []).filter(
      (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)
    );
    const chunks = (graph.chunks || []).filter(
      (chunk) => !chunk.subjectId || chunk.subjectId === subjectId
    );
    const centerEntityIds = (graph.centerEntityIds || []).filter((id) => nodeIds.has(id));
    return {
      id: `subgraph-${subjectId}`,
      title: `${SUBJECT_LABELS[subjectId] || subjectId} 局部图谱`,
      subjectId,
      summary: `${nodes.length} 个实体，${chunks.length} 条证据`,
      nodes,
      edges,
      centerEntityIds,
      chunkIds: chunks.map((chunk) => chunk.id || chunk.chunkId)
    };
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
