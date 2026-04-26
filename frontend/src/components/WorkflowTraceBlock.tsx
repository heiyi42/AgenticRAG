import { useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { AgentExecutionStep, AgentStatus, ExplainabilityDetails, ModeId } from "../types";

const STATUS_LABEL: Record<string, string> = {
  pending: "等待",
  running: "运行中",
  success: "完成",
  error: "错误",
  skipped: "跳过"
};

const MODE_LABEL: Record<ModeId, string> = {
  auto: "Auto",
  instant: "Instant",
  deepsearch: "DeepSearch"
};

interface FlowNode {
  id: string;
  label: string;
  detail: string;
  stepIds: string[];
  inferredFrom?: string[];
  size?: "compact" | "normal" | "wide" | "mid" | "semi" | "long";
}

interface FlowEdge {
  label?: string;
}

interface FlowStart {
  type: "start";
  label: string;
}

interface FlowSpacer {
  type: "spacer";
  width?: "start" | "edge" | "compact" | "normal" | "wide" | "llm";
}

interface FlowTemplate {
  title: string;
  summary: string;
  rows: Array<Array<FlowNode | FlowEdge | FlowStart | FlowSpacer>>;
}

const FLOW_TEMPLATES: Record<ModeId, FlowTemplate> = {
  instant: {
    title: "Instant 链路",
    summary: "先做检索判断和学科路由，再用低成本路径快速生成答案。",
    rows: [
      [
        start("用户问题"),
        edge(),
        node("retrieval_gate", "检索路由", "判断是否需要课程知识库。", ["retrieval_gate"], [], "compact"),
        edge(),
        node("subject_route", "学科路由", "确定本次问题归属的课程知识库。", ["subject_route"], [], "compact"),
        edge(),
        node("instant_answer", "Instant / LightRAG 回答", "使用快速检索或免检索路径生成初步答案。", ["lightrag_retrieve", "answer_generate"], [], "wide"),
        edge(),
        node("final_response", "LLM输出", "保存会话并输出最终回答。", ["final_response"], [], "compact")
      ]
    ]
  },
  auto: {
    title: "Auto 链路",
    summary: "先做检索和学科判断，再由 Auto 策略选择快答、补第二学科或 DeepSearch。",
    rows: [
      [
        start("用户输入"),
        edge(),
        node("retrieval_gate", "检索网关", "Auto 在回答前先判断是否需要课程知识库。", ["retrieval_gate"], [], "normal"),
        edge(),
        node("subject_route", "学科路由", "确定本次问题的主学科，跨学科时最多保留两个候选学科。", ["subject_route"], [], "normal"),
        edge(),
        node("auto_plan", "Auto策略规划", "根据学科、复杂度和置信度选择 Instant 优先或 DeepSearch。", [], ["lightrag_retrieve", "deepsearch_plan", "answer_generate"], "wide")
      ],
      [
        node("instant_trial", "Instant快答", "默认先用主学科 LightRAG 快速回答，降低成本和延迟。", ["lightrag_retrieve"], ["answer_generate"], "wide"),
        edge(),
        node("instant_review", "质量评审", "检查快速回答是否足够完整。", [], ["answer_generate"], "semi"),
        edge("不OK"),
        node("auto_second_subject", "补第二学科", "主学科回答不足且存在第二候选学科时，补查第二学科。", [], ["answer_generate"], "mid"),
        edge(),
        node("auto_merge_review", "合并评审", "合并主学科和第二学科答案后再次判断是否足够。", [], ["answer_generate"], "semi")
      ],
      [
        spacer("wide"),
        spacer("llm"),
        node("ok_output", "LLM输出", "快答或合并评审通过后输出最终回答。", ["final_response"], [], "semi"),
        spacer("wide"),
        spacer("edge"),
        node("deepsearch_fallback", "DeepSearch链路", "复杂问题或快答质量不足时，进入 DeepSearch 后续链路。", ["deepsearch_plan"], [], "long")
      ]
    ]
  },
  deepsearch: {
    title: "DeepSearch 链路",
    summary: "DeepSearch 先做检索网关，再拆解子问题；每个子问题单独学科路由并执行并行检索。",
    rows: [
      [
        start("用户问题"),
        edge(),
        node("retrieval_gate", "检索网关", "先判断是否需要进入课程知识库深度检索。", ["retrieval_gate"], [], "normal"),
        edge(),
        node("deepsearch_plan", "拆解子问题", "将复杂问题拆成多个可检索子问题。", ["deepsearch_plan"], [], "normal"),
        edge(),
        node("deepsearch_subject_route", "子问题学科路由", "对每个子问题分别选择目标学科。", ["deepsearch_subject_route"], ["deepsearch_plan"], "wide")
      ],
      [
        edge(),
        node("deepsearch_retrieve", "子问题并行检索", "按子问题和学科并行执行 LightRAG 检索。", ["deepsearch_retrieve"], ["deepsearch_plan"], "wide"),
        edge(),
        node("deepsearch_review", "证据评审", "判断每个子问题的检索证据是否足够。", ["deepsearch_review"], ["deepsearch_plan"], "compact"),
        edge("OK"),
        node("answer_generate", "综合生成答案", "汇总多路检索证据并生成最终答案。", ["answer_generate"], ["deepsearch_plan"], "wide"),
        edge(),
        node("final_response", "LLM输出", "保存会话并输出最终回答。", ["final_response"], [], "compact")
      ],
      [
        spacer("edge"),
        node("deepsearch_retry", "改写子问题", "对证据不足的子问题改写后补充检索。", ["deepsearch_retry"], [], "wide")
      ]
    ]
  }
};

function buildFlowTemplate(mode: ModeId, details?: ExplainabilityDetails): FlowTemplate {
  if (mode !== "deepsearch") return FLOW_TEMPLATES[mode];
  const locked = Boolean(details?.deepsearchTrace?.subjectLock?.enabled);
  const lockLabels = details?.deepsearchTrace?.subjectLock?.subjectLabels || [];
  const lockText = lockLabels.length ? lockLabels.join("、") : "当前学科";
  const routeLabel = locked ? "子问题学科锁定" : "子问题学科路由";
  const routeDetail = locked
    ? `用户已指定学科，所有子问题仅在 ${lockText} 知识库内检索。`
    : "对每个子问题分别选择目标学科，而不是在请求级提前固定学科。";
  const retrieveLabel = "子问题并行检索";
  const summary = locked
    ? "DeepSearch 先做检索网关，再拆解子问题；由于用户指定学科，后续子问题都锁定在当前知识库内检索。"
    : "DeepSearch 先做检索网关，再拆解子问题；每个子问题单独学科路由并执行并行检索。";

  return {
    ...FLOW_TEMPLATES.deepsearch,
    summary,
    rows: FLOW_TEMPLATES.deepsearch.rows.map((row) =>
      row.map((item) => {
        if (!("id" in item)) return item;
        if (item.id === "deepsearch_subject_route") {
          return { ...item, label: routeLabel, detail: routeDetail };
        }
        if (item.id === "deepsearch_retrieve") {
          return {
            ...item,
            label: retrieveLabel,
            detail: locked
              ? "按拆解后的子问题在用户指定知识库内检索。"
              : "按拆解后的子问题和逐题学科路由执行 LightRAG 检索。"
          };
        }
        return item;
      })
    )
  };
}

export function WorkflowTraceBlock({ details }: { details?: ExplainabilityDetails }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [expandedCanvas, setExpandedCanvas] = useState(false);
  const dragRef = useRef({ active: false, x: 0, y: 0 });
  const mode = normalizeMode(details?.mode);
  const modeUsed = normalizeOptionalMode(details?.modeUsed);
  const template = useMemo(() => buildFlowTemplate(mode, details), [mode, details]);
  const stepMap = useMemo(() => new Map((details?.workflowSteps || []).map((step) => [step.nodeId, step])), [details?.workflowSteps]);
  const selectedNode = template.rows
    .flat()
    .find((item): item is FlowNode => "id" in item && item.id === selectedId);
  const selectedSteps = selectedNode
    ? selectedNode.stepIds
        .map((stepId) => stepMap.get(stepId))
        .filter((step): step is AgentExecutionStep => Boolean(step))
    : [];

  function zoomBy(delta: number) {
    setScale((value) => Math.min(1.35, Math.max(0.75, Number((value + delta).toFixed(2)))));
  }

  function resetCanvas() {
    setScale(1);
    setPan({ x: 0, y: 0 });
  }

  function startPan(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    const target = event.target as HTMLElement;
    if (target.closest("button")) return;
    dragRef.current = { active: true, x: event.clientX, y: event.clientY };
    setIsPanning(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function movePan(event: ReactPointerEvent<HTMLDivElement>) {
    if (!dragRef.current.active) return;
    const nextX = event.clientX;
    const nextY = event.clientY;
    const deltaX = nextX - dragRef.current.x;
    const deltaY = nextY - dragRef.current.y;
    dragRef.current = { active: true, x: nextX, y: nextY };
    setPan((value) => ({ x: value.x + deltaX, y: value.y + deltaY }));
  }

  function stopPan(event: ReactPointerEvent<HTMLDivElement>) {
    if (!dragRef.current.active) return;
    dragRef.current.active = false;
    setIsPanning(false);
    event.currentTarget.releasePointerCapture(event.pointerId);
  }

  return (
    <div className="trace-block">
      <div className="flow-header">
        <div>
          <div className="trace-heading">链路节点路径</div>
          <div className="flow-title">
            {template.title}
            {mode === "auto" && modeUsed && modeUsed !== "auto" ? (
              <span className="flow-mode-used">实际执行：{MODE_LABEL[modeUsed]}</span>
            ) : null}
          </div>
          <div className="flow-summary">{template.summary}</div>
        </div>
      </div>
      <div className="flow-canvas-toolbar">
        <button type="button" onClick={() => setExpandedCanvas((value) => !value)}>
          {expandedCanvas ? "收起画布" : "展开画布"}
        </button>
        <button type="button" onClick={() => zoomBy(-0.1)}>缩小</button>
        <span>{Math.round(scale * 100)}%</span>
        <button type="button" onClick={() => zoomBy(0.1)}>放大</button>
        <button type="button" onClick={resetCanvas}>复位</button>
      </div>
      <div
        className={`flow-canvas ${mode} ${expandedCanvas ? "expanded" : ""} ${isPanning ? "panning" : ""}`}
        onPointerCancel={stopPan}
        onPointerDown={startPan}
        onPointerMove={movePan}
        onPointerUp={stopPan}
      >
        <div
          className={`flow-diagram ${mode}`}
          style={{
            width: `${100 / scale}%`,
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`
          }}
        >
          {mode === "auto" ? <AutoBranchOverlay /> : null}
          {mode === "deepsearch" ? <DeepSearchFeedbackOverlay /> : null}
          {template.rows.map((row, rowIndex) => (
            <div
              className={`flow-row ${mode === "deepsearch" && rowIndex === 2 ? "feedback-row" : ""}`}
              key={`${mode}-${rowIndex}`}
            >
              {row.map((item, index) =>
                "id" in item ? (
                  <FlowNodeButton
                    details={details}
                    isSelected={selectedId === item.id}
                    key={`${item.id}-${rowIndex}-${index}`}
                    node={item}
                    onClick={() => setSelectedId((value) => (value === item.id ? null : item.id))}
                    stepMap={stepMap}
                  />
                ) : "type" in item ? (
                  item.type === "start" ? (
                    <div className="flow-start" key={`start-${rowIndex}-${index}`}>{item.label}</div>
                  ) : (
                    <div className={`flow-spacer ${item.width || "normal"}`} key={`spacer-${rowIndex}-${index}`} />
                  )
                ) : (
                  <div className="flow-edge" key={`edge-${rowIndex}-${index}`}>
                    {item.label ? <span>{item.label}</span> : null}
                  </div>
                )
              )}
            </div>
          ))}
        </div>
      </div>
      {selectedNode ? (
        <div className="trace-detail">
          <div className="font-semibold text-stone-900">{selectedNode.label}</div>
          <p className="mt-1 text-sm leading-6 text-stone-600">{selectedNode.detail}</p>
          <AutoNodeDetail details={details} node={selectedNode} />
          <DeepSearchNodeDetail details={details} node={selectedNode} />
          {selectedSteps.length ? (
            <div className="mt-3 grid gap-2 text-sm text-stone-600">
              {selectedSteps.map((step) => (
                <div className="flow-step-detail" key={step.nodeId}>
                  <div className="font-semibold text-stone-800">{step.nodeName}</div>
                  {step.inputSummary ? <div>输入：{step.inputSummary}</div> : null}
                  {step.outputSummary ? <div>输出：{step.outputSummary}</div> : null}
                  {step.durationMs !== undefined ? <div>耗时：{step.durationMs} ms</div> : null}
                  {step.error ? <div className="text-rose-700">错误：{step.error}</div> : null}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function FlowNodeButton({
  details,
  isSelected,
  node,
  onClick,
  stepMap
}: {
  details?: ExplainabilityDetails;
  isSelected: boolean;
  node: FlowNode;
  onClick: () => void;
  stepMap: Map<string, AgentExecutionStep>;
}) {
  const status = resolveNodeStatus(node, stepMap, details);
  const duration = resolveNodeDuration(node, stepMap, details);
  return (
    <button
      className={`flow-node ${node.size || "normal"} ${status} ${isSelected ? "selected" : ""}`}
      type="button"
      onClick={onClick}
    >
      <span className="flow-node-title">{node.label}</span>
      <span className="flow-node-status">
        {STATUS_LABEL[status]}
        {duration !== null ? ` · ${formatDuration(duration)}` : ""}
      </span>
    </button>
  );
}

function AutoBranchOverlay() {
  return (
    <svg className="flow-auto-overlay" viewBox="0 0 760 360" aria-hidden="true">
      <path
        className="flow-branch-path"
        d="M490 64 L130 115"
      />
      <polyline className="flow-branch-head" points="135,108 130,115 140,118" />
      <text className="flow-branch-label" x="305" y="78">简单</text>

      <path
        className="flow-branch-path"
        d="M560 40 C674 108 666 202 640 230"
      />
      <polyline className="flow-branch-head" points="642,221 640,230 647,228" />
      <text className="flow-branch-label" x="636" y="140">复杂</text>

      <path
        className="flow-branch-path"
        d="M230 180 L230 220"
      />
      <polyline className="flow-branch-head" points="222,210 230,220 238,210" />
      <text className="flow-branch-label" x="240" y="200">OK</text>

      <path
        className="flow-branch-path"
        d="M535 180 L283 222"
      />
      <polyline className="flow-branch-head" points="290,216 283,222 292,224" />
      <text className="flow-branch-label" x="402" y="220">OK</text>

      <path
        className="flow-branch-path"
        d="M535 180 L535 217"
      />
      <polyline className="flow-branch-head" points="528,207 535,217 542,207" />
      <text className="flow-branch-label" x="538" y="197">不OK</text>

      <path
        className="flow-branch-path"
        d="M452 254 L290 254"
      />
      <polyline className="flow-branch-head" points="297,248 290,254 297,260" />
    </svg>
  );
}

function DeepSearchFeedbackOverlay() {
  return (
    <svg className="flow-feedback-overlay" viewBox="0 0 760 250" aria-hidden="true">
      <path
        className="flow-feedback-path"
        d="M284 146 C262 154 239 164 218 174"
      />
      <polyline className="flow-feedback-head" points="214,166 202,181 220,182" />
      <path
        className="flow-feedback-path"
        d="M121 180 V142"
      />
      <polyline className="flow-feedback-head" points="115,151 121,142 127,151" />
      <text className="flow-feedback-label" x="238" y="164">不OK</text>
      <text className="flow-feedback-label muted" x="132" y="162">回到并行检索</text>
    </svg>
  );
}

interface AutoDetailRow {
  label: string;
  value: string;
}

interface AutoDetailPayload {
  title: string;
  rows: AutoDetailRow[];
  pills?: string[];
}

function AutoNodeDetail({
  details,
  node
}: {
  details?: ExplainabilityDetails;
  node: FlowNode;
}) {
  if (normalizeMode(details?.mode) !== "auto") return null;
  const payload = buildAutoNodeDetail(details, node);
  if (!payload) return null;

  return (
    <div className="deeptrace-list">
      <div className="deeptrace-item">
        <div className="deeptrace-item-title">{payload.title}</div>
        {payload.rows.map((row) => (
          <div className="auto-detail-row" key={row.label}>
            <span>{row.label}：</span>
            {row.value}
          </div>
        ))}
        {payload.pills?.length ? (
          <div className="deeptrace-pills">
            {payload.pills.map((pill) => (
              <span className="deeptrace-pill" key={pill}>
                {pill}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function buildAutoNodeDetail(
  details: ExplainabilityDetails | undefined,
  node: FlowNode
): AutoDetailPayload | null {
  const route = details?.subjectRoute || {};
  const autoRoute = details?.autoRoute || {};
  const review = details?.instantReview || {};
  const routeSubjects = subjectIdsFrom(autoRoute.subjects);
  const rankedSubjects = rankedSubjectPills(route);
  const requestedSubjects = subjectIdsFrom(route.requested_subjects);
  const primarySubject = stringValue(route.primary_subject) || details?.detectedSubject;
  const selectedSubjects = routeSubjects.length
    ? routeSubjects
    : requestedSubjects.length
      ? requestedSubjects
      : primarySubject
        ? [String(primarySubject)]
        : [];
  const modeUsed = normalizeOptionalMode(details?.modeUsed);

  switch (node.id) {
    case "retrieval_gate":
      return {
        title: "检索网关判定",
        rows: compactRows([
          ["判定结果", details?.retrievalUsed === false ? "免检索直答" : "需要检索课程知识库"],
          ["置信度", formatConfidence(details?.retrievalGateConfidence)],
          ["判定原因", details?.retrievalGateReason || "未记录"],
          ["后续动作", details?.retrievalUsed === false ? "跳过学科路由，直接生成回答" : "进入学科路由"]
        ])
      };
    case "subject_route":
      return {
        title: "学科路由结果",
        rows: compactRows([
          ["请求学科", requestedSubjects.length ? subjectListLabel(requestedSubjects) : subjectLabel(String(details?.subject || "auto"))],
          ["主学科", primarySubject ? subjectLabel(String(primarySubject)) : "未记录"],
          ["是否跨学科", booleanValue(route.cross_subject) ? "是，最多保留两个候选学科" : "否"],
          ["路由置信度", formatConfidence(route.confidence)],
          ["路由原因", stringValue(route.reason) || "未记录"]
        ]),
        pills: rankedSubjects
      };
    case "auto_plan":
      return {
        title: "Auto 策略规划",
        rows: compactRows([
          ["复杂度", stringValue(autoRoute.complexity) || "未记录"],
          ["策略置信度", formatConfidence(autoRoute.confidence)],
          ["选择链路", autoChainLabel(stringValue(autoRoute.chain), modeUsed)],
          ["策略规则", stringValue(autoRoute.policy) || "未记录"],
          ["规划原因", stringValue(autoRoute.reason) || "未记录"],
          ["节点耗时", formatOptionalDuration(details?.autoTimings?.autoPlanMs)]
        ]),
        pills: selectedSubjects.map(subjectLabel)
      };
    case "instant_trial":
      return {
        title: "Instant 快答执行",
        rows: compactRows([
          ["检索学科", subjectListLabel(selectedSubjects.slice(0, 1)) || "未记录"],
          ["执行路径", details?.retrievalUsed === false ? "免检索直答" : "主学科 LightRAG 快速回答"],
          ["输出去向", "候选答案进入质量评审"],
          ["节点耗时", formatOptionalDuration(details?.autoTimings?.instantTrialMs)]
        ])
      };
    case "instant_review":
      return {
        title: "质量评审",
        rows: compactRows([
          ["启发式检查", stringValue(review.heuristic) || "未记录"],
          ["LLM/规则评审", stringValue(review.review) || "未记录"],
          ["评审结论", autoReviewDecision(details)],
          ["升级原因", details?.autoUpgradeReason || "无"],
          ["节点耗时", formatOptionalDuration(details?.autoTimings?.instantReviewMs)]
        ])
      };
    case "auto_second_subject":
      return {
        title: "补第二学科",
        rows: compactRows([
          ["触发条件", "主学科快答不足，且存在第二候选学科"],
          ["主学科", selectedSubjects[0] ? subjectLabel(selectedSubjects[0]) : "未记录"],
          ["第二学科", selectedSubjects[1] ? subjectLabel(selectedSubjects[1]) : "未触发或未记录"],
          ["当前状态", details?.autoTimings?.autoSecondSubjectMs !== undefined ? "已补查第二学科" : "本次未触发"],
          ["节点耗时", formatOptionalDuration(details?.autoTimings?.autoSecondSubjectMs)]
        ])
      };
    case "auto_merge_review":
      return {
        title: "合并评审",
        rows: compactRows([
          ["输入", selectedSubjects.length > 1 ? `${subjectListLabel(selectedSubjects.slice(0, 2))} 的候选答案` : "未触发多学科合并"],
          ["评审结果", stringValue(review.review) || "未记录"],
          ["后续动作", modeUsed === "deepsearch" ? "合并仍不足，进入 DeepSearch" : "通过，进入 LLM 输出"],
          ["节点耗时", formatOptionalDuration(details?.autoTimings?.autoMergeReviewMs)]
        ])
      };
    case "deepsearch_fallback":
      return {
        title: "DeepSearch 后续链路",
        rows: compactRows([
          ["触发状态", modeUsed === "deepsearch" ? "已进入 DeepSearch" : "跳过"],
          ["触发原因", details?.autoUpgradeReason || stringValue(autoRoute.reason) || "未触发"],
          ["检索范围", subjectListLabel(selectedSubjects) || "未记录"],
          ["节点耗时", formatOptionalDuration(details?.autoTimings?.deepsearchFallbackMs)]
        ])
      };
      case "ok_output":
      return {
        title: "LLM 输出",
        rows: compactRows([
          ["输出来源", modeUsed === "deepsearch" ? "DeepSearch 链路" : "Instant 快答或合并评审"],
          ["最终执行", modeUsed ? MODE_LABEL[modeUsed] : "未记录"],
          ["输出动作", "保存会话并完成流式响应"],
          ["节点耗时", formatOptionalDuration(details?.workflowSteps?.find((step) => step.nodeId === "final_response")?.durationMs)]
        ])
      };
  
    default:
      return null;
  }
}

function DeepSearchNodeDetail({
  details,
  node
}: {
  details?: ExplainabilityDetails;
  node: FlowNode;
}) {
  const mode = normalizeMode(details?.mode);
  const modeUsed = normalizeOptionalMode(details?.modeUsed);
  const trace = details?.deepsearchTrace;
  const isDeepSearch = mode === "deepsearch" || modeUsed === "deepsearch";
  if (!isDeepSearch || !trace) return null;

  if (node.id === "deepsearch_plan") {
    const subQuestions = trace.subQuestions || [];
    return (
      <div className="deeptrace-list">
        {subQuestions.length ? (
          subQuestions.map((item, index) => (
            <div className="deeptrace-item" key={item.id || index}>
              <div className="deeptrace-item-title">
                {item.id || `q${index + 1}`} · {item.question || "未记录子问题"}
              </div>
              <div>使用问题：{item.usedQuestion || item.question || "未记录"}</div>
              <div>
                检索参数：{item.queryMode || "hybrid"} · top_k：
                {item.topK ?? "-"} · chunk_top_k：{item.chunkTopK ?? "-"}
              </div>
            </div>
          ))
        ) : (
          <div className="deeptrace-empty">本次回答没有保存子问题拆解明细。</div>
        )}
      </div>
    );
  }

  if (node.id === "deepsearch_subject_route") {
    const routes = trace.subQuestionRoutes || [];
    const locked = Boolean(trace.subjectLock?.enabled);
    return (
      <div className="deeptrace-list">
        {locked ? (
          <div className="deeptrace-item">
            <div className="deeptrace-item-title">学科锁定</div>
            <div>{trace.subjectLock?.reason || "所有子问题仅在当前知识库内检索。"}</div>
            <div className="deeptrace-pills">
              {(trace.subjectLock?.subjectIds || []).map((subjectId) => (
                <span className="deeptrace-pill" key={subjectId}>
                  {subjectLabel(subjectId)}
                </span>
              ))}
            </div>
          </div>
        ) : null}
        {routes.length ? (
          routes.map((route, index) => {
            const subQuestion = findSubQuestion(trace, route.subQuestionId);
            return (
              <div className="deeptrace-item" key={`${route.subQuestionId}-${index}`}>
                <div className="deeptrace-item-title">
                  {route.subQuestionId || `q${index + 1}`} ·{" "}
                  {subQuestion?.question || "未记录子问题"}
                </div>
                <div>
                  主学科：
                  {route.primarySubjectLabel ||
                    subjectLabel(route.primarySubject) ||
                    "未记录"}
                </div>
                {route.rankedSubjects?.length ? (
                  <div className="deeptrace-pills">
                    {route.rankedSubjects.map((subject) => (
                      <span className="deeptrace-pill" key={subject.subject}>
                        {subject.label || subjectLabel(subject.subject)}
                        {typeof subject.score === "number"
                          ? ` ${Math.round(subject.score * 100)}%`
                          : ""}
                      </span>
                    ))}
                  </div>
                ) : null}
                {route.reason ? <div>路由原因：{route.reason}</div> : null}
              </div>
            );
          })
        ) : (
          <div className="deeptrace-empty">本次回答没有保存子问题学科路由明细。</div>
        )}
      </div>
    );
  }

  if (node.id === "deepsearch_review") {
    const review = trace.review || [];
    return (
      <div className="deeptrace-list">
        {review.length ? (
          review.map((item, index) => {
            const subQuestion = findSubQuestion(trace, item.subQuestionId);
            return (
              <div className="deeptrace-item" key={`${item.subQuestionId}-${index}`}>
                <div className="deeptrace-item-title">
                  {item.subQuestionId || `q${index + 1}`} ·{" "}
                  {subQuestion?.question || "未记录子问题"}
                </div>
                <div>评审结果：{sufficientLabel(item.sufficient)}</div>
                {item.judgeReason ? <div>评审原因：{item.judgeReason}</div> : null}
                {item.rewrittenQuestion ? (
                  <div>不足改写：{item.rewrittenQuestion}</div>
                ) : null}
              </div>
            );
          })
        ) : (
          <div className="deeptrace-empty">本次回答没有保存证据评审明细。</div>
        )}
      </div>
    );
  }

  if (node.id === "deepsearch_retry") {
    const retry = trace.retry;
    const ids = retry?.insufficientSubquestionIds || [];
    return (
      <div className="deeptrace-list">
        <div className="deeptrace-item">
          <div className="deeptrace-item-title">改写/补充检索</div>
          <div>重试轮次：{retry?.queryAttempt ?? 0}</div>
          <div>
            不足子问题：
            {ids.length ? ids.join("、") : "无，证据评审未触发补充检索"}
          </div>
        </div>
      </div>
    );
  }

  return null;
}

function node(
  id: string,
  label: string,
  detail: string,
  stepIds: string[],
  inferredFrom: string[] = [],
  size: FlowNode["size"] = "normal"
): FlowNode {
  return { id, label, detail, stepIds, inferredFrom, size };
}

function edge(label?: string): FlowEdge {
  return { label };
}

function start(label: string): FlowStart {
  return { type: "start", label };
}

function spacer(width: FlowSpacer["width"] = "normal"): FlowSpacer {
  return { type: "spacer", width };
}

const SUBJECT_LABELS: Record<string, string> = {
  C_program: "C语言",
  operating_systems: "操作系统",
  cybersec_lab: "网络安全",
  auto: "自动学科"
};

function subjectLabel(subjectId?: string): string {
  if (!subjectId) return "";
  return SUBJECT_LABELS[subjectId] || subjectId;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function booleanValue(value: unknown): boolean {
  return value === true || value === "true";
}

function subjectIdsFrom(value: unknown): string[] {
  return Array.isArray(value)
    ? value
        .map((item) => String(item || "").trim())
        .filter(Boolean)
    : [];
}

function subjectListLabel(subjectIds: string[]): string {
  return subjectIds.map(subjectLabel).filter(Boolean).join("、");
}

function rankedSubjectPills(route: Record<string, unknown>): string[] {
  const ranked = route.ranked;
  if (!Array.isArray(ranked)) return [];
  return ranked
    .map((item) => {
      if (!item || typeof item !== "object") return "";
      const subject = String((item as Record<string, unknown>).subject || "").trim();
      if (!subject) return "";
      const score = (item as Record<string, unknown>).score;
      return `${subjectLabel(subject)}${formatConfidenceSuffix(score)}`;
    })
    .filter(Boolean);
}

function compactRows(rows: Array<[string, string | undefined]>): AutoDetailRow[] {
  return rows.map(([label, value]) => ({ label, value: value || "未记录" }));
}

function formatConfidence(value: unknown): string {
  if (value === null || value === undefined || value === "") return "未记录";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  return `${Math.round(numeric * 100)}%`;
}

function formatConfidenceSuffix(value: unknown): string {
  const confidence = formatConfidence(value);
  return confidence === "未记录" ? "" : ` ${confidence}`;
}

function formatOptionalDuration(durationMs: number | undefined): string {
  return typeof durationMs === "number" && durationMs >= 0 ? formatDuration(durationMs) : "未记录";
}

function autoChainLabel(chain: string, modeUsed: ModeId | null): string {
  if (chain === "deep") return "复杂问题，直接进入 DeepSearch";
  if (chain === "instant") return "先走 Instant 快答";
  if (chain === "direct") return "免检索直答";
  return modeUsed ? `实际执行 ${MODE_LABEL[modeUsed]}` : "未记录";
}

function autoReviewDecision(details?: ExplainabilityDetails): string {
  const modeUsed = normalizeOptionalMode(details?.modeUsed);
  if (modeUsed === "deepsearch") return "不足，进入 DeepSearch";
  if (details?.autoUpgraded) return "不足，触发升级或补充检索";
  return "通过，进入 LLM 输出";
}

function findSubQuestion(details: NonNullable<ExplainabilityDetails["deepsearchTrace"]>, id?: string) {
  return (details.subQuestions || []).find((item) => item.id === id);
}

function sufficientLabel(value: boolean | null | undefined): string {
  if (value === true) return "充分";
  if (value === false) return "不足";
  return "未评审";
}

function normalizeMode(raw: unknown): ModeId {
  return raw === "instant" || raw === "deepsearch" || raw === "auto" ? raw : "auto";
}

function normalizeOptionalMode(raw: unknown): ModeId | null {
  return raw === "instant" || raw === "deepsearch" || raw === "auto" ? raw : null;
}

function resolveNodeStatus(
  node: FlowNode,
  stepMap: Map<string, AgentExecutionStep>,
  details?: ExplainabilityDetails
): AgentStatus {
  const mode = normalizeMode(details?.mode);
  const modeUsed = normalizeOptionalMode(details?.modeUsed);
  const finalStatus = stepMap.get("final_response")?.status;
  const completed = finalStatus === "success" || details?.status === "done";
  if (node.id === "deepsearch_fallback") {
    if (modeUsed === "instant") return "skipped";
    if (modeUsed === "deepsearch") return completed ? "success" : "running";
  }
  if (node.id === "instant_review_upgrade") {
    if (modeUsed === "instant") return "skipped";
    if (modeUsed === "deepsearch") return completed ? "success" : "running";
  }
  if (node.id === "ok_output") {
    if (modeUsed === "deepsearch") return "skipped";
    if (modeUsed === "instant") return completed ? "success" : "running";
  }
  if (mode === "auto" && completed) {
    const autoStatus = resolveAutoNodeStatus(node, details);
    if (autoStatus) return autoStatus;
  }
  if (mode === "auto" && !completed) {
    const autoStatus = resolveStreamingAutoNodeStatus(node, stepMap, details);
    if (autoStatus) return autoStatus;
  }
  if (mode === "deepsearch" && completed) {
    const deepsearchRetrievalNodes = new Set([
      "deepsearch_plan",
      "deepsearch_subject_route",
      "deepsearch_retrieve",
      "deepsearch_review",
      "deepsearch_retry"
    ]);
    if (details?.retrievalUsed === false && deepsearchRetrievalNodes.has(node.id)) {
      return "skipped";
    }
    if (node.id === "deepsearch_retry" && shouldSkipDeepSearchRetry(stepMap, details)) {
      return "skipped";
    }
  }

  const directSteps = node.stepIds.map((stepId) => stepMap.get(stepId)).filter(Boolean) as AgentExecutionStep[];
  if (directSteps.length) return aggregateStatus(directSteps);

  const inferredSteps = (node.inferredFrom || []).map((stepId) => stepMap.get(stepId)).filter(Boolean) as AgentExecutionStep[];
  if (!inferredSteps.length) return "pending";
  const inferred = aggregateStatus(inferredSteps);
  if (node.id === "deepsearch_fallback" && !stepMap.has("deepsearch_plan")) return "skipped";
  if (node.id === "ok_output" && stepMap.has("deepsearch_plan")) return "skipped";
  if (node.id === "subquestion_route" && inferred === "success") return "success";
  return inferred === "error" ? "error" : inferred === "pending" ? "pending" : inferred;
}

function resolveStreamingAutoNodeStatus(
  node: FlowNode,
  stepMap: Map<string, AgentExecutionStep>,
  details?: ExplainabilityDetails
): AgentStatus | null {
  const postRetrievalNodes = new Set([
    "subject_route",
    "auto_plan",
    "instant_trial",
    "instant_review",
    "auto_second_subject",
    "auto_merge_review",
    "deepsearch_fallback"
  ]);
  if (details?.retrievalUsed === false && postRetrievalNodes.has(node.id)) {
    return "skipped";
  }
  if (node.id !== "auto_plan") return null;

  const downstreamSteps = ["lightrag_retrieve", "deepsearch_plan", "answer_generate"]
    .map((stepId) => stepMap.get(stepId))
    .filter((step): step is AgentExecutionStep => Boolean(step));
  if (downstreamSteps.length) {
    return downstreamSteps.some((step) => step.status === "error") ? "error" : "success";
  }

  const subjectRouteStatus = stepMap.get("subject_route")?.status;
  if (subjectRouteStatus === "success") return "running";
  if (subjectRouteStatus === "error") return "error";
  return "pending";
}

function resolveAutoNodeStatus(
  node: FlowNode,
  details?: ExplainabilityDetails
): AgentStatus | null {
  const postRetrievalNodes = new Set([
    "subject_route",
    "auto_plan",
    "instant_trial",
    "instant_review",
    "auto_second_subject",
    "auto_merge_review",
    "deepsearch_fallback"
  ]);
  if (details?.retrievalUsed === false && postRetrievalNodes.has(node.id)) {
    return "skipped";
  }

  const modeUsed = normalizeOptionalMode(details?.modeUsed);
  const routeChain = String(details?.autoRoute?.chain || "");
  const routePolicy = String(details?.autoRoute?.policy || "");
  const usedSecondary =
    routePolicy.includes("secondary") ||
    routePolicy.includes("dual") ||
    routeChain.includes("dual-subject");
  const plannedDeep = routeChain === "deep";
  const usedDeepSearch = modeUsed === "deepsearch";

  if (node.id === "auto_plan") return "success";
  if (node.id === "instant_trial") {
    return plannedDeep && usedDeepSearch ? "skipped" : "success";
  }
  if (node.id === "instant_review") {
    return plannedDeep && usedDeepSearch ? "skipped" : "success";
  }
  if (node.id === "auto_second_subject" || node.id === "auto_merge_review") {
    return usedSecondary ? "success" : "skipped";
  }
  if (node.id === "deepsearch_fallback") {
    return usedDeepSearch ? "success" : "skipped";
  }
  return null;
}

function shouldSkipDeepSearchRetry(
  stepMap: Map<string, AgentExecutionStep>,
  details?: ExplainabilityDetails
): boolean {
  const retryStep = stepMap.get("deepsearch_retry");
  if (retryStep && retryStep.status !== "pending") return retryStep.status === "skipped";
  const retry = details?.deepsearchTrace?.retry;
  if (retry) {
    const insufficient = retry.insufficientSubquestionIds || [];
    return retry.needsRetry === false || insufficient.length === 0;
  }
  return stepMap.get("deepsearch_review")?.status === "success";
}

function resolveNodeDuration(
  node: FlowNode,
  stepMap: Map<string, AgentExecutionStep>,
  details?: ExplainabilityDetails
): number | null {
  const autoDuration = resolveAutoNodeDuration(node, details);
  if (autoDuration !== null) return autoDuration;

  const steps = node.stepIds
    .map((stepId) => stepMap.get(stepId))
    .filter((step): step is AgentExecutionStep => Boolean(step));
  const durations = steps
    .map((step) => step.durationMs)
    .filter((duration): duration is number => typeof duration === "number" && duration >= 0);
  if (!durations.length) return null;
  return durations.reduce((sum, duration) => sum + duration, 0);
}

function resolveAutoNodeDuration(node: FlowNode, details?: ExplainabilityDetails): number | null {
  if (normalizeMode(details?.mode) !== "auto") return null;
  const timings = details?.autoTimings;
  if (!timings) return null;
  const byNode: Record<string, number | undefined> = {
    auto_plan: timings.autoPlanMs,
    instant_trial: timings.instantTrialMs,
    instant_review: timings.instantReviewMs,
    auto_second_subject: timings.autoSecondSubjectMs,
    auto_merge_review: timings.autoMergeReviewMs,
    deepsearch_fallback: timings.deepsearchFallbackMs
  };
  const duration = byNode[node.id];
  return typeof duration === "number" && duration >= 0 ? duration : null;
}

function formatDuration(durationMs: number): string {
  if (durationMs === 0) return "<1 ms";
  if (durationMs < 1000) return `${durationMs} ms`;
  return `${(durationMs / 1000).toFixed(durationMs < 10_000 ? 1 : 0)} s`;
}

function aggregateStatus(steps: AgentExecutionStep[]): AgentStatus {
  if (steps.some((step) => step.status === "error")) return "error";
  if (steps.some((step) => step.status === "running")) return "running";
  if (steps.every((step) => step.status === "success")) return "success";
  if (steps.some((step) => step.status === "success")) return "success";
  if (steps.some((step) => step.status === "skipped")) return "skipped";
  return "pending";
}
