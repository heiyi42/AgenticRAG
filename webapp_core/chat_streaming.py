from __future__ import annotations

import asyncio
import json
import time
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any, Callable

from . import auto_runtime as auto

from . import config as cfg
from .session_store import ChatSession, safe_int


class ChatStreamingMixin:
    def _match_problem_tutoring_request(
        self,
        text: str,
        *,
        requested_subjects: list[str] | None = None,
        requested_by_user: bool = False,
    ) -> dict[str, Any] | None:
        if not cfg.WEB_ENABLE_PROBLEM_TUTORING or not requested_by_user:
            return None
        return self.problem_tutoring_service.match_request(
            text,
            requested_subjects=requested_subjects,
            requested_by_user=True,
        )

    async def _agenerate_chat_title_from_first_question(self, question: str) -> str:
        q = str(question or "").strip()
        if not q:
            return "新聊天"

        safe_max_len = max(1, cfg.WEB_CHAT_TITLE_MAX_LEN)
        prompt = (
            "你是聊天标题生成器。请根据用户首个问题生成一个会话标题。\n"
            f"要求：\n- 仅输出标题\n- 不超过{safe_max_len}个字\n"
            "- 不带引号和标点\n\n"
            f"问题：{q}"
        )
        try:
            timeout_s = max(1, int(cfg.WEB_CHAT_TITLE_TIMEOUT_S))
            message = await asyncio.wait_for(
                auto.auto_router_llm.ainvoke(prompt),
                timeout=timeout_s,
            )
            raw = self.store.content_to_text(getattr(message, "content", message))
            title = self.store.normalize_chat_title(raw, max_len=safe_max_len)
            if title:
                return title
        except Exception as e:
            print(f"[WARN] 生成聊天标题失败: {e}")
        return self.store.fallback_chat_title(q, max_len=safe_max_len)

    def _apply_fallback_chat_title(
        self,
        session: ChatSession,
        question: str,
    ) -> str:
        safe_max_len = max(1, int(cfg.WEB_CHAT_TITLE_MAX_LEN))
        fallback_title = self.store.fallback_chat_title(question, max_len=safe_max_len)
        with session.lock:
            if self.store.is_placeholder_title(session.title) and fallback_title:
                session.title = fallback_title
                session.updated_at = time.time()
            return session.title

    def _schedule_chat_title_refinement(
        self,
        session: ChatSession,
        question: str,
    ) -> None:
        if self.submit_async is None:
            return

        safe_max_len = max(1, int(cfg.WEB_CHAT_TITLE_MAX_LEN))
        fallback_title = self.store.fallback_chat_title(question, max_len=safe_max_len)

        async def refine_title() -> None:
            generated_title = await self._agenerate_chat_title_from_first_question(question)
            if not generated_title:
                return

            should_persist = False
            with session.lock:
                current_title = str(session.title or "").strip()
                if current_title not in {fallback_title, generated_title} and not self.store.is_placeholder_title(current_title):
                    return
                if current_title != generated_title:
                    session.title = generated_title
                    session.updated_at = time.time()
                    updated_at = session.updated_at
                    should_persist = True
                else:
                    updated_at = session.updated_at

            if should_persist:
                self.store.persist_sessions_safely()
                self._publish_event(
                    "title_updated",
                    {
                        "chat_id": session.chat_id,
                        "title": generated_title,
                        "updated_at": updated_at,
                    },
                )

        future = self.submit_async(refine_title())

        def _log_future_error(done_future: Any) -> None:
            try:
                done_future.result()
            except Exception as e:
                print(f"[WARN] 异步生成聊天标题失败: {e}")

        future.add_done_callback(_log_future_error)

    def _publish_event(self, event: str, data: dict[str, Any]) -> None:
        payload = {
            "event": str(event or "message"),
            "data": dict(data or {}),
        }
        with self._event_subscribers_lock:
            subscribers = list(self._event_subscribers)

        for subscriber in subscribers:
            try:
                subscriber.put_nowait(payload)
            except Full:
                try:
                    subscriber.get_nowait()
                except Empty:
                    pass
                try:
                    subscriber.put_nowait(payload)
                except Full:
                    continue
            except Exception:
                continue

    def _register_event_subscriber(self) -> Queue:
        subscriber: Queue = Queue(maxsize=32)
        with self._event_subscribers_lock:
            self._event_subscribers.add(subscriber)
        return subscriber

    def _unregister_event_subscriber(self, subscriber: Queue) -> None:
        with self._event_subscribers_lock:
            self._event_subscribers.discard(subscriber)

    def iter_chat_update_events(self):
        subscriber = self._register_event_subscriber()
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    payload = subscriber.get(timeout=15)
                except Empty:
                    yield ": keepalive\n\n"
                    continue
                event = str(payload.get("event", "message") or "message")
                data = payload.get("data", {})
                yield self.sse_encode(event, data if isinstance(data, dict) else {})
        finally:
            self._unregister_event_subscriber(subscriber)

    def build_chat_message_stream_handler(
        self,
        chat_id: str,
        payload: dict[str, Any],
    ) -> tuple[Callable[[], Any] | None, tuple[str, int] | None]:
        if not isinstance(payload, dict):
            payload = {}
        question = str(payload.get("message", "")).strip()
        if not question:
            return None, ("message 不能为空", 400)

        chunk_size = safe_int(payload.get("stream_chunk_size"), 24, floor=1)
        session = self.store.get_or_create_session(chat_id)
        mode = self.store.normalize_mode(payload.get("mode", session.mode))
        requested_subjects = self.normalize_requested_subjects(payload.get("subjects"))
        code_analysis_requested = self._safe_bool(payload.get("code_analysis"), False)
        problem_tutoring_requested = self._safe_bool(
            payload.get("problem_tutoring", payload.get("tutoring")),
            False,
        )
        response_language = self._response_language_from_requested_subjects(
            requested_subjects
        )
        explicit_subjects = list(requested_subjects or [])
        default_timeout = cfg.DEFAULT_TIMEOUT_BY_MODE.get(mode, cfg.AUTO_TIMEOUT_S)
        timeout_s = safe_int(payload.get("timeout"), default_timeout, floor=1)

        def event_stream():
            event_queue: Queue = Queue()
            sentinel = object()
            workflow_started_at: dict[str, float] = {}
            workflow_steps: dict[str, dict[str, Any]] = {}
            graph_payload_snapshot: dict[str, Any] | None = None
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            retrieval_used = True
            retrieval_gate_confidence = 1.0
            retrieval_gate_reason = "未启用检索网关"
            subject_route_meta: dict[str, Any] | None = None
            workflow_order = [
                "query_understanding",
                "retrieval_gate",
                "subject_route",
                "lightrag_retrieve",
                "deepsearch_plan",
                "deepsearch_subject_route",
                "deepsearch_retrieve",
                "deepsearch_review",
                "deepsearch_retry",
                "neo4j_subgraph",
                "answer_generate",
                "final_response",
            ]

            def push_event(event: str, data: dict[str, Any]) -> None:
                event_queue.put((event, dict(data or {})))

            def record_workflow_step(step: dict[str, Any]) -> None:
                node_id = str(step.get("nodeId", "") or "").strip()
                if not node_id:
                    return
                current = workflow_steps.get(node_id, {})
                workflow_steps[node_id] = {**current, **step}

            def workflow_steps_snapshot() -> list[dict[str, Any]]:
                ordered: list[dict[str, Any]] = []
                for node_id in workflow_order:
                    if node_id in workflow_steps:
                        ordered.append(dict(workflow_steps[node_id]))
                for node_id, step in workflow_steps.items():
                    if node_id not in workflow_order:
                        ordered.append(dict(step))
                if mode == "deepsearch":
                    ordered = [
                        step
                        for step in ordered
                        if str(step.get("nodeId", "") or "") != "neo4j_subgraph"
                    ]
                return ordered

            def record_graph_payload(payload: dict[str, Any]) -> None:
                nonlocal graph_payload_snapshot
                graph_payload_snapshot = dict(payload or {})

            def graph_payload_to_local_subgraphs(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
                if not isinstance(payload, dict):
                    return []
                nodes = [node for node in payload.get("nodes", []) if isinstance(node, dict)]
                edges = [edge for edge in payload.get("edges", []) if isinstance(edge, dict)]
                chunks = [chunk for chunk in payload.get("chunks", []) if isinstance(chunk, dict)]
                center_ids = {str(item) for item in payload.get("centerEntityIds", []) or []}
                if not nodes:
                    return []
                grouped: dict[str, list[dict[str, Any]]] = {}
                fallback_subjects = payload.get("subjectIds", []) or []
                fallback_subject = str(fallback_subjects[0]) if fallback_subjects else "unknown"
                for node in nodes:
                    subject_id = str(node.get("subjectId", "") or fallback_subject)
                    grouped.setdefault(subject_id, []).append(node)
                subgraphs: list[dict[str, Any]] = []
                for subject_id, group_nodes in grouped.items():
                    node_ids = {str(node.get("id", "")) for node in group_nodes}
                    graph_edges = [
                        edge
                        for edge in edges
                        if str(edge.get("source", "")) in node_ids
                        and str(edge.get("target", "")) in node_ids
                    ]
                    graph_chunks = [
                        chunk
                        for chunk in chunks
                        if not chunk.get("subjectId") or str(chunk.get("subjectId")) == subject_id
                    ]
                    subgraphs.append(
                        {
                            "id": f"subgraph-{subject_id}",
                            "title": f"{self._subject_label(subject_id)} 局部图谱",
                            "subjectId": subject_id,
                            "summary": f"{len(group_nodes)} 个实体，{len(graph_chunks)} 条证据",
                            "nodes": group_nodes,
                            "edges": graph_edges,
                            "centerEntityIds": [item for item in center_ids if item in node_ids],
                            "chunkIds": [
                                str(chunk.get("id") or chunk.get("chunkId") or "")
                                for chunk in graph_chunks
                                if chunk.get("id") or chunk.get("chunkId")
                            ],
                        }
                    )
                return subgraphs

            def _normalize_sufficient(value: Any) -> bool | None:
                if isinstance(value, bool):
                    return value
                text = str(value or "").strip().lower()
                if text == "true":
                    return True
                if text == "false":
                    return False
                return None

            def _normalize_subject_ids(value: Any) -> list[str]:
                if not isinstance(value, list):
                    return []
                return [
                    str(item)
                    for item in value
                    if str(item or "").strip() in self.subject_catalog
                ]

            def _normalize_ranked_subjects(value: Any) -> list[dict[str, Any]]:
                if not isinstance(value, list):
                    return []
                ranked: list[dict[str, Any]] = []
                for item in value:
                    subject_id = ""
                    score = 0.0
                    if isinstance(item, dict):
                        subject_id = str(
                            item.get("subject")
                            or item.get("subject_id")
                            or item.get("id")
                            or ""
                        )
                        raw_score = item.get("score", 0.0)
                    elif isinstance(item, (list, tuple)) and item:
                        subject_id = str(item[0] or "")
                        raw_score = item[1] if len(item) > 1 else 0.0
                    else:
                        continue
                    if subject_id not in self.subject_catalog:
                        continue
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        score = 0.0
                    ranked.append(
                        {
                            "subject": subject_id,
                            "label": self._subject_label(subject_id),
                            "score": score,
                        }
                    )
                return ranked

            def _safe_optional_int(value: Any) -> int | None:
                if value is None or value == "":
                    return None
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None

            def build_deepsearch_trace(
                result: dict[str, Any] | None,
                mode_used: str,
            ) -> dict[str, Any] | None:
                if mode_used != "deepsearch":
                    return None
                raw = (
                    result.get("raw")
                    if isinstance(result, dict) and isinstance(result.get("raw"), dict)
                    else {}
                )
                raw_sub_questions = (
                    raw.get("sub_questions", [])
                    if isinstance(raw.get("sub_questions", []), list)
                    else []
                )
                lock_subject_ids = [
                    subject_id
                    for subject_id in requested_subjects
                    if subject_id in self.subject_catalog
                ]
                subject_lock_enabled = bool(lock_subject_ids)
                subject_lock_labels = [
                    self._subject_label(subject_id) for subject_id in lock_subject_ids
                ]
                sub_questions: list[dict[str, Any]] = []
                sub_question_routes: list[dict[str, Any]] = []
                review_items: list[dict[str, Any]] = []

                for index, raw_item in enumerate(raw_sub_questions):
                    item = raw_item if isinstance(raw_item, dict) else {"question": raw_item}
                    sub_id = str(item.get("id") or f"q{index + 1}")
                    question_text = str(item.get("question", "") or "").strip()
                    used_question = str(
                        item.get("used_question")
                        or item.get("usedQuestion")
                        or question_text
                    ).strip()
                    query_mode = str(
                        item.get("query_mode")
                        or item.get("queryMode")
                        or "hybrid"
                    ).strip()
                    target_subjects = _normalize_subject_ids(
                        item.get("target_subjects") or item.get("targetSubjects")
                    )
                    route_reason = str(
                        item.get("route_reason")
                        or item.get("routeReason")
                        or ""
                    ).strip()
                    ranked_subjects = _normalize_ranked_subjects(
                        item.get("ranked_subjects") or item.get("rankedSubjects")
                    )
                    if subject_lock_enabled:
                        target_subjects = list(lock_subject_ids)
                        if not ranked_subjects:
                            ranked_subjects = [
                                {
                                    "subject": subject_id,
                                    "label": self._subject_label(subject_id),
                                    "score": 1.0 if idx == 0 else 0.0,
                                }
                                for idx, subject_id in enumerate(lock_subject_ids)
                            ]
                        route_reason = (
                            route_reason
                            or f"用户指定学科，锁定为{'、'.join(subject_lock_labels)}"
                        )

                    primary_subject = target_subjects[0] if target_subjects else ""
                    sub_questions.append(
                        {
                            "id": sub_id,
                            "question": question_text,
                            "usedQuestion": used_question,
                            "queryMode": query_mode,
                            "topK": _safe_optional_int(item.get("top_k") or item.get("topK")),
                            "chunkTopK": _safe_optional_int(
                                item.get("chunk_top_k") or item.get("chunkTopK")
                            ),
                        }
                    )
                    sub_question_routes.append(
                        {
                            "subQuestionId": sub_id,
                            "primarySubject": primary_subject,
                            "primarySubjectLabel": (
                                self._subject_label(primary_subject)
                                if primary_subject in self.subject_catalog
                                else ""
                            ),
                            "targetSubjects": target_subjects,
                            "rankedSubjects": ranked_subjects,
                            "reason": route_reason,
                        }
                    )
                    review_items.append(
                        {
                            "subQuestionId": sub_id,
                            "sufficient": _normalize_sufficient(item.get("sufficient")),
                            "judgeReason": str(item.get("judge_reason") or "").strip(),
                            "rewrittenQuestion": str(
                                item.get("rewritten_question") or ""
                            ).strip(),
                        }
                    )

                insufficient_ids = (
                    raw.get("insufficient_subquestion_ids", [])
                    if isinstance(raw.get("insufficient_subquestion_ids", []), list)
                    else []
                )
                if not insufficient_ids:
                    insufficient_ids = [
                        item["subQuestionId"]
                        for item in review_items
                        if item.get("sufficient") is False
                    ]
                query_attempt = _safe_optional_int(raw.get("query_attempt")) or 0
                return {
                    "subQuestions": sub_questions,
                    "subQuestionRoutes": sub_question_routes,
                    "review": review_items,
                    "retry": {
                        "queryAttempt": query_attempt,
                        "needsRetry": bool(raw.get("needs_retry", False)),
                        "insufficientSubquestionIds": [
                            str(item) for item in insufficient_ids
                        ],
                    },
                    "subjectLock": {
                        "enabled": subject_lock_enabled,
                        "subjectIds": lock_subject_ids,
                        "subjectLabels": subject_lock_labels,
                        "reason": (
                            f"所有子问题仅在{'、'.join(subject_lock_labels)}知识库内检索"
                            if subject_lock_enabled
                            else ""
                        ),
                    },
                }

            def build_explainability_details(
                *,
                result: dict[str, Any] | None,
                status: str,
                mode_used: str,
                subject_route: dict[str, Any] | None,
            ) -> dict[str, Any]:
                route = subject_route if isinstance(subject_route, dict) else {}
                primary_subject = str(route.get("primary_subject", "") or "")
                subject = (
                    requested_subjects[0]
                    if len(requested_subjects) == 1
                    else "auto"
                )
                payload = graph_payload_snapshot if isinstance(graph_payload_snapshot, dict) else {}
                graph_ok = bool(payload.get("ok", True)) if payload else True
                chunks = payload.get("chunks", []) if isinstance(payload.get("chunks", []), list) else []
                details = {
                    "mode": mode,
                    "modeUsed": mode_used,
                    "subject": subject,
                    "detectedSubject": primary_subject,
                    "subjectRoute": dict(route),
                    "workflowSteps": workflow_steps_snapshot(),
                    "localSubgraphs": graph_payload_to_local_subgraphs(payload),
                    "chunks": chunks,
                    "graphError": "" if graph_ok else str(payload.get("error", "")),
                    "status": status,
                    "createdAt": created_at,
                    "retrievalUsed": bool(retrieval_used),
                    "retrievalGateConfidence": retrieval_gate_confidence,
                    "retrievalGateReason": retrieval_gate_reason,
                }
                route_info = result.get("route") if isinstance(result, dict) else None
                if mode == "auto" and isinstance(route_info, dict):
                    details["autoRoute"] = {
                        "chain": str(route_info.get("chain", "") or ""),
                        "policy": str(route_info.get("policy", "") or ""),
                        "reason": str(route_info.get("reason", "") or ""),
                        "complexity": str(route_info.get("complexity", "") or ""),
                        "confidence": route_info.get("confidence"),
                        "subjects": route_info.get("subjects", []),
                    }
                    auto_timings = result.get("auto_timings")
                    if isinstance(auto_timings, dict):
                        details["autoTimings"] = dict(auto_timings)
                    details["autoUpgraded"] = bool(result.get("upgraded", False))
                    details["autoUpgradeReason"] = str(result.get("upgrade_reason", "") or "")
                    if isinstance(result.get("instant_review"), dict):
                        details["instantReview"] = dict(result.get("instant_review") or {})
                deepsearch_trace = build_deepsearch_trace(result, mode_used)
                if deepsearch_trace is not None:
                    details["deepsearchTrace"] = deepsearch_trace
                return details

            def build_message_details(
                result: dict[str, Any] | None,
                *,
                status: str,
                mode_used: str,
                subject_route: dict[str, Any] | None,
            ) -> dict[str, Any]:
                base = (
                    dict(result.get("message_details"))
                    if isinstance(result, dict) and isinstance(result.get("message_details"), dict)
                    else {}
                )
                base["explainability"] = build_explainability_details(
                    result=result,
                    status=status,
                    mode_used=mode_used,
                    subject_route=subject_route,
                )
                return base

            def workflow_start(
                node_id: str,
                node_name: str,
                input_summary: str = "",
            ) -> None:
                workflow_started_at[node_id] = time.perf_counter()
                data = {
                    "nodeId": node_id,
                    "nodeName": node_name,
                    "status": "running",
                    "inputSummary": input_summary,
                }
                record_workflow_step(data)
                push_event("workflow_node_start", data)

            def workflow_end(
                node_id: str,
                node_name: str,
                output_summary: str = "",
                *,
                duration_ms: int | None = None,
            ) -> None:
                started_at = workflow_started_at.pop(node_id, time.perf_counter())
                measured_ms = int((time.perf_counter() - started_at) * 1000)
                data = {
                    "nodeId": node_id,
                    "nodeName": node_name,
                    "status": "success",
                    "outputSummary": output_summary,
                    "durationMs": duration_ms if duration_ms is not None else measured_ms,
                }
                record_workflow_step(data)
                push_event("workflow_node_end", data)

            def workflow_complete(
                node_id: str,
                node_name: str,
                *,
                status: str = "success",
                input_summary: str = "",
                output_summary: str = "",
                duration_ms: int | None = None,
            ) -> None:
                data: dict[str, Any] = {
                    "nodeId": node_id,
                    "nodeName": node_name,
                    "status": status,
                }
                if input_summary:
                    data["inputSummary"] = input_summary
                if output_summary:
                    data["outputSummary"] = output_summary
                if duration_ms is not None:
                    data["durationMs"] = duration_ms
                record_workflow_step(data)
                push_event("workflow_node_end", data)

            def workflow_error(
                node_id: str,
                node_name: str,
                error: str,
            ) -> None:
                workflow_started_at.pop(node_id, None)
                data = {
                    "nodeId": node_id,
                    "nodeName": node_name,
                    "status": "error",
                    "error": error,
                }
                record_workflow_step(data)
                push_event("workflow_node_error", data)

            def record_deepsearch_workflow_nodes(result: dict[str, Any]) -> None:
                trace = build_deepsearch_trace(result, "deepsearch")
                if trace is None:
                    return
                raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
                sub_questions = trace.get("subQuestions", [])
                sub_count = len(sub_questions) if isinstance(sub_questions, list) else 0
                if "deepsearch_plan" not in workflow_steps:
                    workflow_complete(
                        "deepsearch_plan",
                        "拆解子问题",
                        output_summary=(
                            f"拆解为 {sub_count} 个子问题"
                            if sub_count
                            else "完成子问题拆解"
                        ),
                        duration_ms=_safe_optional_int(raw.get("planning_ms")),
                    )
                subject_lock = (
                    trace.get("subjectLock")
                    if isinstance(trace.get("subjectLock"), dict)
                    else {}
                )
                lock_enabled = bool(subject_lock.get("enabled"))
                lock_labels = subject_lock.get("subjectLabels", [])
                lock_text = "、".join(
                    str(item) for item in lock_labels if str(item or "").strip()
                )
                route_node_name = "子问题学科锁定" if lock_enabled else "子问题学科路由"
                route_summary = (
                    f"所有子问题锁定到 {lock_text or '当前学科'}"
                    if lock_enabled
                    else f"完成 {sub_count} 个子问题的学科路由"
                )
                workflow_complete(
                    "deepsearch_subject_route",
                    route_node_name,
                    output_summary=route_summary,
                )

                subquery_results = (
                    raw.get("subquery_results", [])
                    if isinstance(raw.get("subquery_results", []), list)
                    else []
                )
                workflow_complete(
                    "deepsearch_retrieve",
                    "子问题并行检索",
                    output_summary=(
                        f"完成 {len(subquery_results)} 条子问题检索结果"
                        if subquery_results
                        else "完成子问题检索"
                    ),
                )

                retry_info = (
                    trace.get("retry") if isinstance(trace.get("retry"), dict) else {}
                )
                insufficient_ids = retry_info.get("insufficientSubquestionIds", [])
                insufficient_count = (
                    len(insufficient_ids) if isinstance(insufficient_ids, list) else 0
                )
                query_attempt = _safe_optional_int(retry_info.get("queryAttempt")) or 0
                workflow_complete(
                    "deepsearch_review",
                    "证据评审",
                    output_summary=(
                        "所有子问题证据充分"
                        if insufficient_count == 0
                        else f"{insufficient_count} 个子问题证据不足"
                    ),
                )
                workflow_complete(
                    "deepsearch_retry",
                    "改写子问题",
                    status="success" if query_attempt > 0 else "skipped",
                    output_summary=(
                        f"已执行 {query_attempt} 轮改写补充检索"
                        if query_attempt > 0
                        else "证据评审未触发改写"
                    ),
                )
                answer_ms = _safe_optional_int(raw.get("answer_ms"))
                workflow_complete(
                    "answer_generate",
                    "综合生成答案",
                    output_summary="汇总子问题证据并生成最终答案",
                    duration_ms=answer_ms,
                )

            async def produce_events() -> None:
                nonlocal retrieval_used
                nonlocal retrieval_gate_confidence
                nonlocal retrieval_gate_reason
                nonlocal subject_route_meta
                workflow_start(
                    "query_understanding",
                    "问题理解",
                    "识别请求类型、模式和上下文",
                )
                code_candidate = self._match_code_analysis_request(
                    question,
                    requested_subjects=requested_subjects,
                    requested_by_user=code_analysis_requested,
                )
                tutoring_candidate = (
                    self._match_problem_tutoring_request(
                        question,
                        requested_subjects=requested_subjects,
                        requested_by_user=problem_tutoring_requested,
                    )
                    if code_candidate is None
                    else None
                )
                fast_result_bundle = (
                    None
                    if code_candidate is not None or tutoring_candidate is not None
                    else self._fast_smalltalk_result_bundle(
                        mode=mode,
                        text=question,
                        response_language=response_language,
                    )
                )
                with session.lock:
                    should_auto_rename = (
                        len(session.turns) == 0
                        and self.store.is_placeholder_title(session.title)
                    )
                    augmented_question = (
                        question
                        if fast_result_bundle is not None
                        else self.store.build_augmented_question(session, question)
                    )
                workflow_end("query_understanding", "问题理解", "请求上下文准备完成")

                streamed_parts: list[str] = []
                streamed_any = False
                started = time.perf_counter()

                def emit_text(text: str) -> None:
                    nonlocal streamed_any
                    piece = str(text or "")
                    if not piece:
                        return
                    streamed_any = True
                    streamed_parts.append(piece)
                    push_event("delta", {"text": piece})

                def route_subject_ids(meta: dict[str, Any] | None) -> list[str]:
                    if requested_subjects:
                        return [
                            item
                            for item in requested_subjects
                            if item in self.subject_catalog
                        ]
                    if not isinstance(meta, dict):
                        return []
                    ranked = meta.get("ranked")
                    if isinstance(ranked, list):
                        selected = [
                            str(item.get("subject", ""))
                            for item in ranked
                            if isinstance(item, dict)
                            and item.get("subject") in self.subject_catalog
                            and float(item.get("score") or 0.0) > 0
                        ]
                        if selected:
                            return selected[:2]
                    primary = str(meta.get("primary_subject", "") or "")
                    if primary in self.subject_catalog:
                        return [primary]
                    if mode == "deepsearch" and retrieval_used:
                        return list(self.subject_catalog.keys())
                    return []

                async def maybe_emit_graph_update(
                    meta: dict[str, Any] | None,
                ) -> None:
                    graph_service = getattr(self, "graph_service", None)
                    if graph_service is None:
                        return
                    subject_ids = route_subject_ids(meta)
                    if not subject_ids:
                        return
                    if not getattr(graph_service, "configured", False):
                        graph_payload = {
                            "ok": False,
                            "error": "Neo4j is not configured.",
                            "nodes": [],
                            "edges": [],
                            "chunks": [],
                            "subjectIds": subject_ids,
                            "centerEntityIds": [],
                        }
                        record_graph_payload(graph_payload)
                        push_event("graph_update", graph_payload)
                        return
                    workflow_start(
                        "neo4j_subgraph",
                        "Neo4j 局部子图",
                        "根据问题和学科查询相关实体图谱",
                    )
                    try:
                        graph_payload = await asyncio.to_thread(
                            graph_service.local_subgraph,
                            subject_ids=subject_ids,
                            query=question,
                            center_entity_ids=[],
                            depth=1,
                            limit=80,
                        )
                        record_graph_payload(graph_payload)
                        push_event("graph_update", graph_payload)
                        workflow_end(
                            "neo4j_subgraph",
                            "Neo4j 局部子图",
                            "局部知识图谱查询完成",
                        )
                    except Exception as graph_exc:
                        message = f"{type(graph_exc).__name__}: {graph_exc}"
                        workflow_error("neo4j_subgraph", "Neo4j 局部子图", message)
                        graph_payload = {
                            "ok": False,
                            "error": message,
                            "nodes": [],
                            "edges": [],
                            "chunks": [],
                            "subjectIds": subject_ids,
                            "centerEntityIds": [],
                        }
                        record_graph_payload(graph_payload)
                        push_event("graph_update", graph_payload)

                try:
                    if code_candidate is not None:
                        workflow_start(
                            "retrieval_gate",
                            "检索判断",
                            "代码分析请求跳过知识库检索",
                        )
                        subject_route_meta = self._build_subject_route_meta(
                            self._build_code_analysis_subject_route(
                                requested_subjects=requested_subjects
                            )
                        )
                        retrieval_used = False
                        retrieval_gate_confidence = 1.0
                        trigger = str(code_candidate.get("trigger") or "").strip()
                        if trigger == "explicit":
                            retrieval_gate_reason = "用户显式触发 C 代码分析"
                        else:
                            retrieval_gate_reason = "检测到 C 代码问题，自动进入代码分析"
                        workflow_end(
                            "retrieval_gate",
                            "检索判断",
                            retrieval_gate_reason,
                        )
                        workflow_start(
                            "answer_generate",
                            "答案生成",
                            "运行 C 代码分析链路",
                        )
                        push_event(
                            "meta",
                            {
                                "retrieval_used": retrieval_used,
                                "retrieval_gate_confidence": retrieval_gate_confidence,
                                "retrieval_gate_reason": retrieval_gate_reason,
                                "request_kind": "code_analysis",
                                "subject_route": subject_route_meta,
                            },
                        )
                        result = await self._run_code_analysis_stream(
                            user_question=question,
                            mode=mode,
                            timeout_s=timeout_s,
                            response_language=response_language,
                            code_candidate=code_candidate,
                            emit_text=emit_text,
                        )
                        result["subject_route"] = subject_route_meta
                        workflow_end("answer_generate", "答案生成", "代码分析完成")
                    elif fast_result_bundle is not None:
                        workflow_start(
                            "retrieval_gate",
                            "检索判断",
                            "本地快路径判断",
                        )
                        (
                            result,
                            retrieval_used,
                            retrieval_gate_confidence,
                            retrieval_gate_reason,
                        ) = fast_result_bundle
                        subject_route_meta = result.get("subject_route")
                        workflow_end(
                            "retrieval_gate",
                            "检索判断",
                            retrieval_gate_reason,
                        )
                        push_event(
                            "meta",
                            {
                                "retrieval_used": retrieval_used,
                                "retrieval_gate_confidence": retrieval_gate_confidence,
                                "retrieval_gate_reason": retrieval_gate_reason,
                                "subject_route": subject_route_meta,
                            },
                        )
                        workflow_start("answer_generate", "答案生成", "输出本地快路径回答")
                        emit_text(str(result.get("answer", "")))
                        workflow_end("answer_generate", "答案生成", "回答输出完成")
                    else:
                        subject_route: dict[str, Any] | None = None
                        if explicit_subjects:
                            retrieval_used = True
                            retrieval_gate_confidence = 1.0
                            retrieval_gate_reason = "用户显式指定学科，强制检索"
                            workflow_start(
                                "retrieval_gate",
                                "检索路由",
                                "用户显式指定学科，强制进入课程知识库检索",
                            )
                            workflow_end(
                                "retrieval_gate",
                                "检索路由",
                                retrieval_gate_reason,
                            )
                            if mode == "deepsearch":
                                subject_route_meta = self._build_subject_route_meta(
                                    self._subject_route_from_explicit_subjects(
                                        explicit_subjects
                                    )
                                )
                            else:
                                workflow_start(
                                    "subject_route",
                                    "学科路由",
                                    "用户显式指定学科后确认主学科",
                                )
                                subject_route = await self.decide_subject_route(
                                    user_question=question,
                                    augmented_question=augmented_question,
                                    mode=mode,
                                    timeout_s=timeout_s,
                                    requested_subjects=explicit_subjects,
                                )
                                workflow_end(
                                    "subject_route",
                                    "学科路由",
                                    str(subject_route.get("reason", "")),
                                )
                        elif tutoring_candidate is not None:
                            workflow_start(
                                "subject_route",
                                "学科路由",
                                "题目辅导请求识别课程范围",
                            )
                            subject_route = await self.decide_subject_route(
                                user_question=question,
                                augmented_question=augmented_question,
                                mode=mode,
                                timeout_s=timeout_s,
                                requested_subjects=requested_subjects,
                            )
                            workflow_end(
                                "subject_route",
                                "学科路由",
                                str(subject_route.get("reason", "")),
                            )
                            retrieval_used = True
                            retrieval_gate_confidence = 1.0
                            retrieval_gate_reason = "用户显式触发题目辅导"
                        elif mode == "deepsearch":
                            if cfg.WEB_ENABLE_RETRIEVAL_GATE:
                                workflow_start(
                                    "retrieval_gate",
                                    "检索判断",
                                    "判断 DeepSearch 是否需要课程知识库",
                                )
                                need_retrieval, gate_conf, gate_reason = (
                                    await self.decide_need_retrieval(
                                        subject_ids=list(self.subject_catalog.keys()),
                                        user_question=question,
                                        augmented_question=augmented_question,
                                        mode=mode,
                                        timeout_s=timeout_s,
                                        response_language=response_language,
                                    )
                                )
                                retrieval_used = bool(need_retrieval)
                                retrieval_gate_confidence = float(gate_conf)
                                retrieval_gate_reason = (
                                    str(gate_reason or "").strip() or "无"
                                )
                                workflow_end(
                                    "retrieval_gate",
                                    "检索判断",
                                    retrieval_gate_reason,
                                )
                                if retrieval_used:
                                    subject_route_meta = self._fast_subject_route_meta(
                                        "DeepSearch 跳过请求级学科路由，拆题后对子问题单独路由"
                                    )
                                else:
                                    subject_route_meta = self._fast_subject_route_meta(
                                        "第一网关判定免检索直答"
                                    )
                            else:
                                subject_route_meta = self._fast_subject_route_meta(
                                    "DeepSearch 未启用请求级学科路由，拆题后对子问题单独路由"
                                )
                        elif cfg.WEB_ENABLE_RETRIEVAL_GATE:
                            workflow_start(
                                "retrieval_gate",
                                "检索判断",
                                "判断问题是否需要课程知识库",
                            )
                            need_retrieval, gate_conf, gate_reason = (
                                await self.decide_need_retrieval(
                                    subject_ids=list(self.subject_catalog.keys()),
                                    user_question=question,
                                    augmented_question=augmented_question,
                                    mode=mode,
                                    timeout_s=timeout_s,
                                    response_language=response_language,
                                )
                            )
                            retrieval_used = bool(need_retrieval)
                            retrieval_gate_confidence = float(gate_conf)
                            retrieval_gate_reason = (
                                str(gate_reason or "").strip() or "无"
                            )
                            workflow_end(
                                "retrieval_gate",
                                "检索判断",
                                retrieval_gate_reason,
                            )
                            if retrieval_used:
                                workflow_start(
                                    "subject_route",
                                    "学科路由",
                                    "选择需要检索的课程知识库",
                                )
                                subject_route = await self.decide_subject_route(
                                    user_question=question,
                                    augmented_question=augmented_question,
                                    mode=mode,
                                    timeout_s=timeout_s,
                                    requested_subjects=requested_subjects,
                                )
                                workflow_end(
                                    "subject_route",
                                    "学科路由",
                                    str(subject_route.get("reason", "")),
                                )
                            else:
                                subject_route_meta = self._fast_subject_route_meta(
                                    "第一网关判定免检索直答"
                                )
                        else:
                            workflow_start(
                                "subject_route",
                                "学科路由",
                                "未启用检索网关，直接选择课程知识库",
                            )
                            subject_route = await self.decide_subject_route(
                                user_question=question,
                                augmented_question=augmented_question,
                                mode=mode,
                                timeout_s=timeout_s,
                                requested_subjects=requested_subjects,
                            )
                            workflow_end(
                                "subject_route",
                                "学科路由",
                                str(subject_route.get("reason", "")),
                            )
                        if subject_route is not None:
                            subject_route_meta = self._build_subject_route_meta(subject_route)

                        push_event(
                            "meta",
                            {
                                "retrieval_used": retrieval_used,
                                "retrieval_gate_confidence": retrieval_gate_confidence,
                                "retrieval_gate_reason": retrieval_gate_reason,
                                "request_kind": (
                                    "problem_tutoring"
                                    if retrieval_used and tutoring_candidate is not None
                                    else "chat"
                                ),
                                "subject_route": subject_route_meta,
                            },
                        )
                        await maybe_emit_graph_update(subject_route_meta)

                        if retrieval_used and tutoring_candidate is not None:
                            workflow_start(
                                "answer_generate",
                                "答案生成",
                                "运行题目辅导链路",
                            )
                            result = await self._run_problem_tutoring_stream(
                                user_question=question,
                                augmented_question=augmented_question,
                                mode=mode,
                                timeout_s=timeout_s,
                                subject_route=subject_route,
                                response_language=response_language,
                                tutoring_candidate=tutoring_candidate,
                                emit_text=emit_text,
                            )
                            result["subject_route"] = subject_route_meta
                            workflow_end("answer_generate", "答案生成", "题目辅导完成")
                        elif retrieval_used:
                            if mode == "deepsearch":
                                lock_subject_ids = [
                                    subject_id
                                    for subject_id in requested_subjects
                                    if subject_id in self.subject_catalog
                                ]
                                lock_text = "、".join(
                                    self._subject_label(subject_id)
                                    for subject_id in lock_subject_ids
                                )

                                def subquestion_count(stage_state: dict[str, Any]) -> int:
                                    sub_questions = stage_state.get("sub_questions", [])
                                    return len(sub_questions) if isinstance(sub_questions, list) else 0

                                def insufficient_count(stage_state: dict[str, Any]) -> int:
                                    ids = stage_state.get("insufficient_subquestion_ids", [])
                                    return len(ids) if isinstance(ids, list) else 0

                                async def deepsearch_stage_callback(
                                    stage: str,
                                    stage_state: dict[str, Any],
                                ) -> None:
                                    if stage == "deepsearch_plan_start":
                                        workflow_start(
                                            "deepsearch_plan",
                                            "拆解子问题",
                                            "将原问题拆成可独立检索的子问题",
                                        )
                                    elif stage == "deepsearch_plan_end":
                                        count = subquestion_count(stage_state)
                                        workflow_end(
                                            "deepsearch_plan",
                                            "拆解子问题",
                                            f"拆解为 {count} 个子问题" if count else "完成子问题拆解",
                                        )
                                    elif stage == "deepsearch_subject_route_start":
                                        workflow_start(
                                            "deepsearch_subject_route",
                                            "子问题学科锁定" if lock_subject_ids else "子问题学科路由",
                                            (
                                                f"用户指定学科，锁定到 {lock_text or '当前学科'}"
                                                if lock_subject_ids
                                                else "为每个子问题选择目标知识库"
                                            ),
                                        )
                                    elif stage == "deepsearch_subject_route_end":
                                        count = subquestion_count(stage_state)
                                        workflow_end(
                                            "deepsearch_subject_route",
                                            "子问题学科锁定" if lock_subject_ids else "子问题学科路由",
                                            (
                                                f"所有子问题锁定到 {lock_text or '当前学科'}"
                                                if lock_subject_ids
                                                else f"完成 {count} 个子问题的学科路由"
                                            ),
                                        )
                                    elif stage == "deepsearch_retrieve_start":
                                        attempt = _safe_optional_int(stage_state.get("query_attempt")) or 0
                                        workflow_start(
                                            "deepsearch_retrieve",
                                            "子问题并行检索",
                                            f"执行第 {attempt + 1} 轮子问题并行检索",
                                        )
                                    elif stage == "deepsearch_retrieve_end":
                                        results = stage_state.get("subquery_results", [])
                                        result_count = len(results) if isinstance(results, list) else 0
                                        workflow_end(
                                            "deepsearch_retrieve",
                                            "子问题并行检索",
                                            (
                                                f"完成 {result_count} 条子问题检索结果"
                                                if result_count
                                                else "完成子问题检索"
                                            ),
                                            duration_ms=_safe_optional_int(stage_state.get("query_total_ms")),
                                        )
                                    elif stage == "deepsearch_review_start":
                                        workflow_start(
                                            "deepsearch_review",
                                            "证据评审",
                                            "评估每个子问题的检索证据是否充分",
                                        )
                                    elif stage == "deepsearch_review_end":
                                        count = insufficient_count(stage_state)
                                        workflow_end(
                                            "deepsearch_review",
                                            "证据评审",
                                            "证据充分，进入综合生成答案"
                                            if count == 0
                                            else f"{count} 个子问题证据不足，进入改写",
                                        )
                                    elif stage == "deepsearch_retry_start":
                                        workflow_start(
                                            "deepsearch_retry",
                                            "改写子问题",
                                            "改写证据不足的子问题并提高检索强度",
                                        )
                                    elif stage == "deepsearch_retry_end":
                                        attempt = _safe_optional_int(stage_state.get("query_attempt")) or 0
                                        workflow_end(
                                            "deepsearch_retry",
                                            "改写子问题",
                                            f"改写完成，进入第 {attempt + 1} 轮并行检索",
                                        )
                                    elif stage == "deepsearch_retry_skipped":
                                        workflow_complete(
                                            "deepsearch_retry",
                                            "改写子问题",
                                            status="skipped",
                                            output_summary="证据评审未触发改写",
                                        )
                                    elif stage == "answer_generate_start":
                                        workflow_start(
                                            "answer_generate",
                                            "综合生成答案",
                                            "汇总子问题证据并生成最终回答",
                                        )
                                    elif stage == "answer_generate_end":
                                        workflow_end(
                                            "answer_generate",
                                            "综合生成答案",
                                            "答案生成完成",
                                            duration_ms=_safe_optional_int(stage_state.get("answer_ms")),
                                        )

                                result = await self._stream_mode_with_retrieval(
                                    mode=mode,
                                    subject_route=subject_route,
                                    requested_subjects=requested_subjects,
                                    user_question=question,
                                    augmented_question=augmented_question,
                                    thread_id=session.chat_id,
                                    timeout_s=timeout_s,
                                    response_language=response_language,
                                    emit_text=emit_text,
                                    workflow_stage_callback=deepsearch_stage_callback,
                                )
                                result["subject_route"] = subject_route_meta
                                if "deepsearch_subject_route" not in workflow_steps:
                                    record_deepsearch_workflow_nodes(result)
                            else:
                                workflow_start(
                                    "lightrag_retrieve",
                                    "LightRAG 检索",
                                    "执行课程知识检索与答案生成",
                                )
                                result = await self._stream_mode_with_retrieval(
                                    mode=mode,
                                    subject_route=subject_route,
                                    requested_subjects=requested_subjects,
                                    user_question=question,
                                    augmented_question=augmented_question,
                                    thread_id=session.chat_id,
                                    timeout_s=timeout_s,
                                    response_language=response_language,
                                    emit_text=emit_text,
                                )
                                result["subject_route"] = subject_route_meta
                                workflow_end(
                                    "lightrag_retrieve",
                                    "LightRAG 检索",
                                    "检索链路完成",
                                )
                        else:
                            workflow_start(
                                "answer_generate",
                                "答案生成",
                                "免检索直答生成",
                            )
                            direct_timeout = max(
                                1,
                                min(
                                    int(timeout_s),
                                    max(1, int(cfg.WEB_DIRECT_ANSWER_TIMEOUT_S)),
                                ),
                            )
                            answer = await self._stream_llm_text(
                                llm_client=auto.auto_router_llm,
                                prompt=self._build_direct_answer_prompt(
                                    user_question=question,
                                    augmented_question=augmented_question,
                                    mode=mode,
                                    thread_id=session.chat_id,
                                    response_language=response_language,
                                ),
                                timeout_s=direct_timeout,
                                emit_text=emit_text,
                            )
                            result = {
                                "mode_used": mode,
                                "answer": answer,
                                "route": {
                                    "chain": "direct",
                                    "reason": "retrieval_gate=no",
                                },
                                "subject_route": subject_route_meta,
                                "upgraded": False,
                                "upgrade_reason": "",
                                "instant_review": {
                                    "heuristic": "",
                                    "review": "direct_answer",
                                },
                            }
                            workflow_end("answer_generate", "答案生成", "直答生成完成")
                except Exception as e:
                    for node_id in list(workflow_started_at):
                        workflow_error(node_id, node_id, f"{type(e).__name__}: {e}")
                    partial_answer = "".join(streamed_parts).strip()
                    error_answer = partial_answer or f"请求失败：{e}"
                    if partial_answer:
                        note = f"\n\n[生成中断：{type(e).__name__}: {e}]"
                        error_answer = f"{partial_answer}{note}"
                        push_event("delta", {"text": note})
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    mode_used = mode
                    assistant_meta = self.store.make_assistant_meta(mode_used, elapsed_ms)
                    message_details = build_message_details(
                        {},
                        status="error",
                        mode_used=mode_used,
                        subject_route=subject_route_meta,
                    )
                    with session.lock:
                        self.store.update_session_after_answer(
                            session,
                            question=question,
                            answer=error_answer,
                            requested_mode=mode,
                            mode_used=mode_used,
                            elapsed_ms=elapsed_ms,
                            assistant_meta=assistant_meta,
                            message_details=message_details,
                        )
                        chat_title = session.title
                    self.store.persist_sessions_safely()
                    response = {
                        "chat_id": session.chat_id,
                        "chat_title": chat_title,
                        "requested_mode": mode,
                        "requested_subjects": requested_subjects,
                        "mode_used": mode_used,
                        "answer": error_answer,
                        "assistant_meta": assistant_meta,
                        "elapsed_ms": str(elapsed_ms),
                        "route": None,
                        "subject_route": subject_route_meta,
                        "upgraded": False,
                        "upgrade_reason": "",
                        "instant_review": None,
                        "retrieval_used": retrieval_used,
                        "retrieval_gate_confidence": retrieval_gate_confidence,
                        "retrieval_gate_reason": retrieval_gate_reason,
                        "message_details": message_details,
                    }
                    push_event("done", response)
                    event_queue.put(sentinel)
                    return

                answer = str(result.get("answer", "")).strip()
                answer = self._strip_leading_question_echo(answer, question)
                if not answer:
                    answer = "未返回有效答案，请稍后重试。"
                if not streamed_any:
                    for chunk in self.iter_answer_chunks(answer, chunk_size):
                        emit_text(chunk)

                elapsed_ms = int((time.perf_counter() - started) * 1000)
                mode_used = self.store.normalize_mode(result.get("mode_used", mode))
                assistant_meta = self.store.make_assistant_meta(mode_used, elapsed_ms)
                route_chain = str((result.get("route") or {}).get("chain", "")).strip()
                if route_chain == "code_analysis":
                    assistant_meta = f"{assistant_meta} | 代码分析"
                elif route_chain == "problem_tutoring":
                    assistant_meta = f"{assistant_meta} | 题目辅导"
                    route_solver_status = str(
                        ((result.get("route") or {}).get("solver_status") or "")
                    ).strip()
                    if route_solver_status == "success":
                        assistant_meta = f"{assistant_meta} | 规则求解"
                elif not retrieval_used:
                    assistant_meta = f"{assistant_meta} | 免检索直答"

                workflow_start("final_response", "最终输出", "保存会话并结束流")
                workflow_end("final_response", "最终输出", "响应完成")
                message_details = build_message_details(
                    result,
                    status="done",
                    mode_used=mode_used,
                    subject_route=(
                        result.get("subject_route")
                        if isinstance(result.get("subject_route"), dict)
                        else subject_route_meta
                    ),
                )
                with session.lock:
                    self.store.update_session_after_answer(
                        session,
                        question=question,
                        answer=answer,
                        requested_mode=mode,
                        mode_used=mode_used,
                        elapsed_ms=elapsed_ms,
                        assistant_meta=assistant_meta,
                        message_details=message_details,
                    )

                if should_auto_rename:
                    self._apply_fallback_chat_title(session, question)
                    self._schedule_chat_title_refinement(session, question)

                self.store.persist_sessions_safely()
                with session.lock:
                    chat_title = session.title
                response = {
                    "chat_id": session.chat_id,
                    "chat_title": chat_title,
                    "requested_mode": mode,
                    "requested_subjects": requested_subjects,
                    "mode_used": mode_used,
                    "answer": answer,
                    "assistant_meta": assistant_meta,
                    "elapsed_ms": str(elapsed_ms),
                    "message_details": message_details,
                    "route": result.get("route"),
                    "subject_route": result.get("subject_route"),
                    "upgraded": bool(result.get("upgraded", False)),
                    "upgrade_reason": result.get("upgrade_reason", ""),
                    "instant_review": result.get("instant_review"),
                    "retrieval_used": retrieval_used,
                    "retrieval_gate_confidence": retrieval_gate_confidence,
                    "retrieval_gate_reason": retrieval_gate_reason,
                }

                meta = dict(response)
                meta.pop("answer", None)
                push_event("meta", meta)
                push_event("done", response)
                event_queue.put(sentinel)

            if self.submit_async is not None:
                future = self.submit_async(produce_events())

                def _consume_future(done_future: Any) -> None:
                    try:
                        done_future.result()
                    except Exception as e:
                        print(f"[WARN] 流式消息后台任务失败: {e}")

                future.add_done_callback(_consume_future)
            else:
                def _runner() -> None:
                    try:
                        self.run_async(produce_events())
                    except Exception as e:
                        print(f"[WARN] 流式消息线程回退失败: {e}")

                Thread(target=_runner, name="chat-stream-fallback", daemon=True).start()

            while True:
                try:
                    payload = event_queue.get(timeout=15)
                except Empty:
                    yield ": keepalive\n\n"
                    continue
                if payload is sentinel:
                    break
                event, data = payload
                yield self.sse_encode(str(event or "message"), data if isinstance(data, dict) else {})

        return event_stream, None

    @staticmethod
    def sse_encode(event: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n"

    @staticmethod
    def iter_answer_chunks(answer: str, chunk_size: int) -> list[str]:
        text = str(answer or "")
        if not text:
            return []
        size = max(1, min(64, int(chunk_size)))
        return [text[i : i + size] for i in range(0, len(text), size)]
