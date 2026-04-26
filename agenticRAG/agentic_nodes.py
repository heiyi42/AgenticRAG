import asyncio
import time
from typing import Any, Dict, List

from lightrag import QueryParam

from agenticRAG.agentic_config import (
    CHUNK_TOPK_RETRY_STEP,
    COMPLEX_MAX_RETRY,
    DEBUG,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_TOP_K,
    MAX_CHUNK_TOP_K,
    MAX_TOP_K,
    MIN_CHUNK_TOP_K,
    MIN_TOP_K,
    RETRY_MAX_ITEMS,
    SIMPLE_MAX_RETRY,
    TOPK_RETRY_STEP,
)
from agenticRAG.agentic_runtime import (
    get_rag,
    llm_evidence_struct,
    llm_subquestion_plan_struct,
)
from agenticRAG.query_utils import (
    build_query_result_row,
    extract_query_response_fields,
    normalize_exception_message,
)
from agenticRAG.agentic_schema import (
    EvidenceCheck,
    PLAN_MAX_ITEMS,
    State,
    SubQuestionQueryPlan,
)


def _clamp_topk(value: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = DEFAULT_TOP_K
    return max(MIN_TOP_K, min(MAX_TOP_K, v))


def _clamp_chunk_topk(value: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = DEFAULT_CHUNK_TOP_K
    return max(MIN_CHUNK_TOP_K, min(MAX_CHUNK_TOP_K, v))


def _mode_fallbacks(initial_mode: str) -> List[str]:
    if initial_mode == "local":
        return ["local", "hybrid", "global"]
    if initial_mode == "global":
        return ["global", "hybrid", "local"]
    return ["hybrid", "local", "global"]


def _next_mode(current_mode: str) -> str:
    seq = _mode_fallbacks(current_mode)
    return seq[1] if len(seq) > 1 else current_mode


def _normalize_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"local", "global", "hybrid"}:
        return mode
    return "hybrid"


def _normalize_requested_mode(state: State) -> str:
    mode = str(state.get("requested_mode", "auto") or "").strip().lower()
    if mode in {"instant", "deepsearch", "auto"}:
        return mode
    return "auto"


def _effective_strategy(state: State) -> str:
    strategy = str(state.get("effective_strategy", "") or "").strip().lower()
    if strategy in {"simple", "deep"}:
        return strategy

    requested_mode = _normalize_requested_mode(state)
    if requested_mode == "instant":
        return "simple"
    if requested_mode == "deepsearch":
        return "deep"
    if str(state.get("question_complexity", "") or "").strip().lower() == "simple":
        return "simple"
    return "deep"


def _normalize_plan_lengths(
    sub_questions: List[str],
    modes: List[str],
    topks: List[int],
    chunk_topks: List[int],
    *,
    fallback_question: str,
) -> dict:
    questions = [str(item).strip() for item in sub_questions if str(item or "").strip()]
    if not questions:
        questions = [fallback_question]
    questions = questions[:PLAN_MAX_ITEMS]

    count = max(1, len(questions))
    normalized_modes = [_normalize_mode(item) for item in modes[:count]]
    normalized_topks = [_clamp_topk(item) for item in topks[:count]]
    normalized_chunk_topks = [_clamp_chunk_topk(item) for item in chunk_topks[:count]]

    while len(normalized_modes) < count:
        normalized_modes.append(normalized_modes[-1] if normalized_modes else "hybrid")
    while len(normalized_topks) < count:
        normalized_topks.append(
            normalized_topks[-1] if normalized_topks else DEFAULT_TOP_K
        )
    while len(normalized_chunk_topks) < count:
        normalized_chunk_topks.append(
            normalized_chunk_topks[-1] if normalized_chunk_topks else DEFAULT_CHUNK_TOP_K
        )

    return {
        "sub_questions": questions,
        "query_modes": normalized_modes,
        "query_topks": normalized_topks,
        "query_chunk_topks": normalized_chunk_topks,
    }


def _blank_subquery_plan_state() -> dict:
    return {
        "subquery_tasks": [],
        "subquery_results": [],
        "query_attempt": 0,
        "needs_retry": False,
        "insufficient_subquestion_ids": [],
        "query_total_ms": "0",
    }


def _normalize_subject_ids(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for item in raw:
        subject_id = str(item or "").strip()
        if not subject_id or subject_id in seen:
            continue
        seen.add(subject_id)
        normalized.append(subject_id)
    return normalized


def _debug_print_subquery_plan(label: str, result: dict) -> None:
    if not DEBUG:
        return
    print(f"\n[DEBUG] {label}：")
    print(
        "[DEBUG] strategy="
        f"{result.get('effective_strategy', '')}, "
        f"requested_mode={result.get('requested_mode', '')}, "
        f"reason={result.get('planning_reason', '')}"
    )
    for item in result.get("sub_questions", []):
        if not isinstance(item, dict):
            continue
        print(
            "[DEBUG] "
            f"{item.get('id', '')}: mode={item.get('query_mode', '')}, "
            f"top_k={item.get('top_k', '')}, chunk_top_k={item.get('chunk_top_k', '')} | "
            f"{item.get('question', '')}"
        )


def build_global_subquestion_plan(state: State) -> dict:
    q = state["question"]
    requested_mode = _normalize_requested_mode(state)
    prompt = (
        "你是 GraphRAG 全局深度查询规划器。请先把原问题拆成适合独立检索的子问题，"
        "再为每个子问题给一个初始的查询参数配置。\n"
        "注意：\n"
        "1) 当前阶段只负责拆分问题，不负责决定学科知识库；学科路由会在后续单独完成。\n"
        "2) 子问题数量为 1 到 5 个，通常优先 2 到 5 个；只有单个检索问题就足够时才返回 1 个。\n"
        "3) 子问题要覆盖不同信息维度，避免重复和弱相关问题。\n"
        "4) mode 只能是 local/global/hybrid，代表每个子问题的初始检索偏好。\n"
        f"5) top_k 必须是整数，范围 {MIN_TOP_K} 到 {MAX_TOP_K}。\n"
        f"6) chunk_top_k 必须是整数，范围 {MIN_CHUNK_TOP_K} 到 {MAX_CHUNK_TOP_K}。\n"
        "7) 输出字段数量必须保持一致，顺序一一对应。\n\n"
        f"原问题：{q}"
    )
    obj: SubQuestionQueryPlan = llm_subquestion_plan_struct.invoke(prompt)
    normalized = _normalize_plan_lengths(
        list(obj.sub_questions),
        list(obj.query_modes),
        list(obj.query_topks),
        list(obj.query_chunk_topks),
        fallback_question=q,
    )
    sub_questions: List[Dict[str, Any]] = []
    for idx, (sub_question, mode, top_k, chunk_top_k) in enumerate(
        zip(
            normalized.get("sub_questions", []),
            normalized.get("query_modes", []),
            normalized.get("query_topks", []),
            normalized.get("query_chunk_topks", []),
        ),
        start=1,
    ):
        sub_questions.append(
            {
                "id": f"sq{idx}",
                "question": str(sub_question),
                "used_question": str(sub_question),
                "query_mode": _normalize_mode(str(mode)),
                "top_k": _clamp_topk(top_k),
                "chunk_top_k": _clamp_chunk_topk(chunk_top_k),
                "target_subjects": [],
                "route_reason": "",
                "ranked_subjects": [],
                "sufficient": "unknown",
                "judge_reason": "",
                "rewritten_question": "",
            }
        )
    result = {
        "requested_mode": requested_mode,
        "detected_complexity": "complex",
        "question_complexity": "complex",
        "effective_strategy": "deep",
        "planning_reason": (
            "用户显式选择 deepsearch，先全局拆分子问题，再对子问题单独路由知识库"
        ),
        "sub_questions": sub_questions,
        **_blank_subquery_plan_state(),
    }
    _debug_print_subquery_plan("global deepsearch 查询规划结果", result)
    return result


def attach_subquestion_routes(
    state: State,
    routed_subjects: Dict[str, Dict[str, Any]],
) -> dict:
    allowed_subject_ids = _normalize_subject_ids(state.get("allowed_subject_ids", []))
    default_subjects = allowed_subject_ids[:1]
    updated: List[Dict[str, Any]] = []

    for raw_item in state.get("sub_questions", []):
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        route = routed_subjects.get(str(item.get("id", "")), {}) or {}
        target_subjects = _normalize_subject_ids(route.get("target_subjects", []))
        primary_subject = str(route.get("primary_subject", "") or "").strip()
        if primary_subject:
            target_subjects = [primary_subject] + target_subjects
        target_subjects = _normalize_subject_ids(target_subjects)
        if allowed_subject_ids:
            target_subjects = [
                subject_id
                for subject_id in target_subjects
                if subject_id in allowed_subject_ids
            ]
        if not target_subjects:
            target_subjects = default_subjects[:]

        item["target_subjects"] = target_subjects[:3]
        item["route_reason"] = str(route.get("reason", "") or "").strip()
        ranked_subjects = route.get("ranked_subjects", [])
        item["ranked_subjects"] = (
            list(ranked_subjects) if isinstance(ranked_subjects, list) else []
        )
        updated.append(item)

    return {"sub_questions": updated}


def build_subquery_tasks(state: State) -> dict:
    allowed_subject_ids = _normalize_subject_ids(state.get("allowed_subject_ids", []))
    tasks: List[Dict[str, Any]] = []

    for raw_item in state.get("sub_questions", []):
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        question = str(item.get("question", "") or "").strip()
        if not question:
            continue
        sub_question_id = str(item.get("id", "") or "").strip() or question
        used_question = str(item.get("used_question", "") or "").strip() or question
        target_subjects = _normalize_subject_ids(item.get("target_subjects", []))
        if not target_subjects:
            target_subjects = allowed_subject_ids[:1]

        mode = _normalize_mode(str(item.get("query_mode", "hybrid")))
        top_k = _clamp_topk(item.get("top_k", DEFAULT_TOP_K))
        chunk_top_k = _clamp_chunk_topk(item.get("chunk_top_k", DEFAULT_CHUNK_TOP_K))

        for subject_id in target_subjects[:3]:
            tasks.append(
                {
                    "task_id": f"{sub_question_id}::{subject_id}",
                    "sub_question_id": sub_question_id,
                    "subject_id": subject_id,
                    "question": question,
                    "used_question": used_question,
                    "mode": mode,
                    "top_k": top_k,
                    "chunk_top_k": chunk_top_k,
                }
            )

    return {"subquery_tasks": tasks}


async def _query_subquery_task(state: State, task: Dict[str, Any]) -> Dict[str, str]:
    t0 = time.perf_counter()
    question = str(task.get("question", "") or "").strip() or "子问题缺失"
    used_question = str(task.get("used_question", "") or "").strip() or question
    subject_id = str(task.get("subject_id", "") or "").strip()
    mode = _normalize_mode(str(task.get("mode", "hybrid")))
    top_k = _clamp_topk(task.get("top_k", DEFAULT_TOP_K))
    chunk_top_k = _clamp_chunk_topk(task.get("chunk_top_k", DEFAULT_CHUNK_TOP_K))
    subject_working_dirs = state.get("subject_working_dirs", {}) or {}
    working_dir = None
    if isinstance(subject_working_dirs, dict):
        working_dir = subject_working_dirs.get(subject_id)

    if not subject_id:
        row = build_query_result_row(
            question_id=str(task.get("task_id", "missing_subject") or "missing_subject"),
            question=question,
            used_question=used_question,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            answer="子问题未命中任何知识库。",
            query_status="failure",
            query_message="缺少 subject_id",
            query_failure_reason="missing_subject",
            sufficient="False",
            judge_reason="子问题未命中任何知识库",
            rewritten_question="",
            retries=state.get("query_attempt", 0),
            trace="missing_subject",
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )
        row.update(
            {
                "task_id": str(task.get("task_id", "")),
                "sub_question_id": str(task.get("sub_question_id", "")),
                "subject_id": "",
            }
        )
        return row

    if subject_working_dirs and working_dir is None:
        row = build_query_result_row(
            question_id=str(task.get("task_id", "missing_working_dir") or "missing_working_dir"),
            question=question,
            used_question=used_question,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            answer=f"知识库缺失: {subject_id}",
            query_status="failure",
            query_message=f"缺少 {subject_id} 对应 working_dir",
            query_failure_reason="missing_working_dir",
            sufficient="False",
            judge_reason="知识库 working_dir 缺失",
            rewritten_question="",
            retries=state.get("query_attempt", 0),
            trace="missing_working_dir",
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )
        row.update(
            {
                "task_id": str(task.get("task_id", "")),
                "sub_question_id": str(task.get("sub_question_id", "")),
                "subject_id": subject_id,
            }
        )
        return row

    rag_obj = await get_rag(working_dir)
    query_resp = await rag_obj.aquery_llm(
        used_question,
        param=QueryParam(mode=mode, top_k=top_k, chunk_top_k=chunk_top_k),
    )
    status, message, failure_reason, answer = extract_query_response_fields(query_resp)
    row = build_query_result_row(
        question_id=str(task.get("task_id", "") or f"{task.get('sub_question_id', '')}::{subject_id}"),
        question=question,
        used_question=used_question,
        mode=mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        answer=answer,
        query_status=status,
        query_message=message,
        query_failure_reason=failure_reason,
        sufficient="unknown",
        judge_reason="",
        rewritten_question="",
        retries=state.get("query_attempt", 0),
        trace=(
            f"attempt={state.get('query_attempt', 0) + 1},subject={subject_id},"
            f"mode={mode},top_k={top_k},chunk_top_k={chunk_top_k}"
        ),
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )
    row.update(
        {
            "task_id": str(task.get("task_id", "")),
            "sub_question_id": str(task.get("sub_question_id", "")),
            "subject_id": subject_id,
        }
    )
    return row


async def query_subquestion_tasks(state: State) -> dict:
    t0 = time.perf_counter()
    tasks = list(state.get("subquery_tasks", []))
    if not tasks:
        return {"subquery_results": [], "query_total_ms": "0"}

    raw_results = await asyncio.gather(
        *[_query_subquery_task(state, task) for task in tasks],
        return_exceptions=True,
    )

    results: List[Dict[str, str]] = []
    for task, item in zip(tasks, raw_results):
        if isinstance(item, Exception):
            err_msg = normalize_exception_message(item)
            row = build_query_result_row(
                question_id=str(task.get("task_id", "") or "task_error"),
                question=str(task.get("question", "") or "子问题缺失"),
                used_question=str(task.get("used_question", "") or task.get("question", "")),
                mode=str(task.get("mode", "hybrid")),
                top_k=task.get("top_k", DEFAULT_TOP_K),
                chunk_top_k=task.get("chunk_top_k", DEFAULT_CHUNK_TOP_K),
                answer=f"查询失败: {err_msg}",
                query_status="failure",
                query_message=f"query 异常: {err_msg}",
                query_failure_reason="exception",
                sufficient="False",
                judge_reason="查询异常",
                rewritten_question="",
                retries=state.get("query_attempt", 0),
                trace="error",
                elapsed_ms=0,
            )
            row.update(
                {
                    "task_id": str(task.get("task_id", "")),
                    "sub_question_id": str(task.get("sub_question_id", "")),
                    "subject_id": str(task.get("subject_id", "")),
                }
            )
            results.append(row)
        else:
            results.append(item)

    total_ms = int((time.perf_counter() - t0) * 1000)
    if DEBUG:
        print(f"[DEBUG] subquery query 节点总耗时: {total_ms} ms")
    return {"subquery_results": results, "query_total_ms": str(total_ms)}


def _group_subquery_results(
    subquery_results: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for item in subquery_results:
        sub_question_id = str(item.get("sub_question_id", "") or "").strip()
        grouped.setdefault(sub_question_id, []).append(item)
    return grouped


async def judge_subquestion_results(state: State) -> dict:
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        return {"needs_retry": False, "insufficient_subquestion_ids": []}

    grouped = _group_subquery_results(list(state.get("subquery_results", [])))
    updated: List[Dict[str, Any]] = []
    insufficient_ids: List[str] = []
    llm_tasks = []
    llm_task_ids: List[str] = []

    for raw_item in sub_questions:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        sub_question_id = str(item.get("id", "") or "").strip()
        related_results = grouped.get(sub_question_id, [])

        if not related_results:
            item["sufficient"] = "False"
            item["judge_reason"] = "子问题未生成可执行检索任务"
            item["rewritten_question"] = ""
            insufficient_ids.append(sub_question_id)
            updated.append(item)
            continue

        combined_answer_parts: List[str] = []
        messages: List[str] = []
        failure_reasons: List[str] = []
        statuses: List[str] = []

        for result in related_results:
            subject_id = str(result.get("subject_id", "") or "").strip() or "unknown"
            answer = str(result.get("answer", "") or "").strip()
            status = str(result.get("query_status", "") or "").strip().lower()
            message = str(result.get("query_message", "") or "").strip()
            failure_reason = str(result.get("query_failure_reason", "") or "").strip()
            statuses.append(status)
            if message and message not in messages:
                messages.append(message)
            if failure_reason and failure_reason not in failure_reasons:
                failure_reasons.append(failure_reason)
            if answer:
                combined_answer_parts.append(f"[{subject_id}]\n{answer}")

        combined_answer = "\n\n".join(combined_answer_parts).strip()
        combined_status = "unknown"
        if any(status == "success" for status in statuses):
            combined_status = "success"
        elif statuses and all(status == "failure" for status in statuses):
            combined_status = "failure"
        elif statuses:
            combined_status = statuses[0]

        synthetic_item = {
            "query_status": combined_status,
            "query_message": "；".join(messages),
            "query_failure_reason": "；".join(failure_reasons),
            "answer": combined_answer,
        }
        rule_ok, rule_reason = _rule_based_evidence(synthetic_item)
        if rule_ok is None:
            llm_tasks.append(
                _judge_evidence(
                    str(item.get("question", "") or ""),
                    str(item.get("used_question", "") or item.get("question", "") or ""),
                    combined_answer,
                )
            )
            llm_task_ids.append(sub_question_id)
            item["sufficient"] = "pending"
            item["judge_reason"] = rule_reason
            item["rewritten_question"] = ""
        else:
            item["sufficient"] = str(rule_ok)
            item["judge_reason"] = rule_reason
            item["rewritten_question"] = ""
            if not rule_ok:
                insufficient_ids.append(sub_question_id)
        updated.append(item)

    if llm_tasks:
        raw_judges = await asyncio.gather(*llm_tasks, return_exceptions=True)
        rows_by_id = {
            str(item.get("id", "") or "").strip(): item
            for item in updated
            if isinstance(item, dict)
        }
        for sub_question_id, judge_item in zip(llm_task_ids, raw_judges):
            row = rows_by_id.get(sub_question_id)
            if row is None:
                continue
            if isinstance(judge_item, Exception):
                row["sufficient"] = "False"
                row["judge_reason"] = f"评估异常: {judge_item}"
                row["rewritten_question"] = ""
                insufficient_ids.append(sub_question_id)
            else:
                sufficient = bool(judge_item.sufficient)
                row["sufficient"] = str(sufficient)
                row["judge_reason"] = judge_item.reason.strip()
                row["rewritten_question"] = judge_item.rewritten_question.strip()
                if not sufficient:
                    insufficient_ids.append(sub_question_id)

    attempt = int(state.get("query_attempt", 0))
    allowed_retry = _allowed_retry_budget(state)
    needs_retry = len(insufficient_ids) > 0 and attempt < allowed_retry
    if DEBUG:
        print(
            f"[DEBUG] subquery judge 结果: insufficient={insufficient_ids}, "
            f"attempt={attempt}, allowed_retry={allowed_retry}, needs_retry={needs_retry}"
        )
    return {
        "sub_questions": updated,
        "insufficient_subquestion_ids": insufficient_ids,
        "needs_retry": needs_retry,
    }


def prepare_subquestion_retry_plan(state: State) -> dict:
    insufficient_ids = set(_normalize_subject_ids(state.get("insufficient_subquestion_ids", [])))
    allowed_subject_ids = _normalize_subject_ids(state.get("allowed_subject_ids", []))
    sub_questions = [
        dict(item) for item in state.get("sub_questions", []) if isinstance(item, dict)
    ]
    scored = sorted(
        [item for item in sub_questions if str(item.get("id", "") or "") in insufficient_ids],
        key=lambda item: len(str(item.get("question", "") or "")),
        reverse=True,
    )
    retry_target_ids = {
        str(item.get("id", "") or "")
        for item in scored[: max(1, RETRY_MAX_ITEMS)]
    }

    updated: List[Dict[str, Any]] = []
    for item in sub_questions:
        current = dict(item)
        sub_question_id = str(current.get("id", "") or "").strip()
        if sub_question_id in retry_target_ids:
            rewritten = str(current.get("rewritten_question", "") or "").strip()
            if rewritten:
                current["used_question"] = rewritten

            current["query_mode"] = _next_mode(str(current.get("query_mode", "hybrid")))
            current["top_k"] = _clamp_topk(
                _clamp_topk(current.get("top_k", DEFAULT_TOP_K)) + TOPK_RETRY_STEP
            )
            current["chunk_top_k"] = _clamp_chunk_topk(
                _clamp_chunk_topk(current.get("chunk_top_k", DEFAULT_CHUNK_TOP_K))
                + CHUNK_TOPK_RETRY_STEP
            )

            target_subjects = _normalize_subject_ids(current.get("target_subjects", []))
            if allowed_subject_ids:
                max_targets = min(2, len(allowed_subject_ids))
                while len(target_subjects) < max_targets:
                    next_subject = next(
                        (
                            subject_id
                            for subject_id in allowed_subject_ids
                            if subject_id not in target_subjects
                        ),
                        "",
                    )
                    if not next_subject:
                        break
                    target_subjects.append(next_subject)
                current["target_subjects"] = target_subjects[: max_targets]

            current["sufficient"] = "unknown"
            current["judge_reason"] = ""
            current["rewritten_question"] = ""

            if DEBUG:
                print(
                    "[DEBUG] subquery retry 规划: "
                    f"id={sub_question_id}, subjects={current.get('target_subjects', [])}, "
                    f"mode={current.get('query_mode', '')}, "
                    f"top_k={current.get('top_k', '')}, "
                    f"chunk_top_k={current.get('chunk_top_k', '')}, "
                    f"used_question={current.get('used_question', '')}"
                )

        updated.append(current)

    return {
        "sub_questions": updated,
        "subquery_tasks": [],
        "subquery_results": [],
        "query_attempt": int(state.get("query_attempt", 0)) + 1,
        "needs_retry": False,
        "insufficient_subquestion_ids": [],
    }


async def _judge_evidence(original_q: str, used_q: str, answer: str) -> EvidenceCheck:
    prompt = (
        "你是检索质量评估器。请判断当前检索结果是否足够回答子问题。\n"
        "标准：相关性、信息完整性、是否缺关键事实。\n"
        "若不足，请给一个更容易检索到答案的改写问题。\n\n"
        f"原子问题：{original_q}\n"
        f"本轮查询问题：{used_q}\n"
        f"检索结果：{answer}"
    )
    return await llm_evidence_struct.ainvoke(prompt)


def _allowed_retry_budget(state: State) -> int:
    if _effective_strategy(state) == "simple":
        return max(0, SIMPLE_MAX_RETRY)
    return max(0, COMPLEX_MAX_RETRY)


def _rule_based_evidence(item: Dict[str, str]) -> tuple[bool | None, str]:
    status = str(item.get("query_status", "")).strip().lower()
    message = str(item.get("query_message", "")).strip()
    failure_reason = str(item.get("query_failure_reason", "")).strip()
    if status == "failure":
        reason = f"结构化状态失败: {message}" if message else "结构化状态失败"
        if failure_reason:
            reason += f" (failure_reason={failure_reason})"
        return (False, reason)

    text = str(item.get("answer", "")).strip()
    if status == "success":
        if not text:
            return (False, "结构化状态成功但答案为空")
        if len(text) >= 260:
            return (True, "结构化状态成功且答案长度充足")
        return (None, "结构化状态成功，需 LLM 进一步评估")

    # Fallback: only when structured status is missing/unknown
    lower = text.lower()
    if not text:
        return (False, "无结构化状态且答案为空")
    bad_markers = [
        "查询失败",
        "未找到",
        "没有找到",
        "检索结果为空",
        "无法回答",
        "none",
        "not found",
    ]
    if any(m in text for m in bad_markers) or any(m in lower for m in bad_markers):
        return (False, "无结构化状态且命中失败关键词")
    if len(text) >= 260:
        return (True, "无结构化状态但答案长度充足，先视为充分")
    return (None, "无结构化状态，需 LLM 进一步评估")


def _final_prompt_evidence_status(item: Dict[str, str]) -> str:
    sufficient = str(item.get("sufficient", "")).strip()
    if sufficient == "True":
        return "充分"
    if sufficient == "False":
        return "不足"
    return "待定"


def _build_final_answer_prompt_from_subquery_state(state: State) -> str:
    sub_questions = [
        dict(item) for item in state.get("sub_questions", []) if isinstance(item, dict)
    ]
    subquery_results = [
        dict(item) for item in state.get("subquery_results", []) if isinstance(item, dict)
    ]
    grouped = _group_subquery_results(subquery_results)

    lines = [f"原问题：{state['question']}"]
    response_language = str(state.get("response_language", "") or "").strip().lower()
    for i, item in enumerate(sub_questions, start=1):
        question = str(item.get("question", "") or "").strip() or "子问题缺失"
        used_question = str(item.get("used_question", "") or "").strip()
        target_subjects = _normalize_subject_ids(item.get("target_subjects", []))
        sub_question_id = str(item.get("id", "") or "").strip()

        lines.append(f"子问题{i}：{question}")
        if used_question and used_question != question:
            lines.append(f"子问题{i}改写查询：{used_question}")
        if target_subjects:
            lines.append(f"子问题{i}目标知识库：{', '.join(target_subjects)}")
        lines.append(f"子问题{i}证据充分性：{_final_prompt_evidence_status(item)}")

        judge_reason = str(item.get("judge_reason", "")).strip()
        if judge_reason:
            lines.append(f"子问题{i}评估说明：{judge_reason}")

        related_results = grouped.get(sub_question_id, [])
        if not related_results:
            lines.append(f"子问题{i}检索结果：无")
            continue

        for j, result in enumerate(related_results, start=1):
            subject_id = str(result.get("subject_id", "") or "").strip() or "unknown"
            query_status = str(result.get("query_status", "")).strip()
            query_failure_reason = str(result.get("query_failure_reason", "")).strip()
            lines.append(f"子问题{i}·证据{j}知识库：{subject_id}")
            if query_status and query_status.lower() != "success":
                lines.append(f"子问题{i}·证据{j}检索状态：{query_status}")
            if query_failure_reason:
                lines.append(f"子问题{i}·证据{j}失败原因：{query_failure_reason}")
            lines.append(f"子问题{i}·证据{j}检索结果：{result.get('answer', '')}")

    lines.append(f"query重试轮数：{state.get('query_attempt', 0)}")
    language_instruction = (
        "Answer entirely in English.\n"
        if response_language == "en"
        else "请全程使用中文回答，必要术语可保留英文缩写，但主体说明必须是中文。\n"
    )
    prompt = (
        "你是一个严谨的问答助手。请只基于给定子问题及其检索证据回答原问题。\n"
        "输出要求：\n"
        "1) 使用 Markdown 小节标题：### 结论 / ### 关键依据 / ### 不确定点 / ### 建议（可选）\n"
        "2) 先给结论，再给关键依据；\n"
        "3) 若某条证据不足或检索失败，必须在不确定点中明确说明；\n"
        "4) 不要编造检索结果中没有的信息；\n"
        "5) 表达尽量紧凑，避免重复复述子问题。\n"
        f"6) {language_instruction.strip()}\n\n"
        + "\n".join(lines)
    )
    return prompt


def build_final_answer_prompt(state: State) -> str:
    return _build_final_answer_prompt_from_subquery_state(state)
