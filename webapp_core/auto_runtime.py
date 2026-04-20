import asyncio
import os
import time
from typing import Callable, Literal

from agenticRAG.agentic_runtime import llm_complexity_struct
from agenticRAG.instant_answer import answer_instant_stream
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

AUTO_QUERY_TIMEOUT_S = int(os.getenv("AUTO_QUERY_TIMEOUT_S", "180"))
AUTO_ROUTE_TIMEOUT_S = int(os.getenv("AUTO_ROUTE_TIMEOUT_S", "20"))
AUTO_INSTANT_BUDGET_S = int(os.getenv("AUTO_INSTANT_BUDGET_S", "60"))
AUTO_DEEP_BUDGET_S = int(os.getenv("AUTO_DEEP_BUDGET_S", "120"))
AUTO_REVIEW_TIMEOUT_S = int(os.getenv("AUTO_REVIEW_TIMEOUT_S", "15"))
AUTO_FAST_REVIEW_TIMEOUT_S = int(os.getenv("AUTO_FAST_REVIEW_TIMEOUT_S", "5"))
AUTO_DEEP_MIN_RESERVE_S = int(os.getenv("AUTO_DEEP_MIN_RESERVE_S", "25"))


def _safe_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


AUTO_ROUTE_BUDGET_RATIO = _safe_env_float("AUTO_ROUTE_BUDGET_RATIO", 0.10)
AUTO_INSTANT_BUDGET_RATIO = _safe_env_float("AUTO_INSTANT_BUDGET_RATIO", 0.35)
AUTO_ROUTE_CONFIDENCE_THRESHOLD = _safe_env_float(
    "AUTO_ROUTE_CONFIDENCE_THRESHOLD", 0.85
)
AUTO_REVIEW_TRIGGER_CONFIDENCE = _safe_env_float(
    "AUTO_REVIEW_TRIGGER_CONFIDENCE", 0.72
)
AUTO_REVIEW_SHORT_ANSWER_CHARS = int(
    os.getenv("AUTO_REVIEW_SHORT_ANSWER_CHARS", "180")
)


class AutoRouteDecision(BaseModel):
    complexity: Literal["simple", "complex"] = Field(description="问题复杂度")
    confidence: float = Field(ge=0.0, le=1.0, description="复杂度判定置信度（0到1）")
    reason: str = Field(description="简短路由理由")


class InstantAnswerReview(BaseModel):
    sufficient: bool = Field(description="instant答案是否足够回答问题")
    reason: str = Field(description="判断理由")
    missing_points: str = Field(
        default="",
        description="若不足，缺失要点（可为空）",
    )


auto_router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_auto_route_struct = auto_router_llm.with_structured_output(AutoRouteDecision)
llm_instant_review_struct = auto_router_llm.with_structured_output(InstantAnswerReview)


def _safe_timeout(timeout_s: int) -> int:
    return max(1, int(timeout_s))


def _remaining_seconds(started: float, total_timeout_s: int) -> int:
    elapsed = int(time.perf_counter() - started)
    return max(0, int(total_timeout_s) - elapsed)


def _budget_by_ratio(
    total_timeout_s: int, ratio: float, floor_s: int, hard_cap_s: int
) -> int:
    raw = int(max(0.01, ratio) * max(1, total_timeout_s))
    return max(1, min(hard_cap_s, max(floor_s, raw)))


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_ratio(value: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return max(0.01, default)
    return max(0.01, min(0.95, v))


def _instant_heuristic_assessment(result: dict) -> tuple[bool | None, str]:
    status = str(result.get("query_status", "")).strip().lower()
    answer = str(result.get("answer", "")).strip()
    message = str(result.get("query_message", "")).strip()

    if status and status != "success":
        return False, f"instant query_status={status or 'unknown'}"
    if not answer:
        return False, "instant 空回答"

    bad_markers = [
        "查询失败",
        "未找到",
        "没有找到",
        "检索结果为空",
        "无法回答",
        "not found",
    ]
    lower = answer.lower()
    if any(m in answer for m in bad_markers) or any(m in lower for m in bad_markers):
        return False, "instant 命中失败关键词"
    if message and "失败" in message:
        return False, f"instant message={message}"
    if len(answer) >= 260:
        return True, "instant 答案长度充足"
    return None, "instant 启发式无法确定，需LLM评审"


def _contains_uncertainty_marker(answer: str) -> bool:
    text = str(answer or "").strip()
    if not text:
        return False

    lower = text.lower()
    zh_markers = [
        "不确定",
        "无法确定",
        "需要进一步",
        "建议进一步",
        "建议改走检索",
        "建议查阅",
        "取决于具体",
        "需结合具体",
        "信息不足",
    ]
    en_markers = [
        "not sure",
        "uncertain",
        "depends on",
        "need more context",
        "requires more context",
        "cannot determine",
        "may need",
    ]
    return any(marker in text for marker in zh_markers) or any(
        marker in lower for marker in en_markers
    )


def _should_review_instant_answer(
    answer: str,
    *,
    cross_subject: bool = False,
    route_confidence: float = 1.0,
    subject_confidence: float = 1.0,
) -> tuple[bool, str]:
    text = str(answer or "").strip()
    if not text:
        return True, "empty_answer"
    if cross_subject:
        return True, "cross_subject"
    if _contains_uncertainty_marker(text):
        return True, "uncertainty_marker"
    if len(text) < max(80, int(AUTO_REVIEW_SHORT_ANSWER_CHARS)):
        return True, f"answer_short<{int(AUTO_REVIEW_SHORT_ANSWER_CHARS)}"
    if _clamp_confidence(subject_confidence) < _clamp_confidence(
        AUTO_REVIEW_TRIGGER_CONFIDENCE
    ):
        return True, "subject_confidence_low"
    if _clamp_confidence(route_confidence) < _clamp_confidence(
        AUTO_REVIEW_TRIGGER_CONFIDENCE
    ):
        return True, "route_confidence_low"
    return False, "high_conf_single_subject_skip_review"


async def _route_complexity(
    question: str, timeout_s: int, route_ratio: float
) -> tuple[str, float, str]:
    route_budget = _budget_by_ratio(
        total_timeout_s=timeout_s,
        ratio=route_ratio,
        floor_s=3,
        hard_cap_s=AUTO_ROUTE_TIMEOUT_S,
    )
    prompt = (
        "你是 Auto 路由器。请判断问题复杂度，并给出置信度。\n"
        "标准：\n"
        "- simple: 单一事实/单一实体，直接检索通常可回答\n"
        "- complex: 需要多跳推理、跨人物关系、时间线或因果分析\n"
        "输出字段要求：\n"
        "- complexity: simple 或 complex\n"
        "- confidence: 0 到 1 的小数\n"
        "- reason: 简短理由\n\n"
        f"问题：{question}"
    )
    try:
        obj = await asyncio.wait_for(
            llm_auto_route_struct.ainvoke(prompt),
            timeout=_safe_timeout(route_budget),
        )
        complexity = obj.complexity if obj.complexity in {"simple", "complex"} else "complex"
        confidence = _clamp_confidence(obj.confidence)
        reason = str(getattr(obj, "reason", "") or "").strip()
        return complexity, confidence, reason or "无"
    except Exception as e:
        fallback_prompt = (
            "请判断这个问题是 simple 还是 complex。\n"
            "标准：\n"
            "- simple: 单一事实/单一实体，直接检索通常可回答\n"
            "- complex: 需要多跳推理、跨人物关系、时间线或因果分析\n"
            "只输出结构化字段。\n\n"
            f"问题：{question}"
        )
        try:
            obj = await asyncio.wait_for(
                llm_complexity_struct.ainvoke(fallback_prompt),
                timeout=_safe_timeout(max(2, route_budget // 2)),
            )
            complexity = (
                obj.complexity if obj.complexity in {"simple", "complex"} else "complex"
            )
            reason = str(getattr(obj, "reason", "") or "").strip()
            return complexity, 0.5, f"fallback路由: {reason or '无'}"
        except Exception as ee:
            return "complex", 0.0, f"路由失败，默认deep。error={e}; fallback_error={ee}"


async def _review_instant_answer(
    question: str, answer: str, timeout_s: int
) -> tuple[bool | None, str]:
    if timeout_s <= 0:
        return None, "评审预算不足"

    prompt = (
        "你是回答充分性评审器。请判断给定答案是否足够回答用户问题。\n"
        "标准：\n"
        "1) 是否直接回答了问题核心\n"
        "2) 是否缺失关键事实\n"
        "3) 是否存在明显回避/空泛描述\n"
        "输出结构化字段 sufficient/reason/missing_points。\n\n"
        f"用户问题：{question}\n\n"
        f"候选答案：{answer}"
    )
    try:
        obj = await asyncio.wait_for(
            llm_instant_review_struct.ainvoke(prompt),
            timeout=_safe_timeout(min(timeout_s, AUTO_REVIEW_TIMEOUT_S)),
        )
        if obj.sufficient:
            return True, f"LLM评审通过: {obj.reason.strip()}"
        missing = str(getattr(obj, "missing_points", "") or "").strip()
        if missing:
            return False, f"LLM评审不足: {obj.reason.strip()} | 缺失: {missing}"
        return False, f"LLM评审不足: {obj.reason.strip()}"
    except Exception as e:
        return None, f"LLM评审失败: {e}"


async def _ask_instant_stream(
    question: str,
    thread_id: str,
    timeout_s: int,
    working_dir: str | None = None,
    emit_text: Callable[[str], None] | None = None,
) -> dict:
    safe_timeout = _safe_timeout(timeout_s)
    started = time.perf_counter()
    result = await asyncio.wait_for(
        answer_instant_stream(question, thread_id=thread_id, working_dir=working_dir),
        timeout=safe_timeout,
    )
    answer = str(result.get("answer", "")).strip()
    response_iterator = result.get("response_iterator")
    if response_iterator is not None:
        streamed_parts: list[str] = []
        remaining = max(1, safe_timeout - int(time.perf_counter() - started))
        async with asyncio.timeout(remaining):
            async for chunk in response_iterator:
                text = str(chunk or "")
                if not text:
                    continue
                streamed_parts.append(text)
                if emit_text is not None:
                    emit_text(text)
        answer = "".join(streamed_parts).strip()
    result["answer"] = answer
    result["elapsed_ms"] = str(int((time.perf_counter() - started) * 1000))
    return result
