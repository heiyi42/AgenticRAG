import argparse
import asyncio
import os
import time
from typing import Callable, Literal

from agenticRAG.agentic_answer import answer_question
from agenticRAG.cli_utils import (
    build_memory_factory,
    is_clear_command,
    is_exit_command,
    print_deep_result,
    reset_thread_after_clear,
)
from agenticRAG.agentic_config import DEFAULT_THREAD_ID
from agenticRAG.agentic_runtime import llm_complexity_struct
from agenticRAG.instant_answer import answer_instant, answer_instant_stream
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

AUTO_ENABLE_SUMMARY_MEMORY = os.getenv("AUTO_ENABLE_SUMMARY_MEMORY", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AUTO_SUMMARY_TRIGGER_TOKENS = int(os.getenv("AUTO_SUMMARY_TRIGGER_TOKENS", "2000"))
AUTO_MAX_TURNS_BEFORE_SUMMARY = int(os.getenv("AUTO_MAX_TURNS_BEFORE_SUMMARY", "4"))
AUTO_KEEP_RECENT_TURNS = int(os.getenv("AUTO_KEEP_RECENT_TURNS", "1"))


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto GraphRAG CLI")
    parser.add_argument("question", nargs="?", help="要提问的问题")
    parser.add_argument(
        "--thread-id",
        default=DEFAULT_THREAD_ID,
        help="会话ID（相同thread_id会复用会话状态）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=AUTO_QUERY_TIMEOUT_S,
        help="单次问题总超时秒数（默认 180）",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="深度链路时打印子问题/检索细节",
    )
    parser.add_argument(
        "--no-summary-memory",
        action="store_true",
        help="关闭总结式短期记忆",
    )
    parser.add_argument(
        "--route-threshold",
        type=float,
        default=AUTO_ROUTE_CONFIDENCE_THRESHOLD,
        help="高置信complex直达deep阈值(0~1,默认 0.85)",
    )
    parser.add_argument(
        "--route-ratio",
        type=float,
        default=AUTO_ROUTE_BUDGET_RATIO,
        help="路由预算占总超时的比例(默认 0.10)",
    )
    parser.add_argument(
        "--instant-ratio",
        type=float,
        default=AUTO_INSTANT_BUDGET_RATIO,
        help="instant预算占总超时的比例(默认 0.35)",
    )
    return parser.parse_args()


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


async def _ask_instant(
    question: str,
    thread_id: str,
    timeout_s: int,
    working_dir: str | None = None,
) -> dict:
    return await asyncio.wait_for(
        answer_instant(question, thread_id=thread_id, working_dir=working_dir),
        timeout=_safe_timeout(timeout_s),
    )


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


async def _ask_deep(
    question: str,
    thread_id: str,
    timeout_s: int,
    working_dir: str | None = None,
) -> dict:
    return await asyncio.wait_for(
        answer_question(question, thread_id=thread_id, working_dir=working_dir),
        timeout=_safe_timeout(timeout_s),
    )


async def _run_deep_with_remaining_budget(
    question: str,
    thread_id: str,
    started: float,
    total_timeout_s: int,
    working_dir: str | None = None,
) -> tuple[dict | None, str]:
    remaining = _remaining_seconds(started, total_timeout_s)
    if remaining <= 0:
        return None, "总超时预算已耗尽，无法继续 deep 链路。"

    deep_budget = min(remaining, AUTO_DEEP_BUDGET_S)
    try:
        deep_result = await _ask_deep(
            question,
            thread_id,
            deep_budget,
            working_dir=working_dir,
        )
        return deep_result, ""
    except asyncio.TimeoutError:
        return None, f"deep 链路超时（>{deep_budget}s）。"
    except Exception as e:
        return None, f"deep 链路失败: {e}"


async def _run() -> None:
    args = _parse_args()
    route_threshold = _clamp_confidence(args.route_threshold)
    route_ratio = _clamp_ratio(args.route_ratio, AUTO_ROUTE_BUDGET_RATIO)
    instant_ratio = _clamp_ratio(args.instant_ratio, AUTO_INSTANT_BUDGET_RATIO)
    current_thread_id = args.thread_id
    pending_question = args.question
    use_summary_memory = AUTO_ENABLE_SUMMARY_MEMORY and not args.no_summary_memory
    memory_for_thread = build_memory_factory(
        use_summary_memory=use_summary_memory,
        summary_trigger_tokens=AUTO_SUMMARY_TRIGGER_TOKENS,
        max_turns_before_summary=AUTO_MAX_TURNS_BEFORE_SUMMARY,
        keep_recent_turns=AUTO_KEEP_RECENT_TURNS,
    )
    memory = memory_for_thread(current_thread_id)

    print(
        f"Auto 模式已启动: thread_id={current_thread_id}, timeout={args.timeout}s, "
        f"summary_memory={use_summary_memory}, route_threshold={route_threshold:.2f}, "
        f"route_ratio={route_ratio:.2f}, instant_ratio={instant_ratio:.2f}"
    )
    print("输入 quit/exit/q 退出，输入 clear 清空会话记忆。")

    while True:
        if pending_question is not None:
            question = pending_question.strip()
            pending_question = None
        else:
            question = input("\n请输入问题: ").strip()

        if not question:
            continue
        if is_exit_command(question):
            print("已退出。")
            return
        if is_clear_command(question):
            current_thread_id, memory = reset_thread_after_clear(
                base_thread_id=args.thread_id,
                current_thread_id=current_thread_id,
                use_summary_memory=use_summary_memory,
                memory_for_thread=memory_for_thread,
            )
            print(f"会话记忆已清空。新thread_id={current_thread_id}")
            continue

        ask_question = (
            memory.build_augmented_question(question)
            if memory is not None
            else question
        )
        started = time.perf_counter()

        complexity, confidence, route_reason = await _route_complexity(
            ask_question, args.timeout, route_ratio
        )
        high_conf_complex = (
            complexity == "complex"
            and confidence >= route_threshold
        )
        route_chain = "deep" if high_conf_complex else "instant"
        route_policy = (
            "high-confidence complex -> deep"
            if high_conf_complex
            else "progressive -> instant first"
        )
        upgraded = False
        upgrade_reason = ""
        final_answer_text = ""

        print(
            f"\nAuto路由: complexity={complexity} | confidence={confidence:.2f} | "
            f"chain={route_chain} | policy={route_policy} | reason={route_reason}"
        )

        if route_chain == "instant":
            remaining = _remaining_seconds(started, args.timeout)
            if remaining <= 0:
                print("总超时预算已耗尽。")
                continue

            base_instant_budget = _budget_by_ratio(
                total_timeout_s=args.timeout,
                ratio=instant_ratio,
                floor_s=5,
                hard_cap_s=AUTO_INSTANT_BUDGET_S,
            )
            reserve_for_deep = min(AUTO_DEEP_MIN_RESERVE_S, max(0, remaining - 1))
            instant_budget = max(
                1, min(base_instant_budget, max(1, remaining - reserve_for_deep))
            )
            try:
                instant_result = await _ask_instant(
                    ask_question,
                    current_thread_id,
                    instant_budget,
                )
            except asyncio.TimeoutError:
                print(f"\ninstant 链路超时（>{instant_budget}s），尝试升级到 deep。")
                deep_result, deep_error = await _run_deep_with_remaining_budget(
                    ask_question, current_thread_id, started, args.timeout
                )
                if deep_result is None:
                    print(deep_error)
                    continue
                upgraded = True
                upgrade_reason = "instant timeout"
                final_answer_text = deep_result.get("final_answer", "")
                print_deep_result(deep_result, show_details=args.details)
            except Exception as e:
                print(f"\ninstant 链路失败: {e}")
                continue
            else:
                heur_sufficient, heur_reason = _instant_heuristic_assessment(instant_result)
                review_sufficient: bool | None = None
                review_reason = "未执行LLM评审"

                review_budget = min(
                    AUTO_FAST_REVIEW_TIMEOUT_S,
                    AUTO_REVIEW_TIMEOUT_S,
                    max(0, _remaining_seconds(started, args.timeout) - 1),
                )
                if heur_sufficient is None and review_budget > 0:
                    should_review, review_gate_reason = _should_review_instant_answer(
                        answer=str(instant_result.get("answer", "")),
                        cross_subject=False,
                        route_confidence=confidence,
                        subject_confidence=1.0,
                    )
                    if should_review:
                        review_sufficient, review_reason = await _review_instant_answer(
                            question=question,
                            answer=str(instant_result.get("answer", "")),
                            timeout_s=review_budget,
                        )
                    else:
                        review_sufficient = True
                        review_reason = f"跳过LLM评审: {review_gate_reason}"
                elif heur_sufficient is None:
                    should_review, review_gate_reason = _should_review_instant_answer(
                        answer=str(instant_result.get("answer", "")),
                        cross_subject=False,
                        route_confidence=confidence,
                        subject_confidence=1.0,
                    )
                    if should_review:
                        review_reason = f"触发LLM评审但预算不足: {review_gate_reason}"
                    else:
                        review_sufficient = True
                        review_reason = f"跳过LLM评审: {review_gate_reason}"

                need_upgrade = False
                upgrade_reason_candidate = ""

                if heur_sufficient is False:
                    need_upgrade = True
                    upgrade_reason_candidate = f"heuristic不足: {heur_reason}"
                elif review_sufficient is False:
                    need_upgrade = True
                    upgrade_reason_candidate = review_reason
                elif review_sufficient is True:
                    need_upgrade = False
                elif heur_sufficient is True:
                    need_upgrade = False
                else:
                    need_upgrade = True
                    upgrade_reason_candidate = (
                        f"评审不可用且启发式不确定: heur={heur_reason}, review={review_reason}"
                    )

                if not need_upgrade:
                    final_answer_text = instant_result.get("answer", "")
                    print(
                        f"\n路由结果: instant | status={instant_result.get('query_status', '-')}"
                    )
                    print("\n回答：")
                    print(instant_result.get("answer", ""))
                    print(
                        f"\n耗时(ms): {instant_result.get('elapsed_ms', '0')}, "
                        f"status={instant_result.get('query_status', '-')}"
                    )
                    print(
                        f"评审: heuristic={heur_reason}; review={review_reason}"
                    )
                else:
                    upgraded = True
                    upgrade_reason = upgrade_reason_candidate
                    remaining = _remaining_seconds(started, args.timeout)
                    if remaining <= 0:
                        print(
                            f"\n自动升级触发但剩余时间不足，保留 instant 结果。原因: {upgrade_reason_candidate}"
                        )
                        final_answer_text = instant_result.get("answer", "")
                    else:
                        print(f"\n自动升级到 deep 链路。原因: {upgrade_reason_candidate}")
                        deep_result, deep_error = await _run_deep_with_remaining_budget(
                            ask_question, current_thread_id, started, args.timeout
                        )
                        if deep_result is None:
                            print(deep_error)
                            continue
                        final_answer_text = deep_result.get("final_answer", "")
                        print_deep_result(deep_result, show_details=args.details)
        else:
            deep_result, deep_error = await _run_deep_with_remaining_budget(
                ask_question, current_thread_id, started, args.timeout
            )
            if deep_result is None:
                print(deep_error)
                continue

            final_answer_text = deep_result.get("final_answer", "")
            print_deep_result(deep_result, show_details=args.details)

        if upgraded:
            print(f"\n升级信息: upgraded_to_deep=True, reason={upgrade_reason}")
        else:
            print("\n升级信息: upgraded_to_deep=False")

        if memory is not None:
            memory.update(question, final_answer_text)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
