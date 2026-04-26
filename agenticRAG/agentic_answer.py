from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from agenticRAG.agentic_nodes import (
    attach_subquestion_routes,
    build_global_subquestion_plan,
    build_subquery_tasks,
    judge_subquestion_results,
    prepare_subquestion_retry_plan,
    query_subquestion_tasks,
)
from agenticRAG.agentic_runtime import use_rag_working_dir


async def run_question_plan_state(
    question: str,
    *,
    requested_mode: str = "deepsearch",
    working_dir: str | None = None,
    routing_question: str | None = None,
    allowed_subject_ids: list[str] | None = None,
    subject_working_dirs: dict[str, str] | None = None,
    response_language: str = "zh",
    route_subquestion_subjects: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if requested_mode != "deepsearch" or route_subquestion_subjects is None:
        raise ValueError(
            "run_question_plan_state only supports routed DeepSearch; "
            "pass requested_mode='deepsearch' and route_subquestion_subjects."
        )

    state: dict[str, Any] = {
        "question": question,
        "requested_mode": requested_mode,
        "response_language": response_language,
    }
    if isinstance(allowed_subject_ids, list):
        state["allowed_subject_ids"] = list(allowed_subject_ids)
    if isinstance(subject_working_dirs, dict):
        state["subject_working_dirs"] = dict(subject_working_dirs)

    with use_rag_working_dir(working_dir):
        state.update(await asyncio.to_thread(build_global_subquestion_plan, state))
        routed_subjects: dict[str, dict[str, Any]] = {}
        base_question = str(routing_question or question)
        for item in state.get("sub_questions", []):
            if not isinstance(item, dict):
                continue
            sub_question_id = str(item.get("id", "") or "").strip()
            sub_question = str(item.get("question", "") or "").strip()
            if not sub_question_id or not sub_question:
                continue
            routed_subjects[sub_question_id] = await route_subquestion_subjects(
                sub_question=sub_question,
                original_question=base_question,
            )
        state.update(
            await asyncio.to_thread(
                attach_subquestion_routes,
                state,
                routed_subjects,
            )
        )
        state.update(await asyncio.to_thread(build_subquery_tasks, state))
        while True:
            state.update(await query_subquestion_tasks(state))
            state.update(await judge_subquestion_results(state))
            if not bool(state.get("needs_retry")):
                break
            state.update(await asyncio.to_thread(prepare_subquestion_retry_plan, state))
            state.update(await asyncio.to_thread(build_subquery_tasks, state))
    return state
