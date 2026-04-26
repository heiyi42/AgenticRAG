from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Awaitable, Callable

from . import auto_runtime as auto

from agenticRAG.agentic_answer import run_question_plan_state
from agenticRAG.agentic_nodes import (
    attach_subquestion_routes,
    build_final_answer_prompt,
    build_global_subquestion_plan,
    build_subquery_tasks,
    judge_subquestion_results,
    prepare_subquestion_retry_plan,
    query_subquestion_tasks,
)
from agenticRAG.agentic_runtime import get_rag, llm, use_rag_working_dir
from agenticRAG.instant_answer import answer_instant, answer_instant_stream

from . import config as cfg


class ChatRetrievalSupportMixin:
    def _pick_problem_tutoring_subject(
        self,
        *,
        subject_route: dict[str, Any] | None,
        tutoring_candidate: dict[str, Any] | None,
    ) -> str:
        route = subject_route or {}
        requested = [
            subject_id
            for subject_id in route.get("requested_subjects", [])
            if subject_id in self.subject_catalog
        ]
        if requested:
            return requested[0]

        analysis = (tutoring_candidate or {}).get("analysis") or {}
        candidate_subject = str(analysis.get("subject_id", "") or "").strip()
        if candidate_subject in self.subject_catalog:
            return candidate_subject

        primary_subject = str(route.get("primary_subject", "") or "").strip()
        if primary_subject in self.subject_catalog:
            return primary_subject

        ranked = list(route.get("ranked", []) or [])
        for item in ranked:
            if isinstance(item, (list, tuple)) and item:
                subject_id = str(item[0])
            elif isinstance(item, dict):
                subject_id = str(item.get("subject", "") or "")
            else:
                subject_id = ""
            if subject_id in self.subject_catalog:
                return subject_id

        return "C_program"

    async def _run_problem_tutoring_stream(
        self,
        *,
        user_question: str,
        augmented_question: str,
        mode: str,
        timeout_s: int,
        subject_route: dict[str, Any] | None,
        response_language: str,
        tutoring_candidate: dict[str, Any] | None = None,
        emit_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        subject_id = self._pick_problem_tutoring_subject(
            subject_route=subject_route,
            tutoring_candidate=tutoring_candidate,
        )
        prep_timeout = max(
            3,
            min(
                int(timeout_s),
                int(cfg.WEB_PROBLEM_TUTORING_PREP_TIMEOUT_S),
            ),
        )
        prepared = await self.problem_tutoring_service.prepare(
            user_question=user_question,
            augmented_question=augmented_question,
            subject_id_hint=subject_id,
            working_dir=self._subject_working_dir(subject_id),
            mode=mode,
            response_language=response_language,
            timeout_s=prep_timeout,
        )
        elapsed_s = int(time.perf_counter() - started)
        final_timeout = max(1, int(timeout_s) - elapsed_s)
        answer = await self._stream_llm_text(
            llm_client=auto.auto_router_llm,
            prompt=str(prepared.get("prompt", "")),
            timeout_s=final_timeout,
            emit_text=emit_text,
        )
        total_ms = int((time.perf_counter() - started) * 1000)
        solver_result = prepared.get("solver_result") or {}
        return {
            "mode_used": mode,
            "answer": answer,
            "request_kind": "problem_tutoring",
            "message_details": prepared.get("learning_outline"),
            "route": {
                "chain": "problem_tutoring",
                "reason": str((tutoring_candidate or {}).get("trigger") or "auto"),
                "subject": subject_id,
                "template_id": str((prepared.get("template") or {}).get("id", "")),
                "problem_type": str((prepared.get("analysis") or {}).get("problem_type", "")),
                "solver_status": str((solver_result or {}).get("status", "")),
                "solver": str((solver_result or {}).get("solver", "")),
            },
            "subject_route": self._build_subject_route_meta(subject_route or {}),
            "upgraded": False,
            "upgrade_reason": "",
            "instant_review": None,
            "elapsed_ms": str(total_ms),
            "raw": {"problem_tutoring": prepared},
        }

    def _build_subject_synthesis_review_prompt(
        self,
        *,
        user_question: str,
        mode: str,
        subject_answers: list[dict[str, str]],
        response_language: str = "zh",
    ) -> str:
        materials = []
        for item in subject_answers:
            materials.append(
                f"学科：{self._subject_label(item['subject_id'])}\n"
                f"回答：{item['answer']}"
            )
        return (
            "你是跨学科答案整合与充分性评审器。请先整合多个课程知识库返回的候选答案，"
            "再判断整合后的答案是否足以回答用户问题。\n"
            "要求：\n"
            "1) 聚合相同结论，去掉重复。\n"
            "2) 若不同学科视角互补，要明确区分层次。\n"
            "3) 不要捏造未在候选答案中出现的课程细节。\n"
            "4) 若答案仍缺关键点，sufficient 必须为 false。\n"
            "5) 输出必须严格遵循下面格式，不要添加额外前言：\n"
            "SUFFICIENT: true 或 false\n"
            "REASON: 一句话说明是否足够\n"
            "ANSWER:\n"
            "<最终 Markdown 回答>\n\n"
            f"回答语言要求：{self._response_language_instruction(response_language)}\n\n"
            f"当前模式：{mode}\n"
            f"用户问题：{user_question}\n\n"
            "候选答案：\n"
            f"{chr(10).join(materials)}"
        )

    @staticmethod
    def _parse_subject_synthesis_review_response(raw_text: str) -> dict[str, Any]:
        text = str(raw_text or "").strip()
        pattern = re.compile(
            r"(?is)SUFFICIENT\s*:\s*(true|false)\s*[\r\n]+"
            r"REASON\s*:\s*(.*?)\s*[\r\n]+"
            r"ANSWER\s*:\s*"
        )
        match = pattern.search(text)
        if match is None:
            return {
                "answer": text,
                "sufficient": None,
                "reason": "跨学科整合评审输出解析失败",
                "parsed": False,
            }
        answer = text[match.end() :].strip()
        return {
            "answer": answer,
            "sufficient": match.group(1).strip().lower() == "true",
            "reason": str(match.group(2) or "").strip() or "无",
            "parsed": True,
        }

    async def _stream_llm_text(
        self,
        *,
        llm_client: Any,
        prompt: str,
        timeout_s: int,
        emit_text: Callable[[str], None] | None = None,
        flush_chars: int = 16,
    ) -> str:
        safe_timeout = max(1, int(timeout_s))
        chunks: list[str] = []
        pending = ""

        async def _flush(force: bool = False) -> None:
            nonlocal pending
            if not pending:
                return
            if not force and len(pending) < max(1, int(flush_chars)):
                return
            if emit_text is not None:
                emit_text(pending)
            pending = ""

        async with asyncio.timeout(safe_timeout):
            async for item in llm_client.astream(prompt):
                text = self.store.content_to_text(getattr(item, "content", item))
                if not text:
                    continue
                chunks.append(text)
                pending += text
                if (
                    len(pending) >= max(1, int(flush_chars))
                    or text.endswith(("\n", "。", "！", "？", ".", "!", "?", "；", ";"))
                ):
                    await _flush(force=True)
            await _flush(force=True)

        return "".join(chunks).strip()

    def _subject_working_dir(self, subject_id: str) -> str:
        return self.subject_catalog[subject_id]["working_dir"]

    @staticmethod
    def _subject_scoped_thread_id(thread_id: str, subject_id: str) -> str:
        return f"{thread_id}::{subject_id}"

    @staticmethod
    def _build_subject_route_meta(subject_route: dict[str, Any]) -> dict[str, Any]:
        ranked = [
            {"subject": subject_id, "score": score}
            for subject_id, score in subject_route.get("ranked", [])
        ]
        return {
            "primary_subject": subject_route.get("primary_subject", ""),
            "cross_subject": bool(subject_route.get("cross_subject", False)),
            "confidence": float(subject_route.get("confidence", 0.0)),
            "reason": str(subject_route.get("reason", "") or ""),
            "requested_subjects": list(subject_route.get("requested_subjects", [])),
            "ranked": ranked,
        }

    async def prewarm_subject_rags(self) -> list[str]:
        subject_ids = list(self.subject_catalog.keys())
        tasks = [
            get_rag(self._subject_working_dir(subject_id))
            for subject_id in subject_ids
        ]
        await asyncio.gather(*tasks)
        return subject_ids

    async def ask_instant_mode(
        self,
        question: str,
        thread_id: str,
        timeout_s: int,
        *,
        working_dir: str | None = None,
    ) -> dict[str, Any]:
        result = await asyncio.wait_for(
            answer_instant(question, thread_id=thread_id, working_dir=working_dir),
            timeout=max(1, int(timeout_s)),
        )
        return {
            "mode_used": "instant",
            "answer": str(result.get("answer", "")).strip(),
            "query_status": result.get("query_status", ""),
            "query_message": result.get("query_message", ""),
            "elapsed_ms": result.get("elapsed_ms", "0"),
            "raw": result,
        }

    async def ask_instant_mode_stream(
        self,
        question: str,
        thread_id: str,
        timeout_s: int,
        *,
        working_dir: str | None = None,
        emit_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        safe_timeout = max(1, int(timeout_s))
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
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "mode_used": "instant",
            "answer": answer,
            "query_status": result.get("query_status", ""),
            "query_message": result.get("query_message", ""),
            "elapsed_ms": str(elapsed_ms),
            "raw": result,
        }

    async def _run_deepsearch_plan_state(
        self,
        *,
        question: str,
        routing_question: str | None = None,
        allowed_subject_ids: list[str] | None = None,
        response_language: str = "zh",
        workflow_stage_callback: Callable[
            [str, dict[str, Any]],
            Awaitable[None],
        ]
        | None = None,
    ) -> dict[str, Any]:
        normalized_subject_ids = [
            subject_id
            for subject_id in (allowed_subject_ids or [])
            if subject_id in self.subject_catalog
        ]
        candidate_subject_ids = normalized_subject_ids or list(self.subject_catalog.keys())
        subject_working_dirs = {
            subject_id: self._subject_working_dir(subject_id)
            for subject_id in candidate_subject_ids
        }

        async def route_subquestion_subjects(
            *,
            sub_question: str,
            original_question: str,
        ) -> dict[str, Any]:
            candidates = candidate_subject_ids
            if len(candidates) == 1:
                subject_id = candidates[0]
                return {
                    "primary_subject": subject_id,
                    "target_subjects": [subject_id],
                    "reason": f"deepsearch 候选学科已锁定为 {self._subject_label(subject_id)}",
                    "ranked_subjects": [{"subject": subject_id, "score": 1.0}],
                }

            route = await self.decide_subject_route(
                user_question=sub_question,
                augmented_question=(
                    f"原问题：{original_question}\n"
                    f"当前检索子问题：{sub_question}\n"
                    "请只判断这个子问题最应该查哪个课程知识库。"
                ),
                mode="deepsearch",
                timeout_s=max(1, int(cfg.WEB_RETRIEVAL_GATE_TIMEOUT_S)),
                requested_subjects=None,
            )
            ranked = [
                (subject_id, float(score))
                for subject_id, score in route.get("ranked", [])
                if subject_id in candidates
            ]
            if not ranked:
                ranked = [
                    (subject_id, 1.0 if idx == 0 else 0.0)
                    for idx, subject_id in enumerate(candidates)
                ]
            primary_subject = ranked[0][0]
            return {
                "primary_subject": primary_subject,
                "target_subjects": [primary_subject],
                "reason": str(route.get("reason", "") or "").strip()
                or f"子问题优先匹配 {self._subject_label(primary_subject)}",
                "ranked_subjects": [
                    {"subject": subject_id, "score": score}
                    for subject_id, score in ranked
                ],
            }

        async def emit_stage(stage: str, state: dict[str, Any]) -> None:
            if workflow_stage_callback is None:
                return
            await workflow_stage_callback(stage, dict(state))

        if workflow_stage_callback is not None:
            state: dict[str, Any] = {
                "question": question,
                "requested_mode": "deepsearch",
                "response_language": response_language,
                "allowed_subject_ids": list(candidate_subject_ids),
                "subject_working_dirs": dict(subject_working_dirs),
            }
            with use_rag_working_dir(None):
                await emit_stage("deepsearch_plan_start", state)
                state.update(await asyncio.to_thread(build_global_subquestion_plan, state))
                await emit_stage("deepsearch_plan_end", state)

                await emit_stage("deepsearch_subject_route_start", state)
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
                await emit_stage("deepsearch_subject_route_end", state)

                state.update(await asyncio.to_thread(build_subquery_tasks, state))
                while True:
                    await emit_stage("deepsearch_retrieve_start", state)
                    state.update(await query_subquestion_tasks(state))
                    await emit_stage("deepsearch_retrieve_end", state)

                    await emit_stage("deepsearch_review_start", state)
                    state.update(await judge_subquestion_results(state))
                    await emit_stage("deepsearch_review_end", state)

                    if not bool(state.get("needs_retry")):
                        await emit_stage("deepsearch_retry_skipped", state)
                        break

                    await emit_stage("deepsearch_retry_start", state)
                    state.update(
                        await asyncio.to_thread(
                            prepare_subquestion_retry_plan,
                            state,
                        )
                    )
                    await emit_stage("deepsearch_retry_end", state)
                    state.update(await asyncio.to_thread(build_subquery_tasks, state))
            return state

        return await run_question_plan_state(
            question,
            requested_mode="deepsearch",
            routing_question=routing_question,
            allowed_subject_ids=candidate_subject_ids,
            subject_working_dirs=subject_working_dirs,
            response_language=response_language,
            route_subquestion_subjects=route_subquestion_subjects,
        )

    async def _stream_routed_deepsearch_mode(
        self,
        *,
        question: str,
        timeout_s: int,
        allowed_subject_ids: list[str] | None = None,
        routing_question: str | None = None,
        response_language: str = "zh",
        emit_text: Callable[[str], None] | None = None,
        workflow_stage_callback: Callable[
            [str, dict[str, Any]],
            Awaitable[None],
        ]
        | None = None,
    ) -> dict[str, Any]:
        safe_timeout = max(1, int(timeout_s))
        started = time.perf_counter()
        async with asyncio.timeout(safe_timeout):
            state = await self._run_deepsearch_plan_state(
                question=question,
                routing_question=routing_question,
                allowed_subject_ids=allowed_subject_ids,
                response_language=response_language,
                workflow_stage_callback=workflow_stage_callback,
            )
            planning_ms = int((time.perf_counter() - started) * 1000)
            elapsed = time.perf_counter() - started
            remaining = max(1, safe_timeout - int(elapsed))
            final_prompt = build_final_answer_prompt(state)
            if not str(final_prompt or "").strip():
                raise ValueError("DeepSearch final answer prompt is empty")
            if workflow_stage_callback is not None:
                await workflow_stage_callback("answer_generate_start", dict(state))
            answer_started = time.perf_counter()
            answer = await self._stream_llm_text(
                llm_client=llm,
                prompt=final_prompt,
                timeout_s=remaining,
                emit_text=emit_text,
            )
            answer_ms = int((time.perf_counter() - answer_started) * 1000)
            if workflow_stage_callback is not None:
                await workflow_stage_callback(
                    "answer_generate_end",
                    {**state, "answer_ms": answer_ms},
                )

        total_ms = int((time.perf_counter() - started) * 1000)
        return {
            "mode_used": "deepsearch",
            "answer": answer,
            "query_total_ms": str(total_ms),
            "raw": {
                "final_answer": answer,
                "query_total_ms": str(total_ms),
                "sub_questions": state.get("sub_questions", []),
                "subquery_results": state.get("subquery_results", []),
                "query_attempt": state.get("query_attempt", 0),
                "planning_ms": str(planning_ms),
                "answer_ms": str(answer_ms),
                "insufficient_subquestion_ids": state.get(
                    "insufficient_subquestion_ids",
                    [],
                ),
                "needs_retry": bool(state.get("needs_retry", False)),
            },
        }

    async def _stream_synthesize_subject_answers_with_review(
        self,
        *,
        user_question: str,
        mode: str,
        subject_answers: list[dict[str, str]],
        timeout_s: int,
        response_language: str = "zh",
        emit_text: Callable[[str], None] | None = None,
        flush_chars: int = 16,
    ) -> dict[str, Any]:
        synth_timeout = max(
            1,
            min(int(timeout_s), max(1, int(cfg.WEB_DIRECT_ANSWER_TIMEOUT_S))),
        )
        prompt = self._build_subject_synthesis_review_prompt(
            user_question=user_question,
            mode=mode,
            subject_answers=subject_answers,
            response_language=response_language,
        )
        marker_pattern = re.compile(
            r"(?is)SUFFICIENT\s*:\s*(true|false)\s*[\r\n]+"
            r"REASON\s*:\s*(.*?)\s*[\r\n]+"
            r"ANSWER\s*:\s*"
        )
        answer_chunks: list[str] = []
        header_buffer = ""
        pending = ""
        answer_started = False
        review_sufficient: bool | None = None
        review_reason = "跨学科整合评审输出解析失败"

        async def _flush(force: bool = False) -> None:
            nonlocal pending
            if not pending:
                return
            if not force and len(pending) < max(1, int(flush_chars)):
                return
            if emit_text is not None:
                emit_text(pending)
            pending = ""

        async with asyncio.timeout(synth_timeout):
            async for item in auto.auto_router_llm.astream(prompt):
                text = self.store.content_to_text(getattr(item, "content", item))
                if not text:
                    continue
                if answer_started:
                    answer_chunks.append(text)
                    pending += text
                    if (
                        len(pending) >= max(1, int(flush_chars))
                        or text.endswith(
                            ("\n", "。", "！", "？", ".", "!", "?", "；", ";")
                        )
                    ):
                        await _flush(force=True)
                    continue

                header_buffer += text
                marker = marker_pattern.search(header_buffer)
                if marker is None:
                    continue

                answer_started = True
                review_sufficient = marker.group(1).strip().lower() == "true"
                review_reason = str(marker.group(2) or "").strip() or "无"
                remainder = header_buffer[marker.end() :]
                header_buffer = ""
                if not remainder:
                    continue
                answer_chunks.append(remainder)
                pending += remainder
                if (
                    len(pending) >= max(1, int(flush_chars))
                    or remainder.endswith(
                        ("\n", "。", "！", "？", ".", "!", "?", "；", ";")
                    )
                ):
                    await _flush(force=True)

            if not answer_started:
                parsed = self._parse_subject_synthesis_review_response(header_buffer)
                answer = str(parsed.get("answer", "")).strip()
                if answer and emit_text is not None:
                    emit_text(answer)
                return {
                    "answer": answer,
                    "sufficient": parsed.get("sufficient"),
                    "reason": str(parsed.get("reason", "")).strip() or "无",
                }

            await _flush(force=True)

        return {
            "answer": "".join(answer_chunks).strip(),
            "sufficient": review_sufficient,
            "reason": review_reason,
        }
