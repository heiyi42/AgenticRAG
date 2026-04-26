from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from . import auto_runtime as auto

from . import config as cfg


class ChatAutoOrchestrationMixin:
    def _start_speculative_auto_secondary_query(
        self,
        *,
        user_question: str,
        question: str,
        thread_id: str,
        timeout_s: int,
        subject_ids: list[str],
        response_language: str = "zh",
    ) -> asyncio.Task[Any] | None:
        if not cfg.WEB_AUTO_SPECULATIVE_SECONDARY or len(subject_ids) <= 1:
            return None
        secondary_subject = subject_ids[1]
        return asyncio.create_task(
            self.ask_instant_mode(
                self._apply_answer_style_to_question(
                    question,
                    user_question=user_question,
                    subject_id=secondary_subject,
                    mode="auto",
                    response_language=response_language,
                ),
                self._subject_scoped_thread_id(thread_id, secondary_subject),
                timeout_s,
                working_dir=self._subject_working_dir(secondary_subject),
            )
        )

    async def _cancel_background_task(self, task: asyncio.Task[Any] | None) -> None:
        if task is None:
            return
        if task.done():
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _resolve_auto_secondary_result(
        self,
        *,
        user_question: str,
        question: str,
        thread_id: str,
        timeout_s: int,
        subject_ids: list[str],
        response_language: str = "zh",
        secondary_result_task: asyncio.Task[Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        if len(subject_ids) <= 1:
            return None, "secondary_not_needed"

        secondary_subject = subject_ids[1]
        speculation_note = ""
        if secondary_result_task is not None:
            try:
                return await secondary_result_task, "secondary_speculative_hit"
            except asyncio.CancelledError:
                speculation_note = "secondary_speculative_cancelled"
            except Exception as e:
                speculation_note = (
                    f"secondary_speculative_failed={type(e).__name__}: {e}"
                )

        try:
            result = await self.ask_instant_mode(
                self._apply_answer_style_to_question(
                    question,
                    user_question=user_question,
                    subject_id=secondary_subject,
                    mode="auto",
                    response_language=response_language,
                ),
                self._subject_scoped_thread_id(thread_id, secondary_subject),
                timeout_s,
                working_dir=self._subject_working_dir(secondary_subject),
            )
            if speculation_note:
                return result, f"{speculation_note} | secondary_direct_fallback"
            return result, "secondary_direct"
        except Exception as e:
            if speculation_note:
                return (
                    None,
                    f"{speculation_note} | secondary_direct_failed={type(e).__name__}: {e}",
                )
            return None, f"secondary_direct_failed={type(e).__name__}: {e}"

    async def _run_multi_subject_instant_stream(
        self,
        *,
        user_question: str,
        question: str,
        thread_id: str,
        timeout_s: int,
        subject_ids: list[str],
        response_language: str = "zh",
        emit_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        subject_id = subject_ids[0]
        scoped_thread_id = self._subject_scoped_thread_id(thread_id, subject_id)
        result = await self.ask_instant_mode_stream(
            self._apply_answer_style_to_question(
                question,
                user_question=user_question,
                subject_id=subject_id,
                mode="instant",
                response_language=response_language,
            ),
            scoped_thread_id,
            timeout_s,
            working_dir=self._subject_working_dir(subject_id),
            emit_text=emit_text,
        )
        result["subject_ids"] = [subject_id]
        result["route"] = {
            "chain": "instant-single-subject",
            "subjects": [subject_id],
            "reason": f"instant 仅查询主学科 {self._subject_label(subject_id)}",
        }
        return result

    async def _run_multi_subject_auto_instant_stream(
        self,
        *,
        user_question: str,
        question: str,
        thread_id: str,
        timeout_s: int,
        subject_ids: list[str],
        primary_answer: str,
        response_language: str = "zh",
        emit_text: Callable[[str], None] | None = None,
        secondary_result_task: asyncio.Task[Any] | None = None,
    ) -> dict[str, Any]:
        selected = subject_ids[:2]
        if len(selected) <= 1:
            return {
                "mode_used": "instant",
                "answer": primary_answer,
                "review_sufficient": True,
                "review_reason": "单学科无需补充整合",
                "auto_timings": {
                    "autoSecondSubjectMs": 0,
                    "autoMergeReviewMs": 0,
                },
                "subject_ids": selected,
                "route": {
                    "chain": "auto-single-subject-instant",
                    "subjects": selected,
                    "reason": "auto 使用主学科单库回答",
                },
                "raw": {"subject_results": []},
            }

        secondary_started = time.perf_counter()
        secondary_result, secondary_note = await self._resolve_auto_secondary_result(
            user_question=user_question,
            question=question,
            thread_id=thread_id,
            timeout_s=timeout_s,
            subject_ids=selected,
            response_language=response_language,
            secondary_result_task=secondary_result_task,
        )
        secondary_ms = int((time.perf_counter() - secondary_started) * 1000)
        subject_answers = [
            {
                "subject_id": selected[0],
                "answer": primary_answer,
            }
        ]
        secondary_answer = (
            str(secondary_result.get("answer", "")).strip()
            if secondary_result is not None
            else ""
        )
        if secondary_answer:
            subject_answers.append(
                {
                    "subject_id": selected[1],
                    "answer": secondary_answer,
                }
            )
        merge_started = time.perf_counter()
        merged = await self._stream_synthesize_subject_answers_with_review(
            user_question=user_question,
            mode="auto",
            subject_answers=subject_answers,
            timeout_s=timeout_s,
            response_language=response_language,
            emit_text=emit_text,
        )
        merge_ms = int((time.perf_counter() - merge_started) * 1000)
        merged_answer = str(merged.get("answer", "")).strip() or str(primary_answer or "").strip()
        review_reason = str(merged.get("reason", "")).strip() or "无"
        if secondary_note and secondary_note not in {
            "secondary_speculative_hit",
            "secondary_direct",
            "secondary_not_needed",
        }:
            review_reason = f"{review_reason} | {secondary_note}"
        return {
            "mode_used": "instant",
            "answer": merged_answer,
            "review_sufficient": merged.get("sufficient"),
            "review_reason": review_reason,
            "auto_timings": {
                "autoSecondSubjectMs": secondary_ms,
                "autoMergeReviewMs": merge_ms,
            },
            "subject_ids": subject_ids[:2],
            "route": {
                "chain": "auto-dual-subject-instant",
                "subjects": selected,
                "reason": (
                    "auto 主学科不足后复用投机第二学科并合并评审"
                    if secondary_result_task is not None
                    else "auto 先查主学科，不足后补第二学科并合并评审"
                ),
            },
            "raw": {
                "subject_results": [secondary_result] if secondary_result is not None else [],
                "secondary_note": secondary_note,
            },
        }

    async def _run_multi_subject_deep_stream(
        self,
        *,
        user_question: str,
        question: str,
        thread_id: str,
        timeout_s: int,
        subject_ids: list[str],
        response_language: str = "zh",
        emit_text: Callable[[str], None] | None = None,
        workflow_stage_callback: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        del thread_id
        normalized_subject_ids = [
            subject_id
            for subject_id in subject_ids
            if subject_id in self.subject_catalog
        ] or list(self.subject_catalog.keys()) or ["operating_systems"]
        primary_subject = normalized_subject_ids[0]
        deep_question = (
            self._apply_answer_style_to_question(
                question,
                user_question=user_question,
                subject_id=primary_subject,
                mode="deepsearch",
                response_language=response_language,
            )
            if len(normalized_subject_ids) == 1
            else self._apply_response_language_to_question(question, response_language)
        )
        result = await self._stream_routed_deepsearch_mode(
            question=deep_question,
            timeout_s=timeout_s,
            allowed_subject_ids=normalized_subject_ids,
            routing_question=question,
            response_language=response_language,
            emit_text=emit_text,
            workflow_stage_callback=workflow_stage_callback,
        )
        result["subject_ids"] = normalized_subject_ids
        result["route"] = {
            "chain": "deepsearch-subquestion-routed",
            "subjects": normalized_subject_ids,
            "reason": "先全局拆分子问题，再对子问题逐个选择知识库并检索",
        }
        return result

    async def _stream_mode_with_retrieval(
        self,
        *,
        mode: str,
        subject_route: dict[str, Any] | None,
        requested_subjects: list[str] | None = None,
        user_question: str,
        augmented_question: str,
        thread_id: str,
        timeout_s: int,
        response_language: str = "zh",
        emit_text: Callable[[str], None] | None = None,
        workflow_stage_callback: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        if mode == "deepsearch":
            deep_subjects = [
                subject_id
                for subject_id in (requested_subjects or [])
                if subject_id in self.subject_catalog
            ] or list(self.subject_catalog.keys())
            return await self._run_multi_subject_deep_stream(
                user_question=user_question,
                question=augmented_question,
                thread_id=thread_id,
                timeout_s=timeout_s,
                subject_ids=deep_subjects,
                response_language=response_language,
                emit_text=emit_text,
                workflow_stage_callback=workflow_stage_callback,
            )

        if subject_route is None:
            raise ValueError(f"{mode} mode requires subject_route")

        instant_subjects = self._subjects_for_instant(subject_route)
        if mode == "instant":
            return await self._run_multi_subject_instant_stream(
                user_question=user_question,
                question=augmented_question,
                thread_id=thread_id,
                timeout_s=timeout_s,
                subject_ids=instant_subjects,
                response_language=response_language,
                emit_text=emit_text,
            )
        return await self._stream_auto_mode(
            subject_route=subject_route,
            user_question=user_question,
            augmented_question=augmented_question,
            thread_id=thread_id,
            timeout_s=timeout_s,
            response_language=response_language,
            emit_text=emit_text,
        )

    async def _plan_auto_route(
        self,
        *,
        subject_route: dict[str, Any],
        augmented_question: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        auto_subjects = self._subjects_for_auto(subject_route)
        deep_subjects = self._subjects_for_deep(subject_route)
        requested_subjects = list(subject_route.get("requested_subjects", []))
        single_subject_fast_path = (
            len(auto_subjects) == 1
            and (
                len(requested_subjects) == 1
                or (
                    not bool(subject_route.get("cross_subject", False))
                    and float(subject_route.get("max_score", 0.0)) >= 0.70
                )
            )
        )
        if single_subject_fast_path:
            complexity = "simple"
            confidence = 1.0 if len(requested_subjects) == 1 else max(
                0.85,
                float(subject_route.get("confidence", 0.0)),
            )
            route_reason = (
                "用户手动指定单学科，auto 优先单库快速回答"
                if len(requested_subjects) == 1
                else "单学科高置信命中，auto 走快速路径"
            )
        else:
            cached_complexity = str(subject_route.get("auto_complexity", "")).strip()
            if cached_complexity:
                complexity = self._normalize_auto_complexity(cached_complexity)
                confidence = auto._clamp_confidence(
                    subject_route.get("auto_complexity_confidence", 0.0)
                )
                route_reason = (
                    str(subject_route.get("auto_complexity_reason", "") or "").strip()
                    or "统一路由复杂度判定"
                )
            else:
                complexity, confidence, route_reason = await auto._route_complexity(
                    augmented_question,
                    timeout_s,
                    cfg.AUTO_ROUTE_RATIO,
                )
        cross_subject = len(auto_subjects) > 1
        high_conf_complex = (
            complexity == "complex" and confidence >= cfg.AUTO_ROUTE_THRESHOLD
        )
        route_chain = (
            "deep"
            if ((cross_subject and complexity == "complex") or high_conf_complex)
            else "instant"
        )
        route_policy = (
            "user-locked single-subject -> instant first"
            if len(requested_subjects) == 1
            else (
                "single-subject high-confidence -> instant first"
                if single_subject_fast_path
                else (
                    "cross-subject complex -> deep"
                    if cross_subject and route_chain == "deep"
                    else (
                        "cross-subject primary-then-secondary"
                        if cross_subject and route_chain == "instant"
                        else (
                            "high-confidence complex -> deep"
                            if high_conf_complex
                            else "progressive -> instant first"
                        )
                    )
                )
            )
        )
        return {
            "auto_subjects": auto_subjects,
            "deep_subjects": deep_subjects,
            "requested_subjects": requested_subjects,
            "single_subject_fast_path": single_subject_fast_path,
            "complexity": complexity,
            "confidence": confidence,
            "route_reason": route_reason,
            "cross_subject": cross_subject,
            "high_conf_complex": high_conf_complex,
            "route_chain": route_chain,
            "route_policy": route_policy,
        }

    async def _run_auto_deep_stream(
        self,
        *,
        user_question: str,
        augmented_question: str,
        thread_id: str,
        started: float,
        timeout_s: int,
        subject_ids: list[str],
        response_language: str = "zh",
        emit_text: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        remaining = auto._remaining_seconds(started, timeout_s)
        if remaining <= 0:
            return None, "总超时预算已耗尽，无法继续 deep 链路。"
        deep_budget = min(remaining, auto.AUTO_DEEP_BUDGET_S)
        try:
            result = await self._run_multi_subject_deep_stream(
                user_question=user_question,
                question=augmented_question,
                thread_id=thread_id,
                timeout_s=deep_budget,
                subject_ids=subject_ids,
                response_language=response_language,
                emit_text=emit_text,
            )
            return result, ""
        except asyncio.TimeoutError:
            return None, f"deep 链路超时（>{deep_budget}s）。"
        except Exception as e:
            return None, f"deep 链路失败: {e}"

    async def _stream_auto_mode(
        self,
        *,
        subject_route: dict[str, Any],
        user_question: str,
        augmented_question: str,
        thread_id: str,
        timeout_s: int,
        response_language: str = "zh",
        emit_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        auto_timings: dict[str, int] = {}
        plan_started = time.perf_counter()
        plan = await self._plan_auto_route(
            subject_route=subject_route,
            augmented_question=augmented_question,
            timeout_s=timeout_s,
        )
        auto_timings["autoPlanMs"] = int((time.perf_counter() - plan_started) * 1000)

        def attach_auto_timings(payload: dict[str, Any]) -> dict[str, Any]:
            payload["auto_timings"] = dict(auto_timings)
            return payload

        auto_subjects = list(plan["auto_subjects"])
        deep_subjects = list(plan["deep_subjects"])
        requested_subjects = list(plan["requested_subjects"])
        complexity = str(plan["complexity"])
        confidence = float(plan["confidence"])
        route_reason = str(plan["route_reason"])
        cross_subject = bool(plan["cross_subject"])
        route_chain = str(plan["route_chain"])
        route_policy = str(plan["route_policy"])

        instant_review = {
            "heuristic": "",
            "review": "未执行LLM评审",
        }

        if route_chain == "instant":
            remaining = auto._remaining_seconds(started, timeout_s)
            if remaining <= 0:
                raise TimeoutError("总超时预算已耗尽。")

            base_instant_budget = auto._budget_by_ratio(
                total_timeout_s=timeout_s,
                ratio=cfg.AUTO_INSTANT_RATIO,
                floor_s=5,
                hard_cap_s=auto.AUTO_INSTANT_BUDGET_S,
            )
            reserve_for_deep = min(auto.AUTO_DEEP_MIN_RESERVE_S, max(0, remaining - 1))
            instant_budget = max(
                1,
                min(base_instant_budget, max(1, remaining - reserve_for_deep)),
            )
            secondary_result_task = self._start_speculative_auto_secondary_query(
                user_question=user_question,
                question=augmented_question,
                thread_id=thread_id,
                timeout_s=instant_budget,
                subject_ids=auto_subjects,
                response_language=response_language,
            )
            try:
                instant_subject = auto_subjects[0]
                instant_started = time.perf_counter()
                instant_result = await auto._ask_instant_stream(
                    self._apply_answer_style_to_question(
                        augmented_question,
                        user_question=user_question,
                        subject_id=instant_subject,
                        mode="auto",
                        response_language=response_language,
                    ),
                    self._subject_scoped_thread_id(thread_id, instant_subject),
                    instant_budget,
                    working_dir=self._subject_working_dir(instant_subject),
                    emit_text=emit_text,
                )
                auto_timings["instantTrialMs"] = int(
                    (time.perf_counter() - instant_started) * 1000
                )
            except asyncio.TimeoutError:
                auto_timings["instantTrialMs"] = int(
                    (time.perf_counter() - instant_started) * 1000
                    if "instant_started" in locals()
                    else 0
                )
                await self._cancel_background_task(secondary_result_task)
                deep_started = time.perf_counter()
                deep_result, deep_error = await self._run_auto_deep_stream(
                    user_question=user_question,
                    augmented_question=augmented_question,
                    thread_id=thread_id,
                    started=started,
                    timeout_s=timeout_s,
                    subject_ids=deep_subjects,
                    response_language=response_language,
                    emit_text=emit_text,
                )
                auto_timings["deepsearchFallbackMs"] = int(
                    (time.perf_counter() - deep_started) * 1000
                )
                if deep_result is None:
                    raise TimeoutError(deep_error)
                return attach_auto_timings({
                    "mode_used": "deepsearch",
                    "answer": str(deep_result.get("answer", "")).strip(),
                    "upgraded": True,
                    "upgrade_reason": "instant timeout",
                    "route": {
                        "complexity": complexity,
                        "confidence": confidence,
                        "chain": route_chain,
                        "policy": route_policy,
                        "subjects": deep_subjects,
                        "reason": route_reason,
                    },
                    "instant_review": instant_review,
                    "elapsed_ms": str(int((time.perf_counter() - started) * 1000)),
                    "raw": {"deep": deep_result},
                })
            except Exception:
                await self._cancel_background_task(secondary_result_task)
                raise

            review_started = time.perf_counter()
            heur_sufficient, heur_reason = auto._instant_heuristic_assessment(instant_result)
            instant_review["heuristic"] = heur_reason
            review_sufficient: bool | None = None
            review_reason = "未执行LLM评审"
            review_budget = min(
                auto.AUTO_FAST_REVIEW_TIMEOUT_S,
                auto.AUTO_REVIEW_TIMEOUT_S,
                max(0, auto._remaining_seconds(started, timeout_s) - 1),
            )
            if heur_sufficient is None and review_budget > 0:
                should_review, review_gate_reason = auto._should_review_instant_answer(
                    answer=str(instant_result.get("answer", "")),
                    cross_subject=cross_subject,
                    route_confidence=confidence,
                    subject_confidence=float(subject_route.get("confidence", 0.0)),
                )
                if should_review:
                    review_sufficient, review_reason = await auto._review_instant_answer(
                        question=user_question,
                        answer=str(instant_result.get("answer", "")),
                        timeout_s=review_budget,
                    )
                else:
                    review_sufficient = True
                    review_reason = f"跳过LLM评审: {review_gate_reason}"
            elif heur_sufficient is None:
                should_review, review_gate_reason = auto._should_review_instant_answer(
                    answer=str(instant_result.get("answer", "")),
                    cross_subject=cross_subject,
                    route_confidence=confidence,
                    subject_confidence=float(subject_route.get("confidence", 0.0)),
                )
                if should_review:
                    review_reason = f"触发LLM评审但预算不足: {review_gate_reason}"
                else:
                    review_sufficient = True
                    review_reason = f"跳过LLM评审: {review_gate_reason}"
            instant_review["review"] = review_reason

            need_upgrade = False
            upgrade_reason = ""
            if heur_sufficient is False:
                need_upgrade = True
                upgrade_reason = f"heuristic不足: {heur_reason}"
            elif review_sufficient is False:
                need_upgrade = True
                upgrade_reason = review_reason
            elif review_sufficient is True:
                need_upgrade = False
            elif heur_sufficient is True:
                need_upgrade = False
            else:
                need_upgrade = True
                upgrade_reason = (
                    f"评审不可用且启发式不确定: heur={heur_reason}, review={review_reason}"
                )
            auto_timings["instantReviewMs"] = int(
                (time.perf_counter() - review_started) * 1000
            )

            if need_upgrade and len(auto_subjects) > 1:
                supplement_budget = max(
                    1,
                    min(
                        instant_budget,
                        max(1, auto._remaining_seconds(started, timeout_s) - 1),
                    ),
                )
                try:
                    supplemented = await self._run_multi_subject_auto_instant_stream(
                        user_question=user_question,
                        question=augmented_question,
                        thread_id=thread_id,
                        timeout_s=supplement_budget,
                        subject_ids=auto_subjects,
                        primary_answer=str(instant_result.get("answer", "")).strip(),
                        response_language=response_language,
                        emit_text=emit_text,
                        secondary_result_task=secondary_result_task,
                    )
                    if isinstance(supplemented.get("auto_timings"), dict):
                        for key, value in supplemented["auto_timings"].items():
                            if isinstance(value, int) and value >= 0:
                                auto_timings[str(key)] = value
                    merged_answer = str(supplemented.get("answer", "")).strip()
                    merged_review_sufficient = supplemented.get("review_sufficient")
                    merged_review_reason = (
                        str(supplemented.get("review_reason", "")).strip() or "无"
                    )
                    merged_heur_sufficient, merged_heur_reason = (
                        auto._instant_heuristic_assessment(
                            {
                                "answer": merged_answer,
                                "query_status": "success",
                                "query_message": "",
                            }
                        )
                    )
                    instant_review["heuristic"] = (
                        f"{heur_reason} -> supplement={merged_heur_reason}"
                    )
                    instant_review["review"] = merged_review_reason
                    if (
                        merged_review_sufficient is not False
                        and merged_heur_sufficient is not False
                    ):
                        return attach_auto_timings({
                            "mode_used": "instant",
                            "answer": merged_answer,
                            "upgraded": False,
                            "upgrade_reason": "",
                            "route": {
                                "complexity": complexity,
                                "confidence": confidence,
                                "chain": route_chain,
                                "policy": "primary-then-secondary",
                                "subjects": auto_subjects,
                                "reason": route_reason,
                            },
                            "instant_review": instant_review,
                            "elapsed_ms": str(int((time.perf_counter() - started) * 1000)),
                            "raw": {"instant": supplemented},
                        })
                    if merged_review_sufficient is False:
                        upgrade_reason = merged_review_reason
                    elif merged_heur_sufficient is False:
                        upgrade_reason = f"补充整合后仍不足: {merged_heur_reason}"
                except Exception as e:
                    instant_review["review"] = (
                        f"{review_reason} | secondary_fallback={type(e).__name__}: {e}"
                    )

            if not need_upgrade:
                await self._cancel_background_task(secondary_result_task)
                return attach_auto_timings({
                    "mode_used": "instant",
                    "answer": str(instant_result.get("answer", "")).strip(),
                    "upgraded": False,
                    "upgrade_reason": "",
                    "route": {
                        "complexity": complexity,
                        "confidence": confidence,
                        "chain": route_chain,
                        "policy": route_policy,
                        "subjects": auto_subjects,
                        "reason": route_reason,
                    },
                    "instant_review": instant_review,
                    "elapsed_ms": str(int((time.perf_counter() - started) * 1000)),
                    "raw": {"instant": instant_result},
                })

            remaining = auto._remaining_seconds(started, timeout_s)
            if remaining <= 0:
                await self._cancel_background_task(secondary_result_task)
                return attach_auto_timings({
                    "mode_used": "instant",
                    "answer": str(instant_result.get("answer", "")).strip(),
                    "upgraded": True,
                    "upgrade_reason": f"{upgrade_reason}; 剩余预算不足，保留instant结果",
                    "route": {
                        "complexity": complexity,
                        "confidence": confidence,
                        "chain": route_chain,
                        "policy": route_policy,
                        "subjects": auto_subjects,
                        "reason": route_reason,
                    },
                    "instant_review": instant_review,
                    "elapsed_ms": str(int((time.perf_counter() - started) * 1000)),
                    "raw": {"instant": instant_result},
                })

            await self._cancel_background_task(secondary_result_task)
            deep_started = time.perf_counter()
            deep_result, deep_error = await self._run_auto_deep_stream(
                user_question=user_question,
                augmented_question=augmented_question,
                thread_id=thread_id,
                started=started,
                timeout_s=timeout_s,
                subject_ids=deep_subjects,
                response_language=response_language,
                emit_text=emit_text,
            )
            auto_timings["deepsearchFallbackMs"] = int(
                (time.perf_counter() - deep_started) * 1000
            )
            if deep_result is None:
                raise TimeoutError(deep_error)
            return attach_auto_timings({
                "mode_used": "deepsearch",
                "answer": str(deep_result.get("answer", "")).strip(),
                "upgraded": True,
                "upgrade_reason": upgrade_reason,
                "route": {
                    "complexity": complexity,
                    "confidence": confidence,
                    "chain": route_chain,
                    "policy": route_policy,
                    "subjects": deep_subjects,
                    "reason": route_reason,
                },
                "instant_review": instant_review,
                "elapsed_ms": str(int((time.perf_counter() - started) * 1000)),
                "raw": {"deep": deep_result},
            })

        deep_started = time.perf_counter()
        deep_result, deep_error = await self._run_auto_deep_stream(
            user_question=user_question,
            augmented_question=augmented_question,
            thread_id=thread_id,
            started=started,
            timeout_s=timeout_s,
            subject_ids=deep_subjects,
            response_language=response_language,
            emit_text=emit_text,
        )
        auto_timings["deepsearchFallbackMs"] = int(
            (time.perf_counter() - deep_started) * 1000
        )
        if deep_result is None:
            raise TimeoutError(deep_error)
        return attach_auto_timings({
            "mode_used": "deepsearch",
            "answer": str(deep_result.get("answer", "")).strip(),
            "upgraded": False,
            "upgrade_reason": "",
            "route": {
                "complexity": complexity,
                "confidence": confidence,
                "chain": route_chain,
                "policy": route_policy,
                "subjects": deep_subjects,
                "reason": route_reason,
            },
            "instant_review": instant_review,
            "elapsed_ms": str(int((time.perf_counter() - started) * 1000)),
            "raw": {"deep": deep_result},
        })
