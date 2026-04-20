from __future__ import annotations

import asyncio
import json
import time
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any, Callable

import models.auto as auto

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

            def push_event(event: str, data: dict[str, Any]) -> None:
                event_queue.put((event, dict(data or {})))

            async def produce_events() -> None:
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

                retrieval_used = True
                retrieval_gate_confidence = 1.0
                retrieval_gate_reason = "未启用检索网关"
                subject_route_meta = None
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

                try:
                    if code_candidate is not None:
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
                    elif fast_result_bundle is not None:
                        (
                            result,
                            retrieval_used,
                            retrieval_gate_confidence,
                            retrieval_gate_reason,
                        ) = fast_result_bundle
                        subject_route_meta = result.get("subject_route")
                        push_event(
                            "meta",
                            {
                                "retrieval_used": retrieval_used,
                                "retrieval_gate_confidence": retrieval_gate_confidence,
                                "retrieval_gate_reason": retrieval_gate_reason,
                                "subject_route": subject_route_meta,
                            },
                        )
                        emit_text(str(result.get("answer", "")))
                    else:
                        subject_route: dict[str, Any] | None = None
                        if explicit_subjects:
                            retrieval_used = True
                            retrieval_gate_confidence = 1.0
                            retrieval_gate_reason = "用户显式指定学科，强制检索"
                            if mode == "deepsearch":
                                subject_route_meta = self._build_subject_route_meta(
                                    self._subject_route_from_explicit_subjects(
                                        explicit_subjects
                                    )
                                )
                            else:
                                subject_route = await self.decide_subject_route(
                                    user_question=question,
                                    augmented_question=augmented_question,
                                    mode=mode,
                                    timeout_s=timeout_s,
                                    requested_subjects=explicit_subjects,
                                )
                        elif tutoring_candidate is not None:
                            subject_route = await self.decide_subject_route(
                                user_question=question,
                                augmented_question=augmented_question,
                                mode=mode,
                                timeout_s=timeout_s,
                                requested_subjects=requested_subjects,
                            )
                            retrieval_used = True
                            retrieval_gate_confidence = 1.0
                            retrieval_gate_reason = "用户显式触发题目辅导"
                        elif mode == "deepsearch":
                            if cfg.WEB_ENABLE_RETRIEVAL_GATE:
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
                            if retrieval_used:
                                subject_route = await self.decide_subject_route(
                                    user_question=question,
                                    augmented_question=augmented_question,
                                    mode=mode,
                                    timeout_s=timeout_s,
                                    requested_subjects=requested_subjects,
                                )
                            else:
                                subject_route_meta = self._fast_subject_route_meta(
                                    "第一网关判定免检索直答"
                                )
                        else:
                            subject_route = await self.decide_subject_route(
                                user_question=question,
                                augmented_question=augmented_question,
                                mode=mode,
                                timeout_s=timeout_s,
                                requested_subjects=requested_subjects,
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

                        if retrieval_used and tutoring_candidate is not None:
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
                        elif retrieval_used:
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
                        else:
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
                except Exception as e:
                    partial_answer = "".join(streamed_parts).strip()
                    error_answer = partial_answer or f"请求失败：{e}"
                    if partial_answer:
                        note = f"\n\n[生成中断：{type(e).__name__}: {e}]"
                        error_answer = f"{partial_answer}{note}"
                        push_event("delta", {"text": note})
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    mode_used = mode
                    assistant_meta = self.store.make_assistant_meta(mode_used, elapsed_ms)
                    with session.lock:
                        self.store.update_session_after_answer(
                            session,
                            question=question,
                            answer=error_answer,
                            requested_mode=mode,
                            mode_used=mode_used,
                            elapsed_ms=elapsed_ms,
                            assistant_meta=assistant_meta,
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

                with session.lock:
                    self.store.update_session_after_answer(
                        session,
                        question=question,
                        answer=answer,
                        requested_mode=mode,
                        mode_used=mode_used,
                        elapsed_ms=elapsed_ms,
                        assistant_meta=assistant_meta,
                        message_details=(
                            dict(result.get("message_details"))
                            if isinstance(result.get("message_details"), dict)
                            else None
                        ),
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
                    "message_details": (
                        dict(result.get("message_details"))
                        if isinstance(result.get("message_details"), dict)
                        else None
                    ),
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
