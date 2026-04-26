from __future__ import annotations

import asyncio
import json
import threading
import unittest
from types import SimpleNamespace

from webapp_core.chat_service import ChatService


class _DummyStore:
    def __init__(self) -> None:
        self.session = SimpleNamespace(
            mode="auto",
            title="existing",
            turns=[{"role": "user", "content": "history"}],
            lock=threading.Lock(),
            chat_id="chat-1",
            updated_at=0.0,
        )
        self.saved_answers: list[dict[str, object]] = []

    def normalize_mode(self, value: object) -> str:
        return str(value or "auto")

    def get_or_create_session(self, chat_id: str) -> SimpleNamespace:
        self.session.chat_id = chat_id
        return self.session

    def build_augmented_question(self, session: object, question: str) -> str:
        del session
        return question

    def is_placeholder_title(self, title: object) -> bool:
        return str(title or "").strip() in {"", "新聊天"}

    def make_assistant_meta(self, mode_used: object, elapsed_ms: object) -> str:
        return f"{mode_used}:{elapsed_ms}"

    def update_session_after_answer(self, session: object, **kwargs: object) -> None:
        del session
        self.saved_answers.append(dict(kwargs))

    def persist_sessions_safely(self) -> None:
        return None


class _FakeProblemTutoringService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def match_request(
        self,
        text: str,
        *,
        requested_subjects: list[str] | None = None,
        requested_by_user: bool = False,
    ) -> dict[str, object]:
        call = {
            "text": text,
            "requested_subjects": list(requested_subjects or []),
            "requested_by_user": requested_by_user,
        }
        self.calls.append(call)
        return {"trigger": "explicit", "question": text}


class ChatStreamingTests(unittest.TestCase):
    @staticmethod
    def _build_service() -> ChatService:
        service = ChatService.__new__(ChatService)
        service.store = _DummyStore()
        service._event_subscribers = set()
        service._event_subscribers_lock = threading.Lock()
        service.problem_tutoring_service = _FakeProblemTutoringService()
        service.subject_catalog = {
            subject_id: {
                "id": subject_id,
                "label": label,
                "working_dir": f"/tmp/{subject_id}",
            }
            for subject_id, label in ChatService.SUBJECT_LABELS.items()
        }
        service.submit_async = None
        service.run_async = asyncio.run
        return service

    def test_iter_answer_chunks_splits_text_by_chunk_size(self) -> None:
        chunks = ChatService.iter_answer_chunks("abcdefghij", 4)

        self.assertEqual(chunks, ["abcd", "efgh", "ij"])

    def test_sse_encode_formats_event_payload(self) -> None:
        payload = ChatService.sse_encode("meta", {"ok": True})

        self.assertEqual(payload, 'event: meta\ndata: {"ok": true}\n\n')

    def test_publish_event_reaches_subscriber_queue(self) -> None:
        service = self._build_service()
        subscriber = service._register_event_subscriber()

        try:
            service._publish_event("title_updated", {"chat_id": "chat-1"})
            payload = subscriber.get_nowait()
        finally:
            service._unregister_event_subscriber(subscriber)

        self.assertEqual(payload["event"], "title_updated")
        self.assertEqual(payload["data"]["chat_id"], "chat-1")

    def test_build_chat_message_stream_handler_rejects_empty_message(self) -> None:
        service = self._build_service()

        handler, error = service.build_chat_message_stream_handler("chat-1", {})

        self.assertIsNone(handler)
        self.assertEqual(error, ("message 不能为空", 400))

    def test_match_problem_tutoring_request_requires_explicit_button(self) -> None:
        service = self._build_service()

        candidate = service._match_problem_tutoring_request(
            "这题怎么做：给定页面访问序列 7 0 1 2 0 3，页框数为 3，用 FIFO 页面置换，求缺页次数。",
            requested_subjects=["operating_systems"],
            requested_by_user=False,
        )

        self.assertIsNone(candidate)
        self.assertEqual(service.problem_tutoring_service.calls, [])

    def test_match_problem_tutoring_request_delegates_on_explicit_button(self) -> None:
        service = self._build_service()

        candidate = service._match_problem_tutoring_request(
            "这题怎么做：给定页面访问序列 7 0 1 2 0 3，页框数为 3，用 FIFO 页面置换，求缺页次数。",
            requested_subjects=["operating_systems"],
            requested_by_user=True,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(len(service.problem_tutoring_service.calls), 1)
        self.assertTrue(service.problem_tutoring_service.calls[0]["requested_by_user"])

    def test_deepsearch_stream_handler_skips_request_level_subject_route_after_gate(self) -> None:
        service = self._build_service()
        captured: dict[str, object] = {}

        service._match_code_analysis_request = lambda *args, **kwargs: None  # type: ignore[method-assign]
        service._match_problem_tutoring_request = lambda *args, **kwargs: None  # type: ignore[method-assign]
        service._fast_smalltalk_result_bundle = lambda *args, **kwargs: None  # type: ignore[method-assign]

        async def fake_decide_need_retrieval(**kwargs: object):
            captured["gate_kwargs"] = dict(kwargs)
            return True, 0.91, "需要检索"

        async def forbidden_decide_subject_route(**kwargs: object):
            raise AssertionError(f"deepsearch 不应在拆题前调用请求级学科路由: {kwargs}")

        async def fake_stream_mode_with_retrieval(**kwargs: object):
            captured["stream_kwargs"] = dict(kwargs)
            callback = kwargs.get("workflow_stage_callback")
            sub_questions = [
                {
                    "id": "q1",
                    "question": "栈溢出在 C 里如何发生？",
                    "used_question": "栈溢出 C",
                    "query_mode": "hybrid",
                    "top_k": 30,
                    "chunk_top_k": 8,
                    "target_subjects": ["C_program"],
                    "route_reason": "匹配 C 语言内存主题",
                    "ranked_subjects": [
                        {"subject": "C_program", "score": 0.92}
                    ],
                    "sufficient": "True",
                    "judge_reason": "证据足够",
                }
            ]
            if callback is not None:
                await callback("deepsearch_plan_start", {})
                await callback("deepsearch_plan_end", {"sub_questions": sub_questions})
                await callback("deepsearch_subject_route_start", {"sub_questions": sub_questions})
                await callback("deepsearch_subject_route_end", {"sub_questions": sub_questions})
                await callback("deepsearch_retrieve_start", {"query_attempt": 0})
                await callback(
                    "deepsearch_retrieve_end",
                    {
                        "subquery_results": [{"sub_question_id": "q1"}],
                        "query_total_ms": "12",
                    },
                )
                await callback("deepsearch_review_start", {})
                await callback(
                    "deepsearch_review_end",
                    {"insufficient_subquestion_ids": []},
                )
                await callback("deepsearch_retry_skipped", {})
                await callback("answer_generate_start", {})
                await callback("answer_generate_end", {"answer_ms": 7})
            return {
                "mode_used": "deepsearch",
                "answer": "深搜回答",
                "route": {"chain": "deepsearch-subquestion-routed"},
                "raw": {
                    "sub_questions": sub_questions,
                    "subquery_results": [{"sub_question_id": "q1"}],
                    "query_attempt": 0,
                    "insufficient_subquestion_ids": [],
                    "needs_retry": False,
                },
            }

        service.decide_need_retrieval = fake_decide_need_retrieval  # type: ignore[method-assign]
        service.decide_subject_route = forbidden_decide_subject_route  # type: ignore[method-assign]
        service._stream_mode_with_retrieval = fake_stream_mode_with_retrieval  # type: ignore[method-assign]

        handler, error = service.build_chat_message_stream_handler(
            "chat-1",
            {
                "message": "解释栈溢出和 C、OS、安全的关系",
                "mode": "deepsearch",
            },
        )

        self.assertIsNone(error)
        self.assertIsNotNone(handler)

        events = list(handler())
        done_payloads = []
        workflow_events: list[tuple[str, str, str]] = []
        for chunk in events:
            event_line = next(
                (line for line in chunk.splitlines() if line.startswith("event: ")),
                "",
            )
            event_name = event_line[len("event: ") :] if event_line else ""
            if event_name.startswith("workflow_node_"):
                data_line = next(
                    (line for line in chunk.splitlines() if line.startswith("data: ")),
                    "",
                )
                data = json.loads(data_line[len("data: ") :])
                workflow_events.append(
                    (
                        event_name,
                        str(data.get("nodeId", "")),
                        str(data.get("status", "")),
                    )
                )
            if not chunk.startswith("event: done\n"):
                continue
            data_line = next(
                (line for line in chunk.splitlines() if line.startswith("data: ")),
                "",
            )
            self.assertTrue(data_line)
            done_payloads.append(json.loads(data_line[len("data: ") :]))

        self.assertEqual(len(done_payloads), 1)
        self.assertIn(
            ("workflow_node_start", "deepsearch_plan", "running"),
            workflow_events,
        )
        self.assertIn(
            ("workflow_node_end", "deepsearch_plan", "success"),
            workflow_events,
        )
        self.assertIn(
            ("workflow_node_start", "deepsearch_subject_route", "running"),
            workflow_events,
        )
        plan_end_index = workflow_events.index(
            ("workflow_node_end", "deepsearch_plan", "success")
        )
        route_start_index = workflow_events.index(
            ("workflow_node_start", "deepsearch_subject_route", "running")
        )
        self.assertLess(plan_end_index, route_start_index)
        self.assertEqual(done_payloads[0]["mode_used"], "deepsearch")
        self.assertEqual(captured["stream_kwargs"]["subject_route"], None)
        self.assertEqual(captured["stream_kwargs"]["requested_subjects"], [])
        self.assertEqual(
            done_payloads[0]["subject_route"]["reason"],
            "DeepSearch 跳过请求级学科路由，拆题后对子问题单独路由",
        )
        explainability = done_payloads[0]["message_details"]["explainability"]
        self.assertEqual(explainability["mode"], "deepsearch")
        self.assertEqual(explainability["status"], "done")
        self.assertTrue(explainability["workflowSteps"])
        self.assertNotIn(
            "neo4j_subgraph",
            [step["nodeId"] for step in explainability["workflowSteps"]],
        )
        self.assertFalse(explainability["deepsearchTrace"]["subjectLock"]["enabled"])
        self.assertEqual(
            explainability["deepsearchTrace"]["subQuestionRoutes"][0]["primarySubject"],
            "C_program",
        )
        self.assertEqual(
            service.store.saved_answers[-1]["message_details"]["explainability"]["mode"],
            "deepsearch",
        )

    def test_deepsearch_explicit_subject_records_retrieval_gate_and_subject_lock(self) -> None:
        service = self._build_service()
        captured: dict[str, object] = {}

        service._match_code_analysis_request = lambda *args, **kwargs: None  # type: ignore[method-assign]
        service._match_problem_tutoring_request = lambda *args, **kwargs: None  # type: ignore[method-assign]
        service._fast_smalltalk_result_bundle = lambda *args, **kwargs: None  # type: ignore[method-assign]

        async def forbidden_decide_need_retrieval(**kwargs: object):
            raise AssertionError(f"显式学科 DeepSearch 不应再调用检索网关 LLM: {kwargs}")

        async def forbidden_decide_subject_route(**kwargs: object):
            raise AssertionError(f"显式学科 DeepSearch 不应调用请求级学科路由: {kwargs}")

        async def fake_stream_mode_with_retrieval(**kwargs: object):
            captured["stream_kwargs"] = dict(kwargs)
            return {
                "mode_used": "deepsearch",
                "answer": "指针回答",
                "route": {"chain": "deepsearch-subquestion-routed"},
                "raw": {
                    "sub_questions": [
                        {
                            "id": "q1",
                            "question": "指针是什么？",
                            "used_question": "指针是什么？",
                            "query_mode": "hybrid",
                            "top_k": 30,
                            "chunk_top_k": 8,
                            "target_subjects": ["C_program"],
                            "route_reason": "deepsearch 候选学科已锁定为 C语言",
                            "ranked_subjects": [
                                {"subject": "C_program", "score": 1.0}
                            ],
                            "sufficient": "False",
                            "judge_reason": "缺少数组指针证据",
                            "rewritten_question": "C 语言数组指针是什么？",
                        }
                    ],
                    "subquery_results": [{"sub_question_id": "q1"}],
                    "query_attempt": 1,
                    "insufficient_subquestion_ids": ["q1"],
                    "needs_retry": False,
                },
            }

        service.decide_need_retrieval = forbidden_decide_need_retrieval  # type: ignore[method-assign]
        service.decide_subject_route = forbidden_decide_subject_route  # type: ignore[method-assign]
        service._stream_mode_with_retrieval = fake_stream_mode_with_retrieval  # type: ignore[method-assign]

        handler, error = service.build_chat_message_stream_handler(
            "chat-1",
            {
                "message": "指针是什么",
                "mode": "deepsearch",
                "subjects": ["C_program"],
            },
        )

        self.assertIsNone(error)
        self.assertIsNotNone(handler)

        done_payloads = []
        for chunk in list(handler()):
            if not chunk.startswith("event: done\n"):
                continue
            data_line = next(
                (line for line in chunk.splitlines() if line.startswith("data: ")),
                "",
            )
            done_payloads.append(json.loads(data_line[len("data: ") :]))

        self.assertEqual(len(done_payloads), 1)
        self.assertEqual(captured["stream_kwargs"]["requested_subjects"], ["C_program"])
        explainability = done_payloads[0]["message_details"]["explainability"]
        workflow_node_ids = [step["nodeId"] for step in explainability["workflowSteps"]]
        self.assertIn("retrieval_gate", workflow_node_ids)
        self.assertIn("deepsearch_review", workflow_node_ids)
        self.assertIn("deepsearch_retry", workflow_node_ids)
        self.assertNotIn("neo4j_subgraph", workflow_node_ids)
        trace = explainability["deepsearchTrace"]
        self.assertTrue(trace["subjectLock"]["enabled"])
        self.assertEqual(trace["subjectLock"]["subjectIds"], ["C_program"])
        self.assertEqual(
            trace["subQuestionRoutes"][0]["targetSubjects"],
            ["C_program"],
        )
        self.assertEqual(trace["review"][0]["rewrittenQuestion"], "C 语言数组指针是什么？")


if __name__ == "__main__":
    unittest.main()
