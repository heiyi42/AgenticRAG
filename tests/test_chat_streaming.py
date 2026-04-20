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
            return {
                "mode_used": "deepsearch",
                "answer": "深搜回答",
                "route": {"chain": "deepsearch-subquestion-routed"},
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
        for chunk in events:
            if not chunk.startswith("event: done\n"):
                continue
            data_line = next(
                (line for line in chunk.splitlines() if line.startswith("data: ")),
                "",
            )
            self.assertTrue(data_line)
            done_payloads.append(json.loads(data_line[len("data: ") :]))

        self.assertEqual(len(done_payloads), 1)
        self.assertEqual(done_payloads[0]["mode_used"], "deepsearch")
        self.assertEqual(captured["stream_kwargs"]["subject_route"], None)
        self.assertEqual(captured["stream_kwargs"]["requested_subjects"], [])
        self.assertEqual(
            done_payloads[0]["subject_route"]["reason"],
            "DeepSearch 跳过请求级学科路由，拆题后对子问题单独路由",
        )


if __name__ == "__main__":
    unittest.main()
