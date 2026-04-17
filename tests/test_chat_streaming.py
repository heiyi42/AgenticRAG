from __future__ import annotations

import threading
import unittest

from webapp_core.chat_service import ChatService


class _DummyStore:
    def normalize_mode(self, value: object) -> str:
        return str(value or "auto")


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


if __name__ == "__main__":
    unittest.main()
