from __future__ import annotations

import importlib
import json
import unittest
import uuid
import warnings
from unittest.mock import AsyncMock, patch


class WebappCodeAnalysisStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Importing Send from langgraph\.constants is deprecated\..*",
                category=Warning,
            )
            cls.webapp = importlib.import_module("webapp")
        cls.client = cls.webapp.app.test_client()

    def setUp(self) -> None:
        self.chat_id = f"test-code-analysis-{uuid.uuid4().hex[:8]}"
        with patch.object(self.webapp.store, "persist_sessions_safely", return_value=None):
            self.webapp.store.create_session(chat_id=self.chat_id, mode="auto")

    def tearDown(self) -> None:
        with patch.object(self.webapp.store, "persist_sessions_safely", return_value=None):
            self.webapp.store.delete_session(self.chat_id)

    @staticmethod
    def _parse_sse_events(payload: str) -> list[tuple[str, dict[str, object]]]:
        events: list[tuple[str, dict[str, object]]] = []
        for chunk in str(payload or "").split("\n\n"):
            lines = [line for line in chunk.splitlines() if line.strip()]
            if not lines:
                continue
            event_name = ""
            data_lines: list[str] = []
            for line in lines:
                if line.startswith("event: "):
                    event_name = line[7:].strip()
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
            if not event_name:
                continue
            data = json.loads("\n".join(data_lines)) if data_lines else {}
            events.append((event_name, data))
        return events

    def test_stream_endpoint_emits_code_analysis_events(self) -> None:
        async def fake_run_code_analysis_stream(**kwargs):
            emit_text = kwargs.get("emit_text")
            if callable(emit_text):
                emit_text("结构分析已生成。")
            return {
                "mode_used": "auto",
                "answer": "结构分析已生成。",
                "request_kind": "code_analysis",
                "route": {
                    "chain": "code_analysis",
                    "reason": "explicit",
                    "tool": "clang",
                    "compile_ok": True,
                    "tool_available": True,
                },
                "subject_route": {
                    "primary_subject": "C_program",
                    "requested_subjects": ["C_program"],
                },
                "upgraded": False,
                "upgrade_reason": "",
                "instant_review": None,
                "raw": {"code_analysis": {"compile_ok": True}},
            }

        with (
            patch.object(self.webapp.store, "persist_sessions_safely", return_value=None),
            patch.object(
                self.webapp.chat_service,
                "_schedule_chat_title_refinement",
                return_value=None,
            ),
            patch.object(
                self.webapp.chat_service,
                "_run_code_analysis_stream",
                new=AsyncMock(side_effect=fake_run_code_analysis_stream),
            ),
        ):
            response = self.client.post(
                f"/api/chats/{self.chat_id}/messages/stream",
                json={
                    "message": "请分析\n```c\nint main(void){return 0;}\n```",
                    "mode": "auto",
                    "subjects": ["C_program"],
                    "code_analysis": True,
                    "stream_chunk_size": 48,
                },
                buffered=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")
        events = self._parse_sse_events(response.get_data(as_text=True))
        event_names = [name for name, _ in events]
        self.assertIn("meta", event_names)
        self.assertIn("delta", event_names)
        self.assertIn("done", event_names)

        meta_events = [data for name, data in events if name == "meta"]
        self.assertTrue(any(data.get("request_kind") == "code_analysis" for data in meta_events))

        delta_events = [data for name, data in events if name == "delta"]
        self.assertTrue(any(data.get("text") == "结构分析已生成。" for data in delta_events))

        done_events = [data for name, data in events if name == "done"]
        self.assertEqual(len(done_events), 1)
        done = done_events[0]
        self.assertEqual(done["answer"], "结构分析已生成。")
        self.assertEqual(done["route"]["chain"], "code_analysis")
        self.assertFalse(done["retrieval_used"])
        self.assertIn("代码分析", str(done["assistant_meta"]))

    def test_short_definition_question_uses_retrieval_gate_instead_of_forced_retrieval(
        self,
    ) -> None:
        fake_decide_need_retrieval = AsyncMock(
            return_value=(False, 0.93, "通用短定义，免检索直答")
        )

        with (
            patch.object(self.webapp.store, "persist_sessions_safely", return_value=None),
            patch.object(
                self.webapp.chat_service,
                "_schedule_chat_title_refinement",
                return_value=None,
            ),
            patch.object(
                self.webapp.chat_service,
                "decide_need_retrieval",
                new=fake_decide_need_retrieval,
            ),
            patch.object(
                self.webapp.chat_service,
                "_stream_llm_text",
                new=AsyncMock(return_value="指针是一个存储地址的变量。"),
            ),
        ):
            response = self.client.post(
                f"/api/chats/{self.chat_id}/messages/stream",
                json={
                    "message": "指针是什么？",
                    "mode": "auto",
                    "stream_chunk_size": 48,
                },
                buffered=True,
            )

        self.assertEqual(response.status_code, 200)
        events = self._parse_sse_events(response.get_data(as_text=True))
        done_events = [data for name, data in events if name == "done"]
        self.assertEqual(len(done_events), 1)
        done = done_events[0]
        self.assertFalse(done["retrieval_used"])
        self.assertEqual(done["route"]["chain"], "direct")
        self.assertEqual(done["retrieval_gate_reason"], "通用短定义，免检索直答")
        fake_decide_need_retrieval.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
