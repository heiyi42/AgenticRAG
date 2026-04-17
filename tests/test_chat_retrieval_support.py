from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from webapp_core import config as cfg
import webapp_core.chat_retrieval_support as retrieval_support_module
from webapp_core.chat_service import ChatService


class _DummyStore:
    @staticmethod
    def content_to_text(value: object) -> str:
        return str(getattr(value, "content", value) or "")


class _FakeProblemTutoringService:
    def __init__(self, prepared: dict[str, object]) -> None:
        self.prepared = prepared
        self.calls: list[dict[str, object]] = []

    async def prepare(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return dict(self.prepared)


class _FakeStreamingLLM:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = list(chunks)

    async def astream(self, prompt: str):
        del prompt
        for chunk in self.chunks:
            yield SimpleNamespace(content=chunk)


class ChatRetrievalSupportTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _build_service() -> ChatService:
        service = ChatService.__new__(ChatService)
        service.subject_catalog = {
            subject_id: {
                "id": subject_id,
                "label": label,
                "working_dir": f"/tmp/{subject_id}",
            }
            for subject_id, label in ChatService.SUBJECT_LABELS.items()
        }
        service.store = _DummyStore()
        return service

    def test_pick_problem_tutoring_subject_prefers_requested_subject(self) -> None:
        service = self._build_service()

        subject_id = service._pick_problem_tutoring_subject(
            subject_route={
                "requested_subjects": ["operating_systems"],
                "primary_subject": "C_program",
                "ranked": [("C_program", 0.9)],
            },
            tutoring_candidate={"analysis": {"subject_id": "cybersec_lab"}},
        )

        self.assertEqual(subject_id, "operating_systems")

    def test_parse_subject_synthesis_review_response(self) -> None:
        parsed = ChatService._parse_subject_synthesis_review_response(
            "SUFFICIENT: true\n"
            "REASON: Coverage is complete.\n"
            "ANSWER:\n"
            "## 结论\n"
            "答案已经足够。\n"
        )

        self.assertTrue(parsed["parsed"])
        self.assertTrue(parsed["sufficient"])
        self.assertEqual(parsed["reason"], "Coverage is complete.")
        self.assertIn("## 结论", parsed["answer"])

    async def test_run_problem_tutoring_stream_uses_prepared_prompt(self) -> None:
        service = self._build_service()
        fake_problem_tutoring = _FakeProblemTutoringService(
            {
                "prompt": "请按模板解答这道题。",
                "template": {"id": "os-banker"},
                "analysis": {"problem_type": "banker"},
                "solver_result": {"status": "success", "solver": "banker"},
                "learning_outline": {
                    "kind": "problem_tutoring",
                    "subject_label": "操作系统",
                    "problem_type_label": "银行家算法题",
                },
            }
        )
        service.problem_tutoring_service = fake_problem_tutoring
        captured: dict[str, object] = {}

        async def fake_stream_llm_text(**kwargs: object) -> str:
            captured.update(kwargs)
            emit_text = kwargs.get("emit_text")
            if callable(emit_text):
                emit_text("规则求解已完成。")
            return "规则求解已完成。"

        service._stream_llm_text = fake_stream_llm_text  # type: ignore[method-assign]
        emitted: list[str] = []

        result = await service._run_problem_tutoring_stream(
            user_question="请解答银行家算法题",
            augmented_question="请解答银行家算法题",
            mode="auto",
            timeout_s=12,
            subject_route={
                "requested_subjects": ["operating_systems"],
                "primary_subject": "operating_systems",
                "confidence": 0.95,
                "reason": "用户显式指定学科",
                "ranked": [("operating_systems", 1.0)],
            },
            response_language="zh",
            tutoring_candidate={"trigger": "explicit"},
            emit_text=emitted.append,
        )

        expected_prep_timeout = max(
            3,
            min(12, int(cfg.WEB_PROBLEM_TUTORING_PREP_TIMEOUT_S)),
        )
        self.assertEqual(len(fake_problem_tutoring.calls), 1)
        self.assertEqual(
            fake_problem_tutoring.calls[0]["working_dir"],
            "/tmp/operating_systems",
        )
        self.assertEqual(
            fake_problem_tutoring.calls[0]["timeout_s"],
            expected_prep_timeout,
        )
        self.assertEqual(result["answer"], "规则求解已完成。")
        self.assertEqual(result["route"]["chain"], "problem_tutoring")
        self.assertEqual(result["route"]["subject"], "operating_systems")
        self.assertEqual(result["route"]["template_id"], "os-banker")
        self.assertEqual(result["route"]["solver_status"], "success")
        self.assertEqual(result["route"]["reason"], "explicit")
        self.assertEqual(result["message_details"]["kind"], "problem_tutoring")
        self.assertEqual(result["message_details"]["subject_label"], "操作系统")
        self.assertEqual(emitted, ["规则求解已完成。"])
        self.assertEqual(captured["prompt"], "请按模板解答这道题。")

    async def test_stream_llm_text_flushes_accumulated_chunks(self) -> None:
        service = self._build_service()
        emitted: list[str] = []

        answer = await service._stream_llm_text(
            llm_client=_FakeStreamingLLM(["Hello", " world.", " Done"]),
            prompt="irrelevant",
            timeout_s=3,
            emit_text=emitted.append,
            flush_chars=6,
        )

        self.assertEqual(answer, "Hello world. Done")
        self.assertEqual(emitted, ["Hello world.", " Done"])

    async def test_ask_instant_mode_uses_agenticrag_instant_helper(self) -> None:
        service = self._build_service()
        with patch.object(
            retrieval_support_module,
            "answer_instant",
            new=AsyncMock(
                return_value={
                    "answer": "统一 instant 答案",
                    "query_status": "success",
                    "query_message": "",
                    "elapsed_ms": "12",
                    "route_mode": "hybrid",
                    "route_reason": "routed",
                }
            ),
        ) as mocked_answer:
            result = await service.ask_instant_mode(
                "测试问题",
                "thread-1",
                10,
                working_dir="/tmp/C_program",
            )

        mocked_answer.assert_awaited_once_with(
            "测试问题",
            thread_id="thread-1",
            working_dir="/tmp/C_program",
        )
        self.assertEqual(result["mode_used"], "instant")
        self.assertEqual(result["answer"], "统一 instant 答案")
        self.assertEqual(result["query_status"], "success")
        self.assertEqual(result["elapsed_ms"], "12")

    async def test_ask_instant_mode_stream_uses_agenticrag_stream_helper(self) -> None:
        service = self._build_service()
        emitted: list[str] = []

        async def fake_iterator():
            for chunk in ["第一段", "第二段"]:
                yield chunk

        with patch.object(
            retrieval_support_module,
            "answer_instant_stream",
            new=AsyncMock(
                return_value={
                    "answer": "",
                    "query_status": "success",
                    "query_message": "",
                    "elapsed_ms": "8",
                    "response_iterator": fake_iterator(),
                }
            ),
        ) as mocked_answer:
            result = await service.ask_instant_mode_stream(
                "测试问题",
                "thread-2",
                10,
                working_dir="/tmp/C_program",
                emit_text=emitted.append,
            )

        mocked_answer.assert_awaited_once_with(
            "测试问题",
            thread_id="thread-2",
            working_dir="/tmp/C_program",
        )
        self.assertEqual(result["answer"], "第一段第二段")
        self.assertEqual(emitted, ["第一段", "第二段"])
        self.assertEqual(result["query_status"], "success")

    async def test_run_deepsearch_plan_state_delegates_to_agenticrag_kernel(self) -> None:
        service = self._build_service()

        with patch.object(
            retrieval_support_module,
            "run_question_plan_state",
            new=AsyncMock(
                return_value={
                    "requested_mode": "deepsearch",
                    "effective_strategy": "deep",
                    "sub_questions": ["Q1", "Q2"],
                }
            ),
        ) as mocked_runner:
            state = await service._run_deepsearch_plan_state(question="测试问题")

        mocked_runner.assert_awaited_once()
        args = mocked_runner.await_args.args
        kwargs = mocked_runner.await_args.kwargs
        self.assertEqual(args, ("测试问题",))
        self.assertEqual(kwargs["requested_mode"], "deepsearch")
        self.assertIsNone(kwargs["routing_question"])
        self.assertEqual(kwargs["response_language"], "zh")
        self.assertCountEqual(
            kwargs["allowed_subject_ids"],
            list(service.subject_catalog.keys()),
        )
        self.assertEqual(
            kwargs["subject_working_dirs"],
            {
                subject_id: f"/tmp/{subject_id}"
                for subject_id in service.subject_catalog
            },
        )
        self.assertTrue(callable(kwargs["route_subquestion_subjects"]))
        self.assertEqual(state["requested_mode"], "deepsearch")
        self.assertEqual(state["effective_strategy"], "deep")
        self.assertEqual(state["sub_questions"], ["Q1", "Q2"])


if __name__ == "__main__":
    unittest.main()
