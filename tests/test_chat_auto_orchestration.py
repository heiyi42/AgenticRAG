from __future__ import annotations

import unittest

from webapp_core.chat_service import ChatService


class ChatAutoOrchestrationTests(unittest.IsolatedAsyncioTestCase):
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
        return service

    async def test_plan_auto_route_prefers_user_locked_single_subject(self) -> None:
        service = self._build_service()

        plan = await service._plan_auto_route(
            subject_route={
                "requested_subjects": ["C_program"],
                "ranked": [("C_program", 1.0)],
                "cross_subject": False,
                "max_score": 1.0,
                "confidence": 0.99,
            },
            augmented_question="解释指针和数组的区别",
            timeout_s=12,
        )

        self.assertEqual(plan["auto_subjects"], ["C_program"])
        self.assertEqual(plan["deep_subjects"], ["C_program"])
        self.assertEqual(plan["complexity"], "simple")
        self.assertEqual(plan["route_chain"], "instant")
        self.assertEqual(plan["route_policy"], "user-locked single-subject -> instant first")
        self.assertEqual(plan["route_reason"], "用户手动指定单学科，auto 优先单库快速回答")

    async def test_run_multi_subject_auto_instant_stream_merges_secondary_answer(self) -> None:
        service = self._build_service()
        captured: dict[str, object] = {}

        async def fake_resolve_auto_secondary_result(**kwargs: object):
            captured["resolve_kwargs"] = dict(kwargs)
            return {"answer": "来自第二学科的补充解释"}, "secondary_direct"

        async def fake_stream_synthesize_subject_answers_with_review(**kwargs: object):
            captured["merge_kwargs"] = dict(kwargs)
            emit_text = kwargs.get("emit_text")
            if callable(emit_text):
                emit_text("整合后的最终回答")
            return {
                "answer": "整合后的最终回答",
                "sufficient": True,
                "reason": "双学科覆盖已经足够",
            }

        service._resolve_auto_secondary_result = fake_resolve_auto_secondary_result  # type: ignore[method-assign]
        service._stream_synthesize_subject_answers_with_review = fake_stream_synthesize_subject_answers_with_review  # type: ignore[method-assign]
        emitted: list[str] = []

        result = await service._run_multi_subject_auto_instant_stream(
            user_question="解释栈溢出为什么会和操作系统有关",
            question="解释栈溢出为什么会和操作系统有关",
            thread_id="thread-1",
            timeout_s=10,
            subject_ids=["C_program", "operating_systems"],
            primary_answer="来自主学科的初始回答",
            response_language="zh",
            emit_text=emitted.append,
        )

        self.assertEqual(result["mode_used"], "instant")
        self.assertEqual(result["answer"], "整合后的最终回答")
        self.assertTrue(result["review_sufficient"])
        self.assertEqual(result["review_reason"], "双学科覆盖已经足够")
        self.assertEqual(result["route"]["chain"], "auto-dual-subject-instant")
        self.assertEqual(
            [item["subject_id"] for item in captured["merge_kwargs"]["subject_answers"]],
            ["C_program", "operating_systems"],
        )
        self.assertEqual(emitted, ["整合后的最终回答"])

    async def test_run_multi_subject_deep_stream_uses_subquestion_routed_chain(self) -> None:
        service = self._build_service()
        captured: dict[str, object] = {}

        async def fake_stream_routed_deepsearch_mode(**kwargs: object):
            captured["kwargs"] = dict(kwargs)
            emit_text = kwargs.get("emit_text")
            if callable(emit_text):
                emit_text("子问题路由后的最终回答")
            return {
                "mode_used": "deepsearch",
                "answer": "子问题路由后的最终回答",
            }

        service._stream_routed_deepsearch_mode = fake_stream_routed_deepsearch_mode  # type: ignore[method-assign]
        emitted: list[str] = []

        result = await service._run_multi_subject_deep_stream(
            user_question="解释栈溢出为什么同时和 C、OS、安全有关",
            question="解释栈溢出为什么同时和 C、OS、安全有关",
            thread_id="thread-1",
            timeout_s=12,
            subject_ids=["C_program", "operating_systems"],
            response_language="zh",
            emit_text=emitted.append,
        )

        self.assertEqual(result["mode_used"], "deepsearch")
        self.assertEqual(result["answer"], "子问题路由后的最终回答")
        self.assertEqual(result["route"]["chain"], "deepsearch-subquestion-routed")
        self.assertEqual(
            captured["kwargs"]["allowed_subject_ids"],
            ["C_program", "operating_systems"],
        )
        self.assertEqual(
            captured["kwargs"]["routing_question"],
            "解释栈溢出为什么同时和 C、OS、安全有关",
        )
        self.assertEqual(emitted, ["子问题路由后的最终回答"])


if __name__ == "__main__":
    unittest.main()
