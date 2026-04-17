from __future__ import annotations

from contextlib import nullcontext
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import agenticRAG.agentic_answer as agentic_answer_module
import agenticRAG.instant_answer as instant_answer_module


class AgenticAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_question_builds_final_answer_without_graph(self) -> None:
        with (
            patch.object(
                agentic_answer_module,
                "run_question_plan_state",
                new=AsyncMock(
                    return_value={
                        "requested_mode": "deepsearch",
                        "query_results": [{"question": "Q1", "answer": "A1"}],
                        "query_attempt": 0,
                    }
                ),
            ) as mocked_runner,
            patch.object(
                agentic_answer_module,
                "build_final_answer_prompt",
                return_value="final prompt",
            ) as mocked_prompt,
            patch.object(
                agentic_answer_module,
                "llm",
                SimpleNamespace(
                    ainvoke=AsyncMock(return_value=SimpleNamespace(content="最终回答"))
                ),
            ),
        ):
            result = await agentic_answer_module.answer_question(
                "测试问题",
                thread_id="thread-1",
                working_dir="/tmp/os",
            )

        mocked_runner.assert_awaited_once_with(
            "测试问题",
            requested_mode="deepsearch",
            working_dir="/tmp/os",
        )
        mocked_prompt.assert_called_once_with(
            {
                "requested_mode": "deepsearch",
                "query_results": [{"question": "Q1", "answer": "A1"}],
                "query_attempt": 0,
            }
        )
        self.assertEqual(result["final_answer"], "最终回答")

    async def test_run_question_plan_state_retries_by_subquestion(self) -> None:
        query_rounds: list[int] = []
        routed: dict[str, dict[str, object]] = {}

        def fake_build_global_subquestion_plan(state: dict[str, object]) -> dict[str, object]:
            self.assertEqual(state["question"], "测试问题")
            self.assertEqual(state["requested_mode"], "deepsearch")
            return {
                "requested_mode": "deepsearch",
                "effective_strategy": "deep",
                "sub_questions": [
                    {
                        "id": "sq1",
                        "question": "Q1",
                        "used_question": "Q1",
                        "query_mode": "local",
                        "top_k": 3,
                        "chunk_top_k": 5,
                        "target_subjects": [],
                    },
                    {
                        "id": "sq2",
                        "question": "Q2",
                        "used_question": "Q2",
                        "query_mode": "global",
                        "top_k": 4,
                        "chunk_top_k": 6,
                        "target_subjects": [],
                    },
                ],
                "subquery_tasks": [],
                "subquery_results": [],
                "query_attempt": 0,
                "needs_retry": False,
                "insufficient_subquestion_ids": [],
                "query_total_ms": "0",
            }

        def fake_attach_subquestion_routes(
            state: dict[str, object],
            routed_subjects: dict[str, dict[str, object]],
        ) -> dict[str, object]:
            del state
            routed.update(routed_subjects)
            return {
                "sub_questions": [
                    {
                        "id": "sq1",
                        "question": "Q1",
                        "used_question": "Q1",
                        "query_mode": "local",
                        "top_k": 3,
                        "chunk_top_k": 5,
                        "target_subjects": ["C_program"],
                    },
                    {
                        "id": "sq2",
                        "question": "Q2",
                        "used_question": "Q2",
                        "query_mode": "global",
                        "top_k": 4,
                        "chunk_top_k": 6,
                        "target_subjects": ["operating_systems"],
                    },
                ]
            }

        def fake_build_subquery_tasks(state: dict[str, object]) -> dict[str, object]:
            return {
                "subquery_tasks": [
                    {
                        "task_id": "sq1::C_program",
                        "sub_question_id": "sq1",
                        "subject_id": "C_program",
                        "question": "Q1",
                        "used_question": state["sub_questions"][0]["used_question"],
                    },
                    {
                        "task_id": "sq2::operating_systems",
                        "sub_question_id": "sq2",
                        "subject_id": "operating_systems",
                        "question": "Q2",
                        "used_question": state["sub_questions"][1]["used_question"],
                    },
                ]
            }

        async def fake_query_subquestion_tasks(state: dict[str, object]) -> dict[str, object]:
            query_rounds.append(int(state.get("query_attempt", 0)))
            return {
                "subquery_results": [
                    {
                        "sub_question_id": "sq1",
                        "subject_id": "C_program",
                        "question": "Q1",
                        "used_question": state["sub_questions"][0]["used_question"],
                        "answer": "第一轮结果" if len(query_rounds) == 1 else "第二轮结果",
                        "query_status": "success",
                        "query_message": "",
                        "query_failure_reason": "",
                    },
                    {
                        "sub_question_id": "sq2",
                        "subject_id": "operating_systems",
                        "question": "Q2",
                        "used_question": "Q2",
                        "answer": "稳定结果",
                        "query_status": "success",
                        "query_message": "",
                        "query_failure_reason": "",
                    },
                ]
            }

        async def fake_judge_subquestion_results(
            state: dict[str, object],
        ) -> dict[str, object]:
            if len(query_rounds) == 1:
                return {
                    "sub_questions": [
                        {
                            **state["sub_questions"][0],
                            "sufficient": "False",
                            "judge_reason": "需要补充第二学科证据",
                            "rewritten_question": "Q1-改写",
                        },
                        {
                            **state["sub_questions"][1],
                            "sufficient": "True",
                            "judge_reason": "证据充分",
                            "rewritten_question": "",
                        },
                    ],
                    "needs_retry": True,
                    "insufficient_subquestion_ids": ["sq1"],
                }
            return {
                "sub_questions": [
                    {
                        **state["sub_questions"][0],
                        "sufficient": "True",
                        "judge_reason": "证据已充分",
                        "rewritten_question": "",
                    },
                    state["sub_questions"][1],
                ],
                "needs_retry": False,
                "insufficient_subquestion_ids": [],
            }

        def fake_prepare_subquestion_retry_plan(state: dict[str, object]) -> dict[str, object]:
            self.assertEqual(state["insufficient_subquestion_ids"], ["sq1"])
            return {
                "query_attempt": 1,
                "sub_questions": [
                    {
                        **state["sub_questions"][0],
                        "used_question": "Q1-改写",
                        "query_mode": "hybrid",
                        "top_k": 5,
                        "chunk_top_k": 8,
                        "target_subjects": ["C_program", "operating_systems"],
                        "sufficient": "unknown",
                        "judge_reason": "",
                        "rewritten_question": "",
                    },
                    state["sub_questions"][1],
                ],
                "subquery_tasks": [],
                "subquery_results": [],
            }

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_route_subquestion_subjects(
            *,
            sub_question: str,
            original_question: str,
        ) -> dict[str, object]:
            self.assertEqual(original_question, "原始问题")
            if sub_question == "Q1":
                return {
                    "primary_subject": "C_program",
                    "target_subjects": ["C_program"],
                    "reason": "Q1 更偏向 C语言",
                }
            return {
                "primary_subject": "operating_systems",
                "target_subjects": ["operating_systems"],
                "reason": "Q2 更偏向操作系统",
            }

        with (
            patch.object(
                agentic_answer_module,
                "use_rag_working_dir",
                return_value=nullcontext(),
            ),
            patch.object(
                agentic_answer_module.asyncio,
                "to_thread",
                side_effect=fake_to_thread,
            ),
            patch.object(
                agentic_answer_module,
                "build_global_subquestion_plan",
                side_effect=fake_build_global_subquestion_plan,
            ),
            patch.object(
                agentic_answer_module,
                "attach_subquestion_routes",
                side_effect=fake_attach_subquestion_routes,
            ),
            patch.object(
                agentic_answer_module,
                "build_subquery_tasks",
                side_effect=fake_build_subquery_tasks,
            ),
            patch.object(
                agentic_answer_module,
                "query_subquestion_tasks",
                side_effect=fake_query_subquestion_tasks,
            ),
            patch.object(
                agentic_answer_module,
                "judge_subquestion_results",
                side_effect=fake_judge_subquestion_results,
            ),
            patch.object(
                agentic_answer_module,
                "prepare_subquestion_retry_plan",
                side_effect=fake_prepare_subquestion_retry_plan,
            ),
        ):
            state = await agentic_answer_module.run_question_plan_state(
                "测试问题",
                requested_mode="deepsearch",
                routing_question="原始问题",
                allowed_subject_ids=["C_program", "operating_systems"],
                subject_working_dirs={
                    "C_program": "/tmp/C_program",
                    "operating_systems": "/tmp/operating_systems",
                },
                route_subquestion_subjects=fake_route_subquestion_subjects,
            )

        self.assertEqual(query_rounds, [0, 1])
        self.assertEqual(state["requested_mode"], "deepsearch")
        self.assertEqual(state["query_attempt"], 1)
        self.assertEqual(routed["sq1"]["primary_subject"], "C_program")
        self.assertEqual(routed["sq2"]["primary_subject"], "operating_systems")
        self.assertEqual(state["sub_questions"][0]["used_question"], "Q1-改写")
        self.assertEqual(state["sub_questions"][0]["target_subjects"], ["C_program", "operating_systems"])
        self.assertEqual(state["subquery_results"][0]["answer"], "第二轮结果")

    async def test_answer_instant_uses_routed_helper_without_graph(self) -> None:
        with patch.object(
            instant_answer_module,
            "_answer_instant_query",
            new=AsyncMock(
                return_value={
                    "route_mode": "hybrid",
                    "route_reason": "routed",
                    "answer": "instant-answer",
                    "elapsed_ms": "11",
                    "query_status": "success",
                    "query_message": "",
                    "query_failure_reason": "",
                    "raw": {},
                }
            ),
        ) as mocked_query:
            result = await instant_answer_module.answer_instant(
                "测试 instant",
                thread_id="thread-2",
                working_dir="/tmp/c",
            )

        mocked_query.assert_awaited_once_with(
            "测试 instant",
            working_dir="/tmp/c",
            stream=False,
        )
        self.assertEqual(result["answer"], "instant-answer")
