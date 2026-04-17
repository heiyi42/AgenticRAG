from __future__ import annotations

import unittest
from unittest.mock import patch

import agenticRAG.agentic_nodes as nodes
from agenticRAG.agentic_config import COMPLEX_MAX_RETRY, SIMPLE_MAX_RETRY
from agenticRAG.agentic_schema import AdaptiveQueryPlan, SubQuestionQueryPlan


class _FakeStructuredInvoker:
    def __init__(self, result: object) -> None:
        self.result = result
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return self.result


class AgenticPlanningTests(unittest.TestCase):
    def test_build_query_plan_routes_by_requested_mode(self) -> None:
        calls: list[str] = []

        def fake_simple(state: dict) -> dict:
            calls.append(f"simple:{state.get('requested_mode')}")
            return {"effective_strategy": "simple"}

        def fake_deep(state: dict) -> dict:
            calls.append(f"deep:{state.get('requested_mode')}")
            return {"effective_strategy": "deep"}

        def fake_auto(state: dict) -> dict:
            calls.append(f"auto:{state.get('requested_mode')}")
            return {"effective_strategy": "deep"}

        with (
            patch.object(nodes, "build_simple_query_plan", side_effect=fake_simple),
            patch.object(nodes, "build_subquestion_query_plan", side_effect=fake_deep),
            patch.object(nodes, "build_auto_query_plan", side_effect=fake_auto),
        ):
            self.assertEqual(
                nodes.build_query_plan({"question": "Q1", "requested_mode": "instant"}),
                {"effective_strategy": "simple"},
            )
            self.assertEqual(
                nodes.build_query_plan(
                    {"question": "Q2", "requested_mode": "deepsearch"}
                ),
                {"effective_strategy": "deep"},
            )
            self.assertEqual(
                nodes.build_query_plan({"question": "Q3", "requested_mode": "auto"}),
                {"effective_strategy": "deep"},
            )

        self.assertEqual(
            calls,
            ["simple:instant", "deep:deepsearch", "auto:auto"],
        )

    def test_build_auto_query_plan_trims_simple_to_single_query(self) -> None:
        fake_planner = _FakeStructuredInvoker(
            AdaptiveQueryPlan(
                complexity="simple",
                reason="单次检索足够",
                sub_questions=["改写后的单问题", "不该保留的额外问题"],
                query_modes=["local", "global"],
                query_topks=[999, 1],
                query_chunk_topks=[-5, 3],
            )
        )

        with patch.object(nodes, "llm_adaptive_plan_struct", fake_planner):
            result = nodes.build_auto_query_plan({"question": "解释什么是指针"})

        self.assertEqual(result["requested_mode"], "auto")
        self.assertEqual(result["question_complexity"], "simple")
        self.assertEqual(result["effective_strategy"], "simple")
        self.assertEqual(result["planning_reason"], "单次检索足够")
        self.assertEqual(result["sub_questions"], ["改写后的单问题"])
        self.assertEqual(result["query_modes"], ["local"])
        self.assertEqual(len(result["query_topks"]), 1)
        self.assertEqual(len(result["query_chunk_topks"]), 1)

    def test_build_subquestion_query_plan_preserves_dynamic_count(self) -> None:
        fake_planner = _FakeStructuredInvoker(
            SubQuestionQueryPlan(
                sub_questions=["Q1", "Q2", "Q3", "Q4"],
                query_modes=["local", "global", "hybrid", "global"],
                query_topks=[1, 2, 3, 4],
                query_chunk_topks=[2, 3, 4, 5],
            )
        )

        with patch.object(nodes, "llm_subquestion_plan_struct", fake_planner):
            result = nodes.build_subquestion_query_plan(
                {"question": "复杂问题", "requested_mode": "deepsearch"}
            )

        self.assertEqual(result["requested_mode"], "deepsearch")
        self.assertEqual(result["effective_strategy"], "deep")
        self.assertEqual(len(result["sub_questions"]), 4)
        self.assertEqual(result["query_modes"], ["local", "global", "hybrid", "global"])
        self.assertIn("deepsearch", result["planning_reason"])

    def test_allowed_retry_budget_prefers_effective_strategy(self) -> None:
        self.assertEqual(
            nodes._allowed_retry_budget(
                {"requested_mode": "deepsearch", "question_complexity": ""}
            ),
            COMPLEX_MAX_RETRY,
        )
        self.assertEqual(
            nodes._allowed_retry_budget(
                {
                    "requested_mode": "deepsearch",
                    "question_complexity": "",
                    "effective_strategy": "simple",
                }
            ),
            SIMPLE_MAX_RETRY,
        )


if __name__ == "__main__":
    unittest.main()
