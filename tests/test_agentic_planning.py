from __future__ import annotations

import unittest
from unittest.mock import patch

import agenticRAG.agentic_nodes as nodes
from agenticRAG.agentic_config import COMPLEX_MAX_RETRY, SIMPLE_MAX_RETRY
from agenticRAG.agentic_schema import SubQuestionQueryPlan


class _FakeStructuredInvoker:
    def __init__(self, result: object) -> None:
        self.result = result
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return self.result


class AgenticPlanningTests(unittest.TestCase):
    def test_build_global_subquestion_plan_preserves_dynamic_count(self) -> None:
        fake_planner = _FakeStructuredInvoker(
            SubQuestionQueryPlan(
                sub_questions=["Q1", "Q2", "Q3", "Q4"],
                query_modes=["local", "global", "hybrid", "global"],
                query_topks=[1, 2, 999, 4],
                query_chunk_topks=[-5, 3, 4, 5],
            )
        )

        with patch.object(nodes, "llm_subquestion_plan_struct", fake_planner):
            result = nodes.build_global_subquestion_plan(
                {"question": "复杂问题", "requested_mode": "deepsearch"}
            )

        self.assertEqual(result["requested_mode"], "deepsearch")
        self.assertEqual(result["effective_strategy"], "deep")
        self.assertEqual(len(result["sub_questions"]), 4)
        self.assertEqual(
            [item["query_mode"] for item in result["sub_questions"]],
            ["local", "global", "hybrid", "global"],
        )
        self.assertEqual(result["sub_questions"][0]["top_k"], nodes.MIN_TOP_K)
        self.assertEqual(result["sub_questions"][2]["top_k"], nodes.MAX_TOP_K)
        self.assertEqual(
            result["sub_questions"][0]["chunk_top_k"],
            nodes.MIN_CHUNK_TOP_K,
        )
        self.assertEqual(result["subquery_tasks"], [])
        self.assertEqual(result["subquery_results"], [])
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
