from __future__ import annotations

import asyncio
import unittest

from webapp_core.chat_service import ChatService


class ChatRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ChatService.__new__(ChatService)
        self.service.subject_catalog = {
            subject_id: {
                "id": subject_id,
                "label": label,
                "working_dir": f"/tmp/{subject_id}",
            }
            for subject_id, label in ChatService.SUBJECT_LABELS.items()
        }

    def test_normalize_requested_subjects_deduplicates_aliases(self) -> None:
        normalized = self.service.normalize_requested_subjects(
            ["c", "操作系统", "cybersec", "C_program", "unknown"]
        )

        self.assertEqual(
            normalized,
            ["C_program", "operating_systems", "cybersec_lab"],
        )

    def test_decide_subject_route_short_circuits_explicit_subjects(self) -> None:
        route = asyncio.run(
            self.service.decide_subject_route(
                user_question="帮我看这道题",
                augmented_question="帮我看这道题",
                mode="auto",
                timeout_s=3,
                requested_subjects=["C_program", "operating_systems"],
            )
        )

        self.assertEqual(route["primary_subject"], "C_program")
        self.assertTrue(route["cross_subject"])
        self.assertEqual(
            route["requested_subjects"],
            ["C_program", "operating_systems"],
        )
        self.assertEqual(route["reason"], "用户显式指定学科")

    def test_build_direct_answer_prompt_keeps_language_instruction(self) -> None:
        prompt = self.service._build_direct_answer_prompt(
            user_question="Explain paging briefly.",
            augmented_question="Explain paging briefly.",
            mode="instant",
            thread_id="thread-1",
            response_language="en",
        )

        self.assertIn("Answer entirely in English.", prompt)
        self.assertIn("thread_id：thread-1", prompt)
        self.assertIn("Explain paging briefly.", prompt)

    def test_score_route_need_retrieval_prefers_direct_answer(self) -> None:
        need_retrieval, route_tag = ChatService._score_route_need_retrieval(
            kb_relevance=0.05,
            direct_answerability=0.92,
            model_need_retrieval=True,
        )

        self.assertFalse(need_retrieval)
        self.assertEqual(route_tag, "score:direct_high")


if __name__ == "__main__":
    unittest.main()
