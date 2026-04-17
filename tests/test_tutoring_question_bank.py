from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


class TutoringQuestionBankTests(unittest.TestCase):
    def test_question_bank_has_three_subjects_with_one_hundred_questions_each(self) -> None:
        path = Path("data/tutoring_question_bank/questions.jsonl")
        self.assertTrue(path.exists())

        questions = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(questions), 300)

        counts = Counter(item["subject_id"] for item in questions)
        self.assertEqual(counts["C_program"], 100)
        self.assertEqual(counts["operating_systems"], 100)
        self.assertEqual(counts["cybersec_lab"], 100)

        required_fields = {
            "id",
            "subject_id",
            "problem_type",
            "knowledge_points",
            "difficulty",
            "question",
            "answer",
            "solution_steps",
            "common_mistakes",
        }
        for item in questions:
            self.assertTrue(required_fields.issubset(item.keys()), item.get("id"))
            self.assertTrue(item["question"], item["id"])
            self.assertTrue(item["answer"], item["id"])
            self.assertTrue(item["solution_steps"], item["id"])


if __name__ == "__main__":
    unittest.main()
