from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.evaluate_problem_tutoring import (
    evaluate_question_bank,
    format_report,
    grade_candidate_answer,
)


class ProblemTutoringEvaluationTests(unittest.TestCase):
    def test_evaluation_reports_core_metrics(self) -> None:
        report = evaluate_question_bank(detail_limit=3)

        self.assertEqual(report["totals"]["total"], 300)
        self.assertGreaterEqual(report["metrics"]["subject_accuracy_pct"], 99.0)
        self.assertGreaterEqual(report["metrics"]["problem_type_accuracy_pct"], 95.0)
        self.assertEqual(report["metrics"]["recommendation_nonempty_pct"], 100.0)
        self.assertGreaterEqual(report["metrics"]["expected_rule_solver_coverage_pct"], 90.0)
        self.assertGreaterEqual(report["metrics"]["answer_eval_coverage_pct"], 30.0)
        self.assertGreaterEqual(report["metrics"]["answer_eval_avg_score_pct"], 60.0)
        self.assertEqual(set(report["by_subject"]), {"C_program", "operating_systems", "cybersec_lab"})

        rendered = format_report(report)
        self.assertIn("题目辅导离线评测", rendered)
        self.assertIn("规则求解器命中数", rendered)
        self.assertIn("按学科统计", rendered)
        self.assertIn("最终答案平均分", rendered)

    def test_grade_candidate_answer_rewards_reference_alignment(self) -> None:
        item = {
            "id": "os_demo",
            "answer": "缺页次数为 5，命中次数为 1。",
            "solution_steps": [
                "先抽取访问序列和页框数。",
                "按 FIFO 逐步记录页框变化。",
                "统计缺页次数和命中次数。",
            ],
            "common_mistakes": [
                "把初始装入页也漏算。",
            ],
        }

        grade = grade_candidate_answer(
            item,
            (
                "先抽取访问序列和页框数，再按 FIFO 逐步记录页框变化。"
                "最终缺页次数为 5，命中次数为 1。"
                "易错点是不要把初始装入页漏算。"
            ),
        )

        self.assertGreaterEqual(grade["answer_recall_pct"], 80.0)
        self.assertGreaterEqual(grade["step_coverage_pct"], 60.0)
        self.assertGreaterEqual(grade["numeric_consistency_pct"], 100.0)
        self.assertTrue(grade["passed"])

    def test_evaluation_uses_candidate_answer_file_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            bank_path = tmp_path / "questions.jsonl"
            candidate_path = tmp_path / "answers.jsonl"
            bank_item = {
                "id": "os_demo",
                "subject_id": "operating_systems",
                "problem_type": "os_page_replacement",
                "knowledge_points": ["页面置换", "FIFO"],
                "difficulty": "medium",
                "question": "页面访问序列 7 0 1 2 0 3，页框数为 3，FIFO 求缺页次数。",
                "answer": "缺页次数为 5，命中次数为 1。",
                "solution_steps": [
                    "抽取访问序列和页框数。",
                    "按 FIFO 逐步模拟。",
                ],
                "common_mistakes": ["漏算初始缺页。"],
            }
            bank_path.write_text(json.dumps(bank_item, ensure_ascii=False), encoding="utf-8")
            candidate_path.write_text(
                json.dumps(
                    {
                        "id": "os_demo",
                        "answer": "先抽取访问序列和页框数，按 FIFO 逐步模拟，最终缺页次数为 5，命中次数为 1。",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = evaluate_question_bank(
                bank_path,
                detail_limit=2,
                candidate_answers_path=candidate_path,
            )

            self.assertEqual(report["candidate_answers_path"], str(candidate_path))
            self.assertEqual(report["totals"]["answer_eval_available"], 1)
            self.assertEqual(report["metrics"]["answer_eval_coverage_pct"], 100.0)
            self.assertGreaterEqual(report["metrics"]["answer_eval_avg_score_pct"], 70.0)


if __name__ == "__main__":
    unittest.main()
