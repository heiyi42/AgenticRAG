from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from webapp_core.problem_tutoring_service import (
    ProblemTemplate,
    ProblemTutoringService,
    TutoringProblemAnalysis,
)


class ProblemTutoringServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ProblemTutoringService()

    def test_detects_page_replacement_problem(self) -> None:
        message = "这题怎么做：给定页面访问序列 7 0 1 2 0 3，页框数为 3，用 FIFO 页面置换，求缺页次数。"
        candidate = self.service.match_request(message)
        self.assertIsNotNone(candidate)

        analysis = self.service.rule_analyze(message)
        self.assertEqual(analysis.subject_id, "operating_systems")
        self.assertEqual(analysis.problem_type, "os_page_replacement")

        template = self.service.select_template(analysis, user_question=message)
        self.assertEqual(template.id, "os_page_replacement")

    def test_detects_c_output_problem(self) -> None:
        message = (
            "下面 C 程序输出什么？\n"
            "```c\n"
            "#include <stdio.h>\n"
            "int main(void) { int a[3] = {1, 2, 3}; printf(\"%d\\n\", a[1]); }\n"
            "```"
        )
        candidate = self.service.match_request(message, requested_subjects=["C_program"])
        self.assertIsNotNone(candidate)

        analysis = self.service.rule_analyze(message, subject_id_hint="C_program")
        self.assertEqual(analysis.subject_id, "C_program")
        self.assertEqual(analysis.problem_type, "c_output")

        template = self.service.select_template(analysis, user_question=message)
        self.assertEqual(template.id, "c_output")

    def test_avoids_product_design_discussion_false_positive(self) -> None:
        message = "题目辅导与过程化解题模块对学习辅助系统有必要吗，好实现吗，咋实现？"
        candidate = self.service.match_request(message)
        self.assertIsNone(candidate)

    def test_avoids_concept_question_false_positive(self) -> None:
        message = "页面置换法算法是啥？"
        candidate = self.service.match_request(message)
        self.assertIsNone(candidate)

    def test_builds_prompt_with_template_sections(self) -> None:
        message = "这题怎么做：银行家算法给出 Available 和 Allocation，求安全序列。"
        analysis = self.service.rule_analyze(message, subject_id_hint="operating_systems")
        template = self.service.select_template(analysis, user_question=message)
        recommendations = self.service.recommend_similar_questions(
            analysis=analysis,
            template=template,
            user_question=message,
            limit=3,
        )
        few_shot_examples = self.service.build_few_shot_examples(recommendations, limit=2)
        prompt = self.service.build_final_prompt(
            user_question=message,
            augmented_question=message,
            analysis=analysis,
            template=template,
            retrieval={"status": "skipped", "message": "test", "answer": ""},
            mode="auto",
            response_language="zh",
            recommendations=recommendations,
            few_shot_examples=few_shot_examples,
            solver_result=self.service.solve_deterministic(
                analysis=analysis,
                template=template,
                user_question=message,
            ),
        )
        self.assertIn("## 题型判断", prompt)
        self.assertIn("银行家算法题", prompt)
        self.assertIn("类题训练方向", prompt)
        self.assertIn("题库相似题推荐", prompt)
        self.assertIn("确定性规则求解结果", prompt)
        self.assertIn("相似题标准解法示例", prompt)
        self.assertIn("solution_steps", prompt)
        self.assertIn("不要照搬示例题的具体数值", prompt)
        self.assertIn("os_005", prompt)

    def test_recommends_real_question_bank_items(self) -> None:
        message = "这题怎么做：给定页面访问序列 7 0 1 2 0 3，页框数为 3，用 FIFO 页面置换，求缺页次数。"
        analysis = self.service.rule_analyze(message)
        template = self.service.select_template(analysis, user_question=message)
        recommendations = self.service.recommend_similar_questions(
            analysis=analysis,
            template=template,
            user_question=message,
            limit=3,
        )
        self.assertTrue(recommendations)
        self.assertEqual(recommendations[0]["subject_id"], "operating_systems")
        self.assertEqual(recommendations[0]["id"], "os_001")
        self.assertIn("question", recommendations[0])

    def test_builds_few_shot_examples_from_question_bank(self) -> None:
        message = "阅读代码：int a[4] = {10, 20, 30, 40}; int *p = &a[1]; p++; printf(\"%d\", *p); 输出什么？"
        analysis = self.service.rule_analyze(message, subject_id_hint="C_program")
        template = self.service.select_template(analysis, user_question=message)
        recommendations = self.service.recommend_similar_questions(
            analysis=analysis,
            template=template,
            user_question=message,
            limit=3,
        )

        examples = self.service.build_few_shot_examples(recommendations, limit=2)

        self.assertGreaterEqual(len(examples), 1)
        self.assertLessEqual(len(examples), 2)
        self.assertIn("answer", examples[0])
        self.assertIn("solution_steps", examples[0])
        self.assertTrue(examples[0]["solution_steps"])
        self.assertIn("use_policy", examples[0])

    def test_builds_learning_outline_from_analysis_solver_and_examples(self) -> None:
        message = "这题怎么做：给定页面访问序列 7 0 1 2 0 3，页框数为 3，用 FIFO 页面置换，求缺页次数。"
        analysis = self.service.rule_analyze(message, subject_id_hint="operating_systems")
        template = self.service.select_template(analysis, user_question=message)
        recommendations = self.service.recommend_similar_questions(
            analysis=analysis,
            template=template,
            user_question=message,
            limit=3,
        )
        few_shot_examples = self.service.build_few_shot_examples(recommendations, limit=2)
        solver_result = self.service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=message,
        )

        outline = self.service.build_learning_outline(
            analysis=analysis,
            template=template,
            recommendations=recommendations,
            few_shot_examples=few_shot_examples,
            solver_result=solver_result,
            retrieval={"status": "success", "answer": "FIFO 会在页框已满时淘汰最早进入内存的页面。"},
        )

        self.assertEqual(outline["kind"], "problem_tutoring")
        self.assertEqual(outline["subject_label"], "操作系统")
        self.assertEqual(outline["problem_type_label"], "页面置换题")
        self.assertTrue(outline["recommended_steps"])
        self.assertTrue(outline["similar_questions"])
        self.assertEqual(outline["similar_questions"][0]["id"], "os_001")
        self.assertIn("缺页", outline["solver"]["summary"])
        self.assertTrue(outline["retrieval"]["summary"])

    def test_solves_fifo_page_replacement_deterministically(self) -> None:
        message = "这题怎么做：给定页面访问序列 7 0 1 2 0 3，页框数为 3，用 FIFO 页面置换，求缺页次数。"
        analysis = self.service.rule_analyze(message)
        template = self.service.select_template(analysis, user_question=message)

        result = self.service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=message,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["solver"], "page_replacement")
        self.assertEqual(result["result"]["algorithm"], "FIFO")
        self.assertEqual(result["result"]["faults"], 5)
        self.assertEqual(result["result"]["hits"], 1)

    def test_solves_rr_scheduling_deterministically(self) -> None:
        message = "P1 到达时间 0、服务时间 5；P2 到达时间 1、服务时间 3。采用时间片轮转 RR，时间片 q=2，求调度过程和完成时间。"
        analysis = self.service.rule_analyze(message, subject_id_hint="operating_systems")
        template = self.service.select_template(analysis, user_question=message)

        result = self.service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=message,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["solver"], "cpu_scheduling")
        self.assertEqual(result["result"]["algorithm"], "RR")
        self.assertEqual(result["result"]["metrics"]["P1"]["completion"], 8)
        self.assertEqual(result["result"]["metrics"]["P2"]["completion"], 7)

    def test_solves_clock_page_replacement_deterministically(self) -> None:
        message = "页面访问序列为 1,2,3,1,4,5，页框数为 3，采用 Clock 页面置换，求缺页次数。"
        analysis = self.service.rule_analyze(message, subject_id_hint="operating_systems")
        template = self.service.select_template(analysis, user_question=message)

        result = self.service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=message,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["solver"], "page_replacement")
        self.assertEqual(result["result"]["algorithm"], "CLOCK")
        self.assertEqual(result["result"]["faults"], 5)
        self.assertEqual(result["result"]["hits"], 1)
        self.assertEqual(result["result"]["trace"][-1]["evicted"], 2)

    def test_solves_srtf_scheduling_deterministically(self) -> None:
        message = "进程 [('P1', 0, 8), ('P2', 1, 4), ('P3', 2, 2)] 采用 SRTF 调度，求完成时间。"
        analysis = self.service.rule_analyze(message, subject_id_hint="operating_systems")
        template = self.service.select_template(analysis, user_question=message)

        result = self.service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=message,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["solver"], "cpu_scheduling")
        self.assertEqual(result["result"]["algorithm"], "SRTF")
        self.assertEqual(result["result"]["metrics"]["P1"]["completion"], 14)
        self.assertEqual(result["result"]["metrics"]["P2"]["completion"], 7)
        self.assertEqual(result["result"]["metrics"]["P3"]["completion"], 4)

    def test_solves_banker_safety_sequence_deterministically(self) -> None:
        message = (
            "银行家算法：Available=(3,3,2)。"
            "Allocation：P0(0,1,0), P1(2,0,0), P2(3,0,2), P3(2,1,1), P4(0,0,2)。"
            "Max：P0(7,5,3), P1(3,2,2), P2(9,0,2), P3(2,2,2), P4(4,3,3)。判断系统是否安全。"
        )
        analysis = self.service.rule_analyze(message, subject_id_hint="operating_systems")
        template = self.service.select_template(analysis, user_question=message)

        result = self.service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=message,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["solver"], "banker")
        self.assertTrue(result["result"]["safe"])
        self.assertEqual(result["result"]["need"]["P1"], (1, 2, 2))
        self.assertEqual(result["result"]["safe_sequence"], ["P1", "P3", "P4", "P0", "P2"])

    def test_solves_banker_request_deterministically(self) -> None:
        message = (
            "银行家算法：Available=(3,3,2)。"
            "Allocation：P0(0,1,0), P1(2,0,0), P2(3,0,2), P3(2,1,1), P4(0,0,2)。"
            "Max：P0(7,5,3), P1(3,2,2), P2(9,0,2), P3(2,2,2), P4(4,3,3)。"
            "P1 请求 Request=(1,0,2)，判断是否可分配。"
        )
        analysis = self.service.rule_analyze(message, subject_id_hint="operating_systems")
        template = self.service.select_template(analysis, user_question=message)

        result = self.service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=message,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["solver"], "banker")
        self.assertEqual(result["result"]["request_process"], "P1")
        self.assertEqual(result["result"]["request"], (1, 0, 2))
        self.assertTrue(result["result"]["grantable"])
        self.assertEqual(result["result"]["trial_available"], (2, 3, 0))
        self.assertEqual(result["result"]["safe_sequence"], ["P1", "P3", "P4", "P0", "P2"])

    def test_builds_pv_sync_structure_deterministically(self) -> None:
        message = "生产者消费者问题，有 5 个缓冲区，用 PV 操作给出信号量设计。"
        analysis = self.service.rule_analyze(message, subject_id_hint="operating_systems")
        template = self.service.select_template(analysis, user_question=message)

        result = self.service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=message,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["solver"], "pv_sync")
        self.assertEqual(result["result"]["scenario"], "producer_consumer")
        semaphores = {item["name"]: item["initial"] for item in result["result"]["semaphores"]}
        self.assertEqual(semaphores["mutex"], 1)
        self.assertEqual(semaphores["empty"], 5)
        self.assertEqual(semaphores["full"], 0)

    def test_solves_dh_modular_arithmetic_deterministically(self) -> None:
        message = "DH 密钥交换实验：公开 p=23，g=5。甲的私钥 a=6，乙的私钥 b=15。求双方公开值和共享密钥。"
        analysis = self.service.rule_analyze(message, subject_id_hint="cybersec_lab")
        template = self.service.select_template(analysis, user_question=message)

        result = self.service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=message,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["solver"], "diffie_hellman")
        self.assertEqual(result["result"]["public_a"], 8)
        self.assertEqual(result["result"]["public_b"], 19)
        self.assertEqual(result["result"]["shared_key"], 2)

class ProblemTutoringEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_question_bank_embedding_index_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            bank_path = tmp_path / "questions.jsonl"
            index_path = tmp_path / "questions.embedding_index.json"
            bank_items = [
                {
                    "id": "os_vec_1",
                    "subject_id": "operating_systems",
                    "problem_type": "os_page_replacement",
                    "knowledge_points": ["页面置换", "FIFO"],
                    "difficulty": "easy",
                    "question": "页面访问序列为 1 2 3，FIFO 求缺页次数。",
                    "answer": "略",
                    "solution_steps": ["略"],
                    "common_mistakes": ["略"],
                },
                {
                    "id": "os_vec_2",
                    "subject_id": "operating_systems",
                    "problem_type": "os_cpu_scheduling",
                    "knowledge_points": ["RR", "时间片"],
                    "difficulty": "medium",
                    "question": "RR 调度题。",
                    "answer": "略",
                    "solution_steps": ["略"],
                    "common_mistakes": ["略"],
                },
            ]
            bank_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in bank_items),
                encoding="utf-8",
            )

            async def fake_embedder(texts: list[str]) -> list[list[float]]:
                self.assertEqual(len(texts), 2)
                return [[1.0, 0.0], [0.0, 1.0]]

            service = ProblemTutoringService(
                question_bank_path=bank_path,
                question_bank_embedding_index_path=index_path,
                question_bank_embed_enabled=True,
                question_bank_embedder=fake_embedder,
            )

            result = await service.build_question_bank_embedding_index()

            self.assertEqual(result["path"], str(index_path))
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["item_count"], 2)
            self.assertEqual(payload["items"][0]["id"], "os_vec_1")
            self.assertEqual(payload["items"][0]["embedding"], [1.0, 0.0])
            self.assertIn("学科:", payload["items"][0]["text"])

    async def test_hybrid_recommendations_use_embedding_when_lexical_is_weak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            bank_path = tmp_path / "questions.jsonl"
            index_path = tmp_path / "questions.embedding_index.json"
            bank_items = [
                {
                    "id": "os_fifo",
                    "subject_id": "operating_systems",
                    "problem_type": "os_page_replacement",
                    "knowledge_points": ["页面置换", "FIFO"],
                    "difficulty": "medium",
                    "question": "FIFO 页面置换，求缺页次数。",
                    "answer": "略",
                    "solution_steps": ["略"],
                    "common_mistakes": ["略"],
                },
                {
                    "id": "os_lru",
                    "subject_id": "operating_systems",
                    "problem_type": "os_page_replacement",
                    "knowledge_points": ["页面置换", "LRU"],
                    "difficulty": "medium",
                    "question": "LRU 页面置换，求缺页次数。",
                    "answer": "略",
                    "solution_steps": ["略"],
                    "common_mistakes": ["略"],
                },
            ]
            bank_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in bank_items),
                encoding="utf-8",
            )
            index_path.write_text(
                json.dumps(
                    {
                        "model": "fake",
                        "items": [
                            {
                                "id": "os_fifo",
                                "subject_id": "operating_systems",
                                "problem_type": "os_page_replacement",
                                "knowledge_points": ["页面置换", "FIFO"],
                                "embedding": [1.0, 0.0],
                            },
                            {
                                "id": "os_lru",
                                "subject_id": "operating_systems",
                                "problem_type": "os_page_replacement",
                                "knowledge_points": ["页面置换", "LRU"],
                                "embedding": [0.0, 1.0],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            async def fake_embedder(texts: list[str]) -> list[list[float]]:
                self.assertEqual(len(texts), 1)
                return [[1.0, 0.0]]

            service = ProblemTutoringService(
                question_bank_path=bank_path,
                question_bank_embedding_index_path=index_path,
                question_bank_embed_enabled=True,
                question_bank_embedder=fake_embedder,
            )
            service.question_bank_embed_trigger_score = 999.0

            analysis = TutoringProblemAnalysis(
                is_problem=True,
                subject_id="operating_systems",
                problem_type="os_page_replacement",
                confidence=0.95,
                target="求缺页次数",
                extracted_conditions=["访问序列", "页框数", "算法"],
                knowledge_points=["页面置换"],
                answer_focus="calculation",
                reason="test",
            )
            template = ProblemTemplate(
                id="os_page_replacement",
                subject_id="operating_systems",
                name="页面置换题",
                problem_type="os_page_replacement",
                aliases=("页面置换",),
                knowledge_keywords=("页面置换", "FIFO", "LRU"),
                steps=("抽条件",),
                output_sections=("结果",),
                solver_hint="test",
                priority=95,
            )

            recommendations = await service.recommend_similar_questions_hybrid(
                analysis=analysis,
                template=template,
                user_question="先进先出缓存淘汰规则下如何计算访问序列结果？",
                limit=2,
            )

            self.assertEqual(recommendations[0]["id"], "os_fifo")
            self.assertEqual(recommendations[0]["subject_id"], "operating_systems")


if __name__ == "__main__":
    unittest.main()
