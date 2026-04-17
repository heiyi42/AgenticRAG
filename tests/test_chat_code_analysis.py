from __future__ import annotations

import unittest

from webapp_core import config as cfg
from webapp_core.chat_service import ChatService


class _FakeCodeAnalysisService:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    async def analyze_code(self, code: str, timeout_s: int) -> dict[str, object]:
        self.calls.append((code, timeout_s))
        return dict(self.result)


class ChatCodeAnalysisTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _build_service(result: dict[str, object]) -> tuple[ChatService, _FakeCodeAnalysisService]:
        service = ChatService.__new__(ChatService)
        fake_service = _FakeCodeAnalysisService(result)
        service.code_analysis_service = fake_service
        return service, fake_service

    def test_clean_code_analysis_question_strips_code_blocks(self) -> None:
        cleaned = ChatService._clean_code_analysis_question(
            "请分析这段代码：\n```c\nint main(void){return 0;}\n```",
            "int main(void){return 0;}",
        )

        self.assertEqual(cleaned, "请分析这段代码")

    async def test_run_code_analysis_stream_handles_missing_compiler(self) -> None:
        service, fake_service = self._build_service(
            {
                "code": "int main(void){return 0;}",
                "tool": "",
                "tool_available": False,
                "tool_error": "clang not found",
                "compile_ok": False,
            }
        )
        emitted: list[str] = []

        result = await service._run_code_analysis_stream(
            user_question="请分析这段代码",
            mode="auto",
            timeout_s=9,
            response_language="zh",
            code_candidate={"code": "int main(void){return 0;}", "trigger": "explicit"},
            emit_text=emitted.append,
        )

        expected_analysis_timeout = max(
            3,
            min(9, int(cfg.WEB_CODE_ANALYSIS_TIMEOUT_S)),
        )
        self.assertEqual(
            fake_service.calls,
            [("int main(void){return 0;}", expected_analysis_timeout)],
        )
        self.assertEqual(result["route"]["chain"], "code_analysis")
        self.assertFalse(result["route"]["tool_available"])
        self.assertIn("没有检测到可用的 C 编译器", result["answer"])
        self.assertEqual(emitted, [result["answer"]])

    async def test_run_code_analysis_stream_builds_prompt_from_tool_payload(self) -> None:
        service, fake_service = self._build_service(
            {
                "code": "int main(void){return 0;}",
                "tool": "clang",
                "tool_available": True,
                "tool_error": "",
                "compiler": "clang",
                "compile_ok": True,
                "compile_errors": [],
                "warnings": ["unused variable"],
                "risk_findings": [],
                "structure": {"functions": ["main"]},
                "tool_runs": {"compile": {"ok": True}},
            }
        )
        captured: dict[str, object] = {}
        emitted: list[str] = []

        async def fake_stream_llm_text(**kwargs: object) -> str:
            captured.update(kwargs)
            emit_text = kwargs.get("emit_text")
            if callable(emit_text):
                emit_text("结构分析已生成。")
            return "结构分析已生成。"

        service._stream_llm_text = fake_stream_llm_text  # type: ignore[method-assign]

        result = await service._run_code_analysis_stream(
            user_question="请分析这段代码：\n```c\nint main(void){return 0;}\n```",
            mode="instant",
            timeout_s=12,
            response_language="zh",
            code_candidate={
                "code": "int main(void){return 0;}",
                "trigger": "auto",
                "task_type": "code_reading",
                "strategy": "fenced",
                "score": 87,
            },
            emit_text=emitted.append,
        )

        expected_analysis_timeout = max(
            3,
            min(12, int(cfg.WEB_CODE_ANALYSIS_TIMEOUT_S)),
        )
        expected_explanation_timeout = max(
            1,
            12 - min(
                expected_analysis_timeout,
                max(1, expected_analysis_timeout // 2),
            ),
        )
        self.assertEqual(
            fake_service.calls,
            [("int main(void){return 0;}", expected_analysis_timeout)],
        )
        self.assertEqual(result["answer"], "结构分析已生成。")
        self.assertTrue(result["route"]["compile_ok"])
        self.assertEqual(result["route"]["tool"], "clang")
        self.assertEqual(emitted, ["结构分析已生成。"])
        self.assertEqual(captured["timeout_s"], expected_explanation_timeout)
        self.assertIn('"compile_ok": true', str(captured["prompt"]))
        self.assertIn('"warnings": [', str(captured["prompt"]))
        self.assertIn("[User question]\n请分析这段代码", str(captured["prompt"]))


if __name__ == "__main__":
    unittest.main()
