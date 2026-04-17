from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from webapp_core.code_analysis_service import CCompiler, CodeAnalysisService


class CodeAnalysisServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = CodeAnalysisService()

    def test_extracts_fenced_c_code(self) -> None:
        message = (
            "帮我看看这段代码为什么报错\n\n"
            "```c\n"
            "#include <stdio.h>\n"
            "int main(void) {\n"
            "    printf(\"hi\\n\");\n"
            "    return 0;\n"
            "}\n"
            "```"
        )
        candidate = self.service.extract_c_code_candidate(message)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["language"], "c")
        self.assertIn("int main", candidate["code"])

    def test_extracts_flattened_unclosed_fenced_c_code(self) -> None:
        message = (
            '请分析代码 ```c #include <stdio.h> int main(void) { '
            'printf("hi\\n"); return 0; }'
        )
        candidate = self.service.extract_c_code_candidate(message)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["strategy"], "fenced_unclosed")
        self.assertNotIn("请分析代码", candidate["code"])
        self.assertIn("#include <stdio.h>\nint main", candidate["code"])

    def test_extracts_flattened_closed_fenced_c_code(self) -> None:
        message = (
            '请分析代码 ```c #include <stdio.h> int main(void) { '
            'printf("hi\\n"); return 0; }```'
        )
        candidate = self.service.extract_c_code_candidate(message)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["strategy"], "fenced")
        self.assertNotIn("请分析代码", candidate["code"])
        self.assertIn("#include <stdio.h>\nint main", candidate["code"])

    def test_plain_extraction_stops_before_explanatory_text(self) -> None:
        message = "int main(void){return 0;}\n(please explain)"
        candidate = self.service.extract_c_code_candidate(message)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["strategy"], "plain")
        self.assertEqual(candidate["code"], "int main(void){return 0;}")

    async def test_reports_compile_error(self) -> None:
        analysis = await self.service.analyze_code(
            "int main(void) { return missing_symbol; }",
            timeout_s=10,
        )
        self.assertFalse(analysis["compile_ok"])
        self.assertTrue(analysis["compile_errors"])

    async def test_reports_missing_compiler_explicitly(self) -> None:
        with patch("webapp_core.code_analysis_service.shutil.which", return_value=None):
            service = CodeAnalysisService(compiler_bin="agenticrag-missing-c-compiler")
        analysis = await service.analyze_code("int main(void) { return 0; }", timeout_s=10)
        self.assertFalse(analysis["tool_available"])
        self.assertFalse(analysis["compile_ok"])
        self.assertIn("未检测到可用的 C 编译器", analysis["tool_error"])
        self.assertFalse(analysis["execution"]["eligible"])

    def test_parses_msvc_diagnostics(self) -> None:
        diagnostics = CodeAnalysisService._parse_diagnostics(
            r"C:\tmp\submission.c(3,5): error C2143: syntax error: missing ';' before 'return'"
        )
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["line"], 3)
        self.assertEqual(diagnostics[0]["column"], 5)
        self.assertEqual(diagnostics[0]["severity"], "error")
        self.assertEqual(diagnostics[0]["category"], "C2143")

    @patch("webapp_core.code_analysis_service.subprocess.run")
    def test_detects_cc_alias_as_clang(self, mock_run) -> None:
        mock_run.return_value.stdout = "Apple clang version 21.0.0"
        mock_run.return_value.stderr = ""
        compiler = CodeAnalysisService._build_compiler("/usr/bin/cc")
        self.assertIsNotNone(compiler)
        self.assertEqual(compiler.kind, "clang")
        self.assertTrue(compiler.supports_ast)

    async def test_extracts_basic_structure(self) -> None:
        analysis = await self.service.analyze_code(
            (
                "int add(int a, int b) { return a + b; }\n"
                "int main(void) {\n"
                "    int sum = add(1, 2);\n"
                "    for (int i = 0; i < 3; ++i) {\n"
                "        if (sum > i) {\n"
                "            sum += i;\n"
                "        }\n"
                "    }\n"
                "    return sum;\n"
                "}\n"
            ),
            timeout_s=10,
        )
        functions = [item["name"] for item in analysis["structure"]["functions"]]
        self.assertIn("add", functions)
        self.assertIn("main", functions)
        self.assertGreaterEqual(analysis["structure"]["counts"]["loops"], 1)
        self.assertGreaterEqual(analysis["structure"]["counts"]["conditionals"], 1)

    async def test_fallback_extracts_structure_without_ast_support(self) -> None:
        service = CodeAnalysisService()
        service.compiler = CCompiler(
            kind="gcc",
            executable="gcc",
            display_name="gcc",
            supports_ast=False,
            supports_static_analysis=True,
        )

        async def fake_run_tool(args: list[str], **kwargs: object) -> dict[str, object]:
            return {
                "args": list(args),
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            }

        service._run_tool = fake_run_tool  # type: ignore[method-assign]
        analysis = await service.analyze_code(
            (
                "int add(int a, int b) { return a + b; }\n"
                "int main(void) {\n"
                "    int sum = add(1, 2);\n"
                "    for (int i = 0; i < 3; ++i) {\n"
                "        if (sum > i) {\n"
                "            sum += i;\n"
                "        }\n"
                "    }\n"
                "    return sum;\n"
                "}\n"
            ),
            timeout_s=10,
        )

        self.assertFalse(analysis["structure"]["ast_available"])
        self.assertEqual(analysis["structure"]["extraction_method"], "regex_fallback")
        functions = [item["name"] for item in analysis["structure"]["functions"]]
        self.assertEqual(functions, ["add", "main"])
        self.assertGreaterEqual(analysis["structure"]["counts"]["loops"], 1)
        self.assertGreaterEqual(analysis["structure"]["counts"]["conditionals"], 1)
        calls = [item["name"] for item in analysis["structure"]["function_calls"]]
        self.assertIn("add", calls)

    async def test_fallback_handles_multiline_signature_and_inline_declarations(self) -> None:
        service = CodeAnalysisService()
        service.compiler = CCompiler(
            kind="gcc",
            executable="gcc",
            display_name="gcc",
            supports_ast=False,
            supports_static_analysis=True,
        )

        async def fake_run_tool(args: list[str], **kwargs: object) -> dict[str, object]:
            return {
                "args": list(args),
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            }

        service._run_tool = fake_run_tool  # type: ignore[method-assign]
        analysis = await service.analyze_code(
            (
                "static inline unsigned long long\n"
                "compute_sum(\n"
                "    const int *values,\n"
                "    size_t n\n"
                ")\n"
                "{\n"
                "    unsigned long long total = 0;\n"
                "    for (size_t i = 0; i < n; ++i) {\n"
                "        total += (unsigned long long)values[i];\n"
                "    }\n"
                "    return total;\n"
                "}\n\n"
                "int main(void) { int x = 1, y = 2; unsigned long long z = compute_sum(&x, 1); "
                "if (z > 0) { return y; } return 0; }\n"
            ),
            timeout_s=10,
        )

        structure = analysis["structure"]
        self.assertFalse(structure["ast_available"])
        self.assertEqual(structure["extraction_method"], "regex_fallback")
        functions = [item["name"] for item in structure["functions"]]
        self.assertEqual(functions, ["compute_sum", "main"])
        variables = {(item["name"], item["scope"]) for item in structure["variables"]}
        self.assertIn(("values", "parameter"), variables)
        self.assertIn(("n", "parameter"), variables)
        self.assertIn(("total", "local"), variables)
        self.assertIn(("x", "local"), variables)
        self.assertIn(("y", "local"), variables)
        self.assertIn(("z", "local"), variables)
        self.assertGreaterEqual(structure["counts"]["loops"], 1)
        self.assertGreaterEqual(structure["counts"]["conditionals"], 1)
        calls = [item["name"] for item in structure["function_calls"]]
        self.assertIn("compute_sum", calls)

    async def test_structure_excludes_included_header_declarations(self) -> None:
        analysis = await self.service.analyze_code(
            '#include <stdio.h>\nint main(void){ printf("hi\\n"); return 0; }',
            timeout_s=10,
        )
        functions = [item["name"] for item in analysis["structure"]["functions"]]
        self.assertEqual(functions, ["main"])
        self.assertEqual(analysis["structure"]["counts"]["functions"], 1)

    async def test_executes_safe_short_program_after_successful_compile(self) -> None:
        service = CodeAnalysisService(
            enable_execution=True,
            execution_timeout_s=2,
            execution_max_code_chars=4000,
        )
        service.compiler = CCompiler(
            kind="gcc",
            executable="gcc",
            display_name="gcc",
            supports_ast=False,
            supports_static_analysis=True,
        )

        async def fake_run_tool(args: list[str], **kwargs: object) -> dict[str, object]:
            if args and args[0] == "gcc" and "-o" in args:
                binary_path = args[args.index("-o") + 1]
                Path(binary_path).write_text("", encoding="utf-8")
                return {
                    "args": list(args),
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                }
            if args and args[0] == "gcc":
                return {
                    "args": list(args),
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                }
            return {
                "args": list(args),
                "exit_code": 0,
                "stdout": "42\n",
                "stderr": "",
                "timed_out": False,
            }

        service._run_tool = fake_run_tool  # type: ignore[method-assign]
        analysis = await service.analyze_code(
            '#include <stdio.h>\nint main(void){ printf("42\\n"); return 0; }',
            timeout_s=10,
        )

        self.assertTrue(analysis["compile_ok"])
        self.assertTrue(analysis["execution"]["enabled"])
        self.assertTrue(analysis["execution"]["eligible"])
        self.assertTrue(analysis["execution"]["build_ok"])
        self.assertTrue(analysis["execution"]["run_ok"])
        self.assertEqual(analysis["execution"]["stdout"], "42")
        self.assertEqual(analysis["tool_runs"]["run"]["stdout"], "42")

    async def test_skips_execution_for_dangerous_code_patterns(self) -> None:
        service = CodeAnalysisService(
            enable_execution=True,
            execution_timeout_s=2,
            execution_max_code_chars=4000,
        )
        service.compiler = CCompiler(
            kind="gcc",
            executable="gcc",
            display_name="gcc",
            supports_ast=False,
            supports_static_analysis=True,
        )
        calls: list[list[str]] = []

        async def fake_run_tool(args: list[str], **kwargs: object) -> dict[str, object]:
            calls.append(list(args))
            return {
                "args": list(args),
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            }

        service._run_tool = fake_run_tool  # type: ignore[method-assign]
        analysis = await service.analyze_code(
            (
                '#include <stdlib.h>\n'
                'int main(void){ system("ls"); return 0; }\n'
            ),
            timeout_s=10,
        )

        self.assertTrue(analysis["compile_ok"])
        self.assertTrue(analysis["execution"]["enabled"])
        self.assertFalse(analysis["execution"]["eligible"])
        self.assertIn("外部命令调用", analysis["execution"]["reason"])
        self.assertTrue(analysis["tool_runs"]["build"]["skipped"])
        self.assertTrue(analysis["tool_runs"]["run"]["skipped"])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
