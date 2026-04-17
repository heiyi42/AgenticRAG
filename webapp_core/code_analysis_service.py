from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CCompiler:
    kind: str
    executable: str
    display_name: str
    supports_ast: bool
    supports_static_analysis: bool


class CodeAnalysisService:
    CODE_BLOCK_RE = re.compile(
        r"```(?P<lang>[A-Za-z0-9_+#-]*)[ \t]*(?:\n|[ \t]+)(?P<code>.*?)```",
        re.DOTALL,
    )
    UNCLOSED_CODE_BLOCK_RE = re.compile(
        r"```(?P<lang>[A-Za-z0-9_+#-]*)[ \t]*(?P<code>.+)$",
        re.DOTALL,
    )
    DIAGNOSTIC_RE = re.compile(
        r"^(?P<file>.+?):(?P<line>\d+):(?P<column>\d+): "
        r"(?P<severity>fatal error|error|warning|note): "
        r"(?P<message>.*?)(?: \[(?P<category>[^\]]+)\])?$"
    )
    MSVC_DIAGNOSTIC_RE = re.compile(
        r"^(?P<file>.+?)\((?P<line>\d+)(?:,(?P<column>\d+))?\): "
        r"(?P<severity>fatal error|error|warning) (?P<category>[A-Z]+\d+): "
        r"(?P<message>.*)$"
    )
    COMPILER_CANDIDATES = (
        "clang",
        "gcc",
        "cc",
        "x86_64-w64-mingw32-gcc",
        "cl",
    )
    NON_C_LANG_HINTS = {
        "python",
        "py",
        "javascript",
        "js",
        "typescript",
        "ts",
        "java",
        "go",
        "rust",
        "rs",
        "bash",
        "sh",
        "shell",
        "json",
        "yaml",
        "yml",
        "html",
        "css",
        "sql",
        "xml",
    }
    EXECUTION_BLOCKLIST: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "检测到文件或目录操作 API",
            re.compile(
                r"\b("
                r"fopen|freopen|open|close|remove|rename|unlink|mkdir|rmdir|"
                r"opendir|readdir|closedir|chdir|fwrite|fread"
                r")\s*\("
            ),
        ),
        (
            "检测到进程或外部命令调用 API",
            re.compile(
                r"\b("
                r"system|popen|execl|execv|execve|execvp|execvpe|fork|spawnl|spawnlp|"
                r"spawnv|spawnvp|CreateProcessA|CreateProcessW|WinExec|ShellExecuteA|ShellExecuteW"
                r")\s*\("
            ),
        ),
        (
            "检测到网络相关 API",
            re.compile(
                r"\b("
                r"socket|connect|bind|listen|accept|send|recv|sendto|recvfrom|"
                r"getaddrinfo|WSAStartup"
                r")\s*\("
            ),
        ),
        (
            "检测到交互式输入 API",
            re.compile(r"\b(scanf|getchar|gets|fgets|getc|fgetc|read)\s*\("),
        ),
        (
            "检测到潜在越权头文件",
            re.compile(
                r"#\s*include\s*<("
                r"unistd\.h|sys/socket\.h|winsock2\.h|windows\.h|dirent\.h|"
                r"fcntl\.h|sys/stat\.h"
                r")>"
            ),
        ),
    )

    def __init__(
        self,
        *,
        compiler_bin: str | None = None,
        clang_bin: str | None = None,
        max_code_chars: int = 12000,
        enable_execution: bool | None = None,
        execution_timeout_s: int | None = None,
        execution_max_code_chars: int = 4000,
        execution_max_output_chars: int = 1200,
    ):
        configured_compiler = (
            compiler_bin
            or clang_bin
            or os.getenv("WEB_CODE_ANALYSIS_COMPILER", "").strip()
            or None
        )
        self.compiler = self._detect_compiler(configured_compiler)
        self.max_code_chars = max(256, int(max_code_chars))
        self.enable_execution = (
            str(os.getenv("WEB_ENABLE_CODE_EXECUTION", "1")).strip().lower()
            in {"1", "true", "yes", "on"}
            if enable_execution is None
            else bool(enable_execution)
        )
        self.execution_timeout_s = max(
            1,
            int(
                execution_timeout_s
                if execution_timeout_s is not None
                else os.getenv("WEB_CODE_EXECUTION_TIMEOUT_S", "2")
            ),
        )
        self.execution_max_code_chars = max(128, int(execution_max_code_chars))
        self.execution_max_output_chars = max(128, int(execution_max_output_chars))

    @property
    def available(self) -> bool:
        return self.compiler is not None

    @classmethod
    def _detect_compiler(cls, preferred_bin: str | None = None) -> CCompiler | None:
        candidates = [str(preferred_bin).strip()] if preferred_bin else []
        candidates.extend(name for name in cls.COMPILER_CANDIDATES if name not in candidates)
        for candidate in candidates:
            resolved = cls._resolve_executable(candidate)
            if not resolved:
                continue
            compiler = cls._build_compiler(resolved)
            if compiler is not None:
                return compiler
        return None

    @staticmethod
    def _resolve_executable(candidate: str) -> str | None:
        text = str(candidate or "").strip()
        if not text:
            return None
        if any(sep in text for sep in ("/", "\\")):
            path = Path(text).expanduser()
            if path.exists():
                return str(path)
            return None
        resolved = shutil.which(text)
        return resolved or None

    @staticmethod
    def _build_compiler(executable: str) -> CCompiler | None:
        name = Path(executable).name.lower()
        stem = name[:-4] if name.endswith(".exe") else name
        if stem.startswith("clang") and stem != "clang-cl":
            return CCompiler(
                kind="clang",
                executable=executable,
                display_name="clang",
                supports_ast=True,
                supports_static_analysis=True,
            )
        if "gcc" in stem:
            return CCompiler(
                kind="gcc",
                executable=executable,
                display_name="gcc",
                supports_ast=False,
                supports_static_analysis=True,
            )
        if stem == "cl":
            return CCompiler(
                kind="msvc",
                executable=executable,
                display_name="MSVC cl",
                supports_ast=False,
                supports_static_analysis=True,
            )
        detected_kind = CodeAnalysisService._probe_compiler_kind(executable)
        if detected_kind == "clang":
            return CCompiler(
                kind="clang",
                executable=executable,
                display_name="clang",
                supports_ast=True,
                supports_static_analysis=True,
            )
        if detected_kind == "gcc":
            return CCompiler(
                kind="gcc",
                executable=executable,
                display_name="gcc",
                supports_ast=False,
                supports_static_analysis=True,
            )
        return None

    @staticmethod
    def _probe_compiler_kind(executable: str) -> str | None:
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

        banner = f"{result.stdout}\n{result.stderr}".lower()
        if "clang" in banner:
            return "clang"
        if "gcc" in banner or "gnu compiler collection" in banner:
            return "gcc"
        return None

    def extract_c_code_candidate(self, message: str) -> dict[str, Any] | None:
        text = str(message or "")
        if not text.strip():
            return None

        candidates: list[dict[str, Any]] = []
        for match in self.CODE_BLOCK_RE.finditer(text):
            language_hint = str(match.group("lang") or "").strip().lower()
            if language_hint in self.NON_C_LANG_HINTS:
                continue
            code = self._normalize_extracted_code(str(match.group("code") or ""))
            if not code:
                continue
            score = self._score_c_code(code, language_hint=language_hint)
            candidates.append(
                {
                    "code": code,
                    "language_hint": language_hint or "",
                    "score": score,
                    "strategy": "fenced",
                }
            )

        if not candidates:
            unclosed_candidate = self._extract_unclosed_fenced_c_candidate(text)
            if unclosed_candidate is not None:
                candidates.append(unclosed_candidate)

        if not candidates:
            plain_candidate = self._extract_plain_c_candidate(text)
            if plain_candidate is not None:
                candidates.append(plain_candidate)

        if not candidates:
            return None

        best = max(candidates, key=lambda item: (item["score"], len(item["code"])))
        if best["score"] < 4:
            return None

        code = best["code"]
        truncated = len(code) > self.max_code_chars
        if truncated:
            code = code[: self.max_code_chars].rstrip()

        return {
            "language": "c",
            "language_hint": best["language_hint"],
            "strategy": best["strategy"],
            "score": int(best["score"]),
            "code": code,
            "truncated": truncated,
        }

    def _extract_unclosed_fenced_c_candidate(self, text: str) -> dict[str, Any] | None:
        match = self.UNCLOSED_CODE_BLOCK_RE.search(str(text or ""))
        if match is None:
            return None
        language_hint = str(match.group("lang") or "").strip().lower()
        if language_hint in self.NON_C_LANG_HINTS:
            return None
        code = self._normalize_extracted_code(str(match.group("code") or ""))
        if not code:
            return None
        score = self._score_c_code(code, language_hint=language_hint)
        return {
            "code": code,
            "language_hint": language_hint or "",
            "score": score,
            "strategy": "fenced_unclosed",
        }

    @classmethod
    def _normalize_extracted_code(cls, code: str) -> str:
        text = str(code or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return ""
        return cls._restore_flattened_preprocessor_lines(text)

    @staticmethod
    def _restore_flattened_preprocessor_lines(code: str) -> str:
        text = str(code or "")
        return re.sub(
            r'(#\s*include\s*(?:<[^>\n]+>|"[^"\n]+"))[ \t]+'
            r"(?=(?:int|void|char|float|double|long|short|unsigned|signed|"
            r"static|extern|typedef|struct|enum|union)\b)",
            r"\1\n",
            text,
        )

    async def analyze_message(
        self,
        message: str,
        *,
        timeout_s: int = 15,
    ) -> dict[str, Any] | None:
        candidate = self.extract_c_code_candidate(message)
        if candidate is None:
            return None

        analysis = await self.analyze_code(candidate["code"], timeout_s=timeout_s)
        analysis["detected"] = {
            "language": candidate["language"],
            "language_hint": candidate["language_hint"],
            "strategy": candidate["strategy"],
            "score": candidate["score"],
            "truncated": candidate["truncated"],
        }
        return analysis

    async def analyze_code(self, code: str, *, timeout_s: int = 15) -> dict[str, Any]:
        safe_code = str(code or "").replace("\r\n", "\n").strip()
        if not safe_code:
            raise ValueError("code 不能为空")
        compiler = self.compiler
        if compiler is None:
            return {
                "language": "c",
                "tool": "",
                "tool_available": False,
                "code": safe_code,
                "compile_ok": False,
                "compile_errors": [],
                "warnings": [],
                "risk_findings": [],
                "structure": self._empty_structure(),
                "tool_runs": {},
                "tool_error": (
                    "未检测到可用的 C 编译器。请安装 clang、gcc 或 MSVC cl，"
                    "并确保它在服务进程的 PATH 中。"
                ),
                "compiler": None,
                "execution": self._execution_result(
                    enabled=self.enable_execution,
                    eligible=False,
                    reason="未检测到可用的 C 编译器，无法执行程序。",
                ),
            }

        safe_timeout = max(3, int(timeout_s))
        with tempfile.TemporaryDirectory(prefix="agenticrag-code-analysis-") as tmp_dir:
            source_path = Path(tmp_dir) / "submission.c"
            source_path.write_text(safe_code, encoding="utf-8")

            compile_timeout = max(2, min(safe_timeout, 8))
            static_timeout = max(2, min(safe_timeout, 10))
            ast_timeout = max(2, min(safe_timeout, 8))

            compile_task = self._run_tool(
                self._compile_args(compiler, source_path, Path(tmp_dir)),
                timeout_s=compile_timeout,
            )
            static_args = self._static_analysis_args(compiler, source_path, Path(tmp_dir))
            static_task = (
                self._run_tool(static_args, timeout_s=static_timeout)
                if static_args
                else self._skipped_tool_run("当前编译器不支持静态分析")
            )
            ast_args = self._ast_args(compiler, source_path, Path(tmp_dir))
            ast_task = (
                self._run_tool(ast_args, timeout_s=ast_timeout)
                if ast_args
                else self._skipped_tool_run("当前编译器不支持 clang AST JSON 结构提取")
            )
            compile_run, static_run, ast_run = await asyncio.gather(
                compile_task,
                static_task,
                ast_task,
            )
            compile_diagnostics = self._parse_diagnostics(
                self._combined_tool_output(compile_run)
            )
            analyzer_diagnostics = self._parse_diagnostics(
                self._combined_tool_output(static_run)
            )
            compile_errors = [
                item
                for item in compile_diagnostics
                if item["severity"] in {"error", "fatal error"}
            ]
            warnings = [item for item in compile_diagnostics if item["severity"] == "warning"]
            risk_findings = [
                item for item in analyzer_diagnostics if item["severity"] == "warning"
            ]

            if compiler.supports_ast and not ast_run.get("skipped"):
                structure = self._extract_structure(
                    ast_run["stdout"],
                    str(source_path),
                    source_len=len(safe_code),
                )
            else:
                structure = self._extract_structure_fallback(safe_code)
                structure["ast_available"] = False
                structure["ast_error"] = self._short_text(
                    ast_run.get("stderr")
                    or ast_run.get("stdout")
                    or "当前编译器不支持 AST 结构提取",
                    max_len=240,
                )
            if compiler.supports_ast and ast_run["exit_code"] != 0 and not structure.get(
                "functions"
            ):
                structure = self._extract_structure_fallback(safe_code)
                structure["ast_available"] = False
                structure["ast_error"] = self._short_text(ast_run["stderr"], max_len=240)

            compile_ok = bool(
                compile_run["exit_code"] == 0
                and not compile_errors
                and not compile_run["timed_out"]
            )
            build_run = await self._skipped_tool_run("未执行可执行文件构建")
            run_run = await self._skipped_tool_run("未执行程序")
            execution = self._execution_result(
                enabled=self.enable_execution,
                eligible=False,
                reason="仅完成静态分析，未尝试执行程序。",
            )
            if compile_ok:
                execution = await self._maybe_execute_code(
                    compiler=compiler,
                    code=safe_code,
                    source_path=source_path,
                    tmp_dir=Path(tmp_dir),
                    safe_timeout=safe_timeout,
                )
                build_run = dict(execution.get("build_run") or build_run)
                run_run = dict(execution.get("run_run") or run_run)

            return {
                "language": "c",
                "tool": compiler.display_name,
                "tool_available": True,
                "code": safe_code,
                "compile_ok": compile_ok,
                "compile_errors": compile_errors,
                "warnings": warnings,
                "risk_findings": risk_findings,
                "structure": structure,
                "compiler": self._compiler_public_info(compiler),
                "tool_runs": {
                    "compile": self._public_tool_run(compile_run),
                    "static_analysis": self._public_tool_run(static_run),
                    "ast": self._public_tool_run(ast_run),
                    "build": self._public_tool_run(build_run),
                    "run": self._public_tool_run(run_run),
                },
                "execution": self._execution_public_result(execution),
            }

    @staticmethod
    def _compiler_public_info(compiler: CCompiler) -> dict[str, Any]:
        return {
            "kind": compiler.kind,
            "name": compiler.display_name,
            "executable": compiler.executable,
            "supports_ast": compiler.supports_ast,
            "supports_static_analysis": compiler.supports_static_analysis,
        }

    @staticmethod
    def _compile_args(compiler: CCompiler, source_path: Path, tmp_dir: Path) -> list[str]:
        if compiler.kind in {"clang", "gcc"}:
            return [
                compiler.executable,
                "-fsyntax-only",
                "-Wall",
                "-Wextra",
                "-std=c11",
                str(source_path),
            ]
        if compiler.kind == "msvc":
            return [
                compiler.executable,
                "/nologo",
                "/TC",
                "/W4",
                "/Zs",
                str(source_path),
            ]
        raise ValueError(f"不支持的 C 编译器: {compiler.kind}")

    @staticmethod
    def _build_binary_args(compiler: CCompiler, source_path: Path, binary_path: Path) -> list[str]:
        if compiler.kind in {"clang", "gcc"}:
            return [
                compiler.executable,
                "-Wall",
                "-Wextra",
                "-std=c11",
                "-O0",
                str(source_path),
                "-o",
                str(binary_path),
            ]
        if compiler.kind == "msvc":
            return [
                compiler.executable,
                "/nologo",
                "/TC",
                str(source_path),
                f"/Fe{binary_path}",
            ]
        raise ValueError(f"不支持的 C 编译器: {compiler.kind}")

    @staticmethod
    def _static_analysis_args(
        compiler: CCompiler,
        source_path: Path,
        tmp_dir: Path,
    ) -> list[str] | None:
        if compiler.kind == "clang":
            return [
                compiler.executable,
                "--analyze",
                "-std=c11",
                str(source_path),
            ]
        if compiler.kind == "gcc":
            return [
                compiler.executable,
                "-fsyntax-only",
                "-fanalyzer",
                "-Wall",
                "-Wextra",
                "-std=c11",
                str(source_path),
            ]
        if compiler.kind == "msvc":
            obj_path = tmp_dir / "submission.obj"
            return [
                compiler.executable,
                "/nologo",
                "/TC",
                "/analyze",
                "/c",
                f"/Fo{obj_path}",
                str(source_path),
            ]
        return None

    @staticmethod
    def _ast_args(
        compiler: CCompiler,
        source_path: Path,
        tmp_dir: Path,
    ) -> list[str] | None:
        if compiler.kind != "clang":
            return None
        return [
            compiler.executable,
            "-Xclang",
            "-ast-dump=json",
            "-fsyntax-only",
            "-std=c11",
            str(source_path),
        ]

    @staticmethod
    def _combined_tool_output(tool_run: dict[str, Any]) -> str:
        stdout = str(tool_run.get("stdout") or "")
        stderr = str(tool_run.get("stderr") or "")
        if stdout and stderr:
            return f"{stderr}\n{stdout}"
        return stderr or stdout

    @staticmethod
    def _public_tool_run(tool_run: dict[str, Any]) -> dict[str, Any]:
        public = {
            "exit_code": int(tool_run.get("exit_code", 0)),
            "timed_out": bool(tool_run.get("timed_out", False)),
            "stderr": CodeAnalysisService._short_text(tool_run.get("stderr", ""), max_len=240),
        }
        if tool_run.get("skipped"):
            public["skipped"] = True
        stdout = CodeAnalysisService._short_text(tool_run.get("stdout", ""), max_len=240)
        if stdout:
            public["stdout"] = stdout
        return public

    @classmethod
    def _assess_execution_eligibility(cls, code: str, *, max_code_chars: int) -> dict[str, Any]:
        text = str(code or "")
        if not text.strip():
            return {"eligible": False, "reason": "代码为空，无法执行。", "blocked_markers": []}
        if len(text) > max_code_chars:
            return {
                "eligible": False,
                "reason": f"代码长度超过执行上限 {max_code_chars} 字符，已跳过运行。",
                "blocked_markers": [],
            }
        if re.search(r"\bmain\s*\(", text) is None:
            return {"eligible": False, "reason": "代码缺少 main 函数，已跳过运行。", "blocked_markers": []}

        blocked_markers: list[str] = []
        for label, pattern in cls.EXECUTION_BLOCKLIST:
            if pattern.search(text):
                blocked_markers.append(label)
        if blocked_markers:
            return {
                "eligible": False,
                "reason": f"{blocked_markers[0]}，为避免越权副作用已跳过执行。",
                "blocked_markers": blocked_markers,
            }
        return {"eligible": True, "reason": "代码满足受限执行条件。", "blocked_markers": []}

    @staticmethod
    def _execution_env(tmp_dir: Path) -> dict[str, str]:
        env: dict[str, str] = {"PATH": os.getenv("PATH", "")}
        if os.name == "nt":
            system_root = os.getenv("SystemRoot")
            if system_root:
                env["SystemRoot"] = system_root
            comspec = os.getenv("ComSpec")
            if comspec:
                env["ComSpec"] = comspec
        env["HOME"] = str(tmp_dir)
        env["TMPDIR"] = str(tmp_dir)
        env["TMP"] = str(tmp_dir)
        env["TEMP"] = str(tmp_dir)
        return env

    @staticmethod
    def _execution_result(
        *,
        enabled: bool,
        eligible: bool,
        reason: str,
        blocked_markers: list[str] | None = None,
        build_ok: bool = False,
        run_ok: bool = False,
        exit_code: int | None = None,
        timed_out: bool = False,
        stdout: str = "",
        stderr: str = "",
        build_run: dict[str, Any] | None = None,
        run_run: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "enabled": bool(enabled),
            "eligible": bool(eligible),
            "reason": str(reason or "").strip(),
            "blocked_markers": list(blocked_markers or []),
            "build_ok": bool(build_ok),
            "run_ok": bool(run_ok),
            "exit_code": exit_code,
            "timed_out": bool(timed_out),
            "stdout": str(stdout or ""),
            "stderr": str(stderr or ""),
            "build_run": dict(build_run or {}),
            "run_run": dict(run_run or {}),
        }

    def _execution_public_result(self, execution: dict[str, Any]) -> dict[str, Any]:
        return {
            "enabled": bool(execution.get("enabled", False)),
            "eligible": bool(execution.get("eligible", False)),
            "reason": self._short_text(str(execution.get("reason", "") or ""), max_len=160),
            "blocked_markers": [
                self._short_text(str(item), max_len=60)
                for item in list(execution.get("blocked_markers", []) or [])[:4]
            ],
            "build_ok": bool(execution.get("build_ok", False)),
            "run_ok": bool(execution.get("run_ok", False)),
            "exit_code": execution.get("exit_code"),
            "timed_out": bool(execution.get("timed_out", False)),
            "stdout": self._short_text(
                str(execution.get("stdout", "") or ""),
                max_len=self.execution_max_output_chars,
            ),
            "stderr": self._short_text(
                str(execution.get("stderr", "") or ""),
                max_len=self.execution_max_output_chars,
            ),
        }

    async def _maybe_execute_code(
        self,
        *,
        compiler: CCompiler,
        code: str,
        source_path: Path,
        tmp_dir: Path,
        safe_timeout: int,
    ) -> dict[str, Any]:
        if not self.enable_execution:
            skipped = await self._skipped_tool_run("当前配置已关闭 C 代码执行")
            return self._execution_result(
                enabled=False,
                eligible=False,
                reason="当前配置已关闭 C 代码执行。",
                build_run=skipped,
                run_run=await self._skipped_tool_run("当前配置已关闭 C 代码执行"),
            )

        eligibility = self._assess_execution_eligibility(
            code,
            max_code_chars=self.execution_max_code_chars,
        )
        if not eligibility["eligible"]:
            return self._execution_result(
                enabled=True,
                eligible=False,
                reason=str(eligibility["reason"]),
                blocked_markers=list(eligibility.get("blocked_markers", []) or []),
                build_run=await self._skipped_tool_run(str(eligibility["reason"])),
                run_run=await self._skipped_tool_run(str(eligibility["reason"])),
            )

        binary_name = "submission.exe" if compiler.kind == "msvc" else "submission.out"
        binary_path = tmp_dir / binary_name
        build_timeout = max(2, min(safe_timeout, 8))
        run_timeout = max(1, min(safe_timeout, self.execution_timeout_s))
        build_run = await self._run_tool(
            self._build_binary_args(compiler, source_path, binary_path),
            timeout_s=build_timeout,
            cwd=str(tmp_dir),
        )
        build_ok = bool(
            build_run["exit_code"] == 0 and not build_run["timed_out"] and binary_path.exists()
        )
        if not build_ok:
            return self._execution_result(
                enabled=True,
                eligible=True,
                reason="代码通过语法检查，但生成可执行文件失败。",
                build_ok=False,
                run_ok=False,
                stdout="",
                stderr=str(build_run.get("stderr") or ""),
                build_run=build_run,
                run_run=await self._skipped_tool_run("构建失败，未执行程序"),
            )

        run_run = await self._run_tool(
            [str(binary_path)],
            timeout_s=run_timeout,
            cwd=str(tmp_dir),
            env=self._execution_env(tmp_dir),
            max_output_chars=self.execution_max_output_chars,
        )
        return self._execution_result(
            enabled=True,
            eligible=True,
            reason="已在受限条件下执行程序。",
            build_ok=True,
            run_ok=bool(run_run["exit_code"] == 0 and not run_run["timed_out"]),
            exit_code=int(run_run.get("exit_code", 0)),
            timed_out=bool(run_run.get("timed_out", False)),
            stdout=str(run_run.get("stdout", "") or ""),
            stderr=str(run_run.get("stderr", "") or ""),
            build_run=build_run,
            run_run=run_run,
        )

    async def _skipped_tool_run(self, reason: str) -> dict[str, Any]:
        return {
            "args": [],
            "exit_code": 0,
            "stdout": "",
            "stderr": str(reason or "已跳过"),
            "timed_out": False,
            "skipped": True,
        }

    async def _run_tool(
        self,
        args: list[str],
        *,
        timeout_s: int,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        max_output_chars: int | None = None,
    ) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=cwd,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            timed_out = True

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if max_output_chars is not None and max_output_chars > 0:
            stdout_text = self._short_text(stdout_text, max_len=max_output_chars)
            stderr_text = self._short_text(stderr_text, max_len=max_output_chars)

        return {
            "args": list(args),
            "exit_code": int(proc.returncode or 0),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "timed_out": timed_out,
        }

    def _extract_plain_c_candidate(self, text: str) -> dict[str, Any] | None:
        lines = text.splitlines()
        if not lines:
            return None

        code_lines: list[str] = []
        started = False
        for line in lines:
            raw_line = line.rstrip("\n")
            stripped = raw_line.strip()
            if not started and self._looks_like_c_line(stripped):
                started = True
            if started:
                if not stripped and code_lines:
                    code_lines.append("")
                    continue
                if self._looks_like_c_line(stripped):
                    code_lines.append(raw_line)
                    continue
                if code_lines and self._looks_like_c_continuation_line(stripped):
                    code_lines.append(raw_line)
                    continue
                if code_lines:
                    break

        code = "\n".join(code_lines).strip()
        if not code:
            return None
        score = self._score_c_code(code, language_hint="")
        return {
            "code": code,
            "language_hint": "",
            "score": score,
            "strategy": "plain",
        }

    @classmethod
    def _looks_like_c_line(cls, line: str) -> bool:
        text = str(line or "").strip()
        if not text:
            return False
        c_markers = (
            "#include",
            "printf(",
            "scanf(",
            "malloc(",
            "free(",
            "return ",
            "return;",
            "int ",
            "char ",
            "void ",
            "double ",
            "float ",
            "for(",
            "for (",
            "while(",
            "while (",
            "if(",
            "if (",
        )
        if any(marker in text for marker in c_markers):
            return True
        return bool(re.search(r"[;{}]", text) and re.search(r"[A-Za-z_]", text))

    @classmethod
    def _looks_like_c_continuation_line(cls, line: str) -> bool:
        text = str(line or "").strip()
        if not text:
            return False
        if text.startswith(("//", "/*", "*", "*/", "#")):
            return True
        if re.search(r"[\u4e00-\u9fff]", text):
            return False
        if re.search(r"[;{}=]", text):
            return True
        if re.match(r"^[A-Za-z_]\w*\s*\(.*\)\s*;?$", text):
            return True
        if re.match(r'^[\'"].*[\'"]\s*,?$', text):
            return True
        if re.match(r"^[\[\]\(\),&|+\-*/%<>!~?:]+$", text):
            return True
        return False

    @classmethod
    def _score_c_code(cls, code: str, *, language_hint: str) -> int:
        text = str(code or "")
        score = 0
        if language_hint in {"c", "h"}:
            score += 5
        elif language_hint in {"cpp", "c++", "cc", "cxx", "hpp"}:
            score -= 4

        positive_markers = [
            (r"#include\s*<", 4),
            (r"\bint\s+main\s*\(", 4),
            (r"\bprintf\s*\(", 2),
            (r"\bscanf\s*\(", 2),
            (r"\bmalloc\s*\(", 2),
            (r"\bfree\s*\(", 2),
            (r"\breturn\b", 1),
            (r"\b(if|for|while|switch)\s*\(", 1),
            (r"[{}]", 1),
            (r";", 1),
            (r"\b(int|char|float|double|void|size_t)\b", 1),
        ]
        negative_markers = [
            (r"\bclass\b", 3),
            (r"\bnamespace\b", 3),
            (r"std::", 4),
            (r"\bcout\s*<<", 4),
            (r"\bdef\b", 3),
            (r"\bimport\b", 1),
            (r"\bconsole\.log\b", 3),
            (r"\bfunction\b", 2),
        ]
        for pattern, weight in positive_markers:
            if re.search(pattern, text):
                score += weight
        for pattern, weight in negative_markers:
            if re.search(pattern, text):
                score -= weight
        if text.count("\n") >= 1:
            score += 1
        return score

    @classmethod
    def _parse_diagnostics(cls, raw_output: str) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for line in str(raw_output or "").splitlines():
            clean_line = line.strip()
            match = cls.DIAGNOSTIC_RE.match(clean_line)
            msvc_match = cls.MSVC_DIAGNOSTIC_RE.match(clean_line) if match is None else None
            if match is None and msvc_match is None:
                continue
            active_match = match or msvc_match
            if active_match is None:
                continue
            severity = str(active_match.group("severity") or "").strip().lower()
            message = str(active_match.group("message") or "").strip()
            category = str(active_match.groupdict().get("category") or "").strip()
            column = active_match.groupdict().get("column") or 1
            item = {
                "line": int(active_match.group("line")),
                "column": int(column),
                "severity": severity,
                "message": message,
            }
            if category:
                item["category"] = category
            key = (
                item["line"],
                item["column"],
                severity,
                message,
                category,
            )
            if key in seen:
                continue
            seen.add(key)
            diagnostics.append(item)
        diagnostics.sort(key=lambda item: (int(item["line"]), int(item["column"]), item["severity"]))
        return diagnostics

    @classmethod
    def _extract_structure(
        cls,
        ast_json: str,
        target_file: str,
        *,
        source_len: int | None = None,
    ) -> dict[str, Any]:
        target_path = Path(target_file).resolve()
        target_name = Path(target_file).name
        structure = cls._empty_structure()
        if not str(ast_json or "").strip():
            return structure

        try:
            root = json.loads(ast_json)
        except Exception as exc:
            structure["ast_available"] = False
            structure["ast_error"] = f"AST JSON 解析失败: {exc}"
            return structure
        structure["extraction_method"] = "ast"

        function_seen: set[tuple[Any, ...]] = set()
        variable_seen: set[tuple[Any, ...]] = set()
        loop_seen: set[tuple[Any, ...]] = set()
        conditional_seen: set[tuple[Any, ...]] = set()
        call_seen: set[tuple[Any, ...]] = set()

        def has_included_from(location: dict[str, Any]) -> bool:
            return isinstance(location.get("includedFrom"), dict)

        def first_int(*values: Any) -> int | None:
            for value in values:
                if isinstance(value, bool):
                    continue
                if isinstance(value, int):
                    return value
            return None

        def node_position(node: dict[str, Any], inherited_file: str) -> tuple[str, int | None, int | None]:
            loc = node.get("loc") if isinstance(node.get("loc"), dict) else {}
            node_range = node.get("range") if isinstance(node.get("range"), dict) else {}
            begin = node_range.get("begin") if isinstance(node_range.get("begin"), dict) else {}
            node_file = str(loc.get("file") or begin.get("file") or inherited_file or "").strip()
            line = loc.get("line") or begin.get("line")
            column = loc.get("col") or begin.get("col")
            if (
                not node_file
                and source_len is not None
                and not has_included_from(loc)
                and not has_included_from(begin)
            ):
                offset = first_int(loc.get("offset"), begin.get("offset"))
                if offset is not None and 0 <= offset <= source_len:
                    node_file = target_file
            return node_file, int(line) if line else None, int(column) if column else None

        def is_user_file(node_file: str) -> bool:
            text = str(node_file or "").strip()
            if not text:
                return False
            try:
                path = Path(text)
                if path.is_absolute():
                    return path.resolve() == target_path
                return path.name == target_name
            except (OSError, RuntimeError, ValueError):
                return text == target_name

        def extract_function_params(node: dict[str, Any]) -> list[str]:
            params: list[str] = []
            for child in node.get("inner", []):
                if not isinstance(child, dict):
                    continue
                if child.get("kind") != "ParmVarDecl":
                    continue
                name = str(child.get("name") or "").strip()
                if name:
                    params.append(name)
            return params

        def extract_callee_name(node: dict[str, Any]) -> str:
            stack: list[Any] = [node]
            while stack:
                current = stack.pop()
                if not isinstance(current, dict):
                    continue
                if current.get("kind") == "DeclRefExpr":
                    referenced = current.get("referencedDecl")
                    if isinstance(referenced, dict):
                        name = str(referenced.get("name") or "").strip()
                        if name:
                            return name
                    name = str(current.get("name") or "").strip()
                    if name:
                        return name
                if current.get("kind") == "MemberExpr":
                    name = str(current.get("name") or "").strip()
                    if name:
                        return name
                for child in reversed(current.get("inner", [])):
                    stack.append(child)
            return ""

        def walk(node: Any, inherited_file: str = "", current_function: str = "") -> None:
            if not isinstance(node, dict):
                return

            kind = str(node.get("kind") or "").strip()
            node_file, line, column = node_position(node, inherited_file)
            user_owned = is_user_file(node_file or inherited_file)
            next_function = current_function

            if kind == "FunctionDecl" and user_owned and not node.get("isImplicit", False):
                name = str(node.get("name") or "").strip()
                if name:
                    key = (name, line, column)
                    if key not in function_seen:
                        function_seen.add(key)
                        structure["functions"].append(
                            {
                                "name": name,
                                "line": line,
                                "column": column,
                                "return_type": str(
                                    (node.get("type") or {}).get("qualType") or ""
                                ).strip(),
                                "parameters": extract_function_params(node),
                            }
                        )
                    next_function = name

            if kind in {"VarDecl", "ParmVarDecl"} and user_owned and not node.get("isImplicit", False):
                name = str(node.get("name") or "").strip()
                if name:
                    scope = "parameter" if kind == "ParmVarDecl" else ("local" if current_function else "global")
                    key = (kind, name, line, column, scope)
                    if key not in variable_seen:
                        variable_seen.add(key)
                        structure["variables"].append(
                            {
                                "name": name,
                                "line": line,
                                "column": column,
                                "type": str(
                                    (node.get("type") or {}).get("qualType") or ""
                                ).strip(),
                                "scope": scope,
                            }
                        )

            if kind in {"ForStmt", "WhileStmt", "DoStmt"} and user_owned:
                key = (kind, line, column)
                if key not in loop_seen:
                    loop_seen.add(key)
                    structure["loops"].append(
                        {
                            "kind": kind,
                            "line": line,
                            "column": column,
                        }
                    )

            if kind in {"IfStmt", "SwitchStmt", "ConditionalOperator"} and user_owned:
                key = (kind, line, column)
                if key not in conditional_seen:
                    conditional_seen.add(key)
                    structure["conditionals"].append(
                        {
                            "kind": kind,
                            "line": line,
                            "column": column,
                        }
                    )

            if kind == "CallExpr" and user_owned:
                callee_name = extract_callee_name(node)
                if callee_name:
                    key = (callee_name, line, column, current_function)
                    if key not in call_seen:
                        call_seen.add(key)
                        structure["function_calls"].append(
                            {
                                "name": callee_name,
                                "line": line,
                                "column": column,
                                "caller": current_function or "",
                            }
                        )

            for child in node.get("inner", []):
                walk(child, node_file or inherited_file, next_function)

        walk(root)
        return cls._finalize_structure(structure)

    @classmethod
    def _extract_structure_fallback(cls, code: str) -> dict[str, Any]:
        source = str(code or "")
        structure = cls._empty_structure()
        structure["extraction_method"] = "regex_fallback"
        if not source.strip():
            return structure

        masked = cls._mask_comments_and_literals(source)
        function_spans: list[dict[str, Any]] = []

        function_pattern = re.compile(
            r"(?ms)(?:^|[;}\n])\s*"
            r"(?P<return_type>[A-Za-z_][\w\s\*\[\]]*?)"
            r"\s+"
            r"(?P<name>[A-Za-z_]\w*)"
            r"\s*\((?P<params>[^;{}]*)\)\s*\{"
        )
        control_keywords = {"if", "for", "while", "switch"}
        for match in function_pattern.finditer(masked):
            name = str(match.group("name") or "").strip()
            if not name or name in control_keywords:
                continue

            brace_offset = masked.find("{", match.end() - 1)
            if brace_offset < 0:
                continue
            body_end = cls._find_matching_brace(masked, brace_offset)
            if body_end is None:
                continue

            line, column = cls._offset_to_line_column(source, match.start("name"))
            params = cls._extract_fallback_params(
                source,
                match.group("params") or "",
                match.start("params"),
            )
            structure["functions"].append(
                {
                    "name": name,
                    "line": line,
                    "column": column,
                    "return_type": re.sub(
                        r"\s+",
                        " ",
                        str(match.group("return_type") or ""),
                    ).strip(),
                    "parameters": [item["name"] for item in params],
                }
            )
            for param in params:
                structure["variables"].append(
                    {
                        "name": param["name"],
                        "line": param["line"],
                        "column": param["column"],
                        "type": param["type"],
                        "scope": "parameter",
                    }
                )
            function_spans.append(
                {
                    "name": name,
                    "start": match.start(),
                    "body_start": brace_offset + 1,
                    "body_end": body_end,
                    "end": body_end + 1,
                }
            )

        loop_patterns = (
            ("ForStmt", re.compile(r"\bfor\s*\(")),
            ("WhileStmt", re.compile(r"\bwhile\s*\(")),
            ("DoStmt", re.compile(r"\bdo\b")),
        )
        for kind, pattern in loop_patterns:
            for match in pattern.finditer(masked):
                line, column = cls._offset_to_line_column(source, match.start())
                structure["loops"].append({"kind": kind, "line": line, "column": column})

        conditional_patterns = (
            ("IfStmt", re.compile(r"\bif\s*\(")),
            ("SwitchStmt", re.compile(r"\bswitch\s*\(")),
        )
        for kind, pattern in conditional_patterns:
            for match in pattern.finditer(masked):
                line, column = cls._offset_to_line_column(source, match.start())
                structure["conditionals"].append(
                    {"kind": kind, "line": line, "column": column}
                )

        sorted_spans = sorted(function_spans, key=lambda item: int(item["start"]))
        cursor = 0
        for span in sorted_spans:
            start = int(span["start"])
            if cursor < start:
                structure["variables"].extend(
                    cls._extract_fallback_declarations_in_range(
                        source=source,
                        masked=masked,
                        start=cursor,
                        end=start,
                        scope="global",
                    )
                )
            body_start = int(span["body_start"])
            body_end = int(span["body_end"])
            if body_start < body_end:
                structure["variables"].extend(
                    cls._extract_fallback_declarations_in_range(
                        source=source,
                        masked=masked,
                        start=body_start,
                        end=body_end,
                        scope="local",
                    )
                )
            cursor = max(cursor, int(span["end"]))
        if cursor < len(source):
            structure["variables"].extend(
                cls._extract_fallback_declarations_in_range(
                    source=source,
                    masked=masked,
                    start=cursor,
                    end=len(source),
                    scope="global",
                )
            )

        call_pattern = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
        call_excludes = control_keywords | {"return", "sizeof"}
        for function_span in function_spans:
            caller = str(function_span["name"])
            body_start = int(function_span["body_start"])
            body_end = int(function_span["body_end"])
            body_text = masked[body_start:body_end]
            for match in call_pattern.finditer(body_text):
                name = str(match.group(1) or "").strip()
                if not name or name in call_excludes:
                    continue
                absolute_offset = body_start + match.start(1)
                line, column = cls._offset_to_line_column(source, absolute_offset)
                structure["function_calls"].append(
                    {
                        "name": name,
                        "line": line,
                        "column": column,
                        "caller": caller,
                    }
                )

        return cls._finalize_structure(structure)

    @classmethod
    def _extract_fallback_declarations_in_range(
        cls,
        *,
        source: str,
        masked: str,
        start: int,
        end: int,
        scope: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if end <= start:
            return results

        type_pattern = (
            r"(?:void|char|short|int|long|float|double|size_t|ssize_t|ptrdiff_t|"
            r"bool|_Bool|FILE|[A-Za-z_]\w*_t|struct\s+\w+|enum\s+\w+|union\s+\w+)"
        )
        qualifier_pattern = (
            r"(?:const|static|volatile|extern|register|restrict|auto|signed|"
            r"unsigned|long|short|inline|_Atomic)"
        )
        declaration_pattern = re.compile(
            r"^\s*(?P<type>"
            r"(?:(?:" + qualifier_pattern + r")\s+)*"
            r"(?:" + type_pattern + r")"
            r"(?:\s+|\s*\*+\s*)+"
            r")(?P<rest>.+?)\s*;\s*$",
            re.DOTALL,
        )
        statement_skip_prefixes = (
            "return",
            "if",
            "for",
            "while",
            "switch",
            "case",
            "goto",
            "do",
            "typedef",
            "#",
        )

        for statement_start, statement_end in cls._iter_statement_ranges(masked, start, end):
            statement_masked = masked[statement_start:statement_end]
            stripped = statement_masked.strip()
            if not stripped or stripped.startswith(statement_skip_prefixes):
                continue
            match = declaration_pattern.match(statement_masked)
            if match is None:
                continue

            declared_type = re.sub(r"\s+", " ", str(match.group("type") or "")).strip()
            rest_start = statement_start + int(match.start("rest"))
            rest_source = str(match.group("rest") or "")
            rest_masked = statement_masked[match.start("rest") : match.end("rest")]

            for declarator_source, declarator_masked, declarator_offset in cls._split_fallback_declarators(
                source=rest_source,
                masked=rest_masked,
                base_offset=rest_start,
            ):
                left_source, left_masked = cls._split_initializer(
                    source=declarator_source,
                    masked=declarator_masked,
                )
                name, relative_offset = cls._extract_fallback_declarator_name(left_masked)
                if not name or relative_offset is None:
                    continue
                absolute_offset = declarator_offset + relative_offset
                line, column = cls._offset_to_line_column(source, absolute_offset)
                results.append(
                    {
                        "name": name,
                        "line": line,
                        "column": column,
                        "type": cls._combine_declared_type_and_declarator(
                            declared_type=declared_type,
                            declarator=left_source,
                            name=name,
                        ),
                        "scope": scope,
                    }
                )
        return results

    @staticmethod
    def _iter_statement_ranges(masked: str, start: int, end: int) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        stmt_start = start
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        for idx in range(start, end):
            ch = masked[idx]
            if ch == "(":
                paren_depth += 1
            elif ch == ")" and paren_depth > 0:
                paren_depth -= 1
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]" and bracket_depth > 0:
                bracket_depth -= 1
            elif ch == "{":
                brace_depth += 1
            elif ch == "}" and brace_depth > 0:
                brace_depth -= 1
            elif ch == ";" and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                ranges.append((stmt_start, idx + 1))
                stmt_start = idx + 1
        return ranges

    @staticmethod
    def _split_fallback_declarators(
        *,
        source: str,
        masked: str,
        base_offset: int,
    ) -> list[tuple[str, str, int]]:
        parts: list[tuple[str, str, int]] = []
        start = 0
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        for idx, ch in enumerate(masked):
            if ch == "(":
                paren_depth += 1
            elif ch == ")" and paren_depth > 0:
                paren_depth -= 1
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]" and bracket_depth > 0:
                bracket_depth -= 1
            elif ch == "{":
                brace_depth += 1
            elif ch == "}" and brace_depth > 0:
                brace_depth -= 1
            elif ch == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                parts.append((source[start:idx], masked[start:idx], base_offset + start))
                start = idx + 1
        parts.append((source[start:], masked[start:], base_offset + start))
        return [(src, msk, offset) for src, msk, offset in parts if src.strip()]

    @staticmethod
    def _split_initializer(*, source: str, masked: str) -> tuple[str, str]:
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        for idx, ch in enumerate(masked):
            if ch == "(":
                paren_depth += 1
            elif ch == ")" and paren_depth > 0:
                paren_depth -= 1
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]" and bracket_depth > 0:
                bracket_depth -= 1
            elif ch == "{":
                brace_depth += 1
            elif ch == "}" and brace_depth > 0:
                brace_depth -= 1
            elif ch == "=" and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                return source[:idx], masked[:idx]
        return source, masked

    @staticmethod
    def _extract_fallback_declarator_name(declarator: str) -> tuple[str, int | None]:
        text = str(declarator or "")
        fn_ptr_match = re.search(r"\(\s*\*+\s*([A-Za-z_]\w*)\s*\)", text)
        if fn_ptr_match is not None:
            return str(fn_ptr_match.group(1) or "").strip(), fn_ptr_match.start(1)
        match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", text)
        if match is None:
            return "", None
        return str(match.group(1) or "").strip(), match.start(1)

    @staticmethod
    def _combine_declared_type_and_declarator(*, declared_type: str, declarator: str, name: str) -> str:
        text = str(declarator or "")
        clean_name = str(name or "")
        if not clean_name:
            return declared_type
        prefix = text.split(clean_name, 1)[0]
        suffix = text.split(clean_name, 1)[1] if clean_name in text else ""
        pointer_part = re.sub(r"\s+", "", prefix)
        array_part_match = re.search(r"(\s*\[[^\]]*\])", suffix)
        array_part = array_part_match.group(1).strip() if array_part_match else ""
        combined = f"{declared_type} {pointer_part}{array_part}".strip()
        return re.sub(r"\s+", " ", combined).strip()

    @staticmethod
    def _mask_comments_and_literals(code: str) -> str:
        text = str(code or "")
        out: list[str] = []
        i = 0
        state = "normal"
        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if state == "normal":
                if ch == "/" and nxt == "/":
                    out.extend([" ", " "])
                    i += 2
                    state = "line_comment"
                    continue
                if ch == "/" and nxt == "*":
                    out.extend([" ", " "])
                    i += 2
                    state = "block_comment"
                    continue
                if ch == '"':
                    out.append(" ")
                    i += 1
                    state = "string"
                    continue
                if ch == "'":
                    out.append(" ")
                    i += 1
                    state = "char"
                    continue
                out.append(ch)
                i += 1
                continue
            if state == "line_comment":
                if ch == "\n":
                    out.append("\n")
                    state = "normal"
                else:
                    out.append(" ")
                i += 1
                continue
            if state == "block_comment":
                if ch == "*" and nxt == "/":
                    out.extend([" ", " "])
                    i += 2
                    state = "normal"
                else:
                    out.append("\n" if ch == "\n" else " ")
                    i += 1
                continue
            if state in {"string", "char"}:
                if ch == "\\" and nxt:
                    out.extend([" ", "\n" if nxt == "\n" else " "])
                    i += 2
                    continue
                if (state == "string" and ch == '"') or (state == "char" and ch == "'"):
                    out.append(" ")
                    i += 1
                    state = "normal"
                else:
                    out.append("\n" if ch == "\n" else " ")
                    i += 1
                continue
        return "".join(out)

    @staticmethod
    def _find_matching_brace(text: str, open_offset: int) -> int | None:
        depth = 0
        for idx in range(max(0, int(open_offset)), len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    return idx
        return None

    @staticmethod
    def _offset_to_line_column(text: str, offset: int) -> tuple[int | None, int | None]:
        safe_offset = max(0, min(int(offset), len(text)))
        line = text.count("\n", 0, safe_offset) + 1
        last_newline = text.rfind("\n", 0, safe_offset)
        column = safe_offset + 1 if last_newline < 0 else safe_offset - last_newline
        return line, column

    @classmethod
    def _extract_fallback_params(
        cls,
        source: str,
        params_text: str,
        start_offset: int,
    ) -> list[dict[str, Any]]:
        text = str(params_text or "").strip()
        if not text or text == "void":
            return []

        params: list[dict[str, Any]] = []
        cursor = 0
        for chunk in str(params_text or "").split(","):
            raw_chunk = chunk
            chunk_text = raw_chunk.strip()
            if not chunk_text or chunk_text == "void":
                cursor += len(raw_chunk) + 1
                continue
            name_match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", chunk_text)
            if name_match is None:
                cursor += len(raw_chunk) + 1
                continue
            name = str(name_match.group(1) or "").strip()
            absolute_offset = start_offset + cursor + raw_chunk.find(name)
            line, column = cls._offset_to_line_column(source, absolute_offset)
            param_type = chunk_text[: name_match.start(1)].strip() or chunk_text
            params.append(
                {
                    "name": name,
                    "line": line,
                    "column": column,
                    "type": re.sub(r"\s+", " ", param_type).strip(),
                }
            )
            cursor += len(raw_chunk) + 1
        return params

    @staticmethod
    def _finalize_structure(structure: dict[str, Any]) -> dict[str, Any]:
        deduped: dict[str, list[dict[str, Any]]] = {}
        dedupe_keys = {
            "functions": lambda item: (
                item.get("name"),
                item.get("line"),
                item.get("column"),
            ),
            "variables": lambda item: (
                item.get("name"),
                item.get("line"),
                item.get("column"),
                item.get("scope"),
            ),
            "loops": lambda item: (
                item.get("kind"),
                item.get("line"),
                item.get("column"),
            ),
            "conditionals": lambda item: (
                item.get("kind"),
                item.get("line"),
                item.get("column"),
            ),
            "function_calls": lambda item: (
                item.get("name"),
                item.get("line"),
                item.get("column"),
                item.get("caller"),
            ),
        }
        for key, key_builder in dedupe_keys.items():
            seen: set[tuple[Any, ...]] = set()
            rows: list[dict[str, Any]] = []
            for item in structure.get(key, []):
                if not isinstance(item, dict):
                    continue
                dedupe_key = key_builder(item)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(item)
            rows.sort(
                key=lambda item: (
                    int(item.get("line") or 0),
                    int(item.get("column") or 0),
                    str(item.get("name") or item.get("kind") or ""),
                )
            )
            deduped[key] = rows

        for key, rows in deduped.items():
            structure[key] = rows
        structure["counts"] = {
            "functions": len(structure["functions"]),
            "variables": len(structure["variables"]),
            "loops": len(structure["loops"]),
            "conditionals": len(structure["conditionals"]),
            "function_calls": len(structure["function_calls"]),
        }
        return structure

    @staticmethod
    def _empty_structure() -> dict[str, Any]:
        return {
            "functions": [],
            "variables": [],
            "loops": [],
            "conditionals": [],
            "function_calls": [],
            "counts": {
                "functions": 0,
                "variables": 0,
                "loops": 0,
                "conditionals": 0,
                "function_calls": 0,
            },
            "ast_available": True,
            "extraction_method": "",
        }

    @staticmethod
    def _short_text(text: Any, *, max_len: int) -> str:
        compact = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(compact) <= max_len:
            return compact
        return f"{compact[: max_len - 3].rstrip()}..."
