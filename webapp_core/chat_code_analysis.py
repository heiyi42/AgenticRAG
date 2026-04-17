from __future__ import annotations

import json
import re
from typing import Any, Callable

import models.auto as auto

from . import config as cfg
from .code_analysis_service import CodeAnalysisService


class ChatCodeAnalysisMixin:
    @staticmethod
    def _clean_code_analysis_question(user_question: str, code: str) -> str:
        text = str(user_question or "")
        clean_code = str(code or "").strip()
        text = CodeAnalysisService.CODE_BLOCK_RE.sub("", text)
        if clean_code:
            text = text.replace(clean_code, "")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        text = re.sub(r"^[\s:：,，。；;、-]+|[\s:：,，。；;、-]+$", "", text)
        return text or "请分析这段 C 代码，定位问题并给出修改建议。"

    def _build_code_analysis_prompt(
        self,
        *,
        user_question: str,
        mode: str,
        response_language: str,
        analysis: dict[str, Any],
    ) -> str:
        code = str(analysis.get("code", "")).strip()
        clean_question = self._clean_code_analysis_question(user_question, code)
        task_type = self._detect_subject_task_type("C_program", clean_question)
        if analysis.get("compile_errors"):
            task_type = "debug"
        style_instruction = self._subject_answer_style_instruction(
            "C_program",
            task_type,
            mode,
            response_language,
        )
        tool_payload = {
            "tool": analysis.get("tool", ""),
            "tool_available": bool(analysis.get("tool_available", False)),
            "tool_error": analysis.get("tool_error", ""),
            "compiler": analysis.get("compiler"),
            "compile_ok": bool(analysis.get("compile_ok", False)),
            "compile_errors": analysis.get("compile_errors", []),
            "warnings": analysis.get("warnings", []),
            "risk_findings": analysis.get("risk_findings", []),
            "structure": analysis.get("structure", {}),
            "execution": analysis.get("execution", {}),
            "detected": analysis.get("detected", {}),
            "tool_runs": analysis.get("tool_runs", {}),
        }
        payload_text = json.dumps(tool_payload, ensure_ascii=False, indent=2)
        return (
            "你是 C 语言代码教学助手。你会收到用户问题、原始 C 代码，以及 C 编译器工具返回的结构化分析结果。\n"
            "回答时必须优先依据工具结果，不要虚构不存在的编译错误、静态风险或运行结果。\n"
            "要求：\n"
            "1) 使用 Markdown；\n"
            "2) 明确区分 `编译错误`、`编译警告`、`静态风险`、`代码结构`；\n"
            "3) 如果某一类问题未发现，明确写“未发现”；\n"
            "4) 修改建议尽量给出关键代码；\n"
            "5) 只有当 `execution` 字段明确给出实际运行结果时，才能写“工具实际运行输出”；否则不要假装真的运行过程序；\n"
            "6) 解释必须面向学生，结论清楚，术语尽量解释。\n\n"
            f"回答语言要求：{self._response_language_instruction(response_language)}\n\n"
            f"[Detected task type]\n{task_type}\n\n"
            f"[Answer format requirement]\n{style_instruction}\n\n"
            f"[User question]\n{clean_question}\n\n"
            f"[C code]\n```c\n{code}\n```\n\n"
            f"[Tool result JSON]\n```json\n{payload_text}\n```"
        )

    @staticmethod
    def _build_code_analysis_unavailable_answer(
        *,
        analysis: dict[str, Any],
        response_language: str,
    ) -> str:
        tool_error = str(analysis.get("tool_error") or "").strip()
        if response_language == "en":
            return (
                "Code analysis did not run because the server did not detect an available C compiler.\n\n"
                f"{tool_error or 'Install clang, gcc, or MSVC cl and make sure it is on PATH.'}\n\n"
                "This is an environment problem, not a compile error in your submitted code. "
                "On Windows, install LLVM, MinGW-w64, or Visual Studio Build Tools, then restart this web service."
            )
        return (
            "这次没有执行代码分析，因为服务端没有检测到可用的 C 编译器。\n\n"
            f"{tool_error or '请安装 clang、gcc 或 MSVC cl，并确保它在服务进程的 PATH 中。'}\n\n"
            "这不是你提交的代码本身的编译错误，而是当前运行环境缺少编译工具。"
            "如果换到 Windows，可以安装 LLVM、MinGW-w64，或安装 Visual Studio Build Tools 后在 Developer Command Prompt 环境启动本项目。"
        )

    async def _run_code_analysis_stream(
        self,
        *,
        user_question: str,
        mode: str,
        timeout_s: int,
        response_language: str,
        code_candidate: dict[str, Any],
        emit_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        analysis_timeout = max(
            3,
            min(int(timeout_s), int(cfg.WEB_CODE_ANALYSIS_TIMEOUT_S)),
        )
        analysis = await self.code_analysis_service.analyze_code(
            code_candidate["code"],
            timeout_s=analysis_timeout,
        )
        analysis["detected"] = {
            "language": "c",
            "language_hint": code_candidate.get("language_hint", ""),
            "strategy": code_candidate.get("strategy", ""),
            "score": int(code_candidate.get("score", 0)),
            "truncated": bool(code_candidate.get("truncated", False)),
            "trigger": code_candidate.get("trigger", ""),
            "task_type": code_candidate.get("task_type", ""),
        }

        if not analysis.get("tool_available", True):
            answer = self._build_code_analysis_unavailable_answer(
                analysis=analysis,
                response_language=response_language,
            )
            if emit_text is not None:
                emit_text(answer)
            return {
                "mode_used": mode,
                "answer": answer,
                "request_kind": "code_analysis",
                "route": {
                    "chain": "code_analysis",
                    "reason": str(code_candidate.get("trigger") or "compiler_unavailable"),
                    "tool": str(analysis.get("tool") or ""),
                    "compile_ok": False,
                    "tool_available": False,
                },
                "subject_route": self._build_subject_route_meta(
                    self._build_code_analysis_subject_route()
                ),
                "upgraded": False,
                "upgrade_reason": "",
                "instant_review": None,
                "raw": {"code_analysis": analysis},
            }

        prompt = self._build_code_analysis_prompt(
            user_question=user_question,
            mode=mode,
            response_language=response_language,
            analysis=analysis,
        )
        explanation_timeout = max(
            1,
            int(timeout_s) - min(analysis_timeout, max(1, analysis_timeout // 2)),
        )
        answer = await self._stream_llm_text(
            llm_client=auto.auto_router_llm,
            prompt=prompt,
            timeout_s=explanation_timeout,
            emit_text=emit_text,
        )
        return {
            "mode_used": mode,
            "answer": answer,
            "request_kind": "code_analysis",
            "route": {
                "chain": "code_analysis",
                "reason": str(code_candidate.get("trigger") or "compiler"),
                "tool": str(analysis.get("tool") or ""),
                "compile_ok": bool(analysis.get("compile_ok", False)),
                "tool_available": bool(analysis.get("tool_available", False)),
            },
            "subject_route": self._build_subject_route_meta(
                self._build_code_analysis_subject_route()
            ),
            "upgraded": False,
            "upgrade_reason": "",
            "instant_review": None,
            "raw": {"code_analysis": analysis},
        }
