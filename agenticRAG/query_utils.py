from __future__ import annotations

from typing import Any, Dict


def extract_query_response_fields(query_resp: dict) -> tuple[str, str, str, str]:
    status = str(query_resp.get("status", "unknown"))
    message = str(query_resp.get("message", ""))
    metadata = query_resp.get("metadata", {}) or {}
    failure_reason = str(metadata.get("failure_reason", ""))
    llm_resp = query_resp.get("llm_response", {}) or {}
    content = llm_resp.get("content")
    answer = str(content) if content is not None else ""
    if not answer and message:
        answer = message
    return status, message, failure_reason, answer


def build_query_result_row(
    *,
    question_id: str,
    question: str,
    used_question: str,
    mode: str,
    top_k: int | str,
    chunk_top_k: int | str,
    answer: str,
    query_status: str,
    query_message: str,
    query_failure_reason: str = "",
    sufficient: str = "unknown",
    judge_reason: str = "",
    rewritten_question: str = "",
    retries: int | str = 0,
    trace: str = "",
    elapsed_ms: int | str = 0,
) -> Dict[str, str]:
    return {
        "id": question_id,
        "question": question,
        "used_question": used_question,
        "mode": mode,
        "top_k": str(top_k),
        "chunk_top_k": str(chunk_top_k),
        "answer": str(answer),
        "query_status": str(query_status),
        "query_message": str(query_message),
        "query_failure_reason": str(query_failure_reason),
        "sufficient": str(sufficient),
        "judge_reason": str(judge_reason),
        "rewritten_question": str(rewritten_question),
        "retries": str(retries),
        "trace": str(trace),
        "elapsed_ms": str(elapsed_ms),
    }


def normalize_exception_message(err: Any) -> str:
    return f"{type(err).__name__}: {err}"
