from __future__ import annotations

import os
import time

from lightrag import QueryParam

from agenticRAG.agentic_config import DEFAULT_THREAD_ID
from agenticRAG.agentic_runtime import get_rag, use_rag_working_dir
from agenticRAG.instant_nodes import route_query_mode_async
from agenticRAG.query_utils import extract_query_response_fields

DEFAULT_INSTANT_THREAD_ID = os.getenv("INSTANT_THREAD_ID", DEFAULT_THREAD_ID)
INSTANT_TOP_K = int(os.getenv("INSTANT_TOP_K", "20"))
INSTANT_CHUNK_TOP_K = int(os.getenv("INSTANT_CHUNK_TOP_K", "10"))
INSTANT_RESPONSE_TYPE = os.getenv(
    "INSTANT_RESPONSE_TYPE",
    (
        "Markdown；建议小节标题使用 emoji："
        "### ✅ 结论、### 📚 关键依据、### ⚠️ 不确定点（可选）、### 💡 建议（可选）；"
        "emoji 主要用于标题，正文尽量不用，全文最多 3 个"
    ),
)


async def _answer_instant_query(
    question: str,
    *,
    working_dir: str | None = None,
    stream: bool = False,
) -> dict:
    t0 = time.perf_counter()
    with use_rag_working_dir(working_dir):
        try:
            route_mode, route_reason = await route_query_mode_async(question)
        except Exception as e:
            route_mode = "hybrid"
            route_reason = f"instant_route_fallback: {type(e).__name__}: {e}"
        rag = await get_rag()
        query_resp = await rag.aquery_llm(
            question,
            param=QueryParam(
                mode=route_mode,
                top_k=INSTANT_TOP_K,
                chunk_top_k=INSTANT_CHUNK_TOP_K,
                enable_rerank=False,
                response_type=INSTANT_RESPONSE_TYPE,
                stream=stream,
            ),
        )
    status, message, failure_reason, answer = extract_query_response_fields(query_resp)
    result = {
        "route_mode": route_mode,
        "route_reason": route_reason,
        "answer": answer,
        "elapsed_ms": str(int((time.perf_counter() - t0) * 1000)),
        "query_status": status,
        "query_message": message,
        "query_failure_reason": failure_reason,
        "raw": query_resp,
    }
    if stream:
        llm_resp = query_resp.get("llm_response", {}) or {}
        response_iterator = llm_resp.get("response_iterator")
        result["response_iterator"] = response_iterator
        result["is_streaming"] = (
            bool(llm_resp.get("is_streaming")) and response_iterator is not None
        )
    return result


async def answer_instant(
    question: str,
    thread_id: str = DEFAULT_INSTANT_THREAD_ID,
    working_dir: str | None = None,
) -> dict:
    del thread_id
    return await _answer_instant_query(question, working_dir=working_dir, stream=False)


async def answer_instant_stream(
    question: str,
    thread_id: str = DEFAULT_INSTANT_THREAD_ID,
    working_dir: str | None = None,
) -> dict:
    del thread_id
    return await _answer_instant_query(question, working_dir=working_dir, stream=True)
