from __future__ import annotations

import uuid
from typing import Any, Callable

from agenticRAG.short_memory import (
    clear_shared_conversation_memory,
    get_shared_conversation_memory,
)
from langchain_openai import ChatOpenAI


MemoryForThread = Callable[[str], Any]


def build_memory_factory(
    *,
    use_summary_memory: bool,
    summary_trigger_tokens: int,
    max_turns_before_summary: int,
    keep_recent_turns: int,
) -> MemoryForThread:
    summary_model = (
        ChatOpenAI(model="gpt-4o-mini", temperature=0) if use_summary_memory else None
    )

    def _memory_for_thread(thread_id: str):
        if summary_model is None:
            return None
        return get_shared_conversation_memory(
            thread_id=thread_id,
            summary_model=summary_model,
            summary_trigger_tokens=summary_trigger_tokens,
            max_turns_before_summary=max_turns_before_summary,
            keep_recent_turns=keep_recent_turns,
            debug=False,
        )

    return _memory_for_thread


def is_exit_command(text: str) -> bool:
    return text.lower() in {"q", "quit", "exit"}


def is_clear_command(text: str) -> bool:
    return text.lower() in {"clear", "/clear"}


def reset_thread_after_clear(
    *,
    base_thread_id: str,
    current_thread_id: str,
    use_summary_memory: bool,
    memory_for_thread: MemoryForThread,
) -> tuple[str, Any]:
    if use_summary_memory:
        clear_shared_conversation_memory(current_thread_id)
    new_thread_id = f"{base_thread_id}-{uuid.uuid4().hex[:8]}"
    return new_thread_id, memory_for_thread(new_thread_id)


def print_deep_result(result: dict, show_details: bool) -> None:
    if show_details:
        print("子问题：")
        for i, sq in enumerate(result.get("sub_questions", []), start=1):
            mode = (
                result.get("query_modes", [])[i - 1]
                if i - 1 < len(result.get("query_modes", []))
                else "hybrid"
            )
            top_k = (
                result.get("query_topks", [])[i - 1]
                if i - 1 < len(result.get("query_topks", []))
                else "-"
            )
            chunk_top_k = (
                result.get("query_chunk_topks", [])[i - 1]
                if i - 1 < len(result.get("query_chunk_topks", []))
                else "-"
            )
            print(f"{i}. [{mode}, top_k={top_k}, chunk_top_k={chunk_top_k}] {sq}")

        print("\n子问题检索结果：")
        for item in result.get("query_results", []):
            print(
                f"{item.get('id', '-')}. "
                f"[{item.get('mode', '-')}, top_k={item.get('top_k', '-')}, "
                f"chunk_top_k={item.get('chunk_top_k', '-')}] "
                f"{item.get('answer', '')}"
            )
            print(
                f"   sufficient={item.get('sufficient', '-')}, retries={item.get('retries', '-')}, "
                f"elapsed_ms={item.get('elapsed_ms', '-')}, used_question={item.get('used_question', '-')}"
            )

        print(f"\nquery节点总耗时(ms): {result.get('query_total_ms', '0')}")

    print("\n最终回答：")
    print(result.get("final_answer", ""))
