from __future__ import annotations

from typing import Any, Callable

from agenticRAG.short_memory import get_shared_conversation_memory
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
