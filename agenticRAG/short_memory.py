from __future__ import annotations
from threading import Event, Lock, Thread
from typing import Any, Dict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

_shared_conversation_memories: Dict[str, "ConversationSummaryMemory"] = {}
_shared_conversation_memories_lock = Lock()


def _normalize_memory_key(thread_id: str) -> str:
    key = (thread_id or "").strip()
    return key or "default"


def get_shared_conversation_memory(
    *,
    thread_id: str,
    summary_model,
    summary_trigger_tokens: int = 2000,
    max_turns_before_summary: int = 4,
    keep_recent_turns: int = 1,
    debug: bool = False,
) -> "ConversationSummaryMemory":
    key = _normalize_memory_key(thread_id)
    with _shared_conversation_memories_lock:
        memory = _shared_conversation_memories.get(key)
        if memory is None:
            memory = ConversationSummaryMemory(
                summary_model=summary_model,
                summary_trigger_tokens=summary_trigger_tokens,
                max_turns_before_summary=max_turns_before_summary,
                keep_recent_turns=keep_recent_turns,
                debug=debug,
            )
            _shared_conversation_memories[key] = memory
        return memory


def clear_shared_conversation_memory(thread_id: str) -> None:
    key = _normalize_memory_key(thread_id)
    with _shared_conversation_memories_lock:
        memory = _shared_conversation_memories.pop(key, None)
    if memory is not None:
        memory.stop()


def shutdown_shared_conversation_memories() -> None:
    with _shared_conversation_memories_lock:
        memories = list(_shared_conversation_memories.values())
        _shared_conversation_memories.clear()
    for memory in memories:
        memory.stop()


class ConversationSummaryMemory:
    """Rolling summary memory for non-message graphs (e.g., deep_search flow)."""

    def __init__(
        self,
        summary_model,
        *,
        summary_trigger_tokens: int = 2000,
        max_turns_before_summary: int = 4,
        keep_recent_turns: int = 1,
        debug: bool = False,
    ):
        self.summary_model = summary_model
        self.summary_trigger_tokens = max(200, int(summary_trigger_tokens))
        self.max_turns_before_summary = max(1, int(max_turns_before_summary))
        self.keep_recent_turns = max(0, int(keep_recent_turns))
        self.debug = debug
        self.summary: str = ""
        self.recent_turns: list[tuple[str, str]] = []
        self._lock = Lock()
        self._worker_lock = Lock()
        self._summary_requested = Event()
        self._summary_stop = Event()
        self._summary_worker: Thread | None = None
        self._state_version = 0

    @staticmethod
    def _to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            out: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        out.append(txt)
                else:
                    out.append(str(item))
            return "\n".join(out).strip()
        return str(content).strip()

    def _current_tokens_from_state(
        self,
        *,
        summary: str,
        recent_turns: list[tuple[str, str]],
    ) -> int:
        msgs: list[BaseMessage] = []
        if summary:
            msgs.append(HumanMessage(content=f"summary:\n{summary}"))
        for q, a in recent_turns:
            msgs.append(HumanMessage(content=q))
            msgs.append(AIMessage(content=a))
        return count_tokens_approximately(msgs)

    def _need_summary_locked(self) -> bool:
        if len(self.recent_turns) >= self.max_turns_before_summary:
            return True
        return (
            self._current_tokens_from_state(
                summary=self.summary,
                recent_turns=self.recent_turns,
            )
            >= self.summary_trigger_tokens
        )

    def _build_summary_prompt(
        self,
        *,
        summary: str,
        recent_turns: list[tuple[str, str]],
    ) -> str:
        turns_text = []
        for i, (q, a) in enumerate(recent_turns, start=1):
            turns_text.append(f"[对话{i}] 用户: {q}")
            turns_text.append(f"[对话{i}] 助手: {a}")
        return (
            "你是会话记忆压缩器。请把已有摘要与新对话整合成一个更短、信息完整的摘要。\n"
            "要求：\n"
            "1) 保留事实、偏好、约束、未解决问题；\n"
            "2) 删除重复与寒暄；\n"
            "3) 使用简明中文要点；\n"
            "4) 不要编造。\n\n"
            f"已有摘要：\n{summary or '（无）'}\n\n"
            f"新增对话：\n{chr(10).join(turns_text)}\n\n"
            "输出新的摘要："
        )

    def _ensure_summary_worker(self) -> None:
        with self._worker_lock:
            if self._summary_worker is not None and self._summary_worker.is_alive():
                return
            self._summary_stop.clear()
            self._summary_requested.clear()
            self._summary_worker = Thread(
                target=self._summary_worker_loop,
                name="conversation-summary-worker",
                daemon=True,
            )
            self._summary_worker.start()

    def _summary_worker_loop(self) -> None:
        while not self._summary_stop.is_set():
            if not self._summary_requested.wait(timeout=0.5):
                continue
            self._summary_requested.clear()
            while not self._summary_stop.is_set():
                with self._lock:
                    if not self._need_summary_locked():
                        break
                    snapshot_version = self._state_version
                    snapshot_summary = self.summary
                    snapshot_turns = list(self.recent_turns)

                prompt = self._build_summary_prompt(
                    summary=snapshot_summary,
                    recent_turns=snapshot_turns,
                )
                try:
                    resp = self.summary_model.invoke(prompt)
                    content = getattr(resp, "content", resp)
                    new_summary = self._to_text(content)
                except Exception as e:
                    if self.debug:
                        print(f"[memory] summarize failed: {e}")
                    break

                with self._lock:
                    if snapshot_version != self._state_version:
                        continue
                    self.summary = new_summary
                    if self.keep_recent_turns > 0:
                        self.recent_turns = self.recent_turns[-self.keep_recent_turns :]
                    else:
                        self.recent_turns = []
                    self._state_version += 1
                    current_tokens = self._current_tokens_from_state(
                        summary=self.summary,
                        recent_turns=self.recent_turns,
                    )
                    recent_turns = len(self.recent_turns)
                    needs_more = self._need_summary_locked()

                if self.debug:
                    print(
                        f"[memory] summarized_async: tokens={current_tokens}, "
                        f"recent_turns={recent_turns}"
                    )
                if not needs_more:
                    break

    def clear(self) -> None:
        with self._lock:
            self.summary = ""
            self.recent_turns.clear()
            self._state_version += 1

    def has_context(self) -> bool:
        with self._lock:
            return bool(self.summary.strip()) or bool(self.recent_turns)

    def snapshot_state(self) -> dict[str, Any]:
        with self._lock:
            safe_recent = [[q, a] for q, a in self.recent_turns]
            return {
                "summary": self.summary,
                "recent_turns": safe_recent,
            }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        summary = str(state.get("summary", "") or "")
        restored_turns: list[tuple[str, str]] = []
        raw_recent_turns = state.get("recent_turns", [])
        if isinstance(raw_recent_turns, list):
            for item in raw_recent_turns:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                restored_turns.append((str(item[0] or ""), str(item[1] or "")))
        with self._lock:
            self.summary = summary
            self.recent_turns = restored_turns
            self._state_version += 1

    def stop(self) -> None:
        with self._worker_lock:
            thread = self._summary_worker
            self._summary_worker = None
            self._summary_stop.set()
            self._summary_requested.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1)
        self._summary_requested.clear()

    def build_augmented_question(self, question: str) -> str:
        q = question.strip()
        with self._lock:
            summary = self.summary.strip()
            recent_turns = list(self.recent_turns)
            keep_recent_turns = self.keep_recent_turns

        if not summary and not recent_turns:
            return q

        parts: list[str] = []
        if summary:
            parts.append(f"历史会话摘要：\n{summary}")

        if recent_turns and keep_recent_turns != 0:
            recent_lines = []
            selected_turns = (
                recent_turns
                if keep_recent_turns < 0
                else recent_turns[-keep_recent_turns:]
            )
            for i, (uq, aa) in enumerate(selected_turns, start=1):
                recent_lines.append(f"最近对话{i} - 用户：{uq}")
                recent_lines.append(f"最近对话{i} - 助手：{aa}")
            parts.append("\n".join(recent_lines))

        parts.append(f"当前问题：{q}")
        parts.append("请优先回答当前问题，必要时再参考历史摘要。")
        return "\n\n".join(parts)

    def update(self, user_question: str, assistant_answer: str) -> None:
        uq = (user_question or "").strip()
        aa = (assistant_answer or "").strip()
        if not uq and not aa:
            return

        with self._lock:
            self.recent_turns.append((uq, aa))
            self._state_version += 1
            should_summarize = self._need_summary_locked()

        if should_summarize:
            self._ensure_summary_worker()
            self._summary_requested.set()
