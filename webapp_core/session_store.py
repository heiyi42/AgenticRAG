from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable

from agenticRAG.short_memory import clear_shared_conversation_memory

from . import config as cfg


def safe_int(raw: Any, default: int, floor: int = 1) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(floor, value)


def safe_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class ChatSession:
    chat_id: str
    title: str
    mode: str = "auto"
    pinned: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    memory: Any = None
    turns: list[tuple[str, str]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock, repr=False)

    def to_public(self, *, include_messages: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "chat_id": self.chat_id,
            "title": self.title,
            "mode": self.mode,
            "pinned": self.pinned,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }
        if include_messages:
            data["messages"] = list(self.messages)
        return data


class SessionStore:
    def __init__(self, memory_for_thread: Callable[[str], Any]) -> None:
        self.memory_for_thread = memory_for_thread
        self._sessions: dict[str, ChatSession] = {}
        self._sessions_lock = Lock()
        self._store_lock = Lock()
        self._persist_worker_lock = Lock()
        self._persist_requested = Event()
        self._persist_stop = Event()
        self._persist_thread: Thread | None = None
        self._persist_debounce_s = (
            max(0, int(cfg.WEB_CHAT_PERSIST_DEBOUNCE_MS)) / 1000.0
        )
        self._chat_counter = 0

    @staticmethod
    def normalize_mode(raw: Any) -> str:
        mode = str(raw or "").strip().lower()
        return mode if mode in cfg.MODE_SET else "auto"

    def mode_label(self, mode: str) -> str:
        return cfg.MODE_LABEL.get(self.normalize_mode(mode), "Auto")

    def make_assistant_meta(self, mode_used: str, elapsed_ms: int) -> str:
        return f"模式: {self.mode_label(mode_used)} | 耗时: {elapsed_ms} ms"

    @staticmethod
    def is_placeholder_title(title: str) -> bool:
        return str(title or "").strip().startswith("新聊天")

    @staticmethod
    def content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "".join(chunks)
        return str(content or "")

    def normalize_chat_title(self, text: str, *, max_len: int) -> str:
        title = str(text or "").strip()
        if not title:
            return ""
        title = title.splitlines()[0].strip()
        title = re.sub(r"\s+", " ", title).strip()
        title = title.strip("`\"'“”‘’「」『』【】[]()（）")
        title = re.sub(r"[，。！？：:；;、,.!?]", "", title)
        title = title.strip()
        if not title:
            return ""
        safe_max_len = max(1, int(max_len))
        if len(title) > safe_max_len:
            title = title[:safe_max_len].rstrip()
        return title

    def fallback_chat_title(self, question: str, *, max_len: int) -> str:
        return self.normalize_chat_title(question, max_len=max_len) or "新聊天"

    def normalize_manual_chat_title(self, raw: Any) -> str:
        title = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not title:
            return ""
        max_len = max(1, int(cfg.WEB_CHAT_RENAME_MAX_LEN))
        if len(title) > max_len:
            title = title[:max_len].rstrip()
        return title

    def normalize_turns(self, raw: Any) -> list[tuple[str, str]]:
        if not isinstance(raw, list):
            return []
        out: list[tuple[str, str]] = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            q = str(item[0] or "").strip()
            a = str(item[1] or "").strip()
            out.append((q, a))
        return out[-cfg.WEB_MAX_LOCAL_TURNS :]

    def normalize_messages(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content", "")).strip()
            meta = str(item.get("meta", "")).strip()
            row: dict[str, Any] = {"role": role, "content": content}
            if meta:
                row["meta"] = meta
            details = item.get("details")
            if isinstance(details, dict):
                row["details"] = details
            out.append(row)
        return out[-cfg.WEB_MAX_LOCAL_MESSAGES :]

    def messages_from_turns(self, turns: list[tuple[str, str]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for q, a in turns:
            out.append({"role": "user", "content": q})
            out.append({"role": "assistant", "content": a})
        return out[-cfg.WEB_MAX_LOCAL_MESSAGES :]

    @staticmethod
    def memory_snapshot(memory: Any) -> dict[str, Any]:
        if memory is None:
            return {}
        snapshot = getattr(memory, "snapshot_state", None)
        if callable(snapshot):
            state = snapshot()
            return state if isinstance(state, dict) else {}
        state: dict[str, Any] = {}
        summary = getattr(memory, "summary", None)
        recent_turns = getattr(memory, "recent_turns", None)
        if isinstance(summary, str):
            state["summary"] = summary
        if isinstance(recent_turns, list):
            safe_recent: list[list[str]] = []
            for item in recent_turns:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                safe_recent.append([str(item[0] or ""), str(item[1] or "")])
            state["recent_turns"] = safe_recent
        return state

    def restore_memory(self, memory: Any, state: Any) -> None:
        if memory is None or not isinstance(state, dict):
            return
        restore_state = getattr(memory, "restore_state", None)
        if callable(restore_state):
            restore_state(state)
            return
        if hasattr(memory, "summary"):
            memory.summary = str(state.get("summary", "") or "")
        if hasattr(memory, "recent_turns"):
            memory.recent_turns = self.normalize_turns(state.get("recent_turns", []))

    def session_to_record(self, session: ChatSession) -> dict[str, Any]:
        with session.lock:
            return {
                "chat_id": session.chat_id,
                "title": session.title,
                "mode": session.mode,
                "pinned": bool(session.pinned),
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "turns": [[q, a] for q, a in session.turns],
                "messages": list(session.messages),
                "memory": self.memory_snapshot(session.memory),
            }

    def persist_sessions_to_disk(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            chat_counter = self._chat_counter
        records = [self.session_to_record(item) for item in sessions]
        payload = {
            "version": 1,
            "chat_counter": chat_counter,
            "sessions": records,
        }

        store_path = Path(cfg.WEB_CHAT_STORE_PATH)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = store_path.with_suffix(store_path.suffix + ".tmp")
        with self._store_lock:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp_path, store_path)

    def _persist_worker_loop(self) -> None:
        while not self._persist_stop.is_set():
            if not self._persist_requested.wait(timeout=0.5):
                continue
            self._persist_requested.clear()

            if self._persist_debounce_s > 0:
                while not self._persist_stop.wait(self._persist_debounce_s):
                    if not self._persist_requested.is_set():
                        break
                    self._persist_requested.clear()

            try:
                self.persist_sessions_to_disk()
            except Exception as e:
                print(f"[WARN] 保存会话失败: {e}")

    def _ensure_persist_worker(self) -> None:
        with self._persist_worker_lock:
            if self._persist_thread is not None and self._persist_thread.is_alive():
                return
            self._persist_stop.clear()
            self._persist_requested.clear()
            self._persist_thread = Thread(
                target=self._persist_worker_loop,
                name="chat-store-writer",
                daemon=True,
            )
            self._persist_thread.start()

    def persist_sessions_safely(self, *, force_sync: bool = False) -> None:
        if force_sync or self._persist_debounce_s <= 0:
            try:
                self.persist_sessions_to_disk()
            except Exception as e:
                print(f"[WARN] 保存会话失败: {e}")
            return
        self._ensure_persist_worker()
        self._persist_requested.set()

    def stop(self) -> None:
        with self._persist_worker_lock:
            thread = self._persist_thread
            self._persist_thread = None
            self._persist_stop.set()
            self._persist_requested.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1)
        self._persist_requested.clear()
        try:
            self.persist_sessions_safely(force_sync=True)
        except Exception as e:
            print(f"[WARN] 保存会话失败: {e}")

    def load_sessions_from_disk(self) -> None:
        store_path = Path(cfg.WEB_CHAT_STORE_PATH)
        if not store_path.exists():
            return

        try:
            with self._store_lock:
                data = json.loads(store_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取会话文件失败: {e}")
            return

        if not isinstance(data, dict):
            return

        raw_sessions = data.get("sessions", [])
        if not isinstance(raw_sessions, list):
            raw_sessions = []
        restored: dict[str, ChatSession] = {}
        for row in raw_sessions:
            if not isinstance(row, dict):
                continue
            chat_id = str(row.get("chat_id", "")).strip()
            if not chat_id:
                continue
            title = str(row.get("title", "")).strip() or "新聊天"
            mode = self.normalize_mode(row.get("mode", "auto"))
            pinned = bool(row.get("pinned", False))
            created_at = safe_float(row.get("created_at"), time.time())
            updated_at = safe_float(row.get("updated_at"), created_at)

            session = ChatSession(
                chat_id=chat_id,
                title=title,
                mode=mode,
                pinned=pinned,
                created_at=created_at,
                updated_at=updated_at,
                memory=self.memory_for_thread(chat_id),
            )
            if session.memory is not None and hasattr(session.memory, "clear"):
                session.memory.clear()
            session.turns = self.normalize_turns(row.get("turns", []))
            session.messages = self.normalize_messages(row.get("messages", []))
            if not session.messages and session.turns:
                session.messages = self.messages_from_turns(session.turns)
            self.restore_memory(session.memory, row.get("memory", {}))
            restored[chat_id] = session

        loaded_counter = safe_int(data.get("chat_counter"), 0, floor=0)
        with self._sessions_lock:
            self._sessions.clear()
            self._sessions.update(restored)
            self._chat_counter = max(loaded_counter, len(restored))

    def get_session(self, chat_id: str) -> ChatSession | None:
        with self._sessions_lock:
            return self._sessions.get(chat_id)

    def _new_chat_id(self) -> str:
        return f"chat-{uuid.uuid4().hex[:8]}"

    def create_session(self, chat_id: str | None = None, mode: str = "auto") -> ChatSession:
        mode = self.normalize_mode(mode)
        with self._sessions_lock:
            self._chat_counter += 1
            real_chat_id = (chat_id or "").strip() or self._new_chat_id()
            if real_chat_id in self._sessions:
                return self._sessions[real_chat_id]
            title = f"新聊天 {self._chat_counter}"
            session = ChatSession(
                chat_id=real_chat_id,
                title=title,
                mode=mode,
                memory=self.memory_for_thread(real_chat_id),
            )
            if session.memory is not None and hasattr(session.memory, "clear"):
                session.memory.clear()
            self._sessions[real_chat_id] = session
        self.persist_sessions_safely()
        return session

    def get_or_create_session(self, chat_id: str) -> ChatSession:
        normalized = (chat_id or "").strip()
        if not normalized:
            return self.create_session()
        existing = self.get_session(normalized)
        if existing is not None:
            return existing
        return self.create_session(chat_id=normalized)

    def delete_session(self, chat_id: str) -> bool:
        normalized = (chat_id or "").strip()
        if not normalized:
            return False

        with self._sessions_lock:
            session = self._sessions.pop(normalized, None)
        if session is None:
            return False

        with session.lock:
            session.turns.clear()
            session.messages.clear()
            if session.memory is not None and hasattr(session.memory, "clear"):
                session.memory.clear()

        clear_shared_conversation_memory(normalized)
        self.persist_sessions_safely()
        return True

    @staticmethod
    def fallback_augmented_question(
        turns: list[tuple[str, str]], question: str
    ) -> str:
        q = (question or "").strip()
        if not turns:
            return q
        selected = turns[-2:]
        lines = []
        for idx, (uq, aa) in enumerate(selected, start=1):
            lines.append(f"最近对话{idx} - 用户：{uq}")
            lines.append(f"最近对话{idx} - 助手：{aa}")
        lines.append(f"当前问题：{q}")
        lines.append("请优先回答当前问题，必要时再参考最近对话。")
        return "\n\n".join(lines)

    def build_augmented_question(self, session: ChatSession, question: str) -> str:
        if session.memory is not None:
            return session.memory.build_augmented_question(question)
        return self.fallback_augmented_question(session.turns, question)

    def update_session_after_answer(
        self,
        session: ChatSession,
        *,
        question: str,
        answer: str,
        requested_mode: str,
        mode_used: str,
        elapsed_ms: int,
        assistant_meta: str | None = None,
        message_details: dict[str, Any] | None = None,
    ) -> None:
        q = question.strip()
        a = answer.strip()
        if q or a:
            session.turns.append((q, a))
            if len(session.turns) > cfg.WEB_MAX_LOCAL_TURNS:
                session.turns = session.turns[-cfg.WEB_MAX_LOCAL_TURNS :]

            session.messages.append({"role": "user", "content": q})
            session.messages.append(
                {
                    "role": "assistant",
                    "content": a,
                    "meta": assistant_meta or self.make_assistant_meta(mode_used, elapsed_ms),
                    **({"details": dict(message_details)} if isinstance(message_details, dict) else {}),
                }
            )
            if len(session.messages) > cfg.WEB_MAX_LOCAL_MESSAGES:
                session.messages = session.messages[-cfg.WEB_MAX_LOCAL_MESSAGES :]

        if session.memory is not None:
            session.memory.update(q, a)
        session.mode = self.normalize_mode(requested_mode)
        session.updated_at = time.time()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: (not bool(s.pinned), -float(s.updated_at)))
        return [item.to_public() for item in sessions]

    def clear_chat(self, chat_id: str) -> dict[str, Any]:
        session = self.get_or_create_session(chat_id)
        with session.lock:
            session.turns.clear()
            session.messages.clear()
            if session.memory is not None:
                session.memory.clear()
            session.updated_at = time.time()
            data = session.to_public(include_messages=True)
        self.persist_sessions_safely()
        return data
