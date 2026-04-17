from __future__ import annotations

import unittest

from webapp_core.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_update_session_after_answer_persists_message_details(self) -> None:
        store = SessionStore(memory_for_thread=lambda _thread_id: None)
        session = store.create_session(chat_id="chat-1")

        store.update_session_after_answer(
            session,
            question="这题怎么做",
            answer="先分析题型。",
            requested_mode="auto",
            mode_used="auto",
            elapsed_ms=123,
            assistant_meta="模式: Auto | 耗时: 123 ms | 题目辅导",
            message_details={
                "kind": "problem_tutoring",
                "subject_label": "操作系统",
                "problem_type_label": "页面置换题",
            },
        )

        self.assertEqual(len(session.messages), 2)
        assistant = session.messages[-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["details"]["kind"], "problem_tutoring")
        self.assertEqual(assistant["details"]["subject_label"], "操作系统")

        normalized = store.normalize_messages(session.messages)
        self.assertEqual(normalized[-1]["details"]["problem_type_label"], "页面置换题")


if __name__ == "__main__":
    unittest.main()
