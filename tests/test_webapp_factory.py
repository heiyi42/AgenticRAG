from __future__ import annotations

import importlib
import unittest
import warnings
from unittest.mock import Mock, patch


class WebappFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Importing Send from langgraph\.constants is deprecated\..*",
                category=Warning,
            )
            cls.webapp = importlib.import_module("webapp")

    def test_create_app_does_not_bootstrap_by_default(self) -> None:
        with (
            patch.object(self.webapp.SessionStore, "load_sessions_from_disk") as load_sessions,
            patch.object(self.webapp, "run_async") as run_async,
        ):
            app = self.webapp.create_app()

        state = self.webapp._get_bootstrap_state(app)
        self.assertFalse(load_sessions.called)
        self.assertFalse(run_async.called)
        self.assertFalse(state["sessions_loaded"])
        self.assertFalse(state["prewarm_attempted"])

    def test_bootstrap_app_runs_explicit_initialization_once(self) -> None:
        app = self.webapp.create_app()
        store = self.webapp.get_store(app)
        chat_service = self.webapp.get_chat_service(app)

        with (
            patch.object(store, "load_sessions_from_disk") as load_sessions,
            patch.object(
                chat_service,
                "prewarm_subject_rags",
                new=Mock(return_value="prewarm-coro"),
            ) as prewarm_subject_rags,
            patch.object(self.webapp, "run_async", return_value=["C_program"]) as run_async,
            patch.object(self.webapp.atexit, "register"),
        ):
            self.webapp.bootstrap_app(app, prewarm=True, load_sessions=True)
            self.webapp.bootstrap_app(app, prewarm=True, load_sessions=True)

        state = self.webapp._get_bootstrap_state(app)
        self.assertEqual(load_sessions.call_count, 1)
        self.assertEqual(prewarm_subject_rags.call_count, 1)
        self.assertEqual(run_async.call_count, 1)
        self.assertTrue(state["sessions_loaded"])
        self.assertTrue(state["prewarm_attempted"])
        self.assertTrue(state["prewarm_succeeded"])


if __name__ == "__main__":
    unittest.main()
