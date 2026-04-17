from __future__ import annotations

import atexit
import time
from threading import Lock

from flask import Flask, Response, current_app, jsonify, render_template, request

from agenticRAG.cli_utils import build_memory_factory
from agenticRAG.short_memory import shutdown_shared_conversation_memories
from webapp_core import config as cfg
from webapp_core.async_runner import async_runner, run_async, submit_async
from webapp_core.chat_service import ChatService
from webapp_core.session_store import SessionStore

_STORE_EXT_KEY = "agenticrag.store"
_CHAT_SERVICE_EXT_KEY = "agenticrag.chat_service"
_BOOTSTRAP_STATE_EXT_KEY = "agenticrag.bootstrap_state"

_shared_cleanup_registered = False
_shared_cleanup_lock = Lock()


def _build_memory_factory():
    return build_memory_factory(
        use_summary_memory=cfg.WEB_ENABLE_SUMMARY_MEMORY,
        summary_trigger_tokens=cfg.WEB_SUMMARY_TRIGGER_TOKENS,
        max_turns_before_summary=cfg.WEB_MAX_TURNS_BEFORE_SUMMARY,
        keep_recent_turns=cfg.WEB_KEEP_RECENT_TURNS,
    )


def get_store(app: Flask | None = None) -> SessionStore:
    target = app or current_app
    return target.extensions[_STORE_EXT_KEY]


def get_chat_service(app: Flask | None = None) -> ChatService:
    target = app or current_app
    return target.extensions[_CHAT_SERVICE_EXT_KEY]


def _get_bootstrap_state(app: Flask) -> dict[str, object]:
    return app.extensions[_BOOTSTRAP_STATE_EXT_KEY]


def _register_cleanup_hooks(store: SessionStore) -> None:
    global _shared_cleanup_registered
    atexit.register(store.stop)
    with _shared_cleanup_lock:
        if _shared_cleanup_registered:
            return
        atexit.register(shutdown_shared_conversation_memories)
        atexit.register(async_runner.stop)
        _shared_cleanup_registered = True


def bootstrap_app(
    app: Flask,
    *,
    prewarm: bool = True,
    load_sessions: bool = True,
    register_cleanup: bool = True,
) -> Flask:
    state = _get_bootstrap_state(app)
    state_lock = state["lock"]

    with state_lock:
        store = get_store(app)
        chat_service = get_chat_service(app)

        if load_sessions and not bool(state["sessions_loaded"]):
            store.load_sessions_from_disk()
            state["sessions_loaded"] = True

        if prewarm and not bool(state["prewarm_attempted"]):
            try:
                warmed_subjects = run_async(chat_service.prewarm_subject_rags())
                print(f"[INFO] 已预热知识库: {', '.join(warmed_subjects)}")
                state["prewarm_succeeded"] = True
            except Exception as e:
                print(f"[WARN] 知识库预热失败: {e}")
                state["prewarm_succeeded"] = False
            finally:
                state["prewarm_attempted"] = True

        if register_cleanup and not bool(state["cleanup_registered"]):
            _register_cleanup_hooks(store)
            state["cleanup_registered"] = True

    return app


def home():
    return render_template("chat.html")


def modes():
    return jsonify(
        {
            "modes": [
                {"id": "auto", "name": "Auto", "description": "自动选择回答策略"},
                {"id": "instant", "name": "Instant", "description": "即刻回答"},
                {
                    "id": "deepsearch",
                    "name": "DeepSearch",
                    "description": "深度检索，回答更全面",
                },
            ]
        }
    )


def list_chats():
    return jsonify({"chats": get_store().list_sessions()})


def create_chat():
    payload = request.get_json(silent=True) or {}
    store = get_store()
    mode = store.normalize_mode(payload.get("mode", "auto"))
    session = store.create_session(mode=mode)
    with session.lock:
        data = session.to_public(include_messages=True)
    return jsonify(data)


def get_chat(chat_id: str):
    session = get_store().get_session(chat_id)
    if session is None:
        return jsonify({"error": "chat 不存在"}), 404
    with session.lock:
        data = session.to_public(include_messages=True)
    return jsonify(data)


def delete_chat(chat_id: str):
    deleted = get_store().delete_session(chat_id)
    if not deleted:
        return jsonify({"error": "chat 不存在"}), 404
    return jsonify({"ok": True, "deleted_chat_id": chat_id})


def delete_chat_alias(chat_id: str):
    deleted = get_store().delete_session(chat_id)
    if not deleted:
        return jsonify({"error": "chat 不存在"}), 404
    return jsonify({"ok": True, "deleted_chat_id": chat_id})


def set_chat_mode(chat_id: str):
    store = get_store()
    session = store.get_or_create_session(chat_id)
    payload = request.get_json(silent=True) or {}
    mode = store.normalize_mode(payload.get("mode", "auto"))
    with session.lock:
        session.mode = mode
        session.updated_at = time.time()
        data = session.to_public()
    store.persist_sessions_safely()
    return jsonify(data)


def set_chat_pin(chat_id: str):
    store = get_store()
    session = store.get_session(chat_id)
    if session is None:
        return jsonify({"error": "chat 不存在"}), 404
    payload = request.get_json(silent=True) or {}
    pinned = bool(payload.get("pinned", True))
    with session.lock:
        session.pinned = pinned
        session.updated_at = time.time()
        data = session.to_public()
    store.persist_sessions_safely()
    return jsonify(data)


def rename_chat(chat_id: str):
    store = get_store()
    session = store.get_session(chat_id)
    if session is None:
        return jsonify({"error": "chat 不存在"}), 404
    payload = request.get_json(silent=True) or {}
    title = store.normalize_manual_chat_title(payload.get("title", ""))
    if not title:
        return jsonify({"error": "标题不能为空"}), 400
    with session.lock:
        session.title = title
        session.updated_at = time.time()
        data = session.to_public()
    store.persist_sessions_safely()
    return jsonify(data)


def clear_chat(chat_id: str):
    data = get_store().clear_chat(chat_id)
    return jsonify({"ok": True, "chat": data})


def chat_message_stream(chat_id: str):
    payload = request.get_json(silent=True) or {}
    event_stream_factory, error = get_chat_service().build_chat_message_stream_handler(
        chat_id,
        payload,
    )
    if error:
        message, status = error
        return jsonify({"error": message}), status

    return Response(
        event_stream_factory(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def chat_updates_stream():
    return Response(
        get_chat_service().iter_chat_update_events(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _register_routes(app: Flask) -> None:
    app.add_url_rule("/", view_func=home, methods=["GET"])
    app.add_url_rule("/api/modes", view_func=modes, methods=["GET"])
    app.add_url_rule("/api/chats", view_func=list_chats, methods=["GET"])
    app.add_url_rule("/api/chats", view_func=create_chat, methods=["POST"])
    app.add_url_rule("/api/chats/<chat_id>", view_func=get_chat, methods=["GET"])
    app.add_url_rule(
        "/api/chats/<chat_id>",
        view_func=delete_chat,
        methods=["DELETE", "POST"],
    )
    app.add_url_rule(
        "/api/chats/<chat_id>/delete",
        view_func=delete_chat_alias,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/chats/<chat_id>/mode",
        view_func=set_chat_mode,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/chats/<chat_id>/pin",
        view_func=set_chat_pin,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/chats/<chat_id>/rename",
        view_func=rename_chat,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/chats/<chat_id>/clear",
        view_func=clear_chat,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/chats/<chat_id>/messages/stream",
        view_func=chat_message_stream,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/events/chat-updates",
        view_func=chat_updates_stream,
        methods=["GET"],
    )


def create_app(
    *,
    bootstrap: bool = False,
    prewarm: bool = False,
    load_sessions: bool = False,
    register_cleanup: bool = True,
) -> Flask:
    app = Flask(__name__, template_folder="templates")
    store = SessionStore(_build_memory_factory())
    chat_service = ChatService(store, run_async, submit_async)

    app.extensions[_STORE_EXT_KEY] = store
    app.extensions[_CHAT_SERVICE_EXT_KEY] = chat_service
    app.extensions[_BOOTSTRAP_STATE_EXT_KEY] = {
        "lock": Lock(),
        "cleanup_registered": False,
        "sessions_loaded": False,
        "prewarm_attempted": False,
        "prewarm_succeeded": False,
    }

    _register_routes(app)

    if bootstrap:
        bootstrap_app(
            app,
            prewarm=prewarm,
            load_sessions=load_sessions,
            register_cleanup=register_cleanup,
        )

    return app


app = create_app()
store = get_store(app)
chat_service = get_chat_service(app)


def main() -> None:
    bootstrap_app(app, prewarm=True, load_sessions=True)
    app.run(host=cfg.WEB_HOST, port=cfg.WEB_PORT, debug=cfg.WEB_DEBUG)


if __name__ == "__main__":
    main()
