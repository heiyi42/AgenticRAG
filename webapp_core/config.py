from __future__ import annotations

import os

from . import auto_runtime as auto

WEB_ENABLE_SUMMARY_MEMORY = os.getenv("WEB_ENABLE_SUMMARY_MEMORY", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WEB_SUMMARY_TRIGGER_TOKENS = int(os.getenv("WEB_SUMMARY_TRIGGER_TOKENS", "2000"))
WEB_MAX_TURNS_BEFORE_SUMMARY = int(os.getenv("WEB_MAX_TURNS_BEFORE_SUMMARY", "4"))
WEB_KEEP_RECENT_TURNS = int(os.getenv("WEB_KEEP_RECENT_TURNS", "1"))
WEB_MAX_LOCAL_TURNS = int(os.getenv("WEB_MAX_LOCAL_TURNS", "24"))
WEB_MAX_LOCAL_MESSAGES = int(os.getenv("WEB_MAX_LOCAL_MESSAGES", "120"))
WEB_CHAT_STORE_PATH = os.getenv("WEB_CHAT_STORE_PATH", "./data/web_chats.json")
WEB_CHAT_PERSIST_DEBOUNCE_MS = int(
    os.getenv("WEB_CHAT_PERSIST_DEBOUNCE_MS", "250")
)
WEB_CHAT_TITLE_MAX_LEN = int(os.getenv("WEB_CHAT_TITLE_MAX_LEN", "12"))
WEB_CHAT_TITLE_TIMEOUT_S = int(os.getenv("WEB_CHAT_TITLE_TIMEOUT_S", "8"))
WEB_CHAT_RENAME_MAX_LEN = int(os.getenv("WEB_CHAT_RENAME_MAX_LEN", "24"))

AUTO_TIMEOUT_S = int(os.getenv("WEB_AUTO_TIMEOUT_S", str(auto.AUTO_QUERY_TIMEOUT_S)))
INSTANT_QUERY_TIMEOUT_S = int(os.getenv("INSTANT_QUERY_TIMEOUT_S", "60"))
DEEP_QUERY_TIMEOUT_S = int(os.getenv("DEEP_QUERY_TIMEOUT_S", "120"))
AUTO_ROUTE_RATIO = auto._clamp_ratio(
    os.getenv("WEB_AUTO_ROUTE_RATIO", str(auto.AUTO_ROUTE_BUDGET_RATIO)),
    auto.AUTO_ROUTE_BUDGET_RATIO,
)
AUTO_INSTANT_RATIO = auto._clamp_ratio(
    os.getenv("WEB_AUTO_INSTANT_RATIO", str(auto.AUTO_INSTANT_BUDGET_RATIO)),
    auto.AUTO_INSTANT_BUDGET_RATIO,
)
AUTO_ROUTE_THRESHOLD = auto._clamp_confidence(
    os.getenv(
        "WEB_AUTO_ROUTE_THRESHOLD",
        str(auto.AUTO_ROUTE_CONFIDENCE_THRESHOLD),
    )
)
WEB_AUTO_SPECULATIVE_SECONDARY = os.getenv(
    "WEB_AUTO_SPECULATIVE_SECONDARY", "1"
).lower() in {"1", "true", "yes", "on"}

WEB_ENABLE_RETRIEVAL_GATE = os.getenv("WEB_ENABLE_RETRIEVAL_GATE", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WEB_RETRIEVAL_GATE_TIMEOUT_S = int(os.getenv("WEB_RETRIEVAL_GATE_TIMEOUT_S", "8"))
WEB_RETRIEVAL_GATE_MIN_CONFIDENCE = auto._clamp_confidence(
    os.getenv("WEB_RETRIEVAL_GATE_MIN_CONFIDENCE", "0.70")
)
WEB_RETRIEVAL_GATE_KB_THRESHOLD = auto._clamp_confidence(
    os.getenv("WEB_RETRIEVAL_GATE_KB_THRESHOLD", "0.40")
)
WEB_RETRIEVAL_GATE_DIRECT_THRESHOLD = auto._clamp_confidence(
    os.getenv("WEB_RETRIEVAL_GATE_DIRECT_THRESHOLD", "0.90")
)
WEB_RETRIEVAL_GATE_MARGIN = max(
    0.0,
    float(os.getenv("WEB_RETRIEVAL_GATE_MARGIN", "0.12")),
)
WEB_RETRIEVAL_GATE_CACHE_TTL_S = int(os.getenv("WEB_RETRIEVAL_GATE_CACHE_TTL_S", "300"))
WEB_RETRIEVAL_GATE_CACHE_MAX_ENTRIES = int(
    os.getenv("WEB_RETRIEVAL_GATE_CACHE_MAX_ENTRIES", "256")
)
WEB_DIRECT_ANSWER_TIMEOUT_S = int(os.getenv("WEB_DIRECT_ANSWER_TIMEOUT_S", "20"))
WEB_ENABLE_CODE_ANALYSIS = os.getenv("WEB_ENABLE_CODE_ANALYSIS", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WEB_CODE_ANALYSIS_TIMEOUT_S = int(os.getenv("WEB_CODE_ANALYSIS_TIMEOUT_S", "15"))
WEB_CODE_ANALYSIS_MAX_CHARS = int(os.getenv("WEB_CODE_ANALYSIS_MAX_CHARS", "12000"))
WEB_CODE_ANALYSIS_COMPILER = os.getenv("WEB_CODE_ANALYSIS_COMPILER", "").strip()
WEB_ENABLE_CODE_EXECUTION = os.getenv("WEB_ENABLE_CODE_EXECUTION", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WEB_CODE_EXECUTION_TIMEOUT_S = int(os.getenv("WEB_CODE_EXECUTION_TIMEOUT_S", "2"))
WEB_CODE_EXECUTION_MAX_CHARS = int(os.getenv("WEB_CODE_EXECUTION_MAX_CHARS", "4000"))
WEB_CODE_EXECUTION_MAX_OUTPUT_CHARS = int(
    os.getenv("WEB_CODE_EXECUTION_MAX_OUTPUT_CHARS", "1200")
)
WEB_ENABLE_PROBLEM_TUTORING = os.getenv(
    "WEB_ENABLE_PROBLEM_TUTORING", "1"
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WEB_PROBLEM_TUTORING_PREP_TIMEOUT_S = int(
    os.getenv("WEB_PROBLEM_TUTORING_PREP_TIMEOUT_S", "18")
)
WEB_STORAGE_ROOT = os.getenv("WEB_STORAGE_ROOT", "./storage")
WEB_KB_SCOPE_DESC = os.getenv(
    "WEB_KB_SCOPE_DESC",
    "当前知识库只覆盖三类固定私有课程语料：C语言、操作系统、网络安全实验；不属于这三类的问题应优先免检索直答。",
)

WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "7860"))
WEB_DEBUG = os.getenv("WEB_DEBUG", "0").lower() in {"1", "true", "yes", "on"}

MODE_SET = {"auto", "instant", "deepsearch"}
MODE_LABEL = {
    "auto": "Auto",
    "instant": "Instant",
    "deepsearch": "DeepSearch",
}

DEFAULT_TIMEOUT_BY_MODE = {
    "auto": AUTO_TIMEOUT_S,
    "instant": INSTANT_QUERY_TIMEOUT_S,
    "deepsearch": DEEP_QUERY_TIMEOUT_S,
}
