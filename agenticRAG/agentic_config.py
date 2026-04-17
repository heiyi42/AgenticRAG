import os

from dotenv import load_dotenv

load_dotenv()


def _safe_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


WORKING_DIR = os.getenv("RAG_WORKING_DIR", "./dickens")
DEBUG = os.getenv("RAG_DEBUG", "1").lower() in {"1", "true", "yes", "on"}
MAX_RETRY = _safe_env_int("RAG_MAX_RETRY", 1)
DEFAULT_THREAD_ID = os.getenv("RAG_THREAD_ID", "rag-demo-1")
DEFAULT_TOP_K = _safe_env_int("RAG_DEFAULT_TOP_K", 40)
MIN_TOP_K = _safe_env_int("RAG_MIN_TOP_K", 10)
MAX_TOP_K = _safe_env_int("RAG_MAX_TOP_K", 80)
TOPK_RETRY_STEP = _safe_env_int("RAG_TOPK_RETRY_STEP", 10)
DEFAULT_CHUNK_TOP_K = _safe_env_int("RAG_DEFAULT_CHUNK_TOP_K", 20)
MIN_CHUNK_TOP_K = _safe_env_int("RAG_MIN_CHUNK_TOP_K", 5)
MAX_CHUNK_TOP_K = _safe_env_int("RAG_MAX_CHUNK_TOP_K", 60)
CHUNK_TOPK_RETRY_STEP = _safe_env_int("RAG_CHUNK_TOPK_RETRY_STEP", 5)
SIMPLE_MAX_RETRY = _safe_env_int("RAG_SIMPLE_MAX_RETRY", 0)
COMPLEX_MAX_RETRY = _safe_env_int("RAG_COMPLEX_MAX_RETRY", MAX_RETRY)
RETRY_MAX_ITEMS = _safe_env_int("RAG_RETRY_MAX_ITEMS", 2)
