import asyncio
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List

from langchain_openai import ChatOpenAI
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import TiktokenTokenizer, Tokenizer, wrap_embedding_func_with_attrs

from agenticRAG.agentic_config import DEBUG, WORKING_DIR
from agenticRAG.agentic_schema import (
    AdaptiveQueryPlan,
    EvidenceCheck,
    QuestionComplexity,
    SimpleQueryPlan,
    SubQuestionQueryPlan,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_subquestion_plan_struct = llm.with_structured_output(SubQuestionQueryPlan)
llm_evidence_struct = llm.with_structured_output(EvidenceCheck)
llm_complexity_struct = llm.with_structured_output(QuestionComplexity)
llm_simple_plan_struct = llm.with_structured_output(SimpleQueryPlan)
llm_adaptive_plan_struct = llm.with_structured_output(AdaptiveQueryPlan)

rag_by_working_dir: dict[str, LightRAG] = {}
rag_init_locks: dict[str, asyncio.Lock] = {}
tiktoken_ready: bool | None = None
current_working_dir: ContextVar[str] = ContextVar(
    "current_rag_working_dir",
    default=os.path.abspath(WORKING_DIR),
)
EMBED_MODEL = os.getenv("LIGHTRAG_EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("LIGHTRAG_EMBED_DIM", "3072"))
EMBED_MAX_TOKEN_SIZE = int(os.getenv("LIGHTRAG_EMBED_MAX_TOKEN_SIZE", "8192"))


class _CharTokenizer:
    """Network-free fallback tokenizer when tiktoken encoding download fails."""

    def encode(self, content: str) -> List[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: List[int]) -> str:
        chars: List[str] = []
        for t in tokens:
            if isinstance(t, int) and 0 <= t <= 0x10FFFF:
                chars.append(chr(t))
        return "".join(chars)


def _check_tiktoken_ready() -> bool:
    global tiktoken_ready
    if tiktoken_ready is not None:
        return tiktoken_ready
    try:
        tok = TiktokenTokenizer("gpt-4o-mini")
        # Warm up once to ensure encoding file is actually available locally.
        tok.encode("tokenizer_warmup")
        tiktoken_ready = True
    except Exception as e:
        tiktoken_ready = False
        if DEBUG:
            print(f"[DEBUG] tiktoken 初始化失败，切换到降级路径: {e}")
    return tiktoken_ready


def _build_tokenizer() -> Tokenizer:
    if _check_tiktoken_ready():
        return TiktokenTokenizer("gpt-4o-mini")
    return Tokenizer(model_name="char-fallback", tokenizer=_CharTokenizer())


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_DIM,
    max_token_size=EMBED_MAX_TOKEN_SIZE,
    model_name=EMBED_MODEL,
)
async def configured_openai_embed(texts, **kwargs):
    kwargs.setdefault("model", EMBED_MODEL)
    kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
    kwargs.setdefault("base_url", os.getenv("OPENAI_BASE_URL"))
    kwargs.setdefault("max_token_size", EMBED_MAX_TOKEN_SIZE)
    return await openai_embed.func(texts, **kwargs)


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_DIM,
    max_token_size=0,
    model_name=EMBED_MODEL,
)
async def openai_embed_no_tiktoken(texts, **kwargs):
    """Disable tiktoken-based truncation to avoid runtime encoding download failures."""
    kwargs.setdefault("model", EMBED_MODEL)
    kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
    kwargs.setdefault("base_url", os.getenv("OPENAI_BASE_URL"))
    kwargs["max_token_size"] = 0
    return await openai_embed.func(texts, **kwargs)


def _build_embedding_func():
    # Keep runtime embedding config aligned with the indexing scripts.
    if _check_tiktoken_ready():
        if DEBUG:
            print(
                f"[DEBUG] 使用 configured_openai_embed: model={EMBED_MODEL}, dim={EMBED_DIM}"
            )
        return configured_openai_embed
    if DEBUG:
        print(
            "[DEBUG] 使用 openai_embed_no_tiktoken "
            f"(model={EMBED_MODEL}, dim={EMBED_DIM}, 禁用 tiktoken 截断)"
        )
    return openai_embed_no_tiktoken


def _normalize_working_dir(working_dir: str | None = None) -> str:
    base = working_dir or current_working_dir.get()
    return os.path.abspath(str(base))


@contextmanager
def use_rag_working_dir(working_dir: str | None):
    token = current_working_dir.set(_normalize_working_dir(working_dir))
    try:
        yield current_working_dir.get()
    finally:
        current_working_dir.reset(token)


async def _ainit_rag(working_dir: str) -> LightRAG:
    r = LightRAG(
        working_dir=working_dir,
        embedding_func=_build_embedding_func(),
        llm_model_func=gpt_4o_mini_complete,
        tokenizer=_build_tokenizer(),
    )
    await r.initialize_storages()
    return r


async def get_rag(working_dir: str | None = None) -> LightRAG:
    resolved_working_dir = _normalize_working_dir(working_dir)
    rag = rag_by_working_dir.get(resolved_working_dir)
    if rag is not None:
        return rag

    lock = rag_init_locks.get(resolved_working_dir)
    if lock is None:
        lock = asyncio.Lock()
        rag_init_locks[resolved_working_dir] = lock

    async with lock:
        rag = rag_by_working_dir.get(resolved_working_dir)
        if rag is None:
            rag = await _ainit_rag(resolved_working_dir)
            rag_by_working_dir[resolved_working_dir] = rag
    return rag
