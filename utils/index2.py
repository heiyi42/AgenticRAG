import os
import re
import asyncio
import logging
import logging.config
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.llm.openai import gpt_4o_complete, gpt_4o_mini_complete, openai_embed


UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent
SUBJECT_NAME = os.getenv("LIGHTRAG_SUBJECT", "cybersec_lab")
CHAPTER_DIR = Path(
    os.getenv(
        "LIGHTRAG_CHAPTER_DIR",
        PROJECT_ROOT / "data" / "subject_chapters" / SUBJECT_NAME,
    )
)
WORKING_DIR = Path(
    os.getenv(
        "LIGHTRAG_WORKING_DIR",
        PROJECT_ROOT / "storage" / "lightrag" / SUBJECT_NAME,
    )
)

EMBED_MODEL = os.getenv("LIGHTRAG_EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("LIGHTRAG_EMBED_DIM", "3072"))
LLM_QUALITY = os.getenv("LIGHTRAG_LLM_QUALITY", "balanced").lower()
LLM_MODEL_FUNC = gpt_4o_complete if LLM_QUALITY == "high" else gpt_4o_mini_complete
LLM_MODEL_NAME = "gpt-4o" if LLM_QUALITY == "high" else "gpt-4o-mini"

CHUNK_TOKEN_SIZE_OVERRIDE = os.getenv("LIGHTRAG_CHUNK_TOKEN_SIZE")
CHUNK_OVERLAP_OVERRIDE = os.getenv("LIGHTRAG_CHUNK_OVERLAP")
ENTITY_GLEANING_OVERRIDE = os.getenv("LIGHTRAG_ENTITY_GLEANING")
MAX_PARALLEL_INSERT_OVERRIDE = os.getenv("LIGHTRAG_MAX_PARALLEL_INSERT")
EMBED_BATCH_NUM_OVERRIDE = os.getenv("LIGHTRAG_EMBED_BATCH_NUM")
EMBED_MAX_ASYNC_OVERRIDE = os.getenv("LIGHTRAG_EMBED_MAX_ASYNC")
LLM_MAX_ASYNC_OVERRIDE = os.getenv("LIGHTRAG_LLM_MAX_ASYNC")
SUMMARY_CONTEXT_SIZE_OVERRIDE = os.getenv("LIGHTRAG_SUMMARY_CONTEXT_SIZE")
SUMMARY_MAX_TOKENS_OVERRIDE = os.getenv("LIGHTRAG_SUMMARY_MAX_TOKENS")
COSINE_THRESHOLD_OVERRIDE = os.getenv("LIGHTRAG_COSINE_THRESHOLD")
FORCE_REBUILD = os.getenv("LIGHTRAG_FORCE_REBUILD", "true").lower() == "true"
KEEP_LLM_CACHE = os.getenv("LIGHTRAG_KEEP_LLM_CACHE", "true").lower() == "true"
DEFAULT_QUERY_MODE = os.getenv("LIGHTRAG_QUERY_MODE", "hybrid")
AUTO_TUNE_PARAMS = os.getenv("LIGHTRAG_AUTO_TUNE", "true").lower() == "true"

PART_RE = re.compile(r"part(?P<part>\d+)(?:_(?P<label>.*))?$", re.IGNORECASE)
SEGMENT_RE = re.compile(
    r"^# Segment:\s*(?P<index>\d+)/(?P<total>\d+)\s+\|\s+title:\s*(?P<title>.+?)\s*$",
    re.M,
)
SECTION_RE = re.compile(r"^(?P<idx>\d+\.\d+)\s+(?P<title>.+?)\s*$", re.M)
EXPERIMENT_RE = re.compile(r"^(?P<title>实验\s*\d+\s+.+?)\s*$", re.M)
STEP_RE = re.compile(r"步骤\s*(?P<num>\d+)")
CODE_TOKEN_RE = re.compile(
    r"\b(?:void|bool|struct|typedef|case|switch|SQL>|CString|DWORD|extern|SELECT|INSERT)\b"
)
BULLET_RE = re.compile(r"^\s*[•*-]\s*", re.M)


@dataclass
class LabDocument:
    path: Path
    course_name: str
    part_index: int
    unit_title: str
    unit_type: str
    doc_id: str
    content: str
    segment_title: str
    section_titles: List[str]
    experiment_headers: List[str]
    step_markers: List[str]
    code_signal_count: int
    previous_title: str = "无"
    next_title: str = "无"

    @property
    def full_title(self) -> str:
        return f"Part {self.part_index:02d} {self.unit_title}".strip()

    def as_insert_text(self) -> str:
        sections = " | ".join(self.section_titles[:20]) if self.section_titles else "无"
        experiments = " | ".join(self.experiment_headers[:8]) if self.experiment_headers else "无"
        steps = " | ".join(self.step_markers[:12]) if self.step_markers else "无"
        header = [
            f"[课程] {self.course_name}",
            f"[分段编号] {self.part_index}",
            f"[分段标题] {self.full_title}",
            f"[分段类型] {self.unit_type}",
            f"[Segment标题] {self.segment_title}",
            f"[来源文件] {self.path.name}",
            f"[上一分段] {self.previous_title}",
            f"[下一分段] {self.next_title}",
            f"[核心小节] {sections}",
            f"[实验标题] {experiments}",
            f"[步骤锚点] {steps}",
            f"[代码信号数] {self.code_signal_count}",
            "[说明] 本分段用于构建网络与信息系统安全综合实验课程知识图谱，重点保留安全概念、实验步骤、协议机制、数据库操作、代码位置与验证方法之间的关系。",
            "[图谱关系建议] 优先抽取 实验分段 -> 小节 -> 原理/机制/步骤/数据表/函数/验证工具/实验要求 的层次关系。",
            "",
        ]
        return "\n".join(header) + self.content.strip() + "\n"


@dataclass
class CorpusProfile:
    doc_count: int
    avg_chars: int
    max_chars: int
    avg_lines: int
    avg_sections: float
    avg_bullets: float
    avg_steps: float
    avg_code_signals: float


@dataclass
class RuntimeConfig:
    chunk_token_size: int
    chunk_overlap: int
    entity_gleaning: int
    max_parallel_insert: int
    embed_batch_num: int
    embed_max_async: int
    llm_max_async: int
    summary_context_size: int
    summary_max_tokens: int
    cosine_threshold: float
    tuning_reason: str


def configure_logging() -> None:
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    log_dir = Path(os.getenv("LOG_DIR", str(WORKING_DIR / "logs")))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "lightrag_cybersec_lab.log"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_file_path),
                    "maxBytes": int(os.getenv("LOG_MAX_BYTES", "10485760")),
                    "backupCount": int(os.getenv("LOG_BACKUP_COUNT", "5")),
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": os.getenv("LIGHTRAG_LOG_LEVEL", "INFO"),
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(os.getenv("LIGHTRAG_LOG_LEVEL", "INFO"))
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")
    print(f"\nLightRAG log file: {log_file_path.resolve()}\n")


def read_text_safely(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode file: {path}")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_file_identity(path: Path, content: str) -> tuple[int, str, str, str]:
    stem = path.stem
    if "__" in stem:
        course_name, suffix = stem.split("__", 1)
    else:
        course_name, suffix = SUBJECT_NAME, stem

    match = PART_RE.search(suffix)
    if not match:
        raise ValueError(f"无法从文件名解析实验分段信息: {path.name}")

    part_index = int(match.group("part"))
    label = (match.group("label") or "").strip("_")

    segment_match = SEGMENT_RE.search(content)
    segment_title = segment_match.group("title").strip() if segment_match else (label or f"part{part_index}")

    if label.lower() == "intro":
        unit_type = "intro"
        unit_title = "课程导论"
    elif label.startswith("实验_"):
        pieces = [piece for piece in label.split("_") if piece]
        exp_no = pieces[1] if len(pieces) >= 2 else str(part_index - 1)
        exp_title = pieces[2] if len(pieces) >= 3 else segment_title
        unit_type = "experiment"
        unit_title = f"实验 {exp_no} {exp_title}"
    else:
        unit_type = "section"
        unit_title = segment_title

    return part_index, course_name, unit_title, unit_type


def extract_section_titles(text: str) -> list[str]:
    titles = [normalize_whitespace(match.group(0)) for match in SECTION_RE.finditer(text)]
    deduped: list[str] = []
    seen: set[str] = set()
    for title in titles:
        if title not in seen:
            deduped.append(title)
            seen.add(title)
    return deduped


def extract_experiment_headers(text: str) -> list[str]:
    headers = [normalize_whitespace(match.group("title")) for match in EXPERIMENT_RE.finditer(text)]
    deduped: list[str] = []
    seen: set[str] = set()
    for header in headers:
        if header not in seen:
            deduped.append(header)
            seen.add(header)
    return deduped


def extract_step_markers(text: str) -> list[str]:
    seen: set[str] = set()
    markers: list[str] = []
    for match in STEP_RE.finditer(text):
        marker = f"步骤 {match.group('num')}"
        if marker not in seen:
            markers.append(marker)
            seen.add(marker)
    return markers


def parse_lab_file(path: Path) -> LabDocument:
    content = read_text_safely(path)
    part_index, course_name, unit_title, unit_type = parse_file_identity(path, content)
    segment_match = SEGMENT_RE.search(content)
    segment_title = segment_match.group("title").strip() if segment_match else unit_title

    return LabDocument(
        path=path,
        course_name=course_name,
        part_index=part_index,
        unit_title=unit_title,
        unit_type=unit_type,
        doc_id=f"{SUBJECT_NAME.lower()}_part{part_index:02d}",
        content=content,
        segment_title=segment_title,
        section_titles=extract_section_titles(content),
        experiment_headers=extract_experiment_headers(content),
        step_markers=extract_step_markers(content),
        code_signal_count=len(CODE_TOKEN_RE.findall(content)),
    )


def load_lab_documents(chapter_dir: Path = CHAPTER_DIR) -> List[LabDocument]:
    if not chapter_dir.exists():
        raise FileNotFoundError(f"Chapter directory not found: {chapter_dir.resolve()}")

    files = [p for p in chapter_dir.glob("*.txt") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No txt files found in: {chapter_dir.resolve()}")

    documents = [parse_lab_file(path) for path in files]
    documents.sort(key=lambda item: (item.part_index, item.path.name))

    for index, document in enumerate(documents):
        document.previous_title = documents[index - 1].full_title if index > 0 else "无"
        document.next_title = documents[index + 1].full_title if index < len(documents) - 1 else "无"

    return documents


def analyze_corpus(documents: List[LabDocument]) -> CorpusProfile:
    char_counts = [len(document.content) for document in documents]
    line_counts = [document.content.count("\n") + 1 for document in documents]

    return CorpusProfile(
        doc_count=len(documents),
        avg_chars=round(sum(char_counts) / len(char_counts)),
        max_chars=max(char_counts),
        avg_lines=round(sum(line_counts) / len(line_counts)),
        avg_sections=sum(len(document.section_titles) for document in documents) / len(documents),
        avg_bullets=sum(len(BULLET_RE.findall(document.content)) for document in documents) / len(documents),
        avg_steps=sum(len(document.step_markers) for document in documents) / len(documents),
        avg_code_signals=sum(document.code_signal_count for document in documents) / len(documents),
    )


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def override_int(raw_value: str | None, default: int) -> int:
    return int(raw_value) if raw_value is not None else default


def override_float(raw_value: str | None, default: float) -> float:
    return float(raw_value) if raw_value is not None else default


def choose_runtime_config(profile: CorpusProfile) -> RuntimeConfig:
    if AUTO_TUNE_PARAMS:
        base_chunk = 620
        if profile.avg_bullets >= 80:
            base_chunk -= 40
        if profile.avg_steps >= 4:
            base_chunk -= 20
        if profile.avg_code_signals >= 5:
            base_chunk -= 20
        if profile.max_chars >= 7000:
            base_chunk += 20

        chunk_token_size = clamp(base_chunk, 520, 720)
        chunk_overlap = clamp(int(chunk_token_size * 0.17), 88, 128)
        entity_gleaning = 2 if (profile.avg_steps >= 3 or profile.avg_sections >= 3) else 1
        max_parallel_insert = 3
        embed_batch_num = 10 if profile.avg_bullets >= 70 else 12
        embed_max_async = 6
        llm_max_async = 3
        summary_context_size = 10000 if profile.max_chars < 8000 else 12000
        summary_max_tokens = 800 if profile.avg_bullets >= 70 else 700
        cosine_threshold = 0.18 if profile.avg_bullets >= 70 else 0.22
        tuning_reason = (
            "实验讲义呈现为提纲式短句、项目符号和步骤说明，知识点切换频繁且 OCR 噪声较多；"
            "采用更小 chunk、略高 overlap 和更低相似度阈值，以减少跨步骤混切并保留实验链路。"
        )
    else:
        chunk_token_size = 560
        chunk_overlap = 96
        entity_gleaning = 2
        max_parallel_insert = 3
        embed_batch_num = 10
        embed_max_async = 6
        llm_max_async = 3
        summary_context_size = 10000
        summary_max_tokens = 800
        cosine_threshold = 0.18
        tuning_reason = "已关闭自动调参，使用实验讲义默认参数。"

    return RuntimeConfig(
        chunk_token_size=override_int(CHUNK_TOKEN_SIZE_OVERRIDE, chunk_token_size),
        chunk_overlap=override_int(CHUNK_OVERLAP_OVERRIDE, chunk_overlap),
        entity_gleaning=override_int(ENTITY_GLEANING_OVERRIDE, entity_gleaning),
        max_parallel_insert=override_int(MAX_PARALLEL_INSERT_OVERRIDE, max_parallel_insert),
        embed_batch_num=override_int(EMBED_BATCH_NUM_OVERRIDE, embed_batch_num),
        embed_max_async=override_int(EMBED_MAX_ASYNC_OVERRIDE, embed_max_async),
        llm_max_async=override_int(LLM_MAX_ASYNC_OVERRIDE, llm_max_async),
        summary_context_size=override_int(SUMMARY_CONTEXT_SIZE_OVERRIDE, summary_context_size),
        summary_max_tokens=override_int(SUMMARY_MAX_TOKENS_OVERRIDE, summary_max_tokens),
        cosine_threshold=override_float(COSINE_THRESHOLD_OVERRIDE, cosine_threshold),
        tuning_reason=tuning_reason,
    )


def clear_previous_index(working_dir: Path, keep_llm_cache: bool = True) -> None:
    if not working_dir.exists():
        return

    targets = [working_dir, working_dir / "rag_storage"]
    removable = {
        "graph_chunk_entity_relation.graphml",
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
        "kv_store_chunks.json",
        "kv_store_docs.json",
    }
    keep_files = {"kv_store_llm_response_cache.json"} if keep_llm_cache else set()

    for base in targets:
        if not base.exists():
            continue
        for item in base.iterdir():
            if item.is_file() and item.name in removable and item.name not in keep_files:
                item.unlink(missing_ok=True)
                print(f"Deleting old file: {item}")


def build_embedding_func() -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=EMBED_DIM,
        max_token_size=8192,
        model_name=EMBED_MODEL,
        func=partial(
            openai_embed.func,
            model=EMBED_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        ),
    )


async def initialize_rag(runtime_config: RuntimeConfig) -> LightRAG:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        embedding_func=build_embedding_func(),
        llm_model_func=LLM_MODEL_FUNC,
        llm_model_name=LLM_MODEL_NAME,
        chunk_token_size=runtime_config.chunk_token_size,
        chunk_overlap_token_size=runtime_config.chunk_overlap,
        entity_extract_max_gleaning=runtime_config.entity_gleaning,
        max_parallel_insert=runtime_config.max_parallel_insert,
        embedding_batch_num=runtime_config.embed_batch_num,
        embedding_func_max_async=runtime_config.embed_max_async,
        llm_model_max_async=runtime_config.llm_max_async,
        summary_context_size=runtime_config.summary_context_size,
        summary_max_tokens=runtime_config.summary_max_tokens,
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": runtime_config.cosine_threshold,
        },
        addon_params={
            "language": "Simplified Chinese",
            "entity_types": [
                "lab_part",
                "section",
                "security_concept",
                "security_mechanism",
                "protocol",
                "algorithm",
                "database_object",
                "function",
                "tool",
                "step",
                "requirement",
                "pitfall",
                "verification_method",
            ],
        },
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        tiktoken_model_name="gpt-4o-mini",
    )

    await rag.initialize_storages()
    return rag


async def ingest_documents(rag: LightRAG, documents: List[LabDocument]) -> list[LabDocument]:
    print("\n=====================")
    print("Indexing by part")
    print("=====================")
    for document in documents:
        print(f"- {document.doc_id}: {document.full_title} -> {document.path.name}")

    for document in documents:
        print(f"\n[Insert] {document.doc_id} | {document.full_title}")
        await rag.ainsert(
            [document.as_insert_text()],
            ids=[document.doc_id],
            file_paths=[str(document.path)],
        )

    return documents


async def run_smoke_query(rag: LightRAG) -> None:
    smoke_query = os.getenv(
        "LIGHTRAG_SMOKE_QUERY",
        "请按实验分段说明该课程中认证、访问控制和安全审计之间的关系，并指出涉及的数据库操作、关键步骤和验证方法。",
    )
    query_param = QueryParam(
        mode=DEFAULT_QUERY_MODE,
        top_k=int(os.getenv("TOP_K", "12")),
        chunk_top_k=int(os.getenv("CHUNK_TOP_K", "8")),
        max_entity_tokens=int(os.getenv("MAX_ENTITY_TOKENS", "4000")),
        max_relation_tokens=int(os.getenv("MAX_RELATION_TOKENS", "4500")),
        user_prompt=(
            "回答时优先按实验分段组织；先列出相关分段，再总结核心安全机制、"
            "实验步骤、数据库/函数位置与验证方法，最后指出可能的实现风险。"
        ),
    )

    print("\n=====================")
    print(f"Query mode: {DEFAULT_QUERY_MODE}")
    print("=====================")
    print(f"Question: {smoke_query}\n")
    print(await rag.aquery(smoke_query, param=query_param))


async def main() -> None:
    rag = None

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Example:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return

    try:
        documents = load_lab_documents(CHAPTER_DIR)
        profile = analyze_corpus(documents)
        runtime_config = choose_runtime_config(profile)

        if FORCE_REBUILD:
            clear_previous_index(WORKING_DIR, keep_llm_cache=KEEP_LLM_CACHE)

        rag = await initialize_rag(runtime_config)

        print("\n=====================")
        print("LightRAG config")
        print("=====================")
        print(f"Subject: {SUBJECT_NAME}")
        print(f"Chapter dir: {CHAPTER_DIR.resolve()}")
        print(f"Working dir: {WORKING_DIR.resolve()}")
        print(f"Embedding model: {EMBED_MODEL} ({EMBED_DIM}d)")
        print(f"LLM model: {LLM_MODEL_NAME}")
        print(
            f"Corpus profile: parts={profile.doc_count}, avg_chars={profile.avg_chars}, "
            f"max_chars={profile.max_chars}, avg_lines={profile.avg_lines}"
        )
        print(
            f"Structure density: avg_sections={profile.avg_sections:.1f}, "
            f"avg_bullets={profile.avg_bullets:.1f}, avg_steps={profile.avg_steps:.1f}, "
            f"avg_code_signals={profile.avg_code_signals:.1f}"
        )
        print(f"Auto tune: {AUTO_TUNE_PARAMS}")
        print(f"Tuning reason: {runtime_config.tuning_reason}")
        print(
            f"chunk_token_size={runtime_config.chunk_token_size}, "
            f"overlap={runtime_config.chunk_overlap}, "
            f"entity_gleaning={runtime_config.entity_gleaning}"
        )
        print(
            f"max_parallel_insert={runtime_config.max_parallel_insert}, "
            f"embedding_batch_num={runtime_config.embed_batch_num}, "
            f"embedding_func_max_async={runtime_config.embed_max_async}, "
            f"llm_model_max_async={runtime_config.llm_max_async}"
        )
        print(
            f"summary_context_size={runtime_config.summary_context_size}, "
            f"summary_max_tokens={runtime_config.summary_max_tokens}, "
            f"cosine_threshold={runtime_config.cosine_threshold}"
        )

        documents = await ingest_documents(rag, documents)
        print(f"\nIndexed parts: {len(documents)}")

        if os.getenv("LIGHTRAG_RUN_SMOKE_QUERY", "true").lower() == "true":
            await run_smoke_query(rag)

    except Exception as exc:
        print(f"An error occurred: {exc}")
        raise
    finally:
        if rag is not None:
            await rag.finalize_storages()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
