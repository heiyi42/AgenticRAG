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


# -----------------------------
# Project paths
# -----------------------------
UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent
SUBJECT_NAME = os.getenv("LIGHTRAG_SUBJECT", "C_program")
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


# -----------------------------
# Tuned params for chapterized, condensed course notes
# -----------------------------
EMBED_MODEL = os.getenv("LIGHTRAG_EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("LIGHTRAG_EMBED_DIM", "3072"))
LLM_QUALITY = os.getenv("LIGHTRAG_LLM_QUALITY", "balanced").lower()  # balanced | high
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


CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000}
CHAPTER_RE = re.compile(r"第(?P<num>[零〇一二两三四五六七八九十百千\d]+)章[_-]?(?P<title>.*)")
SECTION_RE = re.compile(r"^(?P<idx>\d+)\.\s+(?P<title>.+?)\s*$")
SUBSECTION_RE = re.compile(r"^(?P<idx>\d+\.\d+)\s+(?P<title>.+?)\s*$")
EXAMPLE_RE = re.compile(r"^示例\d*[：:]\s*(?P<title>.+?)\s*$")
TAG_TOKEN_RE = re.compile(r"#([^\s#]+)")
BULLET_RE = re.compile(r"^\s*[-*•✅]")


@dataclass
class ChapterDocument:
    path: Path
    course_name: str
    chapter_index: int
    chapter_title: str
    doc_id: str
    content: str
    index_tags: List[str]
    top_sections: List[str]
    sub_sections: List[str]
    code_block_count: int
    previous_title: str = "无"
    next_title: str = "无"

    @property
    def full_title(self) -> str:
        return f"第{self.chapter_index}章 {self.chapter_title}" if self.chapter_title else f"第{self.chapter_index}章"

    def as_insert_text(self) -> str:
        section_summary = " | ".join(self.top_sections[:16]) if self.top_sections else "无"
        sub_section_summary = " | ".join(self.sub_sections[:20]) if self.sub_sections else "无"
        tag_summary = ", ".join(self.index_tags[:32]) if self.index_tags else "无"
        header = [
            f"[课程] {self.course_name}",
            f"[章节编号] {self.chapter_index}",
            f"[章节标题] {self.full_title}",
            f"[来源文件] {self.path.name}",
            "[资料类型] LightRAG缩印版课程资料",
            f"[上一章] {self.previous_title}",
            f"[下一章] {self.next_title}",
            f"[章节标签] {tag_summary}",
            f"[一级小节] {section_summary}",
            f"[二级小节] {sub_section_summary}",
            f"[代码示例块数] {self.code_block_count}",
            "[说明] 本章内容用于构建 C 语言课程知识图谱，重点保留概念、语法、函数、易错点、示例代码之间的关系。",
            "[图谱关系建议] 优先抽取 章节 -> 小节 -> 概念/语法/函数/库/易错点/示例 的层次关系。",
            "",
        ]
        return "\n".join(header) + self.content.strip() + "\n"


@dataclass
class CorpusProfile:
    chapter_count: int
    avg_chars: int
    max_chars: int
    avg_lines: int
    avg_top_sections: float
    avg_sub_sections: float
    avg_code_blocks: float
    avg_tag_count: float
    avg_bullets: float


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
    log_file_path = log_dir / "lightrag_c_program.log"

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


def chinese_numeral_to_int(text: str) -> int:
    if text.isdigit():
        return int(text)

    total = 0
    number = 0
    section = 0

    for char in text:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
        elif char in CHINESE_UNITS:
            unit = CHINESE_UNITS[char]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        else:
            raise ValueError(f"Unsupported Chinese numeral: {text}")

    total += section + number
    return total


def parse_index_tags(text: str) -> List[str]:
    tags: list[str] = []
    seen: set[str] = set()
    capture = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if capture:
                break
            continue

        if "索引标签" in line:
            capture = True
            continue

        if not capture:
            continue

        if line.startswith("#"):
            for tag in TAG_TOKEN_RE.findall(line):
                cleaned = tag.strip()
                if cleaned and cleaned not in seen:
                    tags.append(cleaned)
                    seen.add(cleaned)
        else:
            break

    return tags


def extract_structured_titles(text: str) -> tuple[list[str], list[str]]:
    top_sections: list[str] = []
    sub_sections: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        sub_match = SUBSECTION_RE.match(line)
        if sub_match:
            sub_sections.append(f"{sub_match.group('idx')} {sub_match.group('title')}")
            continue

        top_match = SECTION_RE.match(line)
        if top_match:
            top_sections.append(f"{top_match.group('idx')} {top_match.group('title')}")
            continue

        example_match = EXAMPLE_RE.match(line)
        if example_match:
            sub_sections.append(f"示例 {example_match.group('title')}")

    return top_sections, sub_sections


def parse_chapter_file(path: Path) -> ChapterDocument:
    stem = path.stem
    parts = stem.split("_")
    course_name = parts[0] if parts else SUBJECT_NAME

    match = CHAPTER_RE.search(stem)
    if not match:
        raise ValueError(f"无法从文件名解析章节信息: {path.name}")

    chapter_index = chinese_numeral_to_int(match.group("num"))
    chapter_title = match.group("title").strip(" _-") or f"第{chapter_index}章"

    content = read_text_safely(path)
    doc_id = f"{SUBJECT_NAME.lower()}_ch{chapter_index:02d}"
    index_tags = parse_index_tags(content)
    top_sections, sub_sections = extract_structured_titles(content)

    return ChapterDocument(
        path=path,
        course_name=course_name,
        chapter_index=chapter_index,
        chapter_title=chapter_title,
        doc_id=doc_id,
        content=content,
        index_tags=index_tags,
        top_sections=top_sections,
        sub_sections=sub_sections,
        code_block_count=content.count("```") // 2,
    )


def read_text_safely(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode file: {path}")


def load_chapter_documents(chapter_dir: Path = CHAPTER_DIR) -> List[ChapterDocument]:
    if not chapter_dir.exists():
        raise FileNotFoundError(f"Chapter directory not found: {chapter_dir.resolve()}")

    files = [p for p in chapter_dir.glob("*.txt") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No txt files found in: {chapter_dir.resolve()}")

    chapters = [parse_chapter_file(path) for path in files]
    chapters.sort(key=lambda x: (x.chapter_index, x.path.name))

    for i, chapter in enumerate(chapters):
        chapter.previous_title = chapters[i - 1].full_title if i > 0 else "无"
        chapter.next_title = chapters[i + 1].full_title if i < len(chapters) - 1 else "无"

    return chapters


def analyze_chapter_corpus(chapters: List[ChapterDocument]) -> CorpusProfile:
    char_counts = [len(chapter.content) for chapter in chapters]
    line_counts = [chapter.content.count("\n") + 1 for chapter in chapters]
    bullet_counts = [
        sum(1 for line in chapter.content.splitlines() if BULLET_RE.match(line.strip()))
        for chapter in chapters
    ]

    return CorpusProfile(
        chapter_count=len(chapters),
        avg_chars=round(sum(char_counts) / len(char_counts)),
        max_chars=max(char_counts),
        avg_lines=round(sum(line_counts) / len(line_counts)),
        avg_top_sections=sum(len(chapter.top_sections) for chapter in chapters) / len(chapters),
        avg_sub_sections=sum(len(chapter.sub_sections) for chapter in chapters) / len(chapters),
        avg_code_blocks=sum(chapter.code_block_count for chapter in chapters) / len(chapters),
        avg_tag_count=sum(len(chapter.index_tags) for chapter in chapters) / len(chapters),
        avg_bullets=sum(bullet_counts) / len(chapters),
    )


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def override_int(raw_value: str | None, default: int) -> int:
    return int(raw_value) if raw_value is not None else default


def override_float(raw_value: str | None, default: float) -> float:
    return float(raw_value) if raw_value is not None else default


def choose_runtime_config(profile: CorpusProfile) -> RuntimeConfig:
    if AUTO_TUNE_PARAMS:
        base_chunk = 760
        if profile.avg_chars >= 9000:
            base_chunk -= 80
        if profile.avg_sub_sections >= 18:
            base_chunk -= 40
        if profile.avg_code_blocks >= 12:
            base_chunk -= 40
        if profile.avg_tag_count >= 18:
            base_chunk -= 20

        chunk_token_size = clamp(base_chunk, 560, 860)
        chunk_overlap = clamp(int(chunk_token_size * 0.16), 96, 160)
        entity_gleaning = 2 if (profile.avg_sub_sections >= 12 or profile.avg_tag_count >= 15) else 1
        max_parallel_insert = 2 if profile.max_chars >= 10000 else 3
        embed_batch_num = 12 if profile.avg_code_blocks >= 12 else 16
        embed_max_async = 6 if profile.avg_code_blocks >= 12 else 8
        llm_max_async = 3 if profile.avg_sub_sections >= 12 else 4
        summary_context_size = 14000 if profile.max_chars >= 11000 else 12000
        summary_max_tokens = 900 if profile.avg_chars >= 8500 else 700
        cosine_threshold = 0.2 if profile.avg_tag_count >= 18 else 0.23
        tuning_reason = (
            "章节文本结构密集，包含大量小节标题、索引标签和代码块；"
            "采用偏小 chunk 与中等 overlap，减少跨知识点混切，并提高实体关系抽取稳定性。"
        )
    else:
        chunk_token_size = 760
        chunk_overlap = 120
        entity_gleaning = 1
        max_parallel_insert = 3
        embed_batch_num = 16
        embed_max_async = 8
        llm_max_async = 4
        summary_context_size = 12000
        summary_max_tokens = 700
        cosine_threshold = 0.23
        tuning_reason = "已关闭自动调参，使用保守默认值。"

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
                "chapter",
                "section",
                "concept",
                "keyword",
                "operator",
                "type",
                "statement",
                "function",
                "library",
                "memory_model",
                "error",
                "pitfall",
                "example",
            ],
        },
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        tiktoken_model_name="gpt-4o-mini",
    )

    await rag.initialize_storages()
    return rag


async def ingest_chapters(rag: LightRAG, chapters: List[ChapterDocument]) -> list[ChapterDocument]:
    print("\n=====================")
    print("Indexing by chapter")
    print("=====================")
    for chapter in chapters:
        print(f"- {chapter.doc_id}: {chapter.full_title} -> {chapter.path.name}")

    for chapter in chapters:
        print(f"\n[Insert] {chapter.doc_id} | {chapter.full_title}")
        await rag.ainsert(
            [chapter.as_insert_text()],
            ids=[chapter.doc_id],
            file_paths=[str(chapter.path)],
        )
    return chapters


async def run_smoke_query(rag: LightRAG) -> None:
    smoke_query = os.getenv(
        "LIGHTRAG_SMOKE_QUERY",
        "请按章节说明 C 语言中指针与数组的联系与区别，并给出一个最小可运行示例。",
    )
    query_param = QueryParam(
        mode=DEFAULT_QUERY_MODE,
        top_k=int(os.getenv("TOP_K", "12")),
        chunk_top_k=int(os.getenv("CHUNK_TOP_K", "8")),
        max_entity_tokens=int(os.getenv("MAX_ENTITY_TOKENS", "4000")),
        max_relation_tokens=int(os.getenv("MAX_RELATION_TOKENS", "4500")),
        user_prompt=(
            "回答时优先按章节组织；先列出涉及章节，再概括核心概念，"
            "最后给出简短 C 代码示例与易错点提醒。"
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
        chapters = load_chapter_documents(CHAPTER_DIR)
        profile = analyze_chapter_corpus(chapters)
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
            f"Corpus profile: chapters={profile.chapter_count}, avg_chars={profile.avg_chars}, "
            f"max_chars={profile.max_chars}, avg_lines={profile.avg_lines}"
        )
        print(
            f"Structure density: avg_top_sections={profile.avg_top_sections:.1f}, "
            f"avg_sub_sections={profile.avg_sub_sections:.1f}, "
            f"avg_tags={profile.avg_tag_count:.1f}, avg_code_blocks={profile.avg_code_blocks:.1f}"
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

        chapters = await ingest_chapters(rag, chapters)
        print(f"\nIndexed chapters: {len(chapters)}")

        if os.getenv("LIGHTRAG_RUN_SMOKE_QUERY", "true").lower() == "true":
            await run_smoke_query(rag)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        if rag is not None:
            await rag.finalize_storages()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
