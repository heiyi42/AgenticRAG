import os
import re
import json
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
SUBJECT_NAME = os.getenv("LIGHTRAG_SUBJECT", "operating_systems_pdf_for_index")
CHAPTER_DIR = Path(
    os.getenv(
        "LIGHTRAG_CHAPTER_DIR",
        PROJECT_ROOT / "data" / "subject_chapters" / SUBJECT_NAME,
    )
)
WORKING_DIR = Path(
    os.getenv(
        "LIGHTRAG_WORKING_DIR",
        PROJECT_ROOT / "storage" / "lightrag" / "operating_systems_pdf",
    )
)
MANIFEST_PATH = Path(
    os.getenv("LIGHTRAG_MANIFEST_PATH", CHAPTER_DIR / "index_manifest.json")
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

SECTION_RE = re.compile(
    r"^(?P<idx>\d+\.\d+)(?:\.\d+)?\s*(?:/)?\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)
SUBSECTION_RE = re.compile(
    r"^(?P<idx>\d+\.\d+\.\d+)\s*(?:/)?\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)
VISUAL_TITLE_RE = re.compile(r"^- \[(?:FIGURE|TABLE)\s+[^\]]+\]\s*(?P<title>.+?)\s*$", re.MULTILINE)
VISUAL_KIND_RE = re.compile(r"^- \[(?P<kind>FIGURE|TABLE)\s+[^\]]+\]", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*[•*-]\s+", re.MULTILINE)

BILINGUAL_TERM_MAP = [
    {"en": "operating system", "zh": "操作系统", "patterns": ["operating system", " os "]},
    {"en": "kernel", "zh": "内核", "patterns": ["kernel"]},
    {"en": "system call", "zh": "系统调用", "patterns": ["system call"]},
    {"en": "interrupt", "zh": "中断", "patterns": ["interrupt"]},
    {"en": "instruction cycle", "zh": "指令周期", "patterns": ["instruction cycle"]},
    {"en": "process", "zh": "进程", "patterns": ["process ", "processes", "process state"]},
    {"en": "process control block (PCB)", "zh": "进程控制块", "patterns": ["process control block", " pcb "]},
    {"en": "thread", "zh": "线程", "patterns": ["thread ", "threads", "multithreading"]},
    {"en": "concurrency", "zh": "并发", "patterns": ["concurrency", "concurrent"]},
    {"en": "mutual exclusion", "zh": "互斥", "patterns": ["mutual exclusion"]},
    {"en": "critical section", "zh": "临界区", "patterns": ["critical section"]},
    {"en": "race condition", "zh": "竞争条件", "patterns": ["race condition"]},
    {"en": "semaphore", "zh": "信号量", "patterns": ["semaphore", "semwait", "semsignal"]},
    {"en": "monitor", "zh": "管程", "patterns": ["monitor ", "monitors"]},
    {"en": "message passing", "zh": "消息传递", "patterns": ["message passing", "message-passing"]},
    {"en": "deadlock", "zh": "死锁", "patterns": ["deadlock"]},
    {"en": "starvation", "zh": "饥饿", "patterns": ["starvation"]},
    {"en": "memory management", "zh": "内存管理", "patterns": ["memory management"]},
    {"en": "virtual memory", "zh": "虚拟内存", "patterns": ["virtual memory"]},
    {"en": "paging", "zh": "分页", "patterns": ["paging", "paged"]},
    {"en": "segmentation", "zh": "分段", "patterns": ["segmentation", "segment "]},
    {"en": "page table", "zh": "页表", "patterns": ["page table"]},
    {"en": "translation lookaside buffer (TLB)", "zh": "快表", "patterns": ["translation lookaside buffer", " tlb "]},
    {"en": "page fault", "zh": "缺页", "patterns": ["page fault"]},
    {"en": "thrashing", "zh": "抖动", "patterns": ["thrashing"]},
    {"en": "cache", "zh": "缓存", "patterns": ["cache "]},
    {"en": "frame", "zh": "页框", "patterns": ["frame ", "frames"]},
    {"en": "scheduling", "zh": "调度", "patterns": ["scheduling", "scheduler", "dispatch"]},
    {"en": "round-robin", "zh": "时间片轮转", "patterns": ["round-robin", "round robin"]},
    {"en": "first come first served (FCFS)", "zh": "先来先服务", "patterns": ["first come first served", " fcfs "]},
    {"en": "shortest job first (SJF)", "zh": "最短作业优先", "patterns": ["shortest job first", " sjf "]},
    {"en": "multilevel feedback queue", "zh": "多级反馈队列", "patterns": ["multilevel feedback queue", "feedback scheduling"]},
    {"en": "priority inversion", "zh": "优先级反转", "patterns": ["priority inversion"]},
    {"en": "real-time scheduling", "zh": "实时调度", "patterns": ["real-time scheduling", "real time scheduling"]},
    {"en": "DMA", "zh": "直接内存访问", "patterns": [" dma ", "direct memory access"]},
    {"en": "I/O management", "zh": "I/O 管理", "patterns": ["i/o management", " i/o ", " input/output "]},
    {"en": "disk scheduling", "zh": "磁盘调度", "patterns": ["disk scheduling"]},
    {"en": "RAID", "zh": "磁盘阵列", "patterns": [" raid ", "redundant array of independent disks"]},
    {"en": "file system", "zh": "文件系统", "patterns": ["file system", "filesystem"]},
    {"en": "directory", "zh": "目录", "patterns": ["directory", "directories"]},
    {"en": "inode", "zh": "索引节点", "patterns": ["inode", "inodes"]},
    {"en": "buffer cache", "zh": "缓冲缓存", "patterns": ["buffer cache"]},
    {"en": "embedded operating system", "zh": "嵌入式操作系统", "patterns": ["embedded operating system", "embedded system"]},
    {"en": "security", "zh": "安全", "patterns": ["security", "secure"]},
    {"en": "authentication", "zh": "认证", "patterns": ["authentication", "authenticate"]},
    {"en": "access control", "zh": "访问控制", "patterns": ["access control"]},
    {"en": "malware", "zh": "恶意软件", "patterns": ["malware", "virus", "worm"]},
    {"en": "distributed processing", "zh": "分布式处理", "patterns": ["distributed processing", "distributed system"]},
    {"en": "client/server", "zh": "客户端/服务器", "patterns": ["client/server", "client server"]},
    {"en": "cluster", "zh": "集群", "patterns": ["cluster", "clusters"]},
    {"en": "remote procedure call (RPC)", "zh": "远程过程调用", "patterns": ["remote procedure call", " rpc "]},
    {"en": "middleware", "zh": "中间件", "patterns": ["middleware"]},
]


@dataclass
class OSChapterDocument:
    path: Path
    visual_path: Path | None
    course_name: str
    chapter_index: int
    chapter_title: str
    doc_id: str
    content: str
    visual_notes: str
    section_titles: List[str]
    sub_section_titles: List[str]
    learning_objectives: List[str]
    visual_titles: List[str]
    bilingual_terms: List[str]
    figure_count: int
    table_count: int
    previous_title: str = "无"
    next_title: str = "无"

    @property
    def full_title(self) -> str:
        return f"Chapter {self.chapter_index:02d} {self.chapter_title}".strip()

    @property
    def visual_count(self) -> int:
        return self.figure_count + self.table_count

    def as_insert_text(self) -> str:
        sections = " | ".join(self.section_titles[:18]) if self.section_titles else "无"
        sub_sections = " | ".join(self.sub_section_titles[:24]) if self.sub_section_titles else "无"
        objectives = " | ".join(self.learning_objectives[:8]) if self.learning_objectives else "无"
        visuals = " | ".join(self.visual_titles[:12]) if self.visual_titles else "无"
        bilingual_summary = " | ".join(self.bilingual_terms[:14]) if self.bilingual_terms else "无"
        header = [
            f"[Course] {self.course_name}",
            f"[Chapter Number] {self.chapter_index}",
            f"[Chapter Title] {self.chapter_title}",
            f"[Document Title] {self.full_title}",
            f"[Source File] {self.path.name}",
            f"[Visual Notes File] {self.visual_path.name if self.visual_path else '无'}",
            "[Material Type] Operating systems textbook chapter rebuilt from original PDF",
            f"[Previous Chapter] {self.previous_title}",
            f"[Next Chapter] {self.next_title}",
            f"[Key Sections] {sections}",
            f"[Subsections] {sub_sections}",
            f"[Learning Objectives] {objectives}",
            f"[Visual Summary] {visuals}",
            f"[Bilingual Terms] {bilingual_summary}",
            f"[Figure Count] {self.figure_count}",
            f"[Table Count] {self.table_count}",
            f"[Visual Count] {self.visual_count}",
            "[Indexing Guidance] 回答操作系统问题时，优先区分概念定义、机制流程、设计权衡和教材图示；涉及状态流转、体系结构、调度策略、地址转换、并发同步、I/O 路径和文件系统结构时，优先结合 Visual Notes 组织答案。",
            "[Chinese Query Alignment] 若用户用中文提问，优先将中文术语映射到本章对应的英文教材术语，例如 进程=process、页表=page table、时间片轮转=round-robin，再进行实体关系组织。",
            "[Graph Extraction Guidance] 优先抽取 chapter -> section -> concept/mechanism/algorithm/data_structure/state_transition/example/visual_note 的层次关系，并保留不同机制之间的对比与适用条件。",
            "",
        ]

        chunks = ["\n".join(header) + self.content.strip()]
        if self.bilingual_terms:
            chunks.append("[Bilingual Glossary]\n" + "\n".join(f"- {term}" for term in self.bilingual_terms))
        if self.visual_notes.strip():
            chunks.append(self.visual_notes.strip())
        return "\n\n".join(chunks) + "\n"


@dataclass
class CorpusProfile:
    chapter_count: int
    avg_chars: int
    max_chars: int
    avg_lines: int
    avg_sections: float
    avg_sub_sections: float
    avg_objectives: float
    avg_visuals: float
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
    log_file_path = log_dir / "lightrag_operating_systems_pdf.log"

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


def normalize_search_text(text: str) -> str:
    normalized = normalize_whitespace(text).casefold()
    return f" {normalized} "


def clean_section_title(raw_title: str) -> str:
    title = normalize_whitespace(raw_title)
    title = re.sub(r"\s+\d+$", "", title)
    title = title.strip("/ ").strip()
    return title


def dedupe_keep_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def extract_course_name(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("[Course]"):
            return line.split("]", 1)[1].strip()
    return "Operating Systems: Internals and Design Principles"


def strip_rebuilt_metadata(text: str) -> str:
    lines = text.splitlines()
    while lines and lines[0].startswith("["):
        lines.pop(0)
    return "\n".join(lines).strip()


def split_main_and_guidance(text: str) -> str:
    marker = "\n[Indexing Guidance]"
    core = text.split(marker, 1)[0].rstrip() if marker in text else text.strip()
    return strip_rebuilt_metadata(core)


def extract_section_titles(text: str, chapter_index: int) -> list[str]:
    titles: list[str] = []
    for match in SECTION_RE.finditer(text):
        idx = match.group("idx")
        if idx.count(".") != 1 or not idx.startswith(f"{chapter_index}."):
            continue
        title = clean_section_title(match.group("title"))
        if title:
            titles.append(f"{idx} {title}")
    return dedupe_keep_order(titles)


def extract_subsection_titles(text: str, chapter_index: int) -> list[str]:
    titles = []
    for match in SUBSECTION_RE.finditer(text):
        idx = match.group("idx")
        if not idx.startswith(f"{chapter_index}."):
            continue
        titles.append(f"{idx} {clean_section_title(match.group('title'))}")
    return dedupe_keep_order([title for title in titles if title.strip()])


def extract_learning_objectives(text: str, chapter_index: int) -> list[str]:
    upper_text = text.upper()
    marker = "LEARNING OBJECTIVES"
    start = upper_text.find(marker)
    if start == -1:
        return []

    remaining = text[start:]
    next_section = re.search(
        rf"^\s*{chapter_index}\.\d+\s*(?:/)?\s+.+$",
        remaining,
        re.MULTILINE,
    )
    if next_section and next_section.start() > 0:
        window = remaining[: next_section.start()]
    else:
        window = remaining[:4500]

    segments = [segment.strip() for segment in re.split(r"\n\s*[•*-]\s+", window)]
    if segments:
        segments = segments[1:]

    objectives: list[str] = []
    for segment in segments:
        candidate = normalize_whitespace(segment)
        candidate = re.split(r"\b(?:CHAPTER|[0-9]+\.[0-9]+\s*/)\b", candidate, maxsplit=1)[0]
        if candidate:
            objectives.append(candidate.rstrip("."))

    return dedupe_keep_order(objectives[:10])


def extract_visual_titles(text: str) -> list[str]:
    titles = [normalize_whitespace(match.group("title")) for match in VISUAL_TITLE_RE.finditer(text)]
    return dedupe_keep_order(titles)


def extract_bilingual_terms(
    chapter_title: str,
    section_titles: List[str],
    learning_objectives: List[str],
    visual_titles: List[str],
) -> list[str]:
    search_text = normalize_search_text(
        "\n".join(
            [chapter_title, " ".join(section_titles), " ".join(learning_objectives), " ".join(visual_titles)]
        )
    )
    matches: list[str] = []
    for entry in BILINGUAL_TERM_MAP:
        patterns = [pattern.casefold() for pattern in entry["patterns"]]
        if any(pattern in search_text for pattern in patterns):
            matches.append(f"{entry['zh']} = {entry['en']}")

    if "操作系统 = operating system" not in matches:
        matches.insert(0, "操作系统 = operating system")

    return dedupe_keep_order(matches)


def parse_chapter_documents(
    chapter_dir: Path = CHAPTER_DIR, manifest_path: Path = MANIFEST_PATH
) -> List[OSChapterDocument]:
    if not chapter_dir.exists():
        raise FileNotFoundError(f"Chapter directory not found: {chapter_dir.resolve()}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path.resolve()}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chapter_rows = manifest.get("chapters", [])
    if not chapter_rows:
        raise ValueError(f"No chapters found in manifest: {manifest_path.resolve()}")

    documents: list[OSChapterDocument] = []
    for row in sorted(chapter_rows, key=lambda item: item["chapter_num"]):
        chapter_path = chapter_dir / row["index_file"]
        if not chapter_path.exists():
            raise FileNotFoundError(f"Chapter file not found: {chapter_path.resolve()}")

        visual_path = chapter_dir / row["visual_notes_file"]
        visual_text = read_text_safely(visual_path) if visual_path.exists() else ""
        raw_main_text = read_text_safely(chapter_path)
        main_text = split_main_and_guidance(raw_main_text)
        visual_titles = extract_visual_titles(visual_text)
        chapter_title = normalize_whitespace(row["chapter_title"])
        section_titles = extract_section_titles(main_text, int(row["chapter_num"]))
        sub_section_titles = extract_subsection_titles(main_text, int(row["chapter_num"]))
        learning_objectives = extract_learning_objectives(main_text, int(row["chapter_num"]))

        documents.append(
            OSChapterDocument(
                path=chapter_path,
                visual_path=visual_path if visual_path.exists() else None,
                course_name=extract_course_name(raw_main_text),
                chapter_index=int(row["chapter_num"]),
                chapter_title=chapter_title,
                doc_id=f"operating_systems_chapter_{int(row['chapter_num']):02d}",
                content=main_text,
                visual_notes=visual_text,
                section_titles=section_titles,
                sub_section_titles=sub_section_titles,
                learning_objectives=learning_objectives,
                visual_titles=visual_titles,
                bilingual_terms=extract_bilingual_terms(
                    chapter_title=chapter_title,
                    section_titles=section_titles,
                    learning_objectives=learning_objectives,
                    visual_titles=visual_titles,
                ),
                figure_count=sum(1 for match in VISUAL_KIND_RE.finditer(visual_text) if match.group("kind") == "FIGURE"),
                table_count=sum(1 for match in VISUAL_KIND_RE.finditer(visual_text) if match.group("kind") == "TABLE"),
            )
        )

    for index, document in enumerate(documents):
        document.previous_title = documents[index - 1].full_title if index > 0 else "无"
        document.next_title = documents[index + 1].full_title if index < len(documents) - 1 else "无"

    return documents


def analyze_corpus(documents: List[OSChapterDocument]) -> CorpusProfile:
    char_counts = [len(document.content) + len(document.visual_notes) for document in documents]
    line_counts = [document.as_insert_text().count("\n") + 1 for document in documents]
    bullet_counts = [
        len(BULLET_RE.findall(document.content)) + len(BULLET_RE.findall(document.visual_notes))
        for document in documents
    ]

    return CorpusProfile(
        chapter_count=len(documents),
        avg_chars=round(sum(char_counts) / len(char_counts)),
        max_chars=max(char_counts),
        avg_lines=round(sum(line_counts) / len(line_counts)),
        avg_sections=sum(len(document.section_titles) for document in documents) / len(documents),
        avg_sub_sections=sum(len(document.sub_section_titles) for document in documents) / len(documents),
        avg_objectives=sum(len(document.learning_objectives) for document in documents) / len(documents),
        avg_visuals=sum(document.visual_count for document in documents) / len(documents),
        avg_bullets=sum(bullet_counts) / len(documents),
    )


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def override_int(raw_value: str | None, default: int) -> int:
    return int(raw_value) if raw_value is not None else default


def override_float(raw_value: str | None, default: float) -> float:
    return float(raw_value) if raw_value is not None else default


def choose_runtime_config(profile: CorpusProfile) -> RuntimeConfig:
    if AUTO_TUNE_PARAMS:
        base_chunk = 1280
        if profile.avg_chars >= 110000:
            base_chunk += 40
        if profile.max_chars >= 220000:
            base_chunk += 80
        if profile.avg_visuals >= 18:
            base_chunk -= 80
        if profile.avg_sub_sections >= 18:
            base_chunk -= 60
        if profile.avg_bullets >= 40:
            base_chunk -= 20

        chunk_token_size = clamp(base_chunk, 1080, 1440)
        chunk_overlap = clamp(int(chunk_token_size * 0.16), 160, 230)
        entity_gleaning = 2 if (profile.avg_visuals >= 10 or profile.avg_sections >= 8) else 1
        max_parallel_insert = 2 if profile.max_chars >= 150000 else 3
        embed_batch_num = 8 if profile.max_chars >= 220000 else 10
        embed_max_async = 4 if profile.max_chars >= 220000 else 6
        llm_max_async = 2 if profile.max_chars >= 220000 else 3
        summary_context_size = 22000 if profile.max_chars >= 250000 else 18000
        summary_max_tokens = 1200 if profile.avg_chars >= 100000 else 1000
        cosine_threshold = 0.24 if profile.avg_visuals >= 14 else 0.26
        tuning_reason = (
            "操作系统教材章节篇幅长、概念链条深，且附带大量 Figure/Table 图示；"
            "采用更大的 chunk 保持机制解释完整，同时保留中等 overlap 与二次实体抽取，"
            "减少跨章节概念打散，并让 Visual Notes 更稳定地参与关系建图。"
        )
    else:
        chunk_token_size = 1240
        chunk_overlap = 192
        entity_gleaning = 2
        max_parallel_insert = 2
        embed_batch_num = 10
        embed_max_async = 6
        llm_max_async = 3
        summary_context_size = 18000
        summary_max_tokens = 1000
        cosine_threshold = 0.25
        tuning_reason = "已关闭自动调参，使用长篇教材默认参数。"

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
            "language": "Simplified Chinese and English",
            "entity_types": [
                "chapter",
                "section",
                "subsection",
                "concept",
                "mechanism",
                "algorithm",
                "policy",
                "state",
                "state_transition",
                "data_structure",
                "memory_component",
                "io_component",
                "file_system_component",
                "security_concept",
                "bilingual_term",
                "visual_note",
                "example",
                "os_case",
            ],
        },
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        tiktoken_model_name="gpt-4o-mini",
    )

    await rag.initialize_storages()
    return rag


async def ingest_documents(
    rag: LightRAG, documents: List[OSChapterDocument]
) -> list[OSChapterDocument]:
    print("\n=====================")
    print("Indexing by chapter")
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
        "请按章节说明进程、线程、调度与虚拟内存之间的关系，并指出相关状态图、结构图或表格分别来自哪些章节。",
    )
    query_param = QueryParam(
        mode=DEFAULT_QUERY_MODE,
        top_k=int(os.getenv("TOP_K", "14")),
        chunk_top_k=int(os.getenv("CHUNK_TOP_K", "10")),
        max_entity_tokens=int(os.getenv("MAX_ENTITY_TOKENS", "4500")),
        max_relation_tokens=int(os.getenv("MAX_RELATION_TOKENS", "5000")),
        user_prompt=(
            "回答时先把中文术语对齐到英文教材术语，再按章节组织；先列出涉及章节，再分别说明概念、机制流程和设计权衡，"
            "如果教材中存在相关 Figure/Table，指出其所属章节和用途，最后给出简短总结。"
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
        documents = parse_chapter_documents(CHAPTER_DIR, MANIFEST_PATH)
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
        print(f"Manifest: {MANIFEST_PATH.resolve()}")
        print(f"Working dir: {WORKING_DIR.resolve()}")
        print(f"Embedding model: {EMBED_MODEL} ({EMBED_DIM}d)")
        print(f"LLM model: {LLM_MODEL_NAME}")
        print(
            f"Corpus profile: chapters={profile.chapter_count}, avg_chars={profile.avg_chars}, "
            f"max_chars={profile.max_chars}, avg_lines={profile.avg_lines}"
        )
        print(
            f"Structure density: avg_sections={profile.avg_sections:.1f}, "
            f"avg_sub_sections={profile.avg_sub_sections:.1f}, "
            f"avg_objectives={profile.avg_objectives:.1f}, avg_visuals={profile.avg_visuals:.1f}"
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
        print(f"\nIndexed chapters: {len(documents)}")

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
