from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Operating.Systems.Internals.and.Design.Principles.7th.Edition.pdf"
PYC_PATH = PROJECT_ROOT / "utils" / "__pycache__" / "pdf_to_txt_for_lightrag.cpython-311.pyc"
FULLTEXT_DIR = PROJECT_ROOT / "data" / "rebuilt_fulltext"
CHAPTER_DIR = PROJECT_ROOT / "data" / "subject_chapters" / "operating_systems_pdf_rebuilt"
INDEX_DIR = PROJECT_ROOT / "data" / "subject_chapters" / "operating_systems_pdf_for_index"

CHAPTER_HEADER_RE = re.compile(
    r"CHAPTER\s+(?P<num>\d{1,2})\s*/\s*(?P<title>[A-Z][A-Z0-9 ,:&()'\-/]{3,120})"
)
INTRO_MARKERS = (
    "Guide 1",
    "0.1 Outline of this Book",
    "0.2 Example Systems",
)
FIGURE_CAPTION_RE = re.compile(
    r"\bFigure\s+(?P<label>\d+\.\d+[A-Za-z]?)\s+(?P<title>[A-Z][A-Za-z0-9 ,:&()'\-\/]{4,140})"
)
TABLE_CAPTION_RE = re.compile(
    r"\bTable\s+(?P<label>\d+\.\d+[A-Za-z]?)\s+(?P<title>[A-Z][A-Za-z0-9 ,:&()'\-\/]{4,140})"
)
NOISY_TITLE_RE = re.compile(
    r"(?:shows?|illustrates?|compares?|lists?|see\s+Figure|see\s+Table|discussed|introduced|presented)\b",
    re.I,
)


@dataclass
class ChapterSegment:
    chapter_num: int
    chapter_title: str
    start: int
    end: int
    text: str


@dataclass
class VisualNote:
    kind: str
    label: str
    title: str
    visual_type: str
    context: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild operating systems chapters directly from the original PDF and generate index-ready text."
    )
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--fulltext-dir", type=Path, default=FULLTEXT_DIR)
    parser.add_argument("--chapter-dir", type=Path, default=CHAPTER_DIR)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_title(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"\s+\d{1,4}$", "", text)
    text = re.sub(r"\s+[A-Z]$", "", text)
    text = text.strip(" ,:-/")
    return text.title()


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def ensure_fulltext(pdf_path: Path, fulltext_dir: Path) -> Path:
    fulltext_dir.mkdir(parents=True, exist_ok=True)
    fulltext_path = fulltext_dir / f"{pdf_path.stem}.txt"
    if fulltext_path.exists() and fulltext_path.stat().st_size > 100000:
        return fulltext_path

    script = f"""
import importlib.util, importlib.machinery, pathlib
pyc = pathlib.Path(r'{PYC_PATH}')
loader = importlib.machinery.SourcelessFileLoader('pdf_to_txt_for_lightrag_pyc', str(pyc))
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)
mod.convert_pdf_to_txt(
    pathlib.Path(r'{pdf_path}'),
    pathlib.Path(r'{fulltext_dir}'),
    max_pages=0,
    keep_page_markers=False,
    min_chars=500,
    ocr_engine='none',
    ocr_model='gpt-4o-mini',
    ocr_page_min_chars=80,
    ocr_max_pages=0,
    ocr_max_dim=1600,
    split_mode='none',
    split_min_chars=1200,
)
"""
    subprocess.run(
        ["python3.11", "-c", script],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    return fulltext_path


def clean_fulltext(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered.append("")
            continue
        if stripped.startswith("# Source PDF:"):
            continue
        if stripped.startswith("# Pages processed:"):
            continue
        filtered.append(line)
    cleaned = "\n".join(filtered)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_intro_start(text: str) -> int:
    positions = [text.find(marker) for marker in INTRO_MARKERS if text.find(marker) != -1]
    return min(positions) if positions else 0


def find_primary_chapter_headers(text: str) -> list[tuple[int, str, int]]:
    results: list[tuple[int, str, int]] = []
    seen: set[int] = set()
    for match in CHAPTER_HEADER_RE.finditer(text):
        num = int(match.group("num"))
        if not 1 <= num <= 16:
            continue
        if num in seen:
            continue
        seen.add(num)
        title = sanitize_title(match.group("title"))
        results.append((num, title, match.start()))
    return results


def build_segments(text: str) -> list[ChapterSegment]:
    text = clean_fulltext(text)
    headers = find_primary_chapter_headers(text)
    if not headers:
        raise RuntimeError("No chapter headers detected in full text.")

    segments: list[ChapterSegment] = []
    intro_start = detect_intro_start(text)
    first_start = headers[0][2]
    intro_text = text[intro_start:first_start].strip()
    if intro_text:
        segments.append(
            ChapterSegment(
                chapter_num=0,
                chapter_title="Guide And Intro",
                start=intro_start,
                end=first_start,
                text=intro_text,
            )
        )

    for index, (num, title, start) in enumerate(headers):
        end = headers[index + 1][2] if index + 1 < len(headers) else len(text)
        segment_text = text[start:end].strip()
        segments.append(
            ChapterSegment(
                chapter_num=num,
                chapter_title=title,
                start=start,
                end=end,
                text=segment_text,
            )
        )

    return segments


def looks_like_valid_caption(title: str) -> bool:
    title = normalize_space(title)
    if len(title.split()) < 2:
        return False
    if title.endswith(")"):
        return False
    if NOISY_TITLE_RE.match(title):
        return False
    return True


def infer_visual_type(title: str) -> str:
    lowered = title.lower()
    if any(token in lowered for token in ("state", "transition", "lifecycle")):
        return "state_diagram"
    if any(token in lowered for token in ("queue", "scheduling", "dispatch", "timeline", "timing")):
        return "flow_or_timing_diagram"
    if any(token in lowered for token in ("architecture", "structure", "model", "organization", "topology")):
        return "architecture_diagram"
    if any(token in lowered for token in ("tree", "table", "matrix", "b-tree")):
        return "data_structure_diagram"
    if any(token in lowered for token in ("comparison", "performance", "utilization", "results", "chart")):
        return "comparison_chart"
    if any(token in lowered for token in ("allocation", "translation", "mapping", "paging", "memory")):
        return "mapping_diagram"
    return "general_diagram"


def extract_context(text: str, start: int, end: int, window: int = 260) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = normalize_space(text[left:right])
    if len(snippet) > 320:
        snippet = snippet[:317].rstrip() + "..."
    return snippet


def extract_visual_notes(text: str) -> list[VisualNote]:
    notes: list[VisualNote] = []
    seen: set[tuple[str, str]] = set()
    for kind, pattern in (("figure", FIGURE_CAPTION_RE), ("table", TABLE_CAPTION_RE)):
        for match in pattern.finditer(text):
            label = match.group("label")
            title = normalize_space(match.group("title"))
            if not looks_like_valid_caption(title):
                continue
            key = (kind, label)
            if key in seen:
                continue
            seen.add(key)
            notes.append(
                VisualNote(
                    kind=kind,
                    label=label,
                    title=title,
                    visual_type=infer_visual_type(title),
                    context=extract_context(text, match.start(), match.end()),
                )
            )
    notes.sort(key=lambda item: (item.kind, item.label))
    return notes


def visual_notes_section(notes: list[VisualNote]) -> str:
    if not notes:
        return "[Visual Notes]\n无\n"
    lines = ["[Visual Notes]"]
    for note in notes:
        lines.append(f"- [{note.kind.upper()} {note.label}] {note.title}")
        lines.append(f"  [Visual Type] {note.visual_type}")
        lines.append(f"  [Context] {note.context}")
    return "\n".join(lines) + "\n"


def write_outputs(segments: list[ChapterSegment], chapter_dir: Path, index_dir: Path) -> None:
    chapter_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("Operating_Systems_Chapter_*.txt",):
        for path in chapter_dir.glob(pattern):
            path.unlink(missing_ok=True)
    for pattern in ("Operating_Systems_Chapter_*.txt", "Operating_Systems_Chapter_*__visual_notes.txt", "index_manifest.json"):
        for path in index_dir.glob(pattern):
            path.unlink(missing_ok=True)

    manifest: dict[str, object] = {"chapters": []}

    for segment in segments:
        stem = f"Operating_Systems_Chapter_{segment.chapter_num:02d}_{slugify(segment.chapter_title)}"
        chapter_path = chapter_dir / f"{stem}.txt"
        chapter_header = [
            "[Course] Operating Systems: Internals and Design Principles",
            f"[Chapter Number] {segment.chapter_num}",
            f"[Chapter Title] {segment.chapter_title}",
            "[Source] Rebuilt directly from original PDF full text",
            "",
        ]
        chapter_path.write_text("\n".join(chapter_header) + segment.text + "\n", encoding="utf-8")

        notes = extract_visual_notes(segment.text)
        notes_text = visual_notes_section(notes)
        notes_path = index_dir / f"{stem}__visual_notes.txt"
        notes_path.write_text(notes_text, encoding="utf-8")

        enriched_path = index_dir / f"{stem}.txt"
        enriched_text = (
            "\n".join(chapter_header)
            + segment.text
            + "\n\n[Indexing Guidance] 回答操作系统问题时，优先利用正文与 Visual Notes 一起组织答案；"
              "涉及状态转换、结构图、调度流程、地址转换、文件系统结构与并发机制时，优先参考图示卡片。\n\n"
            + notes_text
        )
        enriched_path.write_text(enriched_text, encoding="utf-8")

        manifest["chapters"].append(
            {
                "chapter_num": segment.chapter_num,
                "chapter_title": segment.chapter_title,
                "chapter_file": chapter_path.name,
                "index_file": enriched_path.name,
                "visual_notes_file": notes_path.name,
                "visual_note_count": len(notes),
            }
        )

    (index_dir / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    fulltext_path = ensure_fulltext(args.pdf, args.fulltext_dir)
    text = fulltext_path.read_text(encoding="utf-8")
    segments = build_segments(text)
    write_outputs(segments, args.chapter_dir, args.index_dir)
    print(f"Full text: {fulltext_path.resolve()}")
    print(f"Chapter dir: {args.chapter_dir.resolve()}")
    print(f"Index dir: {args.index_dir.resolve()}")
    print(f"Rebuilt chapters: {[segment.chapter_num for segment in segments]}")


if __name__ == "__main__":
    main()
