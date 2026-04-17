from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from clean_operating_systems_fragments import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CLEANED_DIR,
    DEFAULT_SOURCE_DIR,
    build_chapter_buckets,
    collect_fragments,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "subject_chapters" / "operating_systems_for_index"

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
class VisualNote:
    kind: str
    label: str
    title: str
    visual_type: str
    context: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an index-ready operating systems corpus with text-cleaned chapters and visual notes."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing fragmented operating systems text exports.",
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=DEFAULT_CLEANED_DIR,
        help="Directory for cleaned intermediate chapter text.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory for final index-ready chapter text.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
    if any(token in lowered for token in ("comparison", "performance", "utilization", "results", "histogram")):
        return "comparison_chart"
    if any(token in lowered for token in ("allocation", "translation", "mapping", "paging", "memory")):
        return "mapping_diagram"
    return "general_diagram"


def extract_context(text: str, start: int, end: int, window: int = 260) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    snippet = normalize_space(snippet)
    if len(snippet) > 320:
        snippet = snippet[:317].rstrip() + "..."
    return snippet


def extract_visual_notes(chapter_text: str) -> list[VisualNote]:
    notes: list[VisualNote] = []
    seen: set[tuple[str, str]] = set()

    for kind, pattern in (("figure", FIGURE_CAPTION_RE), ("table", TABLE_CAPTION_RE)):
        for match in pattern.finditer(chapter_text):
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
                    context=extract_context(chapter_text, match.start(), match.end()),
                )
            )

    notes.sort(key=lambda item: (item.kind, item.label))
    return notes


def load_cleaned_chapters(cleaned_dir: Path) -> list[Path]:
    return sorted(cleaned_dir.glob("Operating_Systems_Chapter_*.txt"))


def chapter_visuals_section(notes: list[VisualNote]) -> str:
    if not notes:
        return "[Visual Notes]\n无\n"

    lines = ["[Visual Notes]"]
    for note in notes:
        lines.append(f"- [{note.kind.upper()} {note.label}] {note.title}")
        lines.append(f"  [Visual Type] {note.visual_type}")
        lines.append(f"  [Context] {note.context}")
    return "\n".join(lines) + "\n"


def build_index_ready_text(chapter_text: str, notes: list[VisualNote]) -> str:
    chapter_text = chapter_text.rstrip() + "\n\n"
    chapter_text += (
        "[Indexing Guidance] 回答操作系统问题时，优先利用章节正文与图示卡片一起组织答案；"
        "如果问题涉及状态转换、体系结构、调度流程、地址转换、文件系统结构或并发关系，"
        "应优先参考 Visual Notes 中的结构化描述。\n\n"
    )
    chapter_text += chapter_visuals_section(notes)
    return chapter_text


def write_index_ready_outputs(cleaned_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "cleaned_dir": str(cleaned_dir),
        "output_dir": str(output_dir),
        "chapters": [],
    }

    for chapter_path in load_cleaned_chapters(cleaned_dir):
        chapter_text = chapter_path.read_text(encoding="utf-8")
        notes = extract_visual_notes(chapter_text)
        enriched_text = build_index_ready_text(chapter_text, notes)
        output_path = output_dir / chapter_path.name
        output_path.write_text(enriched_text, encoding="utf-8")

        note_path = output_dir / f"{chapter_path.stem}__visual_notes.txt"
        note_path.write_text(chapter_visuals_section(notes), encoding="utf-8")

        manifest["chapters"].append(
            {
                "chapter_file": chapter_path.name,
                "index_file": output_path.name,
                "visual_notes_file": note_path.name,
                "visual_note_count": len(notes),
            }
        )

    (output_dir / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_cleaned_chapters(source_dir: Path, cleaned_dir: Path) -> None:
    fragments = collect_fragments(source_dir)
    buckets = build_chapter_buckets(fragments)
    write_outputs(source_dir, cleaned_dir, buckets, fragments)


def main() -> None:
    args = parse_args()
    ensure_cleaned_chapters(args.source_dir, args.cleaned_dir)
    write_index_ready_outputs(args.cleaned_dir, args.output_dir)

    print(f"Cleaned chapters: {args.cleaned_dir.resolve()}")
    print(f"Index-ready chapters: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
