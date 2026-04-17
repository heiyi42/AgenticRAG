from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader, PdfWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Operating.Systems.Internals.and.Design.Principles.7th.Edition.pdf"
INDEX_DIR = PROJECT_ROOT / "data" / "subject_chapters" / "operating_systems_pdf_for_index"
OUTPUT_DIR = PROJECT_ROOT / "data" / "subject_chapters" / "operating_systems_pdf_visuals"
MANIFEST_PATH = INDEX_DIR / "index_manifest.json"
RENDER_MAX_HEIGHT = 1800
FIGURE_SCAN_WINDOW_PT = 340
TABLE_SCAN_WINDOW_PT = 360

VISUAL_LINE_RE = re.compile(r"^- \[(?P<kind>FIGURE|TABLE)\s+(?P<label>[^\]]+)\]\s*(?P<title>.+?)\s*$")


@dataclass
class VisualTarget:
    chapter_num: int
    chapter_title: str
    kind: str
    label: str
    title: str
    notes_file: Path

    @property
    def slug(self) -> str:
        safe_label = re.sub(r"[^A-Za-z0-9]+", "_", self.label).strip("_").lower()
        safe_title = re.sub(r"[^A-Za-z0-9]+", "_", self.title).strip("_").lower()
        safe_title = safe_title[:80].rstrip("_")
        return f"{self.kind.lower()}_{safe_label}_{safe_title}".strip("_")


@dataclass
class TextFragment:
    text: str
    x: float
    y: float


@dataclass
class TextLine:
    text: str
    y: float
    x_min: float
    x_max: float


@dataclass
class CaptionMatch:
    line: TextLine
    line_index: int
    page_index: int
    score: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract figure/table images from the operating systems PDF and build an image index."
    )
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chapter", type=int, action="append", help="Only extract selected chapter numbers.")
    parser.add_argument("--label", action="append", help="Only extract selected figure/table labels, such as 5.15.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N visuals.")
    parser.add_argument("--force", action="store_true", help="Re-render and re-crop existing outputs.")
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    text = normalize_space(text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return text.casefold()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_visual_targets(
    index_dir: Path,
    manifest_path: Path,
    chapters: set[int] | None,
    labels: set[str] | None,
) -> list[VisualTarget]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets: list[VisualTarget] = []
    for chapter in sorted(manifest["chapters"], key=lambda item: item["chapter_num"]):
        chapter_num = int(chapter["chapter_num"])
        if chapters and chapter_num not in chapters:
            continue

        notes_path = index_dir / chapter["visual_notes_file"]
        if not notes_path.exists():
            continue

        for raw_line in notes_path.read_text(encoding="utf-8").splitlines():
            match = VISUAL_LINE_RE.match(raw_line.strip())
            if not match:
                continue
            if labels and match.group("label") not in labels:
                continue
            targets.append(
                VisualTarget(
                    chapter_num=chapter_num,
                    chapter_title=chapter["chapter_title"],
                    kind=match.group("kind"),
                    label=match.group("label"),
                    title=normalize_space(match.group("title")),
                    notes_file=notes_path,
                )
            )
    return targets


def load_existing_rows(output_dir: Path) -> list[dict[str, object]]:
    manifest_path = output_dir / "figure_index.json"
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(data.get("figures", []))


def extract_page_texts(reader: PdfReader) -> list[str]:
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return page_texts


def group_fragments_to_lines(fragments: list[TextFragment]) -> list[TextLine]:
    sorted_fragments = sorted(fragments, key=lambda item: (-item.y, item.x))
    buckets: list[list[TextFragment]] = []

    for fragment in sorted_fragments:
        if not buckets or abs(buckets[-1][0].y - fragment.y) > 2.5:
            buckets.append([fragment])
        else:
            buckets[-1].append(fragment)

    lines: list[TextLine] = []
    for bucket in buckets:
        bucket.sort(key=lambda item: item.x)
        text = "".join(fragment.text for fragment in bucket)
        text = normalize_space(text)
        if not text:
            continue
        x_min = min(fragment.x for fragment in bucket)
        x_max = max(fragment.x + max(len(fragment.text), 1) * 4.0 for fragment in bucket)
        lines.append(TextLine(text=text, y=bucket[0].y, x_min=x_min, x_max=x_max))
    return lines


def extract_page_lines(page) -> list[TextLine]:
    fragments: list[TextFragment] = []

    def visitor(text: str, _cm, tm, _font_dict, _font_size) -> None:
        if text and text.strip():
            fragments.append(TextFragment(text=text, x=float(tm[4]), y=float(tm[5])))

    page.extract_text(visitor_text=visitor)
    return group_fragments_to_lines(fragments)


def find_candidate_pages(page_texts: list[str], target: VisualTarget) -> list[int]:
    primary = normalize_for_match(f"{target.kind.title()} {target.label}")
    secondary = normalize_for_match(target.title[:50])
    candidates: list[int] = []

    for index, page_text in enumerate(page_texts):
        normalized = normalize_for_match(page_text)
        if primary in normalized:
            candidates.append(index)
        elif secondary and secondary in normalized:
            candidates.append(index)
    return candidates


def find_caption_line(
    page_index: int,
    page_height: float,
    lines: list[TextLine],
    target: VisualTarget,
) -> CaptionMatch | None:
    label_text = normalize_for_match(f"{target.kind.title()} {target.label}")
    title_text = normalize_for_match(target.title[:36])
    prefix_pattern = re.compile(
        rf"^{re.escape(label_text)}(?![a-z0-9])"
    )
    exact_pattern = re.compile(
        rf"{re.escape(label_text)}(?![a-z0-9])"
    )

    title_tokens = [token for token in re.findall(r"[a-z0-9]+", title_text) if len(token) >= 4]
    best: CaptionMatch | None = None
    for line_index, line in enumerate(lines):
        normalized = normalize_for_match(line.text)
        neighbors = lines[max(0, line_index - 2) : min(len(lines), line_index + 3)]
        combined = normalize_for_match(" ".join(item.text for item in neighbors))
        overlap = sum(1 for token in title_tokens[:5] if token in combined)

        score = 0
        if prefix_pattern.search(normalized):
            score += 8
        if exact_pattern.search(normalized):
            score += 4
        if title_text and title_text in normalized:
            score += 3
        score += min(overlap, 3)

        word_count = len(line.text.split())
        if len(line.text) <= 70:
            score += 1
        if len(line.text) >= 120 or word_count >= 18:
            score -= 2
        if "chapter" in normalized and not prefix_pattern.search(normalized):
            score -= 3

        vertical_ratio = line.y / page_height if page_height else 0.0
        if target.kind == "FIGURE":
            score += 2 if vertical_ratio < 0.45 else -1
        else:
            score += 2 if vertical_ratio > 0.45 else -1

        if score >= 6 and (best is None or score > best.score):
            best = CaptionMatch(line=line, line_index=line_index, page_index=page_index, score=score)

    return best


def clamp_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(int(round(value)), upper))


def render_page_image(
    reader: PdfReader,
    page_index: int,
    pages_dir: Path,
    force: bool,
) -> Path:
    pages_dir.mkdir(parents=True, exist_ok=True)
    out_path = pages_dir / f"page_{page_index + 1:04d}.png"
    if out_path.exists() and not force:
        return out_path

    with tempfile.TemporaryDirectory(prefix="os_pdf_page_") as tmp_dir:
        single_page_pdf = Path(tmp_dir) / f"page_{page_index + 1:04d}.pdf"
        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])
        with single_page_pdf.open("wb") as handle:
            writer.write(handle)

        subprocess.run(
            [
                "sips",
                "-s",
                "format",
                "png",
                "-Z",
                str(RENDER_MAX_HEIGHT),
                str(single_page_pdf),
                "--out",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    return out_path


def query_image_size(image_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(image_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    width_match = re.search(r"pixelWidth:\s*(\d+)", result.stdout)
    height_match = re.search(r"pixelHeight:\s*(\d+)", result.stdout)
    if not width_match or not height_match:
        raise RuntimeError(f"Unable to read image size: {image_path}")
    return int(width_match.group(1)), int(height_match.group(1))


def filter_relevant_lines(lines: list[TextLine], caption_y: float, kind: str, page_height: float) -> list[TextLine]:
    if kind == "FIGURE":
        window_top = min(page_height - 24.0, caption_y + FIGURE_SCAN_WINDOW_PT)
        candidates = [line for line in lines if caption_y + 6.0 <= line.y <= window_top]
        compact = [line for line in candidates if len(line.text) <= 80 and len(line.text.split()) <= 14]
        return compact or candidates

    window_bottom = max(24.0, caption_y - TABLE_SCAN_WINDOW_PT)
    return [line for line in lines if window_bottom <= line.y <= caption_y - 4.0]


def compute_crop_box(
    target: VisualTarget,
    caption: CaptionMatch,
    lines: list[TextLine],
    image_width: int,
    image_height: int,
    page_width: float,
    page_height: float,
) -> dict[str, int | str | float]:
    scale_x = image_width / float(page_width)
    scale_y = image_height / float(page_height)
    caption_y = caption.line.y

    relevant = filter_relevant_lines(lines, caption_y, target.kind, float(page_height))

    margin_x = int(image_width * 0.06)
    left_px = margin_x
    width_px = image_width - 2 * margin_x

    caption_top_px = image_height - int(round((caption_y + 12.0) * scale_y))
    caption_bottom_px = image_height - int(round((caption_y - 10.0) * scale_y))

    confidence = "medium"
    method = "window_fallback"

    if target.kind == "FIGURE":
        if relevant:
            highest_y = max(line.y for line in relevant)
            lowest_y = min(line.y for line in relevant)
            top_px = image_height - int(round((highest_y + 26.0) * scale_y))
            bottom_px = image_height - int(round((min(lowest_y, caption_y) - 18.0) * scale_y))
            method = "caption_above_text_cluster"
            confidence = "high" if len(relevant) >= 3 else "medium"
        else:
            top_px = max(0, caption_top_px - int(image_height * 0.46))
            bottom_px = min(image_height, caption_bottom_px + 28)
    else:
        if relevant:
            lowest_y = min(line.y for line in relevant)
            top_px = max(0, caption_top_px - 20)
            bottom_px = image_height - int(round((lowest_y - 18.0) * scale_y))
            method = "caption_below_text_cluster"
            confidence = "high" if len(relevant) >= 4 else "medium"
        else:
            top_px = max(0, caption_top_px - 20)
            bottom_px = min(image_height, top_px + int(image_height * 0.42))

    top_px = clamp_int(top_px, 0, image_height - 60)
    bottom_px = clamp_int(bottom_px, top_px + 60, image_height)
    height_px = bottom_px - top_px

    return {
        "left": left_px,
        "top": top_px,
        "width": width_px,
        "height": height_px,
        "method": method,
        "confidence": confidence,
    }


def crop_image(source_path: Path, crop_path: Path, crop_box: dict[str, int | str | float], force: bool) -> None:
    ensure_parent(crop_path)
    if crop_path.exists() and not force:
        return

    subprocess.run(
        [
            "sips",
            "-c",
            str(crop_box["height"]),
            str(crop_box["width"]),
            "--cropOffset",
            str(crop_box["top"]),
            str(crop_box["left"]),
            str(source_path),
            "--out",
            str(crop_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def write_manifest(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    manifest_path = output_dir / "figure_index.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(
        json.dumps({"figures": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def write_report(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    report_path = output_dir / "figure_index.md"
    lines = [
        "# Operating Systems Figure Index",
        "",
        f"Total extracted visuals: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['kind']} {row['label']} | Chapter {int(row['chapter_num']):02d}",
                f"- Title: {row['title']}",
                f"- Page: {row['page_number']}",
                f"- Crop: {row['crop_image']}",
                f"- Confidence: {row['confidence']}",
                "",
            ]
        )
    ensure_parent(report_path)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    chapter_filter = set(args.chapter) if args.chapter else None
    label_filter = set(args.label) if args.label else None
    partial_run = bool(chapter_filter or label_filter or args.limit > 0)

    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    logging.getLogger("pypdf._reader").setLevel(logging.CRITICAL)

    targets = load_visual_targets(args.index_dir, args.manifest, chapter_filter, label_filter)
    if args.limit > 0:
        targets = targets[: args.limit]

    if not targets:
        raise SystemExit("No visuals found to extract.")

    reader = PdfReader(str(args.pdf))
    page_texts = extract_page_texts(reader)

    pages_dir = args.output_dir / "pages"
    crops_dir = args.output_dir / "crops"
    rows: list[dict[str, object]] = load_existing_rows(args.output_dir) if partial_run else []
    row_map = {(str(row["kind"]), str(row["label"])): row for row in rows}

    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {target.kind} {target.label} | {target.title}")
        candidate_pages = find_candidate_pages(page_texts, target)
        if not candidate_pages:
            print("  -> skipped: no candidate page")
            continue

        caption_match: CaptionMatch | None = None
        page_lines: list[TextLine] = []
        best_score = -10**9
        for page_index in candidate_pages:
            page = reader.pages[page_index]
            lines = extract_page_lines(page)
            match = find_caption_line(page_index, float(page.mediabox.height), lines, target)
            if match and match.score > best_score:
                best_score = match.score
                caption_match = match
                page_lines = lines

        if caption_match is None:
            print("  -> skipped: caption line not found")
            continue

        page = reader.pages[caption_match.page_index]
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)

        page_image = render_page_image(reader, caption_match.page_index, pages_dir, args.force)
        image_width, image_height = query_image_size(page_image)

        crop_box = compute_crop_box(
            target=target,
            caption=caption_match,
            lines=page_lines,
            image_width=image_width,
            image_height=image_height,
            page_width=page_width,
            page_height=page_height,
        )

        chapter_dir = crops_dir / f"chapter_{target.chapter_num:02d}"
        crop_path = chapter_dir / f"{target.slug}.png"
        crop_image(page_image, crop_path, crop_box, args.force)

        row_map[(target.kind, target.label)] = {
            "chapter_num": target.chapter_num,
            "chapter_title": target.chapter_title,
            "kind": target.kind,
            "label": target.label,
            "title": target.title,
            "page_index": caption_match.page_index,
            "page_number": caption_match.page_index + 1,
            "page_image": str(page_image.relative_to(args.output_dir)),
            "crop_image": str(crop_path.relative_to(args.output_dir)),
            "crop_box": {
                "left": crop_box["left"],
                "top": crop_box["top"],
                "width": crop_box["width"],
                "height": crop_box["height"],
            },
            "method": crop_box["method"],
            "confidence": crop_box["confidence"],
            "notes_file": str(target.notes_file.relative_to(args.index_dir)),
        }

    rows = sorted(
        row_map.values(),
        key=lambda row: (
            int(row["chapter_num"]),
            0 if str(row["kind"]) == "FIGURE" else 1,
            str(row["label"]),
        ),
    )
    manifest_path = write_manifest(args.output_dir, rows)
    report_path = write_report(args.output_dir, rows)
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Wrote report: {report_path}")
    print(f"Extracted visuals: {len(rows)}")


if __name__ == "__main__":
    main()
