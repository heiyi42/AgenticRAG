from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp_core.problem_tutoring_service import ProblemTutoringService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build question bank embedding index.")
    parser.add_argument(
        "--input",
        default=str(ProblemTutoringService.DEFAULT_QUESTION_BANK_PATH),
        help="Path to the question bank JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=str(ProblemTutoringService.DEFAULT_QUESTION_BANK_EMBEDDING_INDEX_PATH),
        help="Path to the output embedding index JSON file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional embedding model override.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    service = ProblemTutoringService(
        question_bank_path=args.input,
        question_bank_embedding_index_path=args.output,
        question_bank_embed_enabled=True,
    )
    if args.model:
        service.question_bank_embed_model = str(args.model).strip() or service.question_bank_embed_model
    result = await service.build_question_bank_embedding_index()
    print(
        "Built question bank embedding index:",
        f"path={result['path']}",
        f"items={result['item_count']}",
        f"model={result['model']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
