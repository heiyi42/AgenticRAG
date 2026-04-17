from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp_core.problem_tutoring_service import ProblemTutoringService  # noqa: E402


DEFAULT_QUESTION_BANK_PATH = Path("data/tutoring_question_bank/questions.jsonl")


def load_question_bank(path: str | Path) -> list[dict[str, Any]]:
    bank_path = Path(path)
    items: list[dict[str, Any]] = []
    for line_no, line in enumerate(bank_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"题库 JSONL 第 {line_no} 行格式错误: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"题库 JSONL 第 {line_no} 行不是 JSON object")
        items.append(item)
    return items


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def avg(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return round(float(total) / float(count), 2)


def normalize_eval_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "")).lower()


def extract_eval_terms(text: str) -> set[str]:
    raw = str(text or "").lower()
    terms = set(re.findall(r"[a-z_][a-z0-9_]*|\d+", raw))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", raw))
    if cjk:
        terms.update(cjk[i : i + 2] for i in range(max(0, len(cjk) - 1)))
    return {term for term in terms if term}


def load_candidate_answers(path: str | Path) -> dict[str, dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"候选答案文件不存在: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    def normalize_item(item: Any) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(item, dict):
            return None
        item_id = str(item.get("id", "") or "").strip()
        answer = str(
            item.get("answer", item.get("final_answer", item.get("text", ""))) or ""
        ).strip()
        if not item_id or not answer:
            return None
        return (
            item_id,
            {
                "answer": answer,
                "meta": {
                    key: value
                    for key, value in item.items()
                    if key not in {"id", "answer", "final_answer", "text"}
                },
            },
        )

    candidates: dict[str, dict[str, Any]] = {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                candidates[str(key)] = {"answer": value, "meta": {}}
                continue
            normalized = normalize_item({"id": key, **value} if isinstance(value, dict) else None)
            if normalized is not None:
                candidates[normalized[0]] = normalized[1]
        return candidates

    if isinstance(payload, list):
        for item in payload:
            normalized = normalize_item(item)
            if normalized is not None:
                candidates[normalized[0]] = normalized[1]
        return candidates

    for line_no, line in enumerate(text.splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        try:
            item = json.loads(row)
        except json.JSONDecodeError as exc:
            raise ValueError(f"候选答案 JSONL 第 {line_no} 行格式错误: {exc}") from exc
        normalized = normalize_item(item)
        if normalized is not None:
            candidates[normalized[0]] = normalized[1]
    return candidates


def step_coverage_pct(reference_steps: list[str], candidate_answer: str) -> float:
    steps = [str(step or "").strip() for step in list(reference_steps or []) if str(step or "").strip()]
    if not steps:
        return 100.0
    candidate_norm = normalize_eval_text(candidate_answer)
    candidate_terms = extract_eval_terms(candidate_answer)
    covered = 0
    for step in steps:
        step_norm = normalize_eval_text(step)
        if step_norm and len(step_norm) >= 4 and step_norm in candidate_norm:
            covered += 1
            continue
        step_terms = extract_eval_terms(step)
        if not step_terms:
            continue
        overlap = len(step_terms & candidate_terms) / max(1, len(step_terms))
        if overlap >= 0.45:
            covered += 1
    return pct(covered, len(steps))


def grade_candidate_answer(item: dict[str, Any], candidate_answer: str) -> dict[str, Any]:
    expected_answer = str(item.get("answer", "") or "").strip()
    candidate_text = str(candidate_answer or "").strip()
    if not candidate_text:
        return {
            "answer_recall_pct": 0.0,
            "step_coverage_pct": 0.0,
            "numeric_consistency_pct": 0.0,
            "mistake_coverage_pct": 0.0,
            "score_pct": 0.0,
            "passed": False,
            "exact_answer_hit": False,
        }

    expected_norm = normalize_eval_text(expected_answer)
    candidate_norm = normalize_eval_text(candidate_text)
    expected_terms = extract_eval_terms(expected_answer)
    candidate_terms = extract_eval_terms(candidate_text)
    answer_recall = (
        100.0
        if not expected_terms
        else pct(len(expected_terms & candidate_terms), len(expected_terms))
    )
    exact_hit = bool(expected_norm and expected_norm in candidate_norm)

    expected_numbers = {int(value) for value in re.findall(r"-?\d+", expected_answer)}
    candidate_numbers = {int(value) for value in re.findall(r"-?\d+", candidate_text)}
    numeric_consistency = (
        100.0
        if not expected_numbers
        else pct(len(expected_numbers & candidate_numbers), len(expected_numbers))
    )
    steps_pct = step_coverage_pct(list(item.get("solution_steps", []) or []), candidate_text)
    mistakes_pct = step_coverage_pct(list(item.get("common_mistakes", []) or []), candidate_text)
    score = round(
        answer_recall * 0.45
        + steps_pct * 0.30
        + numeric_consistency * 0.15
        + mistakes_pct * 0.10,
        2,
    )
    passed = bool(score >= 60.0 and answer_recall >= 45.0 and numeric_consistency >= 50.0)
    return {
        "answer_recall_pct": round(answer_recall, 2),
        "step_coverage_pct": round(steps_pct, 2),
        "numeric_consistency_pct": round(numeric_consistency, 2),
        "mistake_coverage_pct": round(mistakes_pct, 2),
        "score_pct": score,
        "passed": passed,
        "exact_answer_hit": exact_hit,
    }


def build_solver_baseline_answer(
    service: ProblemTutoringService,
    *,
    solver_result: dict[str, Any],
    learning_outline: dict[str, Any] | None = None,
) -> str | None:
    solver = dict(solver_result or {})
    if str(solver.get("status", "") or "").strip().lower() != "success":
        return None

    lines: list[str] = []
    summary = service._build_solver_summary(solver)
    if summary:
        lines.append(summary)
    result_payload = solver.get("result")
    if isinstance(result_payload, dict) and result_payload:
        lines.append("结构化结果：")
        lines.append(json.dumps(result_payload, ensure_ascii=False))
    steps = [str(step).strip() for step in list(solver.get("steps", []) or []) if str(step).strip()]
    if steps:
        lines.append("关键步骤：")
        lines.extend(f"- {step}" for step in steps[:8])
    outline = dict(learning_outline or {})
    mistakes = [str(item).strip() for item in list(outline.get("common_mistakes", []) or []) if str(item).strip()]
    if mistakes:
        lines.append("易错点：")
        lines.extend(f"- {item}" for item in mistakes[:4])
    text = "\n".join(lines).strip()
    return text or None


def looks_rule_solvable(item: dict[str, Any]) -> bool:
    problem_type = str(item.get("problem_type", "") or "")
    subject_id = str(item.get("subject_id", "") or "")
    question = str(item.get("question", "") or "").lower()
    if problem_type in {"os_page_replacement", "os_cpu_scheduling", "os_banker", "os_pv_sync"}:
        return True
    if subject_id == "cybersec_lab" and (
        "dh" in question or "diffie" in question or "密钥交换" in question
    ):
        return True
    return False


def evaluate_question_bank(
    path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    *,
    detail_limit: int = 12,
    recommendation_limit: int = 3,
    candidate_answers_path: str | Path | None = None,
) -> dict[str, Any]:
    bank_path = Path(path)
    service = ProblemTutoringService(question_bank_path=bank_path)
    items = load_question_bank(bank_path)
    candidate_answers = (
        load_candidate_answers(candidate_answers_path) if candidate_answers_path else {}
    )

    totals = {
        "total": len(items),
        "subject_correct": 0,
        "problem_type_correct": 0,
        "recommendation_nonempty": 0,
        "recommendation_top1_same_subject": 0,
        "recommendation_top1_same_type": 0,
        "solver_success": 0,
        "expected_rule_solvable": 0,
        "expected_rule_solvable_success": 0,
        "llm_fallback_needed": 0,
        "answer_eval_available": 0,
        "answer_eval_passed": 0,
        "answer_eval_exact_hit": 0,
    }
    score_totals = {
        "answer_eval_score": 0.0,
        "answer_eval_recall": 0.0,
        "answer_eval_steps": 0.0,
        "answer_eval_numeric": 0.0,
        "answer_eval_mistakes": 0.0,
    }
    by_subject: dict[str, Counter[str]] = defaultdict(Counter)
    by_problem_type: dict[str, Counter[str]] = defaultdict(Counter)
    solver_counter: Counter[str] = Counter()
    misclassified: list[dict[str, Any]] = []
    weak_recommendations: list[dict[str, Any]] = []
    solver_fallbacks: list[dict[str, Any]] = []
    weak_answer_scores: list[dict[str, Any]] = []

    for item in items:
        qid = str(item.get("id", "") or "")
        question = str(item.get("question", "") or "")
        expected_subject = str(item.get("subject_id", "") or "")
        expected_type = str(item.get("problem_type", "") or "")

        analysis = service.rule_analyze(question)
        template = service.select_template(analysis, user_question=question)
        recommendations = service.recommend_similar_questions(
            analysis=analysis,
            template=template,
            user_question=question,
            limit=recommendation_limit,
        )
        solver_result = service.solve_deterministic(
            analysis=analysis,
            template=template,
            user_question=question,
        )
        few_shot_examples = service.build_few_shot_examples(recommendations, limit=2)
        learning_outline = service.build_learning_outline(
            analysis=analysis,
            template=template,
            recommendations=recommendations,
            few_shot_examples=few_shot_examples,
            solver_result=solver_result,
            retrieval={"status": "skipped", "message": "offline evaluation", "answer": ""},
        )

        actual_subject = str(analysis.subject_id or "")
        actual_type = str(analysis.problem_type or "")
        subject_correct = actual_subject == expected_subject
        type_correct = actual_type == expected_type
        solver_status = str(solver_result.get("status", "") or "")
        solver_name = str(solver_result.get("solver", "") or "")
        solver_success = solver_status == "success"
        expected_solvable = looks_rule_solvable(item)

        totals["subject_correct"] += int(subject_correct)
        totals["problem_type_correct"] += int(type_correct)
        totals["solver_success"] += int(solver_success)
        totals["expected_rule_solvable"] += int(expected_solvable)
        totals["expected_rule_solvable_success"] += int(expected_solvable and solver_success)
        totals["llm_fallback_needed"] += int(not solver_success)

        by_subject[expected_subject]["total"] += 1
        by_subject[expected_subject]["subject_correct"] += int(subject_correct)
        by_subject[expected_subject]["problem_type_correct"] += int(type_correct)
        by_subject[expected_subject]["solver_success"] += int(solver_success)

        by_problem_type[expected_type]["total"] += 1
        by_problem_type[expected_type]["subject_correct"] += int(subject_correct)
        by_problem_type[expected_type]["problem_type_correct"] += int(type_correct)
        by_problem_type[expected_type]["solver_success"] += int(solver_success)

        candidate_entry = candidate_answers.get(qid)
        candidate_answer = ""
        candidate_source = ""
        if candidate_entry is not None:
            candidate_answer = str(candidate_entry.get("answer", "") or "").strip()
            candidate_source = "candidate_file"
        else:
            baseline_answer = build_solver_baseline_answer(
                service,
                solver_result=solver_result,
                learning_outline=learning_outline,
            )
            if baseline_answer:
                candidate_answer = baseline_answer
                candidate_source = "solver_baseline"

        if candidate_answer:
            grade = grade_candidate_answer(item, candidate_answer)
            totals["answer_eval_available"] += 1
            totals["answer_eval_passed"] += int(grade["passed"])
            totals["answer_eval_exact_hit"] += int(grade["exact_answer_hit"])
            score_totals["answer_eval_score"] += float(grade["score_pct"])
            score_totals["answer_eval_recall"] += float(grade["answer_recall_pct"])
            score_totals["answer_eval_steps"] += float(grade["step_coverage_pct"])
            score_totals["answer_eval_numeric"] += float(grade["numeric_consistency_pct"])
            score_totals["answer_eval_mistakes"] += float(grade["mistake_coverage_pct"])
            by_subject[expected_subject]["answer_eval_available"] += 1
            by_subject[expected_subject]["answer_eval_passed"] += int(grade["passed"])
            by_subject[expected_subject]["answer_eval_score"] += float(grade["score_pct"])
            by_problem_type[expected_type]["answer_eval_available"] += 1
            by_problem_type[expected_type]["answer_eval_passed"] += int(grade["passed"])
            by_problem_type[expected_type]["answer_eval_score"] += float(grade["score_pct"])
            if not grade["passed"] or float(grade["score_pct"]) < 70.0:
                weak_answer_scores.append(
                    {
                        "id": qid,
                        "subject_id": expected_subject,
                        "problem_type": expected_type,
                        "source": candidate_source,
                        "score_pct": grade["score_pct"],
                        "answer_recall_pct": grade["answer_recall_pct"],
                        "step_coverage_pct": grade["step_coverage_pct"],
                        "numeric_consistency_pct": grade["numeric_consistency_pct"],
                    }
                )

        solver_counter[solver_name or "none"] += int(solver_success)
        if recommendations:
            top1 = recommendations[0]
            top1_same_subject = str(top1.get("subject_id", "") or "") == expected_subject
            top1_same_type = str(top1.get("problem_type", "") or "") == expected_type
            totals["recommendation_nonempty"] += 1
            totals["recommendation_top1_same_subject"] += int(top1_same_subject)
            totals["recommendation_top1_same_type"] += int(top1_same_type)
            if not top1_same_subject or not top1_same_type:
                weak_recommendations.append(
                    {
                        "id": qid,
                        "expected_subject": expected_subject,
                        "expected_problem_type": expected_type,
                        "top1": {
                            "id": top1.get("id"),
                            "subject_id": top1.get("subject_id"),
                            "problem_type": top1.get("problem_type"),
                        },
                    }
                )
        else:
            weak_recommendations.append(
                {
                    "id": qid,
                    "expected_subject": expected_subject,
                    "expected_problem_type": expected_type,
                    "top1": None,
                }
            )

        if not subject_correct or not type_correct:
            misclassified.append(
                {
                    "id": qid,
                    "expected_subject": expected_subject,
                    "actual_subject": actual_subject,
                    "expected_problem_type": expected_type,
                    "actual_problem_type": actual_type,
                    "question": question,
                }
            )

        if not solver_success:
            solver_fallbacks.append(
                {
                    "id": qid,
                    "subject_id": expected_subject,
                    "problem_type": expected_type,
                    "expected_rule_solvable": expected_solvable,
                    "solver": solver_name,
                    "message": str(solver_result.get("message", "") or ""),
                }
            )

    total = totals["total"]
    nonempty = totals["recommendation_nonempty"]
    expected_solvable_total = totals["expected_rule_solvable"]
    answer_eval_total = totals["answer_eval_available"]
    metrics = {
        "subject_accuracy_pct": pct(totals["subject_correct"], total),
        "problem_type_accuracy_pct": pct(totals["problem_type_correct"], total),
        "recommendation_nonempty_pct": pct(nonempty, total),
        "recommendation_top1_same_subject_pct": pct(
            totals["recommendation_top1_same_subject"],
            nonempty,
        ),
        "recommendation_top1_same_type_pct": pct(
            totals["recommendation_top1_same_type"],
            nonempty,
        ),
        "solver_success_pct": pct(totals["solver_success"], total),
        "expected_rule_solver_coverage_pct": pct(
            totals["expected_rule_solvable_success"],
            expected_solvable_total,
        ),
        "llm_fallback_needed_pct": pct(totals["llm_fallback_needed"], total),
        "answer_eval_coverage_pct": pct(answer_eval_total, total),
        "answer_eval_pass_pct": pct(totals["answer_eval_passed"], answer_eval_total),
        "answer_eval_exact_hit_pct": pct(totals["answer_eval_exact_hit"], answer_eval_total),
        "answer_eval_avg_score_pct": avg(score_totals["answer_eval_score"], answer_eval_total),
        "answer_eval_avg_recall_pct": avg(score_totals["answer_eval_recall"], answer_eval_total),
        "answer_eval_avg_step_coverage_pct": avg(
            score_totals["answer_eval_steps"],
            answer_eval_total,
        ),
        "answer_eval_avg_numeric_pct": avg(
            score_totals["answer_eval_numeric"],
            answer_eval_total,
        ),
        "answer_eval_avg_mistake_pct": avg(
            score_totals["answer_eval_mistakes"],
            answer_eval_total,
        ),
    }

    def counter_table(source: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = {}
        for key in sorted(source):
            row = source[key]
            row_total = int(row["total"])
            answer_eval_available = int(row["answer_eval_available"])
            table[key] = {
                "total": row_total,
                "subject_accuracy_pct": pct(int(row["subject_correct"]), row_total),
                "problem_type_accuracy_pct": pct(int(row["problem_type_correct"]), row_total),
                "solver_success_pct": pct(int(row["solver_success"]), row_total),
                "answer_eval_coverage_pct": pct(answer_eval_available, row_total),
                "answer_eval_pass_pct": pct(int(row["answer_eval_passed"]), answer_eval_available),
                "answer_eval_avg_score_pct": avg(float(row["answer_eval_score"]), answer_eval_available),
            }
        return table

    return {
        "question_bank_path": str(bank_path),
        "candidate_answers_path": str(candidate_answers_path) if candidate_answers_path else "",
        "totals": totals,
        "metrics": metrics,
        "by_subject": counter_table(by_subject),
        "by_problem_type": counter_table(by_problem_type),
        "solver_success_by_solver": dict(sorted(solver_counter.items())),
        "samples": {
            "misclassified": misclassified[: max(0, detail_limit)],
            "weak_recommendations": weak_recommendations[: max(0, detail_limit)],
            "solver_fallbacks": solver_fallbacks[: max(0, detail_limit)],
            "weak_answer_scores": weak_answer_scores[: max(0, detail_limit)],
        },
    }


def format_table(title: str, rows: dict[str, dict[str, Any]]) -> list[str]:
    lines = [title]
    lines.append("名称 | 数量 | 学科识别 | 题型识别 | 规则求解 | 答案评测覆盖 | 答案评测通过 | 答案平均分")
    lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for key, row in rows.items():
        lines.append(
            f"{key} | {row['total']} | {row['subject_accuracy_pct']}% | "
            f"{row['problem_type_accuracy_pct']}% | {row['solver_success_pct']}% | "
            f"{row['answer_eval_coverage_pct']}% | {row['answer_eval_pass_pct']}% | "
            f"{row['answer_eval_avg_score_pct']}%"
        )
    return lines


def format_report(report: dict[str, Any]) -> str:
    totals = report["totals"]
    metrics = report["metrics"]
    samples = report["samples"]
    lines = [
        "# 题目辅导离线评测",
        "",
        f"- 题库：{report['question_bank_path']}",
        f"- 候选答案文件：{report.get('candidate_answers_path') or '未提供（默认使用规则求解基线答案）'}",
        f"- 总题数：{totals['total']}",
        f"- 学科识别准确率：{metrics['subject_accuracy_pct']}%",
        f"- 题型识别准确率：{metrics['problem_type_accuracy_pct']}%",
        f"- 类题推荐非空率：{metrics['recommendation_nonempty_pct']}%",
        f"- 类题推荐 Top1 同学科率：{metrics['recommendation_top1_same_subject_pct']}%",
        f"- 类题推荐 Top1 同题型率：{metrics['recommendation_top1_same_type_pct']}%",
        f"- 规则求解整体命中率：{metrics['solver_success_pct']}%",
        f"- 规则可解题覆盖率：{metrics['expected_rule_solver_coverage_pct']}%",
        f"- 需要 LLM 兜底比例：{metrics['llm_fallback_needed_pct']}%",
        f"- 最终答案评测覆盖率：{metrics['answer_eval_coverage_pct']}%",
        f"- 最终答案评测通过率：{metrics['answer_eval_pass_pct']}%",
        f"- 最终答案平均分：{metrics['answer_eval_avg_score_pct']}%",
        f"- 关键结论召回均值：{metrics['answer_eval_avg_recall_pct']}%",
        f"- 解题步骤覆盖均值：{metrics['answer_eval_avg_step_coverage_pct']}%",
        f"- 数值一致性均值：{metrics['answer_eval_avg_numeric_pct']}%",
        "",
        "## 规则求解器命中数",
    ]
    solver_counts = report["solver_success_by_solver"]
    if solver_counts:
        for solver, count in solver_counts.items():
            lines.append(f"- {solver}: {count}")
    else:
        lines.append("- 无")

    lines.extend(["", *format_table("## 按学科统计", report["by_subject"])])
    lines.extend(["", *format_table("## 按题型统计", report["by_problem_type"])])

    def sample_lines(title: str, values: list[dict[str, Any]]) -> None:
        lines.extend(["", title])
        if not values:
            lines.append("- 无")
            return
        for item in values:
            lines.append(f"- `{item.get('id', '')}`: {json.dumps(item, ensure_ascii=False)}")

    sample_lines("## 识别错误样例", samples["misclassified"])
    sample_lines("## 推荐偏弱样例", samples["weak_recommendations"])
    sample_lines("## 规则求解兜底样例", samples["solver_fallbacks"])
    sample_lines("## 答案评测低分样例", samples["weak_answer_scores"])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线评测题目辅导模块的题库覆盖情况。")
    parser.add_argument(
        "--bank",
        default=str(DEFAULT_QUESTION_BANK_PATH),
        help="题库 JSONL 路径，默认 data/tutoring_question_bank/questions.jsonl",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="输出格式，默认 text",
    )
    parser.add_argument(
        "--detail-limit",
        type=int,
        default=12,
        help="每类问题样例最多输出多少条，默认 12",
    )
    parser.add_argument(
        "--candidate-answers",
        default="",
        help="可选：候选答案 JSON/JSONL 路径。提供后会对真实候选答案做要点评分。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_question_bank(
        args.bank,
        detail_limit=args.detail_limit,
        candidate_answers_path=args.candidate_answers or None,
    )
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
