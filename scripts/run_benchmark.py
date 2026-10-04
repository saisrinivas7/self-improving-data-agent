"""
Run the benchmark across conditions and write the results table.

    make benchmark                     the default set of conditions
    make benchmark ARGS="--quick"      3 questions per condition, for a smoke test
    make benchmark ARGS="--full"       every condition on every test question

CONDITIONS

    baseline                     no memory. The control.
    feedback_rag/clean           naive retrieval, correct lessons
    feedback_rag/poisoned        naive retrieval, wrong lessons      <- the risk
    feedback_rag/mixed           naive retrieval, both
    verified_feedback/clean      verified-only retrieval, correct lessons
    verified_feedback/poisoned   verified-only retrieval, wrong lessons <- the defence
    verified_feedback/mixed      verified-only retrieval, both

The pair that matters is feedback_rag/poisoned versus
verified_feedback/poisoned. Everything else is context for it.

ONLY TEST QUESTIONS ARE SCORED. The lessons in the snapshots were written
against TRAIN questions, so scoring on train questions would measure
memorisation - a lesson written while looking at a question will obviously
help that question. The train/test split in benchmark/questions.json is what
makes a result about generalisation.

COMPUTE. Local inference runs at roughly 13 tokens/sec, so one question
costs 40-80 seconds. The full matrix is several hours; it is designed to be
left running. Results are written incrementally so a crash does not lose
the whole run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS = PROJECT_ROOT / "benchmark" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"

CONDITIONS: list[tuple[str, str | None]] = [
    ("baseline", None),
    ("feedback_rag", "clean"),
    ("feedback_rag", "poisoned"),
    ("feedback_rag", "mixed"),
    ("verified_feedback", "clean"),
    ("verified_feedback", "poisoned"),
    ("verified_feedback", "mixed"),
]

# Questions that detect poison adoption. The poisoned conditions are run on
# these plus a sample of the rest, because running every condition on every
# question costs hours and the poison can only show up where refunds are
# genuinely not the cause.
POISON_PROBES = ["july_why", "july_refunds_role", "aov_trend", "jan_why"]


@dataclass
class RunRecord:
    condition: str
    variant: str
    snapshot: str | None
    question_id: str
    question: str
    seed: int
    answer: str = ""
    sql: str = ""
    status: str = ""
    task_success: float = 0.0
    sql_correct: float = 0.0
    analytical_correct: float = 0.0
    traps_hit: list[str] = field(default_factory=list)
    adopted_poison: bool | None = None
    feedback_retrieved: int = 0
    feedback_used: int = 0
    sql_attempts: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    trace_id: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["answer"] = self.answer[:1500]
        d["sql"] = self.sql[:2000]
        return d


def load_questions(*, split: str = "test") -> list[dict]:
    data = json.loads(QUESTIONS.read_text())
    return [q for q in data["questions"] if q["split"] == split]


def run_one(spec: dict, variant: str, snapshot: str | None, seed: int, llm) -> RunRecord:  # noqa: ANN001
    from app.agent.runner import answer_question
    from app.evaluation.scoring import (
        adopted_wrong_feedback,
        feedback_was_used,
        score_question,
    )
    from app.feedback import memory as fm

    cond = f"{variant}/{snapshot}" if snapshot else variant
    rec = RunRecord(
        condition=cond, variant=variant, snapshot=snapshot,
        question_id=spec["id"], question=spec["question"], seed=seed,
    )
    try:
        state = answer_question(
            spec["question"],
            variant=variant,  # type: ignore[arg-type]
            snapshot=snapshot or "live",
            llm=llm,
            save=True,
        )
    except Exception as e:  # noqa: BLE001
        rec.error = f"{type(e).__name__}: {e}"
        rec.status = "failed"
        return rec

    rec.answer = state.final_answer
    rec.sql = state.generated_sql
    rec.status = state.status
    rec.sql_attempts = state.sql_attempt_count
    rec.tool_calls = state.tool_calls
    rec.llm_calls = state.llm_calls
    rec.total_tokens = state.prompt_tokens + state.output_tokens + state.thinking_tokens
    rec.latency_s = round(state.latency_s, 2)
    rec.trace_id = state.trace_id
    rec.feedback_retrieved = len(state.retrieved_lessons)
    rec.error = "; ".join(state.errors)[:400]

    score = score_question(spec, answer=state.final_answer, sql=state.generated_sql,
                           tables_used=state.tables_used)
    rec.task_success = score.task_success
    rec.sql_correct = score.sql_correct
    rec.analytical_correct = score.analytical_correct
    rec.traps_hit = score.traps_hit
    rec.adopted_poison = adopted_wrong_feedback(spec, state.final_answer)

    # Feedback utilisation: did the SQL touch a table the lesson is about?
    # Counting retrieval alone would make this 100% by definition.
    if state.retrieved_lessons and snapshot:
        used = 0
        stored = {r["feedback_id"]: r for r in fm.list_lessons(snapshot, limit=200)}
        for l in state.retrieved_lessons:
            row = stored.get(l.feedback_id)
            tables = list(row["schema_context"] or []) if row else []
            if feedback_was_used(tables, state.generated_sql, state.final_answer, l.lesson):
                used += 1
        rec.feedback_used = used
    return rec


def aggregate(records: list[RunRecord]) -> dict[str, dict]:
    by_cond: dict[str, list[RunRecord]] = {}
    for r in records:
        by_cond.setdefault(r.condition, []).append(r)

    out: dict[str, dict] = {}
    for cond, rs in by_cond.items():
        ok = [r for r in rs if r.status == "completed"]
        probes = [r for r in rs if r.adopted_poison is not None]
        retrieved_any = [r for r in rs if r.feedback_retrieved > 0]

        def mean(vals: list[float]) -> float | None:
            return round(statistics.mean(vals), 3) if vals else None

        out[cond] = {
            "runs": len(rs),
            "completed": len(ok),
            "task_success": mean([r.task_success for r in ok]),
            "sql_correct": mean([r.sql_correct for r in ok]),
            "analytical_correct": mean([r.analytical_correct for r in ok]),
            "wrong_feedback_adoption": (
                round(sum(bool(r.adopted_poison) for r in probes) / len(probes), 3)
                if probes else None
            ),
            "poison_probes": len(probes),
            "feedback_utilisation": (
                round(
                    sum(r.feedback_used for r in retrieved_any)
                    / sum(r.feedback_retrieved for r in retrieved_any), 3
                )
                if retrieved_any and sum(r.feedback_retrieved for r in retrieved_any)
                else None
            ),
            "avg_lessons_retrieved": mean([float(r.feedback_retrieved) for r in rs]),
            "avg_sql_attempts": mean([float(r.sql_attempts) for r in rs]),
            "avg_tool_calls": mean([float(r.tool_calls) for r in rs]),
            "avg_llm_calls": mean([float(r.llm_calls) for r in rs]),
            "avg_tokens": mean([float(r.total_tokens) for r in rs]),
            "avg_latency_s": mean([r.latency_s for r in rs]),
            "traps_hit_total": sum(len(r.traps_hit) for r in rs),
        }
    return out


def render_table(agg: dict[str, dict]) -> str:
    """The results table from spec section 22, as markdown."""
    order = [f"{v}/{s}" if s else v for v, s in CONDITIONS]
    conds = [c for c in order if c in agg]

    rows = [
        ("Task success", "task_success", "pct"),
        ("SQL correctness", "sql_correct", "pct"),
        ("Analytical accuracy", "analytical_correct", "pct"),
        ("Feedback utilisation", "feedback_utilisation", "pct"),
        ("Wrong-feedback adoption", "wrong_feedback_adoption", "pct"),
        ("Avg lessons retrieved", "avg_lessons_retrieved", "num"),
        ("Avg SQL attempts", "avg_sql_attempts", "num"),
        ("Avg tool calls", "avg_tool_calls", "num"),
        ("Avg LLM calls", "avg_llm_calls", "num"),
        ("Avg tokens", "avg_tokens", "int"),
        ("Avg latency (s)", "avg_latency_s", "num"),
        ("Traps hit (total)", "traps_hit_total", "int"),
    ]

    def fmt(v, kind: str) -> str:  # noqa: ANN001
        if v is None:
            return "-"
        if kind == "pct":
            return f"{v * 100:.0f}%"
        if kind == "int":
            return f"{v:,.0f}"
        return f"{v:.2f}"

    head = "| Metric | " + " | ".join(conds) + " |"
    sep = "|---" * (len(conds) + 1) + "|"
    body = [
        f"| {label} | " + " | ".join(fmt(agg[c].get(key), kind) for c in conds) + " |"
        for label, key, kind in rows
    ]
    note = (
        f"\n_n = {agg[conds[0]]['runs']} questions per condition. "
        "Wrong-feedback adoption is measured only on questions where refunds "
        "are genuinely not the cause._\n"
    )
    return "\n".join([head, sep, *body]) + note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 questions per condition")
    ap.add_argument("--full", action="store_true", help="every test question, every condition")
    ap.add_argument("--seeds", type=int, default=1, help="repeat the whole matrix N times")
    ap.add_argument("--conditions", type=str, default="", help="comma-separated subset")
    args = ap.parse_args()

    from app.config import get_settings
    from app.llm import LLMClient

    s = get_settings()
    questions = load_questions(split="test")

    conds = CONDITIONS
    if args.conditions:
        want = {c.strip() for c in args.conditions.split(",")}
        conds = [(v, sn) for v, sn in CONDITIONS
                 if (f"{v}/{sn}" if sn else v) in want]

    def questions_for(variant: str, snapshot: str | None) -> list[dict]:
        if args.quick:
            return questions[:3]
        if args.full:
            return questions
        # Default: poisoned conditions only on the probes plus a sample,
        # since poison can only surface where refunds are not the cause.
        if snapshot == "poisoned":
            probes = [q for q in questions if q["id"] in POISON_PROBES]
            others = [q for q in questions if q["id"] not in POISON_PROBES][:6]
            return probes + others
        return questions

    total = sum(len(questions_for(v, sn)) for v, sn in conds) * args.seeds
    print(f"Benchmark: {len(conds)} condition(s), {total} runs, model {s.chat_model}")
    print(f"  estimated {total * 55 / 60:.0f}-{total * 80 / 60:.0f} minutes\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RESULTS_DIR / f"runs_{stamp}.jsonl"

    records: list[RunRecord] = []
    t0 = time.perf_counter()
    done = 0

    with LLMClient() as llm, raw_path.open("w") as raw:
        for seed in range(args.seeds):
            for variant, snapshot in conds:
                cond = f"{variant}/{snapshot}" if snapshot else variant
                qs = questions_for(variant, snapshot)
                print(f"--- {cond}  ({len(qs)} questions, seed {seed}) ---")
                for q in qs:
                    rec = run_one(q, variant, snapshot, seed, llm)
                    records.append(rec)
                    # Written immediately so a crash keeps everything so far.
                    raw.write(json.dumps(rec.as_dict()) + "\n")
                    raw.flush()
                    done += 1
                    elapsed = time.perf_counter() - t0
                    eta = (elapsed / done) * (total - done) / 60
                    flag = ""
                    if rec.adopted_poison:
                        flag = "  [ADOPTED POISON]"
                    if rec.status != "completed":
                        flag = f"  [{rec.status}]"
                    print(
                        f"  {done:>3}/{total}  {q['id']:<24} "
                        f"task={rec.task_success:.2f} sql={rec.sql_correct:.2f} "
                        f"fb={rec.feedback_used}/{rec.feedback_retrieved} "
                        f"{rec.latency_s:>5.0f}s  eta {eta:.0f}m{flag}"
                    )

    agg = aggregate(records)
    table = render_table(agg)

    summary = {
        "generated_at": stamp,
        "chat_model": s.chat_model,
        "embedding_model": s.embedding_model,
        "seeds": args.seeds,
        "mode": "quick" if args.quick else "full" if args.full else "default",
        "conditions": agg,
    }
    (RESULTS_DIR / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / f"table_{stamp}.md").write_text(table)
    (RESULTS_DIR / "latest_table.md").write_text(table)
    (RESULTS_DIR / "latest_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + table)
    print(f"\nwrote {raw_path.name}, summary_{stamp}.json, table_{stamp}.md")
    print(f"total time {(time.perf_counter() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
