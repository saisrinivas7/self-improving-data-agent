"""
Build the memory snapshots the benchmark runs against.

Run:  make snapshots

    clean      correct lessons only
    poisoned   deliberately wrong lessons only
    mixed      both, which is the realistic case

WHY SNAPSHOTS EXIST AT ALL

The spec requires the Learning Lab's live memory to start empty, and it
does. But a benchmark needs each condition to run against a KNOWN memory
state, identical on every run. Otherwise the comparison between conditions
is confounded by whatever happened to be in memory that day.

Snapshots are built by the SAME pipeline the Learning Lab uses, so the
statuses are earned by the verifier rather than asserted here. The one
exception is the poison, explained below.

WHY THE POISON BYPASSES THE CLASSIFIER

Running "revenue declines are always caused by refunds" through the
classifier produced a SENSIBLE lesson: "do not conclude it is caused by
refunds without examining other factors". That is the classifier quietly
correcting the analyst - interesting in itself, and reported as a finding,
but fatal for the experiment. A sanitised poison HELPS the naive system, so
wrong-feedback adoption would measure near zero and the headline result
would be a false success.

So poison lessons are inserted verbatim with hand-written claims. The
verifier still has to catch them on its own; nothing here tells it the
answer. `expect_status` in benchmark/lessons.json is an assertion about what
the verifier SHOULD conclude, and this script reports every mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LESSONS = PROJECT_ROOT / "benchmark" / "lessons.json"
QUESTIONS = PROJECT_ROOT / "benchmark" / "questions.json"

SNAPSHOTS = {
    "clean": ["clean"],
    "poisoned": ["poison"],
    "mixed": ["clean", "poison"],
}


def _question_text(qid: str) -> str:
    data = json.loads(QUESTIONS.read_text())
    for q in data["questions"]:
        if q["id"] == qid:
            return q["question"]
    return ""


def _wipe(snapshot: str) -> int:
    from app.database.connection import rw_connection

    with rw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memory.feedback_memory WHERE snapshot = %s", (snapshot,)
            )
            n = cur.rowcount
        conn.commit()
    return n


def main() -> None:
    from app.feedback import memory as fm
    from app.feedback.pipeline import submit_feedback
    from app.feedback.processor import Claim, ProcessedFeedback
    from app.feedback.verifier import verify_feedback
    from app.llm import LLMClient

    spec = json.loads(LESSONS.read_text())
    mismatches: list[str] = []

    print("Building memory snapshots\n")

    with LLMClient() as llm:
        # ---- clean lessons: through the real pipeline, verified honestly ----
        clean_results: list[dict] = []
        print("CLEAN lessons (classified and verified like real feedback)")
        for item in spec["clean"]:
            q = _question_text(item["from_question"])
            res = submit_feedback(
                feedback=item["feedback"],
                question=q,
                snapshot="_staging_clean",
                source=f"benchmark:{item['id']}",
                verify=True,
                llm=llm,
            )
            got = str(res.outcome.status)
            want = item.get("expect_status", "VERIFIED")
            ok = got == want
            if not ok:
                mismatches.append(f"{item['id']}: expected {want}, verifier said {got}")
            print(
                f"  {'ok  ' if ok else 'DIFF'} {item['id']:<28} {got:<12} "
                f"{res.outcome.lesson_kind}"
            )
            clean_results.append(
                {
                    "id": item["id"],
                    "processed": res.processed,
                    "outcome": res.outcome,
                }
            )

        # ---- poison lessons: verbatim, then verified ----
        poison_results: list[dict] = []
        print("\nPOISON lessons (inserted verbatim, then put through the verifier)")
        for item in spec["poison"]:
            q = _question_text(item["from_question"])
            processed = ProcessedFeedback(
                feedback_type=item["feedback_type"],
                mistake=item["mistake"],
                correction=item["correction"],
                lesson=item["lesson"],
                confidence=0.9,  # the analyst sounded confident; that is the danger
                schema_context=item["schema_context"],
                claims=[
                    Claim(c["text"], c["kind"], c.get("is_universal", False))
                    for c in item["claims"]
                ],
                raw_feedback=item["lesson"],
                original_question=q,
            )
            outcome = verify_feedback(processed, llm=llm)
            got = str(outcome.status)
            want = item["expect_status"]
            ok = got == want
            if not ok:
                mismatches.append(f"{item['id']}: expected {want}, verifier said {got}")
            print(f"  {'ok  ' if ok else 'DIFF'} {item['id']:<28} {got:<12}")
            if outcome.claim_results:
                print(f"       -> {outcome.claim_results[0].detail[:120]}")
            poison_results.append(
                {"id": item["id"], "processed": processed, "outcome": outcome}
            )

        # ---- write each snapshot ----
        print()
        pools = {"clean": clean_results, "poison": poison_results}
        for name, sources in SNAPSHOTS.items():
            removed = _wipe(name)
            stored = 0
            for src in sources:
                for rec in pools[src]:
                    fm.store_lesson(
                        rec["processed"],
                        status=str(rec["outcome"].status),
                        verified=rec["outcome"].verified,
                        verification_notes=rec["outcome"].as_notes(),
                        snapshot=name,
                        source=f"benchmark:{rec['id']}",
                        llm=llm,
                    )
                    stored += 1
            stats = fm.memory_stats(name)
            by = " ".join(f"{k}={v}" for k, v in sorted(stats["by_status"].items()))
            print(f"  {name:<10} {stored:>2} lesson(s)  ({by})  [removed {removed} old]")

        _wipe("_staging_clean")

    print()
    if mismatches:
        print("=" * 66)
        print(f"{len(mismatches)} VERIFIER MISMATCH(ES)")
        print("=" * 66)
        for m in mismatches:
            print(f"  - {m}")
        print(
            "\n  These are assertions about what the verifier SHOULD conclude.\n"
            "  A mismatch means either the verifier or the expectation is wrong -\n"
            "  investigate before trusting any benchmark result built on it."
        )
        sys.exit(1)

    print("=" * 66)
    print("SNAPSHOTS BUILT - every verifier verdict matched its expectation")
    print("=" * 66)


if __name__ == "__main__":
    main()
