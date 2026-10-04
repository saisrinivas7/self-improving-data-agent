"""
The full feedback lifecycle, in one call.

    raw analyst text
         |
         v
    classify + generalise        processor.py
         |
         v
    detect conflicts             conflict_resolver.py
         |
         v
    verify each claim            verifier.py
         |
         v
    decide status                policy.py
         |
         v
    store with audit trail       memory.py

This exists so the Learning Lab, the snapshot builder and the benchmark all
go through exactly the same path. If the Lab stored feedback differently
from the benchmark, the experiment would be measuring the difference between
two code paths rather than between two retrieval policies.

Conflict detection runs BEFORE verification on purpose. A lesson that
contradicts a stored one is not resolved by checking its arithmetic - which
of two contradictory statements is right is a human decision - so there is
no point spending model calls verifying it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.feedback import memory as fm
from app.feedback.conflict_resolver import Conflict, find_conflicts, mark_mutual_conflict
from app.feedback.policy import LessonKind, Status, VerificationOutcome
from app.feedback.processor import ProcessedFeedback, process_feedback
from app.feedback.verifier import verify_feedback
from app.llm import LLMClient


@dataclass
class FeedbackResult:
    processed: ProcessedFeedback
    outcome: VerificationOutcome
    feedback_id: str
    conflicts: list[Conflict] = field(default_factory=list)
    stored: bool = True

    @property
    def status(self) -> Status:
        return self.outcome.status

    def summary(self) -> str:
        """One line for the terminal or the UI."""
        return (
            f"{self.outcome.status} "
            f"({self.outcome.lesson_kind}, confidence {self.outcome.confidence:.2f}) "
            f"- {self.outcome.reason}"
        )


def submit_feedback(
    *,
    feedback: str,
    question: str,
    sql: str = "",
    answer: str = "",
    trace_id: str | None = None,
    snapshot: str = "live",
    source: str = "human_analyst",
    verify: bool = True,
    llm: LLMClient | None = None,
) -> FeedbackResult:
    """Process, verify and store one piece of analyst feedback.

    verify=False stores it as PENDING without checking - which is what the
    naive feedback-RAG condition represents, and what the Learning Lab did
    before verification existed.
    """
    own = llm is None
    llm = llm or LLMClient()

    try:
        processed = process_feedback(
            feedback=feedback,
            question=question,
            sql=sql,
            answer=answer,
            trace_id=trace_id,
            llm=llm,
        )

        if not verify:
            fid = fm.store_lesson(
                processed, status="PENDING", verified=False,
                snapshot=snapshot, source=source, llm=llm,
            )
            from app.feedback.policy import CONFIDENCE, Verdict  # noqa: PLC0415

            outcome = VerificationOutcome(
                status=Status.PENDING, verified=False,
                confidence=processed.confidence,
                lesson_kind=LessonKind.UNKNOWN,
                reason="stored without verification",
            )
            return FeedbackResult(processed, outcome, fid)

        conflicts = find_conflicts(
            processed.lesson,
            original_question=processed.original_question,
            snapshot=snapshot,
            llm=llm,
        )

        outcome = verify_feedback(
            processed,
            conflicts_with=[c.existing_id for c in conflicts],
            llm=llm,
        )

        fid = fm.store_lesson(
            processed,
            status=str(outcome.status),
            verified=outcome.verified,
            verification_notes=outcome.as_notes(),
            conflicts_with=[c.existing_id for c in conflicts],
            snapshot=snapshot,
            source=source,
            llm=llm,
        )

        # Override the clarity confidence from the classifier with the
        # verification confidence. Clarity measures how clearly the analyst
        # expressed themselves; this measures whether it holds up. Retrieval
        # ranking should weight the latter.
        fm.update_status(
            fid,
            status=str(outcome.status),
            verified=outcome.verified,
            confidence=outcome.confidence,
            verification_notes=outcome.as_notes(),
        )

        if conflicts:
            mark_mutual_conflict(fid, conflicts)

        return FeedbackResult(processed, outcome, fid, conflicts)
    finally:
        if own:
            llm.close()
