"""
Feedback memory: storing and retrieving lessons (spec sections 10, 14).

This is the "persistent memory" the project is named after. Nothing is
fine-tuned; what changes between runs is which lessons end up in the prompt.

RETRIEVAL, END TO END

    question
       |
       v
    embed the question                  (one vector, 1024 numbers)
       |
       v
    pgvector cosine search              (nearest stored lessons)
       |
       v
    filter: snapshot, embedding model, similarity floor
       |
       v
    rank: similarity + status + confidence + schema overlap
       |
       v
    top k lessons -> the prompt

Three details that are easy to get wrong:

  - Filtering by embedding_model is not optional. Two models can both
    produce 1024 numbers while meaning completely different things by them.
    Mixing them returns confident nonsense with no error anywhere.

  - A similarity FLOOR matters more than top-k. Nearest-neighbour search
    always returns something; without a floor, an unrelated lesson gets
    injected on every question just for being least-unrelated. That is one
    of the main ways naive feedback RAG makes an agent worse.

  - Snapshots keep the experiment honest. 'live' is the Learning Lab's own
    memory, which starts empty as the spec requires. Benchmarks read from
    named snapshots ('clean', 'poisoned') so each condition has a known
    memory state and the live one is never touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.agent.state import RetrievedLesson
from app.config import get_settings
from app.database.connection import rw_connection
from app.feedback.processor import ProcessedFeedback
from app.llm import LLMClient

# Below this cosine similarity a lesson is treated as unrelated and dropped
# before it can reach the prompt.
DEFAULT_SIMILARITY_FLOOR = 0.45


@dataclass
class StoredLesson:
    feedback_id: str
    feedback_type: str
    lesson: str
    status: str
    confidence: float
    similarity: float = 0.0
    schema_context: list[str] | None = None
    mistake: str = ""
    correction: str = ""
    raw_feedback: str = ""
    original_question: str = ""
    verification_notes: dict | None = None
    created_at: Any = None

    def prompt_text(self) -> str:
        """How the lesson appears to the model.

        Status and confidence are included on purpose. The prompt tells the
        model these are fallible evidence, so it needs to see how much to
        trust each one.
        """
        return (
            f"{self.lesson}  "
            f"[type: {self.feedback_type}, status: {self.status}, "
            f"confidence: {self.confidence:.2f}, similarity: {self.similarity:.2f}]"
        )


def embed_text(text: str, llm: LLMClient | None = None) -> list[float]:
    own = llm is None
    llm = llm or LLMClient()
    try:
        return llm.embed([text])[0]
    finally:
        if own:
            llm.close()


def embedding_input(lesson: str, original_question: str = "") -> str:
    """What text gets embedded for a stored lesson.

    Both the lesson and the question it came from. Embedding the lesson
    alone makes retrieval worse: lessons are phrased as instructions
    ("When investigating X, also check Y") while incoming queries are
    phrased as questions ("Why did X happen?"), and those two forms sit
    further apart in embedding space than their meanings deserve. Including
    the original question gives the vector something question-shaped to
    match against.
    """
    return f"{original_question}\n{lesson}".strip()


# ------------------------------------------------------------------ writing

def store_lesson(
    p: ProcessedFeedback,
    *,
    status: str = "PENDING",
    verified: bool = False,
    verification_notes: dict | None = None,
    conflicts_with: list[str] | None = None,
    snapshot: str = "live",
    source: str = "human_analyst",
    llm: LLMClient | None = None,
) -> str:
    """Write one lesson to memory. Returns its feedback_id.

    status defaults to PENDING: a lesson is not VERIFIED just because it was
    stored. Phase 7's verifier is what promotes it.
    """
    s = get_settings()
    vec = embed_text(embedding_input(p.lesson, p.original_question), llm)

    with rw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory.feedback_memory (
                    feedback_type, original_question, mistake, correction,
                    lesson, raw_feedback, source, trace_id,
                    status, verified, confidence, verification_notes,
                    conflicts_with, schema_context, context,
                    embedding, embedding_model, snapshot
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                RETURNING feedback_id
                """,
                (
                    p.feedback_type, p.original_question, p.mistake, p.correction,
                    p.lesson, p.raw_feedback, source, p.trace_id,
                    status, verified, p.confidence,
                    json.dumps(verification_notes or {}, default=str),
                    conflicts_with or [], p.schema_context,
                    json.dumps({"claims": [c.__dict__ for c in p.claims]}, default=str),
                    vec, s.embedding_model, snapshot,
                ),
            )
            fid = str(cur.fetchone()[0])
        conn.commit()
    return fid


def update_status(
    feedback_id: str,
    *,
    status: str,
    verified: bool,
    confidence: float | None = None,
    verification_notes: dict | None = None,
    conflicts_with: list[str] | None = None,
) -> None:
    sets = ["status = %s", "verified = %s", "updated_at = now()"]
    vals: list[Any] = [status, verified]
    if confidence is not None:
        sets.append("confidence = %s")
        vals.append(confidence)
    if verification_notes is not None:
        sets.append("verification_notes = %s")
        vals.append(json.dumps(verification_notes, default=str))
    if conflicts_with is not None:
        sets.append("conflicts_with = %s")
        vals.append(conflicts_with)
    vals.append(feedback_id)

    with rw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE memory.feedback_memory SET {', '.join(sets)} "
                f"WHERE feedback_id = %s",
                vals,
            )
        conn.commit()


def bump_usage(feedback_id: str, *, retrieved: bool = False,
               applied: bool = False, rejected: bool = False) -> None:
    """Keep the counters behind the feedback-utilisation metric."""
    with rw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory.feedback_memory
                SET times_retrieved = times_retrieved + %s,
                    times_applied   = times_applied + %s,
                    times_rejected  = times_rejected + %s
                WHERE feedback_id = %s
                """,
                (int(retrieved), int(applied), int(rejected), feedback_id),
            )
        conn.commit()


# ------------------------------------------------------------------ reading

def retrieve_lessons(
    question: str,
    *,
    k: int | None = None,
    snapshot: str = "live",
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
    include_statuses: tuple[str, ...] = ("PENDING", "VERIFIED"),
    tables_in_play: list[str] | None = None,
    llm: LLMClient | None = None,
) -> list[StoredLesson]:
    """Find lessons relevant to this question.

    include_statuses is what separates the two feedback systems:
      - plain feedback RAG passes ('PENDING','VERIFIED'), i.e. it will use
        anything that was stored
      - the verified system passes ('VERIFIED',) only, so rejected and
        conflicting lessons cannot influence it

    REJECTED and CONFLICTING are never included by default. They stay in the
    table for auditing and for the Learning Lab to display, not for use.
    """
    s = get_settings()
    k = s.feedback_top_k if k is None else k

    # Nothing stored yet is the normal case early on - skip the embedding
    # call entirely rather than pay for it on every question.
    with rw_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memory.feedback_memory WHERE snapshot = %s",
            (snapshot,),
        )
        if cur.fetchone()[0] == 0:
            return []

    vec = embed_text(question, llm)

    from psycopg.rows import dict_row

    with rw_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT feedback_id, feedback_type, lesson, status, confidence,
                       schema_context, mistake, correction, raw_feedback,
                       original_question, verification_notes, created_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM memory.feedback_memory
                WHERE snapshot = %s
                  AND status = ANY(%s)
                  AND embedding IS NOT NULL
                  AND embedding_model = %s
                  AND superseded_by IS NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec, snapshot, list(include_statuses), s.embedding_model, vec, k * 3),
            )
            rows = cur.fetchall()

    out: list[StoredLesson] = []
    for r in rows:
        sim = float(r["similarity"])
        if sim < similarity_floor:
            continue
        out.append(
            StoredLesson(
                feedback_id=str(r["feedback_id"]),
                feedback_type=r["feedback_type"],
                lesson=r["lesson"],
                status=r["status"],
                confidence=float(r["confidence"]),
                similarity=sim,
                schema_context=list(r["schema_context"] or []),
                mistake=r["mistake"],
                correction=r["correction"],
                raw_feedback=r["raw_feedback"] or "",
                original_question=r["original_question"],
                verification_notes=r["verification_notes"],
                created_at=r["created_at"],
            )
        )

    out = rank_lessons(out, tables_in_play=tables_in_play)
    return out[:k]


def rank_lessons(
    lessons: list[StoredLesson], *, tables_in_play: list[str] | None = None
) -> list[StoredLesson]:
    """Re-rank beyond raw similarity (spec section 14).

    V1 combines similarity, verification status and confidence, plus schema
    overlap. Pure similarity is not enough: a vaguely-worded unverified
    guess can sit closer in embedding space than a precise verified lesson,
    and injecting the wrong one is exactly how feedback memory degrades an
    agent instead of improving it.
    """
    status_weight = {"VERIFIED": 1.0, "PENDING": 0.6, "CONFLICTING": 0.3, "REJECTED": 0.0}
    tables = set(tables_in_play or [])

    def score(l: StoredLesson) -> float:
        s = 0.55 * l.similarity
        s += 0.25 * status_weight.get(l.status, 0.3)
        s += 0.10 * l.confidence
        if tables and l.schema_context:
            overlap = len(tables & set(l.schema_context)) / len(set(l.schema_context))
            s += 0.10 * overlap
        return s

    return sorted(lessons, key=score, reverse=True)


def to_retrieved(lessons: list[StoredLesson], *, used: bool = True) -> list[RetrievedLesson]:
    """Convert to the agent-state form, for tracing."""
    return [
        RetrievedLesson(
            feedback_id=l.feedback_id,
            lesson=l.lesson,
            feedback_type=l.feedback_type,
            status=l.status,
            confidence=l.confidence,
            similarity=l.similarity,
            used=used,
        )
        for l in lessons
    ]


# ------------------------------------------------------------------ browsing

def list_lessons(snapshot: str = "live", limit: int = 100) -> list[dict]:
    """Everything in memory, for the Learning Lab's inspector."""
    from psycopg.rows import dict_row

    with rw_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT feedback_id, feedback_type, status, verified, confidence,
                       lesson, mistake, correction, raw_feedback,
                       original_question, schema_context, verification_notes,
                       conflicts_with, times_retrieved, times_applied,
                       times_rejected, created_at
                FROM memory.feedback_memory
                WHERE snapshot = %s
                ORDER BY created_at DESC LIMIT %s
                """,
                (snapshot, limit),
            )
            return [dict(r) for r in cur.fetchall()]


def memory_stats(snapshot: str = "live") -> dict:
    with rw_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, count(*) FROM memory.feedback_memory
            WHERE snapshot = %s GROUP BY status
            """,
            (snapshot,),
        )
        by_status = {r[0]: r[1] for r in cur.fetchall()}
    return {"total": sum(by_status.values()), "by_status": by_status}


def delete_lesson(feedback_id: str) -> None:
    with rw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memory.feedback_memory WHERE feedback_id = %s",
                (feedback_id,),
            )
        conn.commit()
