"""
Conflict detection between lessons (spec section 12, memory policy rule 5).

The problem this solves:

    stored:  "Always analyse refunds when revenue declines."
    new:     "Refunds are irrelevant for the Electronics category."

Both can be retrieved for the same question, and the agent would receive two
contradictory instructions with no way to choose. Worse, the newer one might
simply overwrite the older one's influence by being more similar to the
question - so which advice the agent follows would come down to embedding
luck rather than which is right.

HOW IT WORKS, AND WHY IT IS TWO STAGES

    stage 1   vector search for lessons about the same thing
    stage 2   ask the model whether they actually contradict

Stage 1 alone is not enough, because similarity is not contradiction. "Check
refunds when revenue falls" and "Check refund reasons alongside support
tickets" are highly similar and entirely compatible. Stage 2 alone is not
affordable: it would mean comparing every new lesson against every stored
one. So stage 1 narrows to a handful of candidates and stage 2 judges those.

WHAT HAPPENS ON A CONFLICT

Neither lesson is applied, and both are flagged. This is deliberate. Which
of two contradictory statements is correct is usually a human decision - the
newer one may be a refinement, a correction, or simply wrong - and silently
picking one would be exactly the unexamined trust this project exists to
avoid. The Learning Lab shows the pair so a person can resolve it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.feedback import memory as fm
from app.llm import LLMClient

# How similar two lessons must be before it is worth asking whether they
# contradict. Below this they are about different things.
CANDIDATE_SIMILARITY_FLOOR = 0.62
MAX_CANDIDATES = 5

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contradicts": {"type": "boolean"},
        "relationship": {
            "type": "string",
            "enum": ["contradicts", "refines", "duplicates", "unrelated", "compatible"],
        },
        "explanation": {"type": "string"},
    },
    "required": ["contradicts", "relationship", "explanation"],
}

JUDGE_SYSTEM = """You compare two instructions given to a data analyst and \
decide whether following both at once is impossible. Be strict: two \
instructions about the same topic are not contradictory unless obeying one \
means disobeying the other."""

JUDGE_USER = """Instruction A, already stored:
"{existing}"

Instruction B, newly proposed:
"{new}"

Decide the relationship:

- contradicts: an analyst could not follow both. One says to do something \
the other says not to do, or they assert opposite facts about the same thing.
- refines: B narrows or qualifies A without negating it. For example A says \
"always check refunds" and B says "check refunds especially for Electronics".
- duplicates: B says essentially the same thing as A.
- compatible: both can be followed, and they are about related things.
- unrelated: they concern different topics.

Set contradicts to true ONLY for the first case. A narrower exception to a \
general rule is "refines", not "contradicts"."""


@dataclass
class Conflict:
    existing_id: str
    existing_lesson: str
    new_lesson: str
    relationship: str
    explanation: str
    similarity: float


def find_conflicts(
    lesson: str,
    *,
    original_question: str = "",
    snapshot: str = "live",
    exclude_id: str | None = None,
    llm: LLMClient | None = None,
) -> list[Conflict]:
    """Find stored lessons that this one contradicts.

    Only returns genuine contradictions. Refinements, duplicates and merely
    related lessons are examined and discarded, which is the point of the
    second stage.
    """
    own = llm is None
    llm = llm or LLMClient()
    conflicts: list[Conflict] = []

    try:
        # Stage 1: narrow by similarity. Includes every status, because a
        # new lesson can contradict one that is still PENDING.
        candidates = fm.retrieve_lessons(
            fm.embedding_input(lesson, original_question),
            k=MAX_CANDIDATES,
            snapshot=snapshot,
            similarity_floor=CANDIDATE_SIMILARITY_FLOOR,
            include_statuses=None,
            use_verification=False,
            llm=llm,
        )

        # Stage 2: judge each candidate.
        for cand in candidates:
            if exclude_id and cand.feedback_id == exclude_id:
                continue
            if cand.lesson.strip() == lesson.strip():
                continue
            resp = llm.complete(
                JUDGE_USER.format(existing=cand.lesson, new=lesson),
                system=JUDGE_SYSTEM,
                json_schema=JUDGE_SCHEMA,
                max_tokens=400,
            )
            try:
                verdict = resp.json()
            except ValueError:
                continue
            if verdict.get("contradicts") and verdict.get("relationship") == "contradicts":
                conflicts.append(
                    Conflict(
                        existing_id=cand.feedback_id,
                        existing_lesson=cand.lesson,
                        new_lesson=lesson,
                        relationship=str(verdict.get("relationship")),
                        explanation=str(verdict.get("explanation", ""))[:500],
                        similarity=cand.similarity,
                    )
                )
    finally:
        if own:
            llm.close()

    return conflicts


def mark_mutual_conflict(new_id: str, conflicts: list[Conflict]) -> None:
    """Record the conflict on BOTH lessons.

    Flagging only the new one would leave the stored lesson looking healthy
    and still eligible for retrieval, so the contradiction would keep
    influencing answers from one side.
    """
    if not conflicts:
        return

    from app.database.connection import rw_connection

    existing_ids = [c.existing_id for c in conflicts]
    with rw_connection() as conn:
        with conn.cursor() as cur:
            # The new lesson points at all the ones it contradicts.
            cur.execute(
                """
                UPDATE memory.feedback_memory
                SET conflicts_with = %s, status = 'CONFLICTING',
                    verified = false, updated_at = now()
                WHERE feedback_id = %s
                """,
                (existing_ids, new_id),
            )
            # Each stored lesson points back at the new one, and is demoted
            # so it stops being applied while the contradiction stands.
            for cid in existing_ids:
                cur.execute(
                    """
                    UPDATE memory.feedback_memory
                    SET conflicts_with = array_append(
                            array_remove(conflicts_with, %s::uuid), %s::uuid),
                        status = CASE WHEN status = 'REJECTED'
                                      THEN status ELSE 'CONFLICTING' END,
                        verified = false,
                        updated_at = now()
                    WHERE feedback_id = %s
                    """,
                    (new_id, new_id, cid),
                )
        conn.commit()
