"""
The one entry point everything uses to answer a question.

Picks which of the three systems to run, and that choice is the ONLY
difference between them:

    baseline           no retrieval at all. The control.
    feedback_rag       retrieve anything stored (PENDING or VERIFIED).
                       The naive version - it will happily use a lesson
                       nobody checked.
    verified_feedback  retrieve VERIFIED only, re-ranked by confidence.
                       REJECTED and CONFLICTING lessons cannot reach it.

Everything downstream - prompts, SQL generation, retries, analysis, the
answer step - is identical across all three. That is deliberate and it is
what makes the comparison mean anything: if the numbers differ, the cause
is the memory, not a prompt someone tweaked.
"""

from __future__ import annotations

from app.agent.baseline import run_agent
from app.agent.state import AgentState, SystemVariant
from app.config import get_settings
from app.feedback import memory as fm
from app.llm import LLMClient
from app.observability.tracing import save_trace

# Which stored statuses each system is allowed to see.
STATUS_POLICY: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "feedback_rag": ("PENDING", "VERIFIED"),
    "verified_feedback": ("VERIFIED",),
}


def answer_question(
    question: str,
    *,
    variant: SystemVariant = "baseline",
    snapshot: str = "live",
    llm: LLMClient | None = None,
    save: bool = True,
    verbose: bool = False,
) -> AgentState:
    """Answer one question with the chosen system."""
    s = get_settings()
    own = llm is None
    llm = llm or LLMClient()

    try:
        lesson_texts: list[str] = []
        extra_tables: list[str] = []
        retrieved: list[fm.StoredLesson] = []

        statuses = STATUS_POLICY.get(variant, ())
        if statuses:
            retrieved = fm.retrieve_lessons(
                question,
                snapshot=snapshot,
                include_statuses=statuses,
                llm=llm,
            )
            lesson_texts = [l.prompt_text() for l in retrieved]
            # Make the lessons actionable: a lesson about refunds is inert
            # unless the refunds table is in the schema the agent sees.
            for l in retrieved:
                for t in l.schema_context or []:
                    if t not in extra_tables:
                        extra_tables.append(t)
            if verbose and retrieved:
                print(f"  retrieved {len(retrieved)} lesson(s):")
                for l in retrieved:
                    print(f"    [{l.similarity:.2f} {l.status}] {l.lesson[:80]}")

        state = run_agent(
            question,
            llm=llm,
            lessons=lesson_texts,
            extra_tables=extra_tables,
            system_variant=variant,
            verbose=verbose,
        )
        state.snapshot = snapshot
        state.retrieved_lessons = fm.to_retrieved(retrieved, used=True)

        # Usage counters, for the feedback-utilisation metric.
        for l in retrieved:
            fm.bump_usage(l.feedback_id, retrieved=True, applied=True)

        if save:
            save_trace(state)
        return state
    finally:
        if own:
            llm.close()
