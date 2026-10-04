"""
The Learning Lab.

    make ui      ->  http://localhost:8501

What you can do here:

  1. Ask an analytical question, choosing which of the three systems answers.
  2. See the SQL it wrote, the rows it got, the numbers computed from them,
     and which stored lessons (if any) it retrieved.
  3. Accept or reject the answer, and type a correction in plain English.
  4. Watch that correction get classified into a reusable lesson, and see
     whether it is stored.
  5. Browse everything in memory, and delete anything.
  6. Ask a related question and see whether the lesson comes back and
     changes the answer.

Deliberately plain. The spec says not to over-polish the UI, and the
interesting part of this project is the feedback pipeline, not the styling.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.agent.runner import answer_question  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.feedback import memory as fm  # noqa: E402
from app.feedback.processor import process_feedback  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.observability.tracing import get_trace, recent_traces  # noqa: E402

st.set_page_config(page_title="Learning Lab", layout="wide")

SAMPLE_QUESTIONS = [
    "Why did revenue decrease in March 2026?",
    "What were the main drivers of the March 2026 revenue decline?",
    "What was total revenue each month?",
    "Which product categories have the highest refund rates?",
    "Which products have high sales but poor margins?",
    "Which categories have unusually high support ticket volume?",
    "Did the June 2026 promotions increase revenue or just reduce margin?",
    "Which customer segments are most likely to churn?",
    "Why did revenue fall in July 2026?",
]

VARIANT_HELP = {
    "baseline": "No memory at all. The control condition.",
    "feedback_rag": "Uses any stored lesson, verified or not. The naive version.",
    "verified_feedback": "Uses only VERIFIED lessons, re-ranked by confidence.",
}


# --------------------------------------------------------------- sidebar

def sidebar() -> tuple[str, str]:
    s = get_settings()
    st.sidebar.title("Learning Lab")

    variant = st.sidebar.radio(
        "Which system answers?",
        ["baseline", "feedback_rag", "verified_feedback"],
        index=1,
        format_func=lambda v: {
            "baseline": "1. Baseline (no memory)",
            "feedback_rag": "2. Feedback RAG (any lesson)",
            "verified_feedback": "3. Verified feedback only",
        }[v],
    )
    st.sidebar.caption(VARIANT_HELP[variant])

    snapshot = st.sidebar.selectbox(
        "Memory snapshot", ["live", "clean", "poisoned", "mixed"], index=0,
        help="'live' is your own memory and starts empty. The others are for benchmarks.",
    )

    stats = fm.memory_stats(snapshot)
    st.sidebar.metric(f"Lessons in '{snapshot}'", stats["total"])
    if stats["by_status"]:
        st.sidebar.write(
            " · ".join(f"{k} {v}" for k, v in sorted(stats["by_status"].items()))
        )

    st.sidebar.divider()
    st.sidebar.caption(
        f"model: {s.chat_model}\n\n"
        f"embeddings: {s.embedding_model} ({s.embedding_dim}d)\n\n"
        f"provider: {s.llm_provider}"
    )
    return variant, snapshot


# ------------------------------------------------------------------ ask

def render_ask(variant: str, snapshot: str) -> None:
    st.subheader("Ask a question")

    picked = st.selectbox(
        "Examples", ["(type my own)"] + SAMPLE_QUESTIONS, index=1
    )
    default = "" if picked == "(type my own)" else picked
    question = st.text_area("Question", value=default, height=70)

    if st.button("Analyse", type="primary", disabled=not question.strip()):
        with st.spinner("Thinking - the local model takes ~20-60s per question..."):
            try:
                state = answer_question(
                    question.strip(), variant=variant, snapshot=snapshot  # type: ignore[arg-type]
                )
                st.session_state["state"] = state
                st.session_state.pop("processed", None)
                st.session_state.pop("stored_id", None)
            except Exception as e:  # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")
                return

    state = st.session_state.get("state")
    if state is None:
        st.info("Ask something to begin. Memory starts empty, so the first answer has no lessons to draw on.")
        return

    # ---- the answer ----
    if state.status == "failed":
        st.error(state.final_answer)
    else:
        st.markdown("### Answer")
        st.success(state.final_answer)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SQL attempts", state.sql_attempt_count)
    c2.metric("Rows", len(state.sql_result_rows))
    c3.metric("Latency", f"{state.latency_s:.0f}s")
    c4.metric("Tokens", f"{state.prompt_tokens + state.output_tokens:,}")

    # ---- lessons it retrieved ----
    st.markdown("### Lessons retrieved")
    if not state.retrieved_lessons:
        if variant == "baseline":
            st.caption("Baseline never retrieves. That is the point of the control condition.")
        else:
            st.caption("Nothing retrieved - memory is empty, or nothing was similar enough.")
    else:
        for l in state.retrieved_lessons:
            st.markdown(
                f"- **{l.status}** · similarity `{l.similarity:.2f}` · "
                f"confidence `{l.confidence:.2f}` · {l.feedback_type}  \n  {l.lesson}"
            )

    # ---- the working ----
    with st.expander("SQL the agent wrote", expanded=True):
        st.code(state.generated_sql or "(none)", language="sql")
        if state.sql_attempt_count > 1:
            st.caption(f"Took {state.sql_attempt_count} attempts:")
            for a in state.sql_attempts:
                if not a.ok:
                    st.error(f"attempt {a.attempt}: {a.error}")

    with st.expander("Data it got back"):
        if state.sql_result_rows:
            st.dataframe(pd.DataFrame(state.sql_result_rows), use_container_width=True)
        else:
            st.caption("no rows")

    with st.expander("Computed analysis"):
        st.text(state.analysis_text or "(none)")

    with st.expander("Schema the agent was shown"):
        st.caption(f"tables: {', '.join(state.tables_used)}")
        st.text(state.schema_text)

    with st.expander("Trace"):
        tr = get_trace(state.trace_id)
        if tr:
            st.caption(f"trace {state.trace_id}")
            for ev in tr["events"]:
                icon = "✗" if ev["error"] else "·"
                st.markdown(f"**{icon} {ev['seq']}. {ev['node']}**")
                if ev["error"]:
                    st.error(ev["error"])
        else:
            st.caption("not saved")

    render_feedback(state)


# ------------------------------------------------------------- feedback

def render_feedback(state) -> None:  # noqa: ANN001
    st.divider()
    st.subheader("Was this analysis useful?")

    verdict = st.radio(
        "Verdict", ["Yes - it was right", "No - it was wrong or incomplete"],
        index=1, horizontal=True, label_visibility="collapsed",
    )
    rejected = verdict.startswith("No")

    feedback = st.text_area(
        "Your correction, in plain English",
        placeholder=(
            "e.g. You only looked at March in isolation. Compare it to February, "
            "and check refunds too - refund volume rose sharply in March."
        ),
        height=100,
    )

    st.caption(
        "Try giving deliberately WRONG feedback too, such as "
        "\"revenue declines are always caused by refunds\", and watch what the "
        "system does with it."
    )

    if st.button("Submit feedback", type="primary", disabled=not feedback.strip()):
        with st.spinner("Classifying the feedback..."):
            try:
                with LLMClient() as llm:
                    p = process_feedback(
                        feedback=feedback.strip(),
                        question=state.question,
                        sql=state.generated_sql,
                        answer=state.final_answer,
                        trace_id=state.trace_id,
                        llm=llm,
                    )
                    st.session_state["processed"] = p
                    # No verification yet - that is Phase 7. For now the
                    # lesson is stored as PENDING, which is honest: it has
                    # been recorded, not validated.
                    fid = fm.store_lesson(p, status="PENDING", llm=llm)
                    st.session_state["stored_id"] = fid
            except Exception as e:  # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")
                return

    p = st.session_state.get("processed")
    if p is None:
        return

    st.markdown("### How your feedback was understood")
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Type**  \n`{p.feedback_type}`")
        st.markdown(f"**Clarity confidence**  \n`{p.confidence:.2f}`")
        st.markdown(f"**Tables it concerns**  \n`{', '.join(p.schema_context) or 'none'}`")
    with b:
        st.markdown(f"**What it thinks went wrong**  \n{p.mistake}")
        st.markdown(f"**What you said to do instead**  \n{p.correction}")

    st.markdown("**The reusable lesson it extracted** - this is what gets stored and retrieved later:")
    st.info(p.lesson)

    if p.claims:
        st.markdown("**Checkable statements it found in your feedback:**")
        for c in p.claims:
            tag = {"empirical": "can be checked against the data",
                   "causal": "asserts a cause",
                   "procedural": "advice - not true or false"}.get(c.kind, c.kind)
            uni = " · claims this ALWAYS holds" if c.is_universal else ""
            st.markdown(f"- `{c.kind}` ({tag}){uni}  \n  {c.text}")

    fid = st.session_state.get("stored_id")
    if fid:
        st.success(f"Stored as **PENDING** · id `{fid[:8]}`")
        st.caption(
            "PENDING, not VERIFIED: nothing has checked whether your claim is "
            "actually true yet. Verification is the next thing being built - "
            "until then this memory accepts anything, which is exactly the "
            "failure mode the project is about."
        )


# -------------------------------------------------------------- memory

def render_memory(snapshot: str) -> None:
    st.subheader(f"Stored lessons - snapshot '{snapshot}'")
    rows = fm.list_lessons(snapshot)
    if not rows:
        st.info("Memory is empty. Give some feedback on the Ask tab and it will appear here.")
        return

    for r in rows:
        head = (
            f"{r['status']} · {r['feedback_type']} · "
            f"confidence {float(r['confidence']):.2f} · "
            f"used {r['times_applied']}x"
        )
        with st.expander(head):
            st.markdown(f"**Lesson**  \n{r['lesson']}")
            st.markdown(f"**From the question**  \n{r['original_question']}")
            if r["raw_feedback"]:
                st.markdown(f"**What you actually typed**  \n_{r['raw_feedback']}_")
            st.markdown(f"**Mistake it was correcting**  \n{r['mistake']}")
            st.markdown(f"**Correction**  \n{r['correction']}")
            st.caption(
                f"tables: {', '.join(r['schema_context'] or []) or 'none'} · "
                f"retrieved {r['times_retrieved']}x · "
                f"created {r['created_at']:%Y-%m-%d %H:%M} · id {str(r['feedback_id'])[:8]}"
            )
            if r["verification_notes"]:
                st.json(r["verification_notes"])
            if st.button("Delete this lesson", key=f"del-{r['feedback_id']}"):
                fm.delete_lesson(str(r["feedback_id"]))
                st.rerun()


# -------------------------------------------------------------- history

def render_history() -> None:
    st.subheader("Recent runs")
    rows = recent_traces(40)
    if not rows:
        st.info("No runs yet.")
        return
    st.dataframe(
        pd.DataFrame(rows)[
            ["created_at", "system_variant", "status", "question",
             "sql_attempts", "latency_s"]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ----------------------------------------------------------------- main

variant, snapshot = sidebar()
tab_ask, tab_mem, tab_hist = st.tabs(["Ask & correct", "Stored lessons", "History"])
with tab_ask:
    render_ask(variant, snapshot)
with tab_mem:
    render_memory(snapshot)
with tab_hist:
    render_history()
