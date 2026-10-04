"""
Trace storage (spec section 23).

Saves one row per agent run, plus a child row per step, into the `memory`
schema. Deliberately plain: two tables in the Postgres we already run.

Why it matters beyond debugging: when an analyst gives feedback in the
Learning Lab, they are correcting a specific run. The stored trace is what
lets us attach that feedback to the exact SQL and answer it was about, so a
lesson carries real provenance instead of just a timestamp.
"""

from __future__ import annotations

import json
from typing import Any

from app.agent.state import AgentState
from app.database.connection import rw_connection


def _j(v: Any) -> str:
    return json.dumps(v, default=str)


def save_trace(state: AgentState) -> str:
    """Persist a finished run. Returns the trace id."""
    s = state.summary()
    with rw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory.traces (
                    trace_id, question, system_variant, snapshot,
                    final_answer, generated_sql, status, error,
                    sql_attempts, tool_calls, llm_calls,
                    prompt_tokens, output_tokens, thinking_tokens, latency_s,
                    chat_model, embedding_model,
                    feedback_retrieved, feedback_used, feedback_rejected
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (trace_id) DO UPDATE SET
                    final_answer = EXCLUDED.final_answer,
                    generated_sql = EXCLUDED.generated_sql,
                    status = EXCLUDED.status,
                    latency_s = EXCLUDED.latency_s
                """,
                (
                    state.trace_id, state.question, state.system_variant, state.snapshot,
                    state.final_answer, state.generated_sql, state.status,
                    "; ".join(state.errors)[:4000] or None,
                    state.sql_attempt_count, state.tool_calls, state.llm_calls,
                    state.prompt_tokens, state.output_tokens, state.thinking_tokens,
                    round(state.latency_s, 3),
                    state.chat_model, state.embedding_model,
                    [l.feedback_id for l in state.retrieved_lessons],
                    [l.feedback_id for l in state.lessons_used],
                    [l.feedback_id for l in state.lessons_rejected],
                ),
            )

            # ---- step events, in order ----
            seq = 0
            events: list[tuple[str, dict, dict, str, float]] = []

            events.append((
                "inspect_schema",
                {"question": state.question},
                {"tables": state.tables_used},
                "", 0.0,
            ))

            if state.retrieved_lessons:
                events.append((
                    "retrieve_feedback",
                    {"question": state.question},
                    {
                        "retrieved": [
                            {
                                "id": l.feedback_id, "similarity": round(l.similarity, 4),
                                "status": l.status, "confidence": l.confidence,
                                "used": l.used, "lesson": l.lesson[:300],
                                "rejected_reason": l.rejected_reason,
                            }
                            for l in state.retrieved_lessons
                        ]
                    },
                    "", 0.0,
                ))

            for a in state.sql_attempts:
                events.append((
                    f"generate_sql" if a.ok else "generate_sql (failed)",
                    {"attempt": a.attempt},
                    {"sql": a.sql, "rows": a.row_count},
                    a.error, a.latency_s,
                ))

            if state.analysis_findings:
                events.append((
                    "analyze_results",
                    {"rows": len(state.sql_result_rows)},
                    {"findings": state.analysis_findings},
                    "", 0.0,
                ))

            if state.final_answer:
                events.append((
                    "generate_answer",
                    {},
                    {"answer": state.final_answer},
                    "", 0.0,
                ))

            for node, inp, outp, err, lat in events:
                seq += 1
                cur.execute(
                    """
                    INSERT INTO memory.trace_events
                        (trace_id, seq, node, input, output, error, latency_s)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trace_id, seq) DO NOTHING
                    """,
                    (state.trace_id, seq, node, _j(inp), _j(outp), err or None, lat),
                )
        conn.commit()
    return state.trace_id


def get_trace(trace_id: str) -> dict | None:
    """Fetch a run and its steps, for the UI and the /trace endpoint."""
    from psycopg.rows import dict_row

    with rw_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM memory.traces WHERE trace_id = %s", (trace_id,))
            trace = cur.fetchone()
            if not trace:
                return None
            cur.execute(
                "SELECT * FROM memory.trace_events WHERE trace_id = %s ORDER BY seq",
                (trace_id,),
            )
            trace["events"] = cur.fetchall()
            return dict(trace)


def recent_traces(limit: int = 25) -> list[dict]:
    from psycopg.rows import dict_row

    with rw_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT trace_id, question, system_variant, status,
                       latency_s, sql_attempts, created_at
                FROM memory.traces ORDER BY created_at DESC LIMIT %s
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
