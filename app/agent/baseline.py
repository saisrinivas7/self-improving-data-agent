"""
The baseline agent (spec section 15).

    question -> schema -> SQL -> execute -> analysis -> answer

No feedback retrieval. This is the control condition: whatever the other two
systems score, this is what they have to beat, and the difference is the
only evidence that feedback memory does anything.

Written as a plain sequence of function calls rather than a graph. Phase 3
wraps these same steps as LangGraph nodes; keeping the logic here means the
graph conversion is wiring, not a rewrite, and the two cannot drift apart.

The `lessons` argument is the single seam for the later systems. The baseline
passes nothing, which makes its prompts byte-identical to a version of this
file with no feedback feature at all.
"""

from __future__ import annotations

import time

from app.agent import prompts
from app.agent.state import AgentState, SQLAttempt
from app.config import get_settings
from app.llm import LLMClient
from app.tools.python_analysis_tool import analyse
from app.tools.schema_tool import get_schema_context
from app.tools.sql_tool import format_rows, run_sql


def run_agent(
    question: str,
    *,
    llm: LLMClient | None = None,
    lessons: list[str] | None = None,
    extra_tables: list[str] | None = None,
    system_variant: str = "baseline",
    verbose: bool = False,
) -> AgentState:
    """Answer one question. Always returns a state, even on failure."""
    s = get_settings()
    state = AgentState(question=question, system_variant=system_variant)  # type: ignore[arg-type]
    state.chat_model = s.chat_model
    state.embedding_model = s.embedding_model
    feedback_block = prompts.render_feedback_block(lessons or [])

    own_client = llm is None
    llm = llm or LLMClient()
    t_start = time.perf_counter()

    def log(msg: str) -> None:
        if verbose:
            print(f"  [{time.perf_counter() - t_start:6.1f}s] {msg}")

    try:
        # ---------------------------------------------------- 1. schema
        log("inspecting schema")
        state.schema_text, state.tables_used = get_schema_context(
            question, extra_tables=extra_tables
        )
        state.tool_calls += 1
        log(f"tables: {', '.join(state.tables_used)}")

        # ------------------------------------- 2-3. generate + run SQL
        # Bounded retry: on failure the error text goes back to the model so
        # it can fix its own mistake. Capped at max_sql_attempts so a model
        # that cannot get it right fails fast instead of looping.
        result = None
        for attempt in range(1, s.max_sql_attempts + 1):
            if attempt == 1:
                user = prompts.SQL_USER.format(
                    question=question,
                    schema=state.schema_text,
                    feedback_block=feedback_block,
                )
            else:
                last = state.sql_attempts[-1]
                user = prompts.SQL_RETRY.format(
                    question=question,
                    schema=state.schema_text,
                    feedback_block=feedback_block,
                    failed_sql=last.sql,
                    error=last.error,
                )

            log(f"generating SQL (attempt {attempt}/{s.max_sql_attempts})")
            resp = llm.complete(user, system=prompts.SQL_SYSTEM, max_tokens=900)
            state.record_llm(resp)
            sql = resp.text.strip()

            log("running SQL")
            result = run_sql(sql)
            state.tool_calls += 1
            state.sql_attempts.append(
                SQLAttempt(
                    attempt=attempt,
                    sql=result.sql or sql,
                    ok=result.ok,
                    error=result.error,
                    row_count=result.row_count,
                    latency_s=result.latency_s,
                )
            )

            if result.ok:
                log(f"ok: {result.row_count} rows")
                break
            log(f"failed: {result.error[:90]}")
            state.errors.append(f"sql attempt {attempt}: {result.error}")

        if result is None or not result.ok:
            state.status = "failed"
            state.final_answer = (
                f"I could not produce a working query after "
                f"{s.max_sql_attempts} attempts. Last error: "
                f"{result.error if result else 'unknown'}"
            )
            return state

        state.generated_sql = result.sql
        state.sql_result_rows = result.rows
        state.sql_result_table = format_rows(result.rows)
        if result.truncated:
            state.errors.append(f"result truncated at {s.sql_row_limit} rows")

        # ------------------------------------------------- 4. analysis
        log("analysing results")
        state.analysis_text, findings = analyse(result.rows, question=question)
        state.analysis_findings = [
            {"kind": f.kind, "text": f.text, "values": f.values} for f in findings
        ]
        state.tool_calls += 1

        # --------------------------------------------------- 5. answer
        log("writing answer")
        resp = llm.complete(
            prompts.ANSWER_USER.format(
                question=question,
                result_table=state.sql_result_table[:4000],
                analysis=state.analysis_text,
                feedback_block=feedback_block,
            ),
            system=prompts.ANSWER_SYSTEM,
            max_tokens=500,
        )
        state.record_llm(resp)
        state.final_answer = resp.text.strip()
        state.status = "completed"

    except Exception as e:  # noqa: BLE001
        state.status = "failed"
        state.errors.append(f"{type(e).__name__}: {e}")
        state.final_answer = f"The run failed: {type(e).__name__}: {e}"
    finally:
        state.latency_s = time.perf_counter() - t_start
        if own_client:
            llm.close()

    return state
