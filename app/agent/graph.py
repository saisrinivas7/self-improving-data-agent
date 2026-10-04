"""
The agent as an explicit LangGraph state machine (spec sections 6 and 8).

    START
      -> understand_question
      -> retrieve_feedback
      -> inspect_schema
      -> generate_sql
      -> validate_sql ---(invalid, under retry cap)---> generate_sql
      -> execute_sql  ---(error, under retry cap)-----> generate_sql
      -> analyze_results
      -> validate_analysis
      -> generate_answer
      -> END

WHY A GRAPH AT ALL, GIVEN THE PLAIN VERSION WORKS

app/agent/baseline.py does the same thing as a sequence of function calls,
and for a single happy path that is simpler. The graph earns its keep in
three places:

  1. The retry loop becomes a declared edge with a cap, rather than a `for`
     loop whose bound is easy to get wrong. "Can this agent loop forever?"
     is answerable by reading the edges.
  2. Every node boundary is a natural trace point, which is what the
     observability requirement wants.
  3. Validation becomes its own node rather than an `if` inside another
     step, so "the SQL was rejected before it ran" is visible in the trace
     as a distinct event.

The node bodies deliberately delegate to the same tools the plain pipeline
uses. The two share all their logic, so they cannot drift apart and produce
different answers.
"""

from __future__ import annotations

import time
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agent import prompts
from app.agent.state import AgentState, SQLAttempt
from app.config import get_settings
from app.llm import LLMClient
from app.tools.python_analysis_tool import analyse
from app.tools.schema_tool import get_schema_context
from app.tools.sql_tool import format_rows, run_sql, validate_sql


# Mutable per-run context that is not part of the scored state: the LLM
# client, and the retrieved lesson text. Kept off AgentState so the state
# stays serialisable for tracing.
class _Ctx:
    def __init__(
        self,
        llm: LLMClient,
        lessons: list[str] | None = None,
        extra_tables: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.feedback_block = prompts.render_feedback_block(lessons or [])
        self.extra_tables = extra_tables or []
        self.t0 = time.perf_counter()


_ctx: _Ctx | None = None


def _c() -> _Ctx:
    if _ctx is None:
        raise RuntimeError("graph run context not set; call run_graph()")
    return _ctx


# ---------------------------------------------------------------- the nodes

def understand_question(state: AgentState) -> AgentState:
    """Record the question and set up bookkeeping.

    No LLM call. An earlier design had the model restate the question as a
    "plan", which cost a 20-second round trip and produced nothing the later
    nodes used. The node is kept because it is the graph's entry point and a
    natural place for a trace event, and because question rewriting is a
    plausible future improvement.
    """
    s = get_settings()
    state.chat_model = s.chat_model
    state.embedding_model = s.embedding_model
    return state


def retrieve_feedback(state: AgentState) -> AgentState:
    """Lessons are retrieved by the caller, before the graph starts.

    Retrieval needs the embedding model and the snapshot policy, which is
    the runner's job. The node exists so the step appears in the graph and
    the trace even for the baseline, where it is a no-op.
    """
    return state


def inspect_schema(state: AgentState) -> AgentState:
    state.schema_text, state.tables_used = get_schema_context(
        state.question, extra_tables=_c().extra_tables
    )
    state.tool_calls += 1
    return state


def generate_sql(state: AgentState) -> AgentState:
    ctx = _c()
    if not state.sql_attempts:
        user = prompts.SQL_USER.format(
            question=state.question,
            schema=state.schema_text,
            feedback_block=ctx.feedback_block,
        )
    else:
        last = state.sql_attempts[-1]
        user = prompts.SQL_RETRY.format(
            question=state.question,
            schema=state.schema_text,
            feedback_block=ctx.feedback_block,
            failed_sql=last.sql,
            error=last.error,
        )
    resp = ctx.llm.complete(user, system=prompts.SQL_SYSTEM, max_tokens=900)
    state.record_llm(resp)
    state.generated_sql = resp.text.strip()
    return state


def validate_sql_node(state: AgentState) -> AgentState:
    """Reject unsafe or malformed SQL before it reaches the database.

    A separate node so a rejection is its own trace event. When it fails,
    the error is written as an attempt so generate_sql can see what went
    wrong and fix it.
    """
    v = validate_sql(state.generated_sql)
    if not v.ok:
        state.sql_attempts.append(
            SQLAttempt(
                attempt=len(state.sql_attempts) + 1,
                sql=state.generated_sql,
                ok=False,
                error=f"rejected by validation: {v.reason}",
            )
        )
        state.errors.append(f"sql validation: {v.reason}")
    return state


def execute_sql(state: AgentState) -> AgentState:
    result = run_sql(state.generated_sql)
    state.tool_calls += 1
    state.sql_attempts.append(
        SQLAttempt(
            attempt=len(state.sql_attempts) + 1,
            sql=result.sql or state.generated_sql,
            ok=result.ok,
            error=result.error,
            row_count=result.row_count,
            latency_s=result.latency_s,
        )
    )
    if result.ok:
        state.generated_sql = result.sql
        state.sql_result_rows = result.rows
        state.sql_result_table = format_rows(result.rows)
        if result.truncated:
            state.errors.append("result truncated at the row limit")
    else:
        state.errors.append(f"sql execution: {result.error}")
    return state


def analyze_results(state: AgentState) -> AgentState:
    state.analysis_text, findings = analyse(state.sql_result_rows, question=state.question)
    state.analysis_findings = [
        {"kind": f.kind, "text": f.text, "values": f.values} for f in findings
    ]
    state.tool_calls += 1
    return state


def validate_analysis(state: AgentState) -> AgentState:
    """Catch the case where there is nothing to report.

    Cheap and deterministic: no rows, or no findings, means the answer node
    should say so rather than inventing a conclusion. This is where a
    richer check (does the analysis actually address the question?) would
    go.
    """
    if not state.sql_result_rows:
        state.errors.append("query returned no rows")
    return state


def generate_answer(state: AgentState) -> AgentState:
    ctx = _c()
    resp = ctx.llm.complete(
        prompts.ANSWER_USER.format(
            question=state.question,
            result_table=state.sql_result_table[:4000],
            analysis=state.analysis_text,
            feedback_block=ctx.feedback_block,
        ),
        system=prompts.ANSWER_SYSTEM,
        max_tokens=500,
    )
    state.record_llm(resp)
    state.final_answer = resp.text.strip()
    state.status = "completed"
    return state


def give_up(state: AgentState) -> AgentState:
    """Terminal node for exhausted retries."""
    s = get_settings()
    last = state.sql_attempts[-1].error if state.sql_attempts else "unknown"
    state.status = "failed"
    state.final_answer = (
        f"I could not produce a working query after {s.max_sql_attempts} "
        f"attempts. Last error: {last}"
    )
    return state


# ------------------------------------------------------------------- edges

def after_validate(state: AgentState) -> Literal["execute_sql", "generate_sql", "give_up"]:
    """Route on validation, honouring the retry cap.

    The cap is checked here rather than inside a node, so the bound on
    looping is a property of the graph instead of an implementation detail.
    """
    s = get_settings()
    last_failed = bool(state.sql_attempts) and not state.sql_attempts[-1].ok
    if not last_failed:
        return "execute_sql"
    if len(state.sql_attempts) >= s.max_sql_attempts:
        return "give_up"
    return "generate_sql"


def after_execute(state: AgentState) -> Literal["analyze_results", "generate_sql", "give_up"]:
    s = get_settings()
    if state.sql_attempts and state.sql_attempts[-1].ok:
        return "analyze_results"
    if len(state.sql_attempts) >= s.max_sql_attempts:
        return "give_up"
    return "generate_sql"


# ------------------------------------------------------------------- build

def build_graph():  # noqa: ANN201
    """Compile the state machine. Structure only - no services touched."""
    g = StateGraph(AgentState)

    g.add_node("understand_question", understand_question)
    g.add_node("retrieve_feedback", retrieve_feedback)
    g.add_node("inspect_schema", inspect_schema)
    g.add_node("generate_sql", generate_sql)
    g.add_node("validate_sql", validate_sql_node)
    g.add_node("execute_sql", execute_sql)
    g.add_node("analyze_results", analyze_results)
    g.add_node("validate_analysis", validate_analysis)
    g.add_node("generate_answer", generate_answer)
    g.add_node("give_up", give_up)

    g.add_edge(START, "understand_question")
    g.add_edge("understand_question", "retrieve_feedback")
    g.add_edge("retrieve_feedback", "inspect_schema")
    g.add_edge("inspect_schema", "generate_sql")
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges("validate_sql", after_validate)
    g.add_conditional_edges("execute_sql", after_execute)
    g.add_edge("analyze_results", "validate_analysis")
    g.add_edge("validate_analysis", "generate_answer")
    g.add_edge("generate_answer", END)
    g.add_edge("give_up", END)

    return g.compile()


_compiled = None


def get_graph():  # noqa: ANN201
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run_graph(
    question: str,
    *,
    llm: LLMClient,
    lessons: list[str] | None = None,
    extra_tables: list[str] | None = None,
    system_variant: str = "baseline",
) -> AgentState:
    """Run one question through the graph."""
    global _ctx
    state = AgentState(question=question, system_variant=system_variant)  # type: ignore[arg-type]
    _ctx = _Ctx(llm, lessons, extra_tables)
    t0 = time.perf_counter()
    try:
        # The recursion limit is a hard backstop in case a routing bug
        # creates a cycle the retry cap does not catch.
        result = get_graph().invoke(state, {"recursion_limit": 25})
        out = result if isinstance(result, AgentState) else AgentState(**result)
    except Exception as e:  # noqa: BLE001
        state.status = "failed"
        state.errors.append(f"{type(e).__name__}: {e}")
        state.final_answer = f"The run failed: {type(e).__name__}: {e}"
        out = state
    finally:
        _ctx = None
    out.latency_s = time.perf_counter() - t0
    return out
