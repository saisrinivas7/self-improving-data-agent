"""
Tests for the LangGraph state machine's structure and routing.

The point of putting the agent in a graph was to make "can this loop
forever?" answerable by reading the edges. These tests make it answerable by
running them. They check the routing functions as pure predicates, so they
need no database, no model and no network.

The node bodies are not tested here - those need services, and are covered
by the integration tests.
"""

from __future__ import annotations

import pytest

from app.agent.graph import after_execute, after_validate, build_graph
from app.agent.state import AgentState, SQLAttempt

EXPECTED_NODES = {
    "understand_question",
    "retrieve_feedback",
    "inspect_schema",
    "generate_sql",
    "validate_sql",
    "execute_sql",
    "analyze_results",
    "validate_analysis",
    "generate_answer",
    "give_up",
}


def _state(*attempts: SQLAttempt) -> AgentState:
    s = AgentState(question="why did revenue fall?")
    s.sql_attempts = list(attempts)
    return s


def fail(n: int) -> SQLAttempt:
    return SQLAttempt(attempt=n, sql="bad sql", ok=False, error="boom")


def ok(n: int) -> SQLAttempt:
    return SQLAttempt(attempt=n, sql="SELECT 1", ok=True, row_count=3)


# ---------------------------------------------------------------- structure

def test_graph_compiles() -> None:
    build_graph()


def test_all_spec_nodes_present() -> None:
    """The spec names these nodes; the graph should actually have them."""
    g = build_graph().get_graph()
    nodes = {n for n in g.nodes if not n.startswith("__")}
    assert EXPECTED_NODES <= nodes, f"missing: {EXPECTED_NODES - nodes}"


# ------------------------------------------------------------------ routing

def test_valid_sql_goes_to_execution() -> None:
    assert after_validate(_state()) == "execute_sql"
    assert after_validate(_state(ok(1))) == "execute_sql"


def test_invalid_sql_retries() -> None:
    assert after_validate(_state(fail(1))) == "generate_sql"
    assert after_validate(_state(fail(1), fail(2))) == "generate_sql"


def test_retries_are_capped() -> None:
    """The whole reason the cap lives on the edge rather than in a node."""
    assert after_validate(_state(fail(1), fail(2), fail(3))) == "give_up"
    assert after_execute(_state(fail(1), fail(2), fail(3))) == "give_up"


def test_successful_execution_proceeds_to_analysis() -> None:
    assert after_execute(_state(ok(1))) == "analyze_results"


def test_failed_execution_retries_then_gives_up() -> None:
    assert after_execute(_state(fail(1))) == "generate_sql"
    assert after_execute(_state(fail(1), fail(2))) == "generate_sql"
    assert after_execute(_state(fail(1), fail(2), fail(3))) == "give_up"


@pytest.mark.parametrize("n", range(1, 12))
def test_no_attempt_count_loops_forever(n: int) -> None:
    """Beyond the cap, every route must terminate rather than loop.

    Parameterised past the cap on purpose: an off-by-one in the comparison
    would let the graph keep regenerating SQL indefinitely, and the
    recursion limit would be the only thing stopping it.
    """
    s = _state(*[fail(i) for i in range(1, n + 1)])
    v, e = after_validate(s), after_execute(s)
    if n >= 3:
        assert v == "give_up"
        assert e == "give_up"
    else:
        assert v == "generate_sql"
        assert e == "generate_sql"


def test_mixed_history_routes_on_the_last_attempt() -> None:
    """A success after failures should proceed, not retry."""
    assert after_execute(_state(fail(1), ok(2))) == "analyze_results"
    assert after_validate(_state(fail(1), ok(2))) == "execute_sql"
