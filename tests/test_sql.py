"""
SQL validation tests (spec section 26).

The agent writes its own SQL. These tests are the first of the two defences
described in app/tools/sql_tool.py, and they are the one place where a
regression is genuinely dangerous: a parser change that starts accepting
DELETE would not break any other test in the suite.

No LLM and no network here. Pure function tests over the validator, so they
run in under a second and can be trusted in CI.
"""

from __future__ import annotations

import pytest

from app.tools.sql_tool import strip_markdown, validate_sql

# ---------------------------------------------------------------- rejected

DANGEROUS = [
    # plain writes
    ("DROP TABLE orders", "drop"),
    ("DELETE FROM orders", "delete"),
    ("UPDATE orders SET total_amount = 0", "update"),
    ("INSERT INTO orders VALUES (1)", "insert"),
    ("TRUNCATE orders", "truncate"),
    ("ALTER TABLE orders ADD COLUMN x int", "alter"),
    ("CREATE TABLE evil (id int)", "create"),
    # multiple statements - the classic injection shape
    ("SELECT 1; DROP TABLE orders", "two statements"),
    ("SELECT count(*) FROM orders; DELETE FROM refunds", "two statements"),
    # obfuscation that defeats a keyword blocklist but not a parser
    ("/**/DELETE FROM orders", "comment prefix"),
    ("dElEtE FROM orders", "mixed case"),
    ("  \n\t DELETE   FROM   orders ", "whitespace"),
    # writes hidden inside a CTE
    ("WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x", "cte delete"),
    ("WITH x AS (INSERT INTO orders VALUES (1) RETURNING *) SELECT * FROM x", "cte insert"),
    # reaching outside the business tables
    ("SELECT * FROM memory.feedback_memory", "memory schema"),
    ("SELECT * FROM memory.traces", "memory schema"),
    ("SELECT * FROM pg_shadow", "system table"),
    ("SELECT * FROM information_schema.tables", "information_schema"),
    # filesystem / side-effect functions
    ("SELECT pg_read_file('/etc/passwd')", "read file"),
    ("SELECT pg_sleep(60)", "sleep"),
    # nonsense
    ("", "empty"),
    ("not sql at all !!!", "garbage"),
]


@pytest.mark.parametrize("sql,label", DANGEROUS, ids=[l for _, l in DANGEROUS])
def test_dangerous_sql_is_rejected(sql: str, label: str) -> None:
    v = validate_sql(sql)
    assert not v.ok, f"{label}: should have been rejected but passed -> {sql!r}"
    assert v.reason, "a rejection must explain itself so the agent can retry"


# ----------------------------------------------------------------- allowed

SAFE = [
    "SELECT count(*) FROM orders",
    "SELECT status, count(*) FROM orders GROUP BY status",
    "SELECT * FROM orders LIMIT 10",
    """WITH monthly AS (
         SELECT date_trunc('month', order_date) AS m, sum(total_amount) AS rev
         FROM orders WHERE status IN ('completed','refunded') GROUP BY 1
       ) SELECT m, rev FROM monthly ORDER BY m""",
    """SELECT p.category, sum(i.quantity * i.unit_price) AS revenue
       FROM order_items i
       JOIN products p USING (product_id)
       JOIN orders o USING (order_id)
       WHERE o.status = 'completed'
       GROUP BY 1 ORDER BY revenue DESC""",
    "SELECT r.reason, count(*) FROM refunds r GROUP BY 1",
    # a string literal that merely contains a keyword must not trip it
    "SELECT 'delete' AS word FROM orders LIMIT 1",
    "SELECT 'drop table' AS s FROM products LIMIT 1",
]


@pytest.mark.parametrize("sql", SAFE, ids=[f"safe{i}" for i in range(len(SAFE))])
def test_safe_sql_is_allowed(sql: str) -> None:
    v = validate_sql(sql)
    assert v.ok, f"legitimate query rejected: {v.reason}\n{sql}"
    assert v.tables_used, "a valid query must report the tables it reads"


def test_cte_names_are_not_mistaken_for_tables() -> None:
    """A CTE alias is not a real table and must not trigger the allowlist."""
    v = validate_sql(
        "WITH totally_made_up AS (SELECT 1 AS x) SELECT x FROM totally_made_up, orders LIMIT 1"
    )
    assert v.ok, v.reason
    assert "totally_made_up" not in v.tables_used


def test_tables_used_is_reported() -> None:
    v = validate_sql(
        "SELECT 1 FROM orders o JOIN refunds r USING (order_id) "
        "JOIN customers c USING (customer_id)"
    )
    assert v.ok, v.reason
    assert set(v.tables_used) == {"orders", "refunds", "customers"}


# ------------------------------------------------------------ markdown

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("```sql\nSELECT 1 FROM orders\n```", "SELECT 1 FROM orders"),
        ("```\nSELECT 1 FROM orders\n```", "SELECT 1 FROM orders"),
        ("SELECT 1 FROM orders;", "SELECT 1 FROM orders"),
        ("  SELECT 1 FROM orders  ", "SELECT 1 FROM orders"),
    ],
)
def test_markdown_fences_are_stripped(raw: str, expected: str) -> None:
    """Small models add ``` fences however firmly the prompt forbids it."""
    assert strip_markdown(raw) == expected


def test_fenced_sql_still_validates() -> None:
    v = validate_sql("```sql\nSELECT count(*) FROM orders\n```")
    assert v.ok, v.reason
