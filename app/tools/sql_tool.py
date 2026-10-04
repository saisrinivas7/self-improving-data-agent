"""
Tool 2 - SQL Tool (spec section 10).

Validates agent-written SQL, then runs it.

THREE LAYERS OF PROTECTION, because the SQL comes from a language model:

  1. Parse it. sqlglot turns the text into a syntax tree, and we inspect
     that tree. A keyword blocklist is the obvious approach and the wrong
     one: comments, string literals, casing and whitespace all defeat it
     ("/**/DELETE", "dElEtE", "SELECT 'delete'"). A parser is not fooled by
     any of those, and it also rejects SQL that is simply malformed before
     we waste a database round trip.

  2. Run it as agent_ro. That role's transactions are read-only at the
     engine level, so even a statement that slipped past the parser cannot
     write. It also cannot see the memory schema.

  3. Bound it. One statement only, a row limit, and a statement timeout, so
     a cartesian join cannot hang the agent.

Layers 1 and 2 are independent on purpose. Layer 1 can be wrong; layer 2
cannot be talked around.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

from app.config import get_settings

# NOTE: app.database.connection is imported lazily inside run_sql, not here.
# Validation is a pure function over a string and must stay importable
# without a database driver, so the safety tests can run anywhere - including
# in CI, which has no Postgres. A module-level import of psycopg would couple
# the validator to the database and break exactly that.

# Statement types that are never acceptable. Checked against the parsed
# tree, not the raw text.
FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Grant, exp.Merge,
)

# Tables the agent may read. Anything else - the memory schema, Postgres
# catalogs holding other people's data - is refused before execution.
ALLOWED_TABLES = {
    "customers", "products", "orders", "order_items",
    "refunds", "support_tickets", "promotions",
}


@dataclass
class SQLValidation:
    ok: bool
    reason: str = ""
    normalised_sql: str = ""
    tables_used: list[str] = field(default_factory=list)


@dataclass
class SQLResult:
    ok: bool
    sql: str
    rows: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    latency_s: float = 0.0
    error: str = ""


def strip_markdown(sql: str) -> str:
    """Remove ```sql fences. Small models add them however firmly told not to."""
    s = sql.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip().rstrip(";").strip()


def validate_sql(sql: str) -> SQLValidation:
    """Parse and inspect. Returns a reason the agent can act on when invalid."""
    sql = strip_markdown(sql)
    if not sql:
        return SQLValidation(False, "empty SQL")

    # One statement only. Splitting with the parser handles semicolons
    # inside string literals correctly, which a str.split(';') would not.
    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception as e:  # noqa: BLE001
        return SQLValidation(False, f"could not parse SQL: {e}")

    statements = [s for s in statements if s is not None]
    if len(statements) == 0:
        return SQLValidation(False, "no statement found")
    if len(statements) > 1:
        return SQLValidation(
            False, f"{len(statements)} statements found; exactly one SELECT is allowed"
        )

    tree = statements[0]

    # Must be a SELECT (a WITH ... SELECT is fine).
    if not isinstance(tree, (exp.Select, exp.Union, exp.With, exp.Subquery)):
        return SQLValidation(
            False, f"only SELECT is allowed, got {type(tree).__name__.upper()}"
        )

    # No write node anywhere in the tree, including inside CTEs.
    for node_type in FORBIDDEN_NODES:
        found = list(tree.find_all(node_type))
        if found:
            return SQLValidation(
                False, f"{node_type.__name__.upper()} is not allowed"
            )

    # Functions that write or read the filesystem.
    for fn in tree.find_all(exp.Anonymous):
        nm = (fn.name or "").lower()
        if nm in {"pg_read_file", "pg_ls_dir", "pg_read_binary_file", "lo_import",
                  "lo_export", "dblink", "pg_sleep", "copy"}:
            return SQLValidation(False, f"function {nm}() is not allowed")

    # Only the business tables.
    tables_used: list[str] = []
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        name = (t.name or "").lower()
        schema = (t.db or "").lower()
        if not name or name in cte_names:
            continue
        if schema and schema != "public":
            return SQLValidation(False, f"schema '{schema}' is not accessible")
        if name not in ALLOWED_TABLES:
            return SQLValidation(
                False,
                f"table '{name}' is not available. Readable tables: "
                + ", ".join(sorted(ALLOWED_TABLES)),
            )
        if name not in tables_used:
            tables_used.append(name)

    if not tables_used:
        return SQLValidation(False, "query does not read any known table")

    return SQLValidation(
        True, normalised_sql=tree.sql(dialect="postgres", pretty=True),
        tables_used=tables_used,
    )


def _add_limit(sql: str, limit: int) -> tuple[str, bool]:
    """Append a LIMIT when the query has none.

    Asks for limit+1 rows so we can tell "exactly at the limit" from
    "there was more", and report truncation honestly instead of letting the
    agent analyse a silently clipped result.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:  # noqa: BLE001
        return sql, False
    if tree.args.get("limit") is not None:
        return sql, False
    return tree.limit(limit + 1).sql(dialect="postgres"), True


def run_sql(sql: str, *, row_limit: int | None = None) -> SQLResult:
    """Validate, then execute read-only with a limit and timeout."""
    s = get_settings()
    row_limit = s.sql_row_limit if row_limit is None else row_limit

    v = validate_sql(sql)
    if not v.ok:
        return SQLResult(False, sql=sql, error=f"SQL rejected: {v.reason}")

    final_sql, limit_added = _add_limit(v.normalised_sql, row_limit)

    t0 = time.perf_counter()
    try:
        from psycopg.rows import dict_row

        from app.database.connection import ro_connection

        with ro_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"SET LOCAL statement_timeout = {s.sql_timeout_ms}")
                cur.execute(final_sql)  # type: ignore[arg-type]
                rows: list[dict[str, Any]] = cur.fetchall()
                cols = [d.name for d in (cur.description or [])]
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()[0]
        return SQLResult(
            False, sql=final_sql, error=msg,
            latency_s=time.perf_counter() - t0,
        )

    truncated = limit_added and len(rows) > row_limit
    if truncated:
        rows = rows[:row_limit]

    return SQLResult(
        ok=True, sql=final_sql, rows=rows, columns=cols,
        row_count=len(rows), truncated=truncated,
        latency_s=time.perf_counter() - t0,
    )


def format_rows(rows: list[dict], *, max_rows: int = 30) -> str:
    """Render rows as a compact table for the prompt."""
    if not rows:
        return "(no rows returned)"
    cols = list(rows[0].keys())
    shown = rows[:max_rows]
    widths = {
        c: max(len(c), *(len(_fmt(r.get(c))) for r in shown)) for c in cols
    }
    head = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    body = [
        " | ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols) for r in shown
    ]
    out = [head, sep, *body]
    if len(rows) > max_rows:
        out.append(f"... {len(rows) - max_rows} more rows")
    return "\n".join(out)


def _fmt(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:,.2f}"
    from decimal import Decimal

    if isinstance(v, Decimal):
        return f"{float(v):,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)
