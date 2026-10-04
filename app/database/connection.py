"""
Database connections.

There are deliberately TWO ways to reach Postgres, and which one you use
is a security decision, not a style choice:

    rw_connection()   role: analyst   schema creation, seeding, feedback
                                      memory writes, trace logging
    ro_connection()   role: agent_ro  EVERY query the agent generates

The agent writes its own SQL from an LLM prompt. We parse and validate that
SQL before running it, but a parser can be fooled. agent_ro is the
independent second layer: its transactions are read-only at the engine
level, and it has no access to the `memory` schema at all, so agent SQL
cannot read stored feedback (which would leak rejected poisoned lessons
into the agent's context through a side channel).

Use ro_connection() for anything the model authored. Use rw_connection()
only for code we wrote ourselves.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import get_settings


@contextmanager
def rw_connection(*, autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """Read-write connection as `analyst`. For our own code only."""
    s = get_settings()
    with psycopg.connect(
        s.database_url, autocommit=autocommit, connect_timeout=10
    ) as conn:
        register_vector(conn)
        yield conn


@contextmanager
def ro_connection() -> Iterator[psycopg.Connection]:
    """Read-only connection as `agent_ro`. For agent-generated SQL.

    Cannot write, and cannot see the `memory` schema.
    """
    s = get_settings()
    with psycopg.connect(s.database_url_ro, connect_timeout=10) as conn:
        yield conn


def fetch_all(sql: str, params: Any = None, *, read_only: bool = True) -> list[dict]:
    """Run a SELECT and return rows as dicts.

    read_only=True (the default) routes through agent_ro. Pass False only
    for our own queries against the memory schema.
    """
    ctx = ro_connection() if read_only else rw_connection()
    with ctx as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def fetch_one(sql: str, params: Any = None, *, read_only: bool = True) -> dict | None:
    rows = fetch_all(sql, params, read_only=read_only)
    return rows[0] if rows else None


def ping() -> dict[str, Any]:
    """Health check used by /health and the verify scripts."""
    out: dict[str, Any] = {}
    with rw_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT current_database(), current_user")
        db, user = cur.fetchone()
        out["database"] = db
        out["rw_user"] = user
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
        row = cur.fetchone()
        out["pgvector"] = row[0] if row else None
    with ro_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT current_user")
        out["ro_user"] = cur.fetchone()[0]
    return out
