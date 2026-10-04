"""
Create the MEMORY schema (feedback_memory, traces, trace_events).

Run:  make init-memory              refuses if any feedback exists
      make init-memory FORCE=1      wipes it anyway

WHY THIS IS SEPARATE FROM init_db.py, AND GUARDED

The business tables are synthetic and regenerable from a seed, so dropping
them costs nothing. Feedback memory is the opposite: every row is analyst
feedback a human typed into the Learning Lab, reasoning about a specific
answer the agent gave. None of it can be regenerated.

An earlier version applied both schemas from one script, which meant the
routine `make seed` quietly deleted every stored lesson. In a project whose
entire subject is a persistent, hard-won memory, that is the worst possible
footgun - so this script counts the rows first and refuses.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_DIR = PROJECT_ROOT / "app" / "database" / "schema" / "memory"


def main() -> None:
    import psycopg

    from app.config import get_settings
    from app.database.connection import ro_connection, rw_connection

    s = get_settings()
    force = os.environ.get("FORCE", "").strip() in {"1", "true", "yes"}

    files = sorted(SCHEMA_DIR.glob("*.sql"))
    if not files:
        print(f"No .sql files found in {SCHEMA_DIR}", file=sys.stderr)
        sys.exit(1)

    # ---- count what we would destroy ----
    existing = {"feedback": 0, "traces": 0}
    with rw_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('memory.feedback_memory'), to_regclass('memory.traces')"
        )
        fm_exists, tr_exists = cur.fetchone()
        if fm_exists:
            cur.execute("SELECT count(*) FROM memory.feedback_memory")
            existing["feedback"] = cur.fetchone()[0]
        if tr_exists:
            cur.execute("SELECT count(*) FROM memory.traces")
            existing["traces"] = cur.fetchone()[0]

    if existing["feedback"] and not force:
        print(
            f"REFUSING: memory.feedback_memory holds {existing['feedback']} row(s) "
            f"and memory.traces holds {existing['traces']}.\n\n"
            "  These are analyst feedback and agent traces. They cannot be\n"
            "  regenerated from a seed the way the business data can.\n\n"
            "  To wipe them anyway:   make init-memory FORCE=1\n"
            "  To inspect them first: make psql, then\n"
            "      SELECT status, feedback_type, lesson FROM memory.feedback_memory;",
            file=sys.stderr,
        )
        sys.exit(1)

    if existing["feedback"]:
        print(f"FORCE set - destroying {existing['feedback']} feedback row(s)\n")

    print(f"Applying memory schema  (EMBEDDING_DIM={s.embedding_dim})\n")

    with rw_connection() as conn:
        for f in files:
            sql = f.read_text().replace("{EMBEDDING_DIM}", str(s.embedding_dim))
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            print(f"  applied {f.name}")

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema='memory' AND table_type='BASE TABLE'
                ORDER BY table_name
                """
            )
            print(f"  tables: {', '.join(r[0] for r in cur.fetchall())}")

            # The vector column must match the pinned embedding dimension,
            # or every insert fails much later with a confusing error.
            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname='memory' AND c.relname='feedback_memory'
                  AND a.attname='embedding'
                """
            )
            row = cur.fetchone()
            got = row[0] if row else "MISSING"
            print(f"  feedback_memory.embedding is {got}")
            if f"({s.embedding_dim})" not in got:
                print(f"  FAILED: expected vector({s.embedding_dim})", file=sys.stderr)
                sys.exit(1)

    # ---- the security boundary ----
    print("\n  Checking the agent role cannot read feedback memory...")
    with ro_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM memory.feedback_memory")
            print("  FAILED: agent_ro CAN read memory.feedback_memory", file=sys.stderr)
            sys.exit(1)
        except psycopg.errors.Error as e:
            print(f"  correctly blocked: {str(e).strip().splitlines()[0]}")

    print("\n" + "=" * 62)
    print("MEMORY SCHEMA OK  (feedback_memory is empty, as the spec requires)")
    print("=" * 62)


if __name__ == "__main__":
    main()
