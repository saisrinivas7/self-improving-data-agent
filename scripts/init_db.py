"""
Create the BUSINESS schema (public).

Run:  make init-db     (also run as the first step of `make seed`)

This drops and recreates the Lumen & Co. tables, so it destroys the business
data. That is fine: the business data is synthetic and regenerable from a
seed.

It deliberately does NOT touch the `memory` schema. Feedback memory holds
analyst feedback typed by a human, which cannot be regenerated, so it has
its own script (scripts/init_memory.py) with a guard. An earlier version
applied both, which meant `make seed` silently deleted every stored lesson.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_DIR = PROJECT_ROOT / "app" / "database" / "schema" / "business"


def main() -> None:
    from app.database.connection import rw_connection

    files = sorted(SCHEMA_DIR.glob("*.sql"))
    if not files:
        print(f"No .sql files found in {SCHEMA_DIR}", file=sys.stderr)
        sys.exit(1)

    print("Applying business schema (public)\n")

    with rw_connection() as conn:
        for f in files:
            with conn.cursor() as cur:
                cur.execute(f.read_text())
            conn.commit()
            print(f"  applied {f.name}")

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema='public' AND table_type='BASE TABLE'
                ORDER BY table_name
                """
            )
            tables = [r[0] for r in cur.fetchall()]
            print(f"\n  {len(tables)} tables: {', '.join(tables)}")

            # The schema must not document itself. Column comments live in
            # pg_description, which is world-readable, and a schema tool
            # would paste them into every prompt - handing the agent the very
            # knowledge the Learning Lab exists to teach it. See the header
            # of business/01_business.sql.
            cur.execute(
                """
                SELECT count(*) FROM pg_description d
                JOIN pg_class c ON c.oid = d.objoid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND d.objsubid > 0
                """
            )
            n_comments = cur.fetchone()[0]
            if n_comments:
                print(
                    f"\n  FAILED: {n_comments} column comment(s) found in public.\n"
                    "  These leak the planted trap answers to the agent.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("  no column comments in public (traps stay hidden from the agent)")

    print("\n" + "=" * 62)
    print("BUSINESS SCHEMA OK")
    print("=" * 62)
    print("  Next: make generate")


if __name__ == "__main__":
    main()
