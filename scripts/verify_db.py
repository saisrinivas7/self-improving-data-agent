"""
Phase 0 database verification.

Run:  make verify-db

Checks four things, in order of how annoying they are to debug later:

1. Postgres is reachable from Python with the read-write role.
2. pgvector is installed AND the vector type round-trips through psycopg,
   including numpy arrays.
3. The read-only role can SELECT.
4. The read-only role CANNOT write.

Check 4 is the one worth caring about. The agent executes SQL written by an
LLM. We validate that SQL with a parser before running it, but a parser can
be fooled, so the read-only role is an independent second layer. This script
asserts that layer actually works rather than assuming the GRANTs were right.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  \033[32mOK\033[0m   {msg}")


def bad(msg: str) -> None:
    failures.append(msg)
    print(f"  \033[31mFAIL\033[0m {msg}")


def main() -> None:
    from dotenv import load_dotenv

    if not ENV_PATH.exists():
        print(f"FAILED: {ENV_PATH} not found. Run: cp .env.example .env", file=sys.stderr)
        sys.exit(1)
    load_dotenv(ENV_PATH)

    import numpy as np
    import psycopg
    from pgvector.psycopg import register_vector

    rw_url = os.environ.get("DATABASE_URL")
    ro_url = os.environ.get("DATABASE_URL_RO")
    if not rw_url or not ro_url:
        print("FAILED: DATABASE_URL / DATABASE_URL_RO missing from .env", file=sys.stderr)
        sys.exit(1)

    # ---------- 1 & 2: read-write connection, pgvector ----------
    print("\nRead-write connection (role: analyst)")
    try:
        with psycopg.connect(rw_url, connect_timeout=10) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT current_user, current_database(), version()")
                user, db, ver = cur.fetchone()
                ok(f"connected as {user} to {db}")
                ok(f"{ver.split(' on ')[0]}")

                cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
                row = cur.fetchone()
                if row:
                    ok(f"pgvector {row[0]}")
                else:
                    bad("pgvector extension not installed")

                # numpy -> vector -> distance operator
                cur.execute(
                    "SELECT %s::vector <=> %s::vector",
                    (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
                )
                d = cur.fetchone()[0]
                if abs(d - 1.0) < 1e-6:
                    ok(f"numpy -> vector round-trip, cosine distance = {d}")
                else:
                    bad(f"unexpected cosine distance for orthogonal vectors: {d}")

                dim = os.environ.get("EMBEDDING_DIM")
                if dim:
                    ok(f"EMBEDDING_DIM = {dim} (set by verify-llm)")
                else:
                    print("  \033[33mnote\033[0m EMBEDDING_DIM not set yet - run: make verify-llm")
    except Exception as e:  # noqa: BLE001
        bad(f"read-write connection failed: {type(e).__name__}: {e}")
        print("\n  Is the container running?  make db-up")
        sys.exit(1)

    # ---------- 3 & 4: read-only role ----------
    print("\nRead-only connection (role: agent_ro)")
    try:
        # Create a probe table as the privileged user first.
        with psycopg.connect(rw_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS _verify_probe")
                cur.execute("CREATE TABLE _verify_probe(id int)")
                cur.execute("INSERT INTO _verify_probe VALUES (1)")
            conn.commit()

        with psycopg.connect(ro_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_user")
                ok(f"connected as {cur.fetchone()[0]}")

                cur.execute("SELECT count(*) FROM _verify_probe")
                ok(f"SELECT allowed (returned {cur.fetchone()[0]} row)")

            # Each write attempt needs its own transaction, since the failure
            # aborts the current one.
            for stmt, label in [
                ("INSERT INTO _verify_probe VALUES (2)", "INSERT"),
                ("UPDATE _verify_probe SET id = 9", "UPDATE"),
                ("DELETE FROM _verify_probe", "DELETE"),
                ("DROP TABLE _verify_probe", "DROP"),
                ("CREATE TABLE _verify_probe2(id int)", "CREATE"),
            ]:
                try:
                    with conn.cursor() as cur:
                        cur.execute(stmt)
                    bad(f"{label} SUCCEEDED as agent_ro - SECURITY HOLE")
                except psycopg.errors.Error:
                    ok(f"{label} correctly blocked")
                finally:
                    conn.rollback()
    except Exception as e:  # noqa: BLE001
        bad(f"read-only checks failed: {type(e).__name__}: {e}")
    finally:
        try:
            with psycopg.connect(rw_url, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    cur.execute("DROP TABLE IF EXISTS _verify_probe")
                    cur.execute("DROP TABLE IF EXISTS _verify_probe2")
                conn.commit()
        except Exception:  # noqa: BLE001
            pass

    # ---------- 5: the memory schema must be unreachable from agent_ro ----------
    # feedback_memory holds REJECTED poisoned lessons as well as verified
    # ones. If agent-authored SQL could read it, "what lessons do you have?"
    # would pull rejected content into the agent's context and the
    # wrong-feedback-adoption metric would measure a leak, not the policy.
    print("\nMemory schema isolation")
    try:
        with psycopg.connect(ro_url, connect_timeout=10) as conn:
            for tbl in ("feedback_memory", "traces", "trace_events"):
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"SELECT count(*) FROM memory.{tbl}")
                    bad(f"agent_ro CAN read memory.{tbl} - LEAK")
                except psycopg.errors.Error:
                    ok(f"memory.{tbl} correctly unreachable")
                finally:
                    conn.rollback()
    except Exception as e:  # noqa: BLE001
        bad(f"memory isolation check failed: {type(e).__name__}: {e}")

    # ---------- 6: the business schema must not document its own traps ----------
    print("\nSchema self-documentation (must be absent)")
    try:
        with psycopg.connect(rw_url, connect_timeout=10) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM pg_description d
                JOIN pg_class c ON c.oid = d.objoid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND d.objsubid > 0
                """
            )
            n = cur.fetchone()[0]
            if n:
                bad(f"{n} column comment(s) in public leak the planted trap answers")
            else:
                ok("no column comments in public")
    except Exception as e:  # noqa: BLE001
        bad(f"comment check failed: {type(e).__name__}: {e}")

    print()
    if failures:
        print("=" * 62)
        print(f"DB LAYER: {len(failures)} FAILURE(S)")
        print("=" * 62)
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("=" * 62)
    print("DB LAYER OK")
    print("=" * 62)


if __name__ == "__main__":
    main()
