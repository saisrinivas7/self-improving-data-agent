"""
Load the generated CSVs into Postgres.

Run:  make load

Uses COPY rather than INSERT, which matters at ~250k rows: COPY streams the
file in one statement instead of a round trip per row.

Load order follows the foreign keys - customers and products first, then
orders, then everything that references an order. Indexes already exist from
init_db.py; for a load this size that is fine, and keeping them avoids a
rebuild step that would need more code than it saves time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "generated"

# (table, csv columns in file order, columns that are empty-string -> NULL)
TABLES: list[tuple[str, list[str], list[str]]] = [
    ("customers", ["customer_id", "name", "country", "signup_date", "customer_segment"], ["country"]),
    ("products", ["product_id", "product_name", "category", "price", "cost"], []),
    ("orders", ["order_id", "customer_id", "order_date", "status", "total_amount"], []),
    ("order_items", ["order_id", "product_id", "quantity", "unit_price"], []),
    ("refunds", ["refund_id", "order_id", "refund_date", "refund_amount", "reason"], []),
    (
        "support_tickets",
        ["ticket_id", "customer_id", "order_id", "created_at", "category", "resolution"],
        ["order_id"],
    ),
    ("promotions", ["promotion_id", "product_id", "start_date", "end_date", "discount_percent"], []),
]


def main() -> None:
    from app.database.connection import rw_connection

    missing = [t for t, _, _ in TABLES if not (DATA_DIR / f"{t}.csv").exists()]
    if missing:
        print(f"Missing CSVs for: {', '.join(missing)}", file=sys.stderr)
        print("Run: make generate", file=sys.stderr)
        sys.exit(1)

    print("Loading into Postgres\n")
    t_start = time.perf_counter()

    with rw_connection() as conn:
        # Truncate in reverse dependency order so FKs never block it.
        with conn.cursor() as cur:
            names = ", ".join(t for t, _, _ in reversed(TABLES))
            cur.execute(f"TRUNCATE {names} RESTART IDENTITY CASCADE")
        conn.commit()
        print("  truncated existing rows")

        total = 0
        for table, cols, nullable in TABLES:
            path = DATA_DIR / f"{table}.csv"
            t0 = time.perf_counter()
            # FORCE_NULL turns the CSV's empty strings into real NULLs for the
            # columns that allow them (customers.country, tickets.order_id).
            force_null = f", FORCE_NULL ({', '.join(nullable)})" if nullable else ""
            copy_sql = (
                f"COPY {table} ({', '.join(cols)}) FROM STDIN "
                f"WITH (FORMAT csv, HEADER true{force_null})"
            )
            with conn.cursor() as cur, open(path, "rb") as f:
                with cur.copy(copy_sql) as copy:
                    while chunk := f.read(1 << 20):
                        copy.write(chunk)
                cur.execute(f"SELECT count(*) FROM {table}")
                n = cur.fetchone()[0]
            conn.commit()
            total += n
            print(f"  {table:<18} {n:>8,} rows  ({time.perf_counter() - t0:.2f}s)")

        # ANALYZE so the planner has statistics. Without this, the agent's
        # first few queries get bad plans and misleading latency numbers.
        with conn.cursor() as cur:
            cur.execute("ANALYZE")
        conn.commit()
        print("\n  ANALYZE done (planner statistics refreshed)")

        # ---- integrity checks: catch a bad load before the agent sees it ----
        print("\n  integrity checks:")
        checks: list[tuple[str, str, str]] = [
            (
                "order_items sum matches orders.total_amount",
                """SELECT count(*) FROM (
                     SELECT o.order_id
                     FROM orders o JOIN order_items i USING (order_id)
                     GROUP BY o.order_id, o.total_amount
                     HAVING abs(sum(i.quantity * i.unit_price) - o.total_amount) > 0.02
                   ) x""",
                "0",
            ),
            (
                "every refund has a parent order",
                "SELECT count(*) FROM refunds r LEFT JOIN orders o USING (order_id) WHERE o.order_id IS NULL",
                "0",
            ),
            (
                "no refund predates its order",
                "SELECT count(*) FROM refunds r JOIN orders o USING (order_id) WHERE r.refund_date < o.order_date",
                "0",
            ),
            (
                "no order predates customer signup",
                """SELECT count(*) FROM orders o JOIN customers c USING (customer_id)
                   WHERE o.order_date < c.signup_date""",
                "0",
            ),
        ]
        failed = 0
        with conn.cursor() as cur:
            for label, sql, expect in checks:
                cur.execute(sql)
                got = str(cur.fetchone()[0])
                ok = got == expect
                failed += not ok
                mark = "\033[32mOK\033[0m  " if ok else "\033[31mFAIL\033[0m"
                print(f"    {mark} {label}" + ("" if ok else f"  (got {got}, expected {expect})"))

    print(f"\n  {total:,} rows loaded in {time.perf_counter() - t_start:.1f}s")
    if failed:
        print(f"\n  {failed} integrity check(s) FAILED", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 62)
    print("DATABASE LOADED")
    print("=" * 62)
    print("  Next: make verify-effects   (measure the planted effects)")


if __name__ == "__main__":
    main()
