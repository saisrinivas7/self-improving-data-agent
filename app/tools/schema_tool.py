"""
Tool 1 - Schema Tool (spec section 10).

Tells the agent what tables and columns exist.

Two rules it follows:

1. Only show RELEVANT tables. Pasting all seven tables plus every column
   into every prompt wastes context and makes the model's job harder. We
   score tables against the question's words and send the best few, always
   pulling in whatever they join to.

2. Never reveal the answers. This introspects live Postgres, and the
   business schema deliberately carries no column comments, so what comes
   back is structure and example values only - never an explanation of what
   a column means. If the agent is to learn that `total_amount` is gross of
   refunds, it has to learn that from analyst feedback, which is the point
   of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# app.database.connection is imported lazily inside describe_tables. Table
# selection (pick_tables and the keyword scoring) is pure string work and
# must stay importable without a database driver, so it can be tested
# anywhere - including CI, which has no Postgres.

# Words in a question that point at a table. Crude on purpose: a scoring
# heuristic is inspectable and cheap, where an LLM call to pick tables
# would add latency and another failure mode.
TABLE_HINTS: dict[str, set[str]] = {
    "orders": {
        "order", "orders", "revenue", "sales", "sold", "aov", "basket",
        "purchase", "purchased", "status", "cancelled", "pending", "month",
        "monthly", "trend", "growth", "decline", "decrease", "increase",
    },
    "order_items": {
        "item", "items", "quantity", "units", "unit", "line", "basket",
        "product", "products", "sku", "price", "paid",
    },
    "products": {
        "product", "products", "category", "categories", "price", "cost",
        "margin", "margins", "profit", "profitable", "sku", "catalogue",
    },
    "customers": {
        "customer", "customers", "country", "countries", "segment",
        "segments", "signup", "cohort", "churn", "churned", "retention",
        "retained", "lifetime", "ltv", "loyal", "new", "region",
    },
    "refunds": {
        "refund", "refunds", "refunded", "return", "returns", "returned",
        "reason", "reasons", "late", "damaged", "chargeback",
    },
    "support_tickets": {
        "ticket", "tickets", "support", "complaint", "complaints",
        "shipping", "delay", "delays", "escalated", "resolution", "issue",
    },
    "promotions": {
        "promotion", "promotions", "promo", "discount", "discounts",
        "discounted", "campaign", "sale", "offer", "markdown",
    },
}

# Tables that are nearly useless alone. If one is selected, bring its
# partner too, otherwise the model writes a query it cannot join.
COMPANIONS: dict[str, set[str]] = {
    "order_items": {"orders", "products"},
    "refunds": {"orders"},
    "support_tickets": {"customers"},
    "promotions": {"products"},
    "products": {"order_items"},
}


@dataclass
class Column:
    name: str
    type: str
    nullable: bool
    samples: list[str] = field(default_factory=list)

    def render(self) -> str:
        bits = [f"{self.name} {self.type}"]
        if self.nullable:
            bits.append("NULL allowed")
        if self.samples:
            bits.append("e.g. " + ", ".join(self.samples))
        return "  " + "  |  ".join(bits)


@dataclass
class Table:
    name: str
    columns: list[Column]
    row_count: int
    foreign_keys: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"TABLE {self.name}  ({self.row_count:,} rows)"]
        lines += [c.render() for c in self.columns]
        if self.foreign_keys:
            lines.append("  joins: " + "; ".join(self.foreign_keys))
        return "\n".join(lines)


def _score(question: str, table: str) -> int:
    words = set(question.lower().replace("?", " ").replace(",", " ").split())
    return len(words & TABLE_HINTS.get(table, set()))


def pick_tables(question: str, max_tables: int = 5) -> list[str]:
    """Choose the tables worth describing for this question."""
    scored = sorted(
        ((t, _score(question, t)) for t in TABLE_HINTS),
        key=lambda kv: kv[1],
        reverse=True,
    )
    chosen = [t for t, s in scored if s > 0][:max_tables]
    # Nothing matched: fall back to the core three rather than guessing.
    if not chosen:
        chosen = ["orders", "order_items", "products"]
    # Pull in companions so joins are possible.
    for t in list(chosen):
        for c in COMPANIONS.get(t, set()):
            if c not in chosen:
                chosen.append(c)
    return chosen[: max_tables + 2]


def describe_tables(tables: list[str], *, samples_per_column: int = 3) -> list[Table]:
    """Introspect live Postgres for the given tables."""
    from app.database.connection import ro_connection  # noqa: PLC0415

    out: list[Table] = []
    with ro_connection() as conn, conn.cursor() as cur:
        for name in tables:
            cur.execute(
                """
                SELECT column_name,
                       CASE WHEN data_type = 'character varying' THEN 'text'
                            WHEN data_type = 'timestamp without time zone' THEN 'timestamp'
                            WHEN data_type = 'numeric' THEN 'numeric'
                            ELSE data_type END,
                       is_nullable = 'YES'
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
                """,
                (name,),
            )
            cols = [Column(n, t, nl) for n, t, nl in cur.fetchall()]
            if not cols:
                continue

            cur.execute(f"SELECT count(*) FROM {name}")
            n_rows = cur.fetchone()[0]

            # Example values make the difference between the model guessing
            # at an enum and knowing it. Only for low-cardinality text and
            # date columns; sampling 90k order ids teaches nothing.
            for c in cols:
                if c.type in ("text", "date"):
                    cur.execute(
                        f"SELECT DISTINCT {c.name} FROM {name} "
                        f"WHERE {c.name} IS NOT NULL "
                        f"ORDER BY {c.name} LIMIT {samples_per_column + 6}"
                    )
                    vals = [str(r[0]) for r in cur.fetchall()]
                    if len(vals) <= samples_per_column + 6:
                        c.samples = vals[:samples_per_column]

            cur.execute(
                """
                SELECT kcu.column_name, ccu.table_name, ccu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public' AND tc.table_name = %s
                """,
                (name,),
            )
            fks = [f"{name}.{a} -> {b}.{c}" for a, b, c in cur.fetchall()]

            out.append(Table(name, cols, n_rows, fks))
    return out


def get_schema_context(
    question: str, *, max_tables: int = 5, extra_tables: list[str] | None = None
) -> tuple[str, list[str]]:
    """Main entry point.

    Returns (text for the prompt, list of table names used). The table list
    is recorded on the trace, and later used to judge whether a stored
    lesson is even relevant to the tables in play.

    extra_tables exists to make retrieved lessons ACTIONABLE. Keyword
    scoring picks tables from the question's words, so "why did revenue
    decrease in March?" yields only `orders`. A retrieved lesson saying
    "also check refund trends" would then be impossible to follow: the
    agent cannot write SQL against a table it was never shown, so the
    lesson would appear in the prompt and change nothing.
    Each lesson carries its own schema_context, and those tables are passed
    in here so the lesson can actually be acted on.
    """
    names = pick_tables(question, max_tables=max_tables)
    for t in extra_tables or []:
        t = t.lower().strip()
        if t in TABLE_HINTS and t not in names:
            names.append(t)
            # Bring companions too, or the new table cannot be joined.
            for c in COMPANIONS.get(t, set()):
                if c not in names:
                    names.append(c)
    tables = describe_tables(names)
    text = "\n\n".join(t.render() for t in tables)
    return text, [t.name for t in tables]
