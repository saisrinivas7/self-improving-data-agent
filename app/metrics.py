"""
The canonical way to compute every business metric in this project.

WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL

The verifier has to decide whether "refunds jumped in March" is true. The
tempting implementation is to let the model write a SQL query and read the
answer. That does not work here, for a specific reason: the model falls into
the planted traps. We watched it group refunds by `refund_date` instead of
the order's month, and filter `status = 'completed'` while dropping
`'refunded'` orders which are also real sales.

A verifier that makes those mistakes produces confident, wrong verdicts. It
would reject true feedback and accept false feedback, and every number
downstream would be noise.

So the division of labour is:

    the model      decides WHICH metric a claim is about
    this module    computes the number

The model's output is constrained to the enum below, so it cannot invent a
metric, and it never writes arithmetic. The same functions are used by
scripts/verify_effects.py, which means the ground truth and the verifier can
never disagree about what "net revenue" means.

THE THREE DEFINITIONS BAKED IN HERE
Each one is a trap the agent routinely falls into:

  1. Real sales are status IN ('completed','refunded'). A refunded order
     still happened. 'pending' and 'cancelled' are not sales.
  2. Net revenue subtracts refunds. orders.total_amount is gross.
  3. A refund belongs to the month of its ORDER, not the month it was paid.
     refunds.refund_date lags by 12-45 days and usually crosses a month.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from app.database.connection import rw_connection

# Months at the end of the data window have incomplete refund data, because
# a refund dated after the window has not been processed yet. Any claim about
# refund rates in these months is unanswerable rather than false.
INCOMPLETE_REFUND_MONTHS = frozenset({"2026-08", "2026-09"})

FIRST_MONTH = "2025-10"
LAST_COMPLETE_MONTH = "2026-07"


class Metric(StrEnum):
    """Every metric a claim may be about. The model picks from this list."""

    NET_REVENUE = "net_revenue"
    GROSS_REVENUE = "gross_revenue"
    ORDER_COUNT = "order_count"
    AVG_ORDER_VALUE = "avg_order_value"
    REFUND_COUNT = "refund_count"
    REFUND_AMOUNT = "refund_amount"
    REFUND_SHARE = "refund_share"
    TICKET_COUNT = "ticket_count"
    MARGIN_PCT = "margin_pct"


# The month-by-month series. One query, the canonical definitions.
_MONTHLY_SQL = """
WITH sales AS (
    SELECT o.order_id,
           to_char(date_trunc('month', o.order_date), 'YYYY-MM') AS month,
           o.total_amount
    FROM orders o
    WHERE o.status IN ('completed', 'refunded')
),
refunded AS (
    SELECT s.month,
           count(r.refund_id)                 AS refund_count,
           coalesce(sum(r.refund_amount), 0)  AS refund_amount
    FROM sales s
    JOIN refunds r ON r.order_id = s.order_id
    GROUP BY s.month
),
tickets AS (
    SELECT to_char(date_trunc('month', created_at), 'YYYY-MM') AS month,
           count(*) AS ticket_count
    FROM support_tickets GROUP BY 1
),
margin AS (
    SELECT to_char(date_trunc('month', o.order_date), 'YYYY-MM') AS month,
           sum(i.quantity * i.unit_price)             AS revenue,
           sum(i.quantity * (i.unit_price - p.cost))  AS profit
    FROM orders o
    JOIN order_items i USING (order_id)
    JOIN products p USING (product_id)
    WHERE o.status IN ('completed','refunded')
    GROUP BY 1
)
SELECT s.month,
       count(*)                                        AS order_count,
       sum(s.total_amount)                             AS gross_revenue,
       avg(s.total_amount)                             AS avg_order_value,
       coalesce(f.refund_count, 0)                     AS refund_count,
       coalesce(f.refund_amount, 0)                    AS refund_amount,
       sum(s.total_amount) - coalesce(f.refund_amount, 0) AS net_revenue,
       coalesce(f.refund_amount, 0) / sum(s.total_amount) AS refund_share,
       coalesce(t.ticket_count, 0)                     AS ticket_count,
       CASE WHEN m.revenue > 0 THEN m.profit / m.revenue ELSE NULL END AS margin_pct
FROM sales s
LEFT JOIN refunded f ON f.month = s.month
LEFT JOIN tickets  t ON t.month = s.month
LEFT JOIN margin   m ON m.month = s.month
GROUP BY s.month, f.refund_count, f.refund_amount, t.ticket_count, m.revenue, m.profit
ORDER BY s.month
"""


@dataclass(frozen=True)
class MonthMetrics:
    month: str
    order_count: int
    gross_revenue: float
    avg_order_value: float
    refund_count: int
    refund_amount: float
    net_revenue: float
    refund_share: float
    ticket_count: int
    margin_pct: float | None

    def get(self, metric: Metric) -> float | None:
        return getattr(self, metric.value, None)


@lru_cache(maxsize=1)
def monthly_series() -> dict[str, MonthMetrics]:
    """Every metric for every month. Cached: the data does not change at runtime."""
    out: dict[str, MonthMetrics] = {}
    with rw_connection() as conn, conn.cursor() as cur:
        cur.execute(_MONTHLY_SQL)
        for r in cur.fetchall():
            out[r[0]] = MonthMetrics(
                month=r[0],
                order_count=int(r[1]),
                gross_revenue=float(r[2]),
                avg_order_value=float(r[3]),
                refund_count=int(r[4]),
                refund_amount=float(r[5]),
                net_revenue=float(r[6]),
                refund_share=float(r[7]),
                ticket_count=int(r[8]),
                margin_pct=float(r[9]) if r[9] is not None else None,
            )
    return out


def prior_month(month: str) -> str | None:
    months = sorted(monthly_series())
    if month not in months:
        return None
    i = months.index(month)
    return months[i - 1] if i > 0 else None


def all_months(*, complete_refunds_only: bool = False) -> list[str]:
    months = sorted(monthly_series())
    if complete_refunds_only:
        months = [m for m in months if m not in INCOMPLETE_REFUND_MONTHS]
    return months


@dataclass(frozen=True)
class Change:
    metric: Metric
    month: str
    compare_to: str
    before: float
    after: float
    pct_change: float
    direction: str  # "increase" | "decrease" | "flat"


def measure_change(metric: Metric, month: str, compare_to: str | None = None) -> Change | None:
    """How a metric moved between two months. None when it cannot be computed."""
    series = monthly_series()
    compare_to = compare_to or prior_month(month) or ""
    if month not in series or compare_to not in series:
        return None
    before = series[compare_to].get(metric)
    after = series[month].get(metric)
    if before is None or after is None or before == 0:
        return None
    pct = (after - before) / abs(before) * 100
    direction = "increase" if pct > 1.0 else "decrease" if pct < -1.0 else "flat"
    return Change(metric, month, compare_to, float(before), float(after), pct, direction)


# ----------------------------------------------------------------- attribution

@dataclass(frozen=True)
class Attribution:
    """How a net revenue change splits between its causes.

    THE DECOMPOSITION

    Net revenue is  gross - refunds,  and  gross = order_count x avg_order_value.
    So a change in net revenue has exactly three sources:

        volume   = (orders_after - orders_before) x aov_before
        aov      = orders_after x (aov_after - aov_before)
        refunds  = -(refunds_after - refunds_before)

    and those three sum to the net change exactly, which `residual` asserts.

    An earlier version of this put the whole gross change into `volume`,
    which looked fine for March (where AOV barely moved) but reported July's
    decline as volume-driven when July's order count actually ROSE 1% and
    the cause was a 15% drop in average order value. A verifier using that
    formula would have refuted true feedback about AOV.

    Shares are signed fractions of the net change, so for a decline a
    positive share means "this contributed to the fall".
    """

    month: str
    compare_to: str
    net_change: float
    net_change_pct: float
    volume_share: float
    refund_share_of_change: float
    aov_share: float
    # How far the three parts miss the actual net change, as a fraction.
    # Should be ~0; anything else means the decomposition is broken.
    residual: float = 0.0

    def share_of(self, factor: str) -> float:
        return {
            "order_volume": self.volume_share,
            "volume": self.volume_share,
            "refunds": self.refund_share_of_change,
            "avg_order_value": self.aov_share,
            "aov": self.aov_share,
        }.get(factor, 0.0)

    @property
    def dominant_factor(self) -> str:
        return max(
            (("order_volume", self.volume_share),
             ("refunds", self.refund_share_of_change),
             ("avg_order_value", self.aov_share)),
            key=lambda kv: kv[1],
        )[0]


def attribute_net_change(month: str, compare_to: str | None = None) -> Attribution | None:
    """Split a month's net revenue change into volume, refunds and AOV.

    This is the function that decides whether "refunds caused the decline" is
    true. It is also used by scripts/verify_effects.py, so the claim checker
    and the ground truth cannot drift apart.
    """
    series = monthly_series()
    compare_to = compare_to or prior_month(month) or ""
    if month not in series or compare_to not in series:
        return None
    a, b = series[compare_to], series[month]
    if a.net_revenue == 0 or a.gross_revenue == 0 or b.gross_revenue == 0:
        return None

    net_change = b.net_revenue - a.net_revenue
    if net_change == 0:
        return None

    # gross = orders x aov, so the gross change splits cleanly in two.
    volume_part = (b.order_count - a.order_count) * a.avg_order_value
    aov_part = b.order_count * (b.avg_order_value - a.avg_order_value)
    refund_part = -(b.refund_amount - a.refund_amount)

    # Signed shares of the net change: positive means "contributed to the
    # move in the same direction as the overall change".
    denom = net_change
    residual = (volume_part + aov_part + refund_part - net_change) / abs(net_change)

    return Attribution(
        month=month,
        compare_to=compare_to,
        net_change=net_change,
        net_change_pct=net_change / a.net_revenue * 100,
        volume_share=volume_part / denom,
        refund_share_of_change=refund_part / denom,
        aov_share=aov_part / denom,
        residual=residual,
    )


# -------------------------------------------------------------- counterexamples

# A month only counts as a counterexample to "declines are always caused by
# refunds" if refunds explain little of the decline. Testing "refunds did not
# rise" would be wrong: in this dataset July's refund COUNT rose 11.8% while
# refunds explained almost none of the revenue drop. Attribution is the
# meaningful test, not the raw count.
COUNTEREXAMPLE_MAX_REFUND_SHARE = 0.25


def decline_months(*, complete_refunds_only: bool = True) -> list[str]:
    """Months where net revenue fell against the previous month."""
    out = []
    for m in all_months(complete_refunds_only=complete_refunds_only):
        c = measure_change(Metric.NET_REVENUE, m)
        if c and c.pct_change < 0:
            out.append(m)
    return out


def find_counterexamples(
    factor: str = "refunds",
    *,
    max_share: float = COUNTEREXAMPLE_MAX_REFUND_SHARE,
) -> list[dict]:
    """Months that refute "revenue declines are always caused by <factor>".

    Returns every qualifying month with its measured numbers, so a rejection
    can cite real evidence rather than just asserting the claim is false.
    """
    found: list[dict] = []
    for m in decline_months():
        attr = attribute_net_change(m)
        if attr is None:
            continue
        share = attr.share_of(factor)
        if share < max_share:
            found.append(
                {
                    "month": m,
                    "compare_to": attr.compare_to,
                    "net_change_pct": round(attr.net_change_pct, 2),
                    f"{factor}_share_of_decline": round(share, 3),
                    "dominant_factor": attr.dominant_factor,
                }
            )
    return found


# ------------------------------------------------------------------ by category

@lru_cache(maxsize=1)
def category_month_tickets() -> dict[tuple[str, str], dict]:
    """Ticket rate per order, split by whether the order contains a category.

    Needed for claims about a particular category, e.g. "Electronics had
    unusually many shipping complaints".
    """
    sql = """
    WITH flags AS (
        SELECT o.order_id,
               to_char(date_trunc('month', o.order_date), 'YYYY-MM') AS month,
               p.category
        FROM orders o
        JOIN order_items i USING (order_id)
        JOIN products p USING (product_id)
        GROUP BY 1, 2, 3
    )
    SELECT f.month, f.category,
           count(DISTINCT f.order_id)      AS orders,
           count(st.ticket_id)             AS tickets
    FROM flags f
    LEFT JOIN support_tickets st ON st.order_id = f.order_id
    GROUP BY 1, 2
    """
    out: dict[tuple[str, str], dict] = {}
    with rw_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        for month, cat, orders, tickets in cur.fetchall():
            out[(month, cat)] = {
                "orders": int(orders),
                "tickets": int(tickets),
                "rate": (int(tickets) / int(orders)) if orders else 0.0,
            }
    return out


@lru_cache(maxsize=1)
def refund_reasons_by_month() -> dict[str, dict[str, int]]:
    """Refund reason mix per ORDER month."""
    out: dict[str, dict[str, int]] = {}
    with rw_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT to_char(date_trunc('month', o.order_date), 'YYYY-MM') AS month,
                   r.reason, count(*)
            FROM refunds r JOIN orders o USING (order_id)
            GROUP BY 1, 2
            """
        )
        for month, reason, n in cur.fetchall():
            out.setdefault(month, {})[reason] = int(n)
    return out


def clear_cache() -> None:
    """Drop cached series. Call after reseeding the database."""
    monthly_series.cache_clear()
    category_month_tickets.cache_clear()
    refund_reasons_by_month.cache_clear()
