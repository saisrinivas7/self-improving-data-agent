"""
Measure the planted effects back out of the loaded database.

Run:  make verify-effects

WHY THIS EXISTS, and why it is not optional

The generator aims at the targets in data/effects_manifest.yml, but random
noise moves the realised numbers. If the benchmark's ground truth were
copied from the intended values, every scored answer would be compared
against a number that is not actually in the database - and the agent would
be marked wrong for being right.

So this script is the only source of ground truth. It runs real SQL, prints
what it measured, asserts the manifest's thresholds, and writes
benchmark/ground_truth_measured.json.

It also measures how MATERIAL each planted trap is. A trap that shifts
revenue by 0.3% teaches nothing, because a wrong answer and a right answer
look the same. Each one has to be big enough that getting it wrong is
visibly wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "benchmark" / "ground_truth_measured.json"

# ----------------------------------------------------------------------
# The canonical revenue definition for this dataset.
#
# Three decisions are baked in here, and each one is a trap the agent can
# fall into:
#   1. Only completed and refunded orders are real sales. pending and
#      cancelled are not.
#   2. orders.total_amount is GROSS. Net revenue subtracts refunds.
#   3. A refund belongs to the month of its ORDER, not the month it was
#      paid. refunds.refund_date lags by 5-30 days and crosses months.
# ----------------------------------------------------------------------
MONTHLY_REVENUE_SQL = """
WITH sales AS (
    SELECT o.order_id,
           date_trunc('month', o.order_date)::date AS month,
           o.total_amount
    FROM orders o
    WHERE o.status IN ('completed', 'refunded')
),
refunded AS (
    SELECT s.month,
           count(r.refund_id)          AS refund_count,
           coalesce(sum(r.refund_amount), 0) AS refund_amount
    FROM sales s
    JOIN refunds r ON r.order_id = s.order_id
    GROUP BY s.month
)
SELECT s.month,
       count(*)                               AS orders,
       sum(s.total_amount)                    AS gross,
       round(avg(s.total_amount), 2)          AS aov,
       coalesce(f.refund_count, 0)            AS refund_count,
       coalesce(f.refund_amount, 0)           AS refund_amount,
       sum(s.total_amount) - coalesce(f.refund_amount, 0) AS net,
       round(coalesce(f.refund_amount, 0) / sum(s.total_amount) * 100, 2) AS refund_pct
FROM sales s
LEFT JOIN refunded f ON f.month = s.month
GROUP BY s.month, f.refund_count, f.refund_amount
ORDER BY s.month
"""


def main() -> None:
    from app.database.connection import rw_connection

    failures: list[str] = []
    gt: dict = {"source": "measured from seeded database", "effects": {}, "traps": {}}

    def check(label: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"    \033[32mPASS\033[0m  {label}" + (f"  {detail}" if detail else ""))
        else:
            failures.append(f"{label} {detail}")
            print(f"    \033[31mFAIL\033[0m  {label}  {detail}")

    with rw_connection() as conn, conn.cursor() as cur:
        # ------------------------------------------------------ monthly table
        cur.execute(MONTHLY_REVENUE_SQL)
        rows = cur.fetchall()
        months = {
            r[0].strftime("%Y-%m"): {
                "orders": int(r[1]),
                "gross": float(r[2]),
                "aov": float(r[3]),
                "refund_count": int(r[4]),
                "refund_amount": float(r[5]),
                "net": float(r[6]),
                "refund_pct": float(r[7]),
            }
            for r in rows
        }
        gt["monthly"] = months

        print("Monthly revenue, measured with the canonical definition")
        print("(completed+refunded orders; refunds attributed to the ORDER month)\n")
        print(f"  {'month':<9} {'orders':>7} {'AOV':>7} {'gross':>12} {'refunds':>9} {'ref%':>6} {'net':>12} {'net Δ':>8}")
        prev = None
        for mk, m in months.items():
            d = f"{(m['net'] / prev - 1) * 100:+6.1f}%" if prev else ""
            print(
                f"  {mk:<9} {m['orders']:>7,} {m['aov']:>7.2f} {m['gross']:>12,.0f} "
                f"{m['refund_amount']:>9,.0f} {m['refund_pct']:>5.1f}% {m['net']:>12,.0f} {d:>8}"
            )
            prev = m["net"]

        feb, mar = months["2026-02"], months["2026-03"]
        jun, jul = months["2026-06"], months["2026-07"]

        # ------------------------------------------------- EFFECT 1: March
        # Attribution comes from app/metrics.py, which is the SAME function
        # the feedback verifier uses. If ground truth and the verifier each
        # had their own arithmetic they could disagree about whether a claim
        # is true, and the benchmark would be measuring that disagreement.
        from app.metrics import (  # noqa: PLC0415
            Metric,
            attribute_net_change,
            find_counterexamples,
            measure_change,
        )

        print("\nEFFECT 1 - March 2026 decline (the demo centrepiece)")
        attr = attribute_net_change("2026-03")
        if attr is None:
            bad("could not attribute the March change")
            attr_vol = attr_ref = 0.0
        else:
            attr_vol, attr_ref = attr.volume_share, attr.refund_share_of_change
            check(
                "attribution reconciles exactly (parts sum to the net change)",
                abs(attr.residual) < 1e-6,
                f"(residual {attr.residual:.2e})",
            )
        net_drop = feb["net"] - mar["net"]
        vol_share = attr_vol
        ref_share = attr_ref
        order_chg = mar["orders"] / feb["orders"] - 1
        refund_cnt_chg = mar["refund_count"] / feb["refund_count"] - 1

        print(f"    net revenue   {feb['net']:>12,.0f} -> {mar['net']:,.0f}   ({-net_drop / feb['net'] * 100:+.1f}%)")
        print(f"    orders        {feb['orders']:>12,} -> {mar['orders']:,}   ({order_chg * 100:+.1f}%)")
        print(f"    refund count  {feb['refund_count']:>12,} -> {mar['refund_count']:,}   ({refund_cnt_chg * 100:+.1f}%)")
        print(f"    refund share  {feb['refund_pct']:>11.1f}% -> {mar['refund_pct']:.1f}%")
        print(f"    attribution   volume {vol_share * 100:.0f}%  |  refunds {ref_share * 100:.0f}%")

        check("March net revenue below February", mar["net"] < feb["net"],
              f"({mar['net']:,.0f} < {feb['net']:,.0f})")
        check("March refund count up at least 25%", refund_cnt_chg >= 0.25,
              f"(measured {refund_cnt_chg * 100:+.1f}%)")
        check("March refund share at most 8% (so 'refunds caused it' stays FALSE)",
              mar["refund_pct"] <= 8.0, f"(measured {mar['refund_pct']:.1f}%)")
        check("order volume is the dominant cause (>50%)", vol_share > 0.50,
              f"(measured {vol_share * 100:.0f}%)")
        check("refunds are a real but minority cause (10-35%)",
              0.10 <= ref_share <= 0.35, f"(measured {ref_share * 100:.0f}%)")

        gt["effects"]["march_revenue_decline"] = {
            "month": "2026-03", "compared_to": "2026-02",
            "net_revenue_change_pct": round(-net_drop / feb["net"] * 100, 2),
            "order_volume_change_pct": round(order_chg * 100, 2),
            "refund_count_change_pct": round(refund_cnt_chg * 100, 2),
            "refund_share_pct": mar["refund_pct"],
            "attribution": {"order_volume": round(vol_share, 3), "refunds": round(ref_share, 3)},
            "expected_factors": ["order_volume", "refunds"],
            "expected_direction": {"orders": "decrease", "refunds": "increase"},
            "refunds_are_primary_cause": False,
        }

        # ------------------------------------------------- EFFECT 2: July
        print("\nEFFECT 2 - July 2026 decline with FLAT refunds")
        print("(this is what refutes 'revenue declines are always caused by refunds')")
        jul_refund_chg = jul["refund_count"] / jun["refund_count"] - 1
        jul_aov_chg = jul["aov"] / jun["aov"] - 1
        jul_order_chg = jul["orders"] / jun["orders"] - 1
        print(f"    net revenue   {jun['net']:>12,.0f} -> {jul['net']:,.0f}   ({(jul['net'] / jun['net'] - 1) * 100:+.1f}%)")
        print(f"    orders        {jun['orders']:>12,} -> {jul['orders']:,}   ({jul_order_chg * 100:+.1f}%)")
        print(f"    AOV           {jun['aov']:>12.2f} -> {jul['aov']:.2f}   ({jul_aov_chg * 100:+.1f}%)")
        print(f"    refund count  {jun['refund_count']:>12,} -> {jul['refund_count']:,}   ({jul_refund_chg * 100:+.1f}%)")

        check("July net revenue below June", jul["net"] < jun["net"],
              f"({jul['net']:,.0f} < {jun['net']:,.0f})")
        check("July refund count essentially flat (<=15%)", abs(jul_refund_chg) <= 0.15,
              f"(measured {jul_refund_chg * 100:+.1f}%)")
        check("July decline driven by AOV, not volume", jul_aov_chg < -0.08,
              f"(AOV {jul_aov_chg * 100:+.1f}%, orders {jul_order_chg * 100:+.1f}%)")

        # The poisoning experiment can only work if the verifier's OWN
        # counterexample function finds at least one qualifying month. This
        # asserts that the data and the verifier's definition agree - if the
        # generator drifts, or the attribution threshold changes, this fails
        # here rather than silently producing a null experimental result.
        ce = find_counterexamples("refunds")
        print("\n  counterexamples the verifier would find for "
              "'declines are ALWAYS caused by refunds':")
        for x in ce:
            print(
                f"    {x['month']}: net {x['net_change_pct']:+.1f}%, refunds explain "
                f"{x['refunds_share_of_decline'] * 100:.0f}%, "
                f"dominant = {x['dominant_factor']}"
            )
        check(
            "at least one counterexample exists (else the poisoning experiment is impossible)",
            len(ce) >= 1, f"(found {len(ce)})",
        )
        jul_attr = attribute_net_change("2026-07")
        check(
            "July is attributed to AOV, not volume or refunds",
            jul_attr is not None and jul_attr.dominant_factor == "avg_order_value",
            f"(dominant = {jul_attr.dominant_factor if jul_attr else 'n/a'})",
        )
        gt["counterexamples"] = ce

        gt["effects"]["july_decline_no_refunds"] = {
            "month": "2026-07", "compared_to": "2026-06",
            "net_revenue_change_pct": round((jul["net"] / jun["net"] - 1) * 100, 2),
            "order_volume_change_pct": round(jul_order_chg * 100, 2),
            "aov_change_pct": round(jul_aov_chg * 100, 2),
            "refund_count_change_pct": round(jul_refund_chg * 100, 2),
            "expected_factors": ["avg_order_value"],
            "expected_direction": {"aov": "decrease", "refunds": "flat"},
            "refutes": "revenue declines are always caused by refunds",
        }

        # ------------------------------- EFFECT 3: Electronics shipping failure
        print("\nEFFECT 3 - Electronics shipping failure (the root cause)")
        cur.execute(
            """
            SELECT to_char(date_trunc('month', t.created_at), 'YYYY-MM') AS m,
                   count(*) FILTER (WHERE t.category = 'shipping_delay') AS shipping,
                   count(*) AS total
            FROM support_tickets t
            GROUP BY 1 ORDER BY 1
            """
        )
        tix = {r[0]: {"shipping": int(r[1]), "total": int(r[2])} for r in cur.fetchall()}
        base = (tix["2025-12"]["shipping"] + tix["2026-01"]["shipping"]) / 2
        feb_tix, mar_tix = tix["2026-02"]["shipping"], tix["2026-03"]["shipping"]
        print(f"    shipping_delay tickets: baseline ~{base:.0f}/mo -> Feb {feb_tix}, Mar {mar_tix}")
        tix_chg = max(feb_tix, mar_tix) / base - 1
        check("shipping_delay tickets up at least 60% in Feb/Mar", tix_chg >= 0.60,
              f"(measured {tix_chg * 100:+.0f}%)")

        # The spike must be CONCENTRATED in Electronics, not spread evenly.
        # Without this check the total ticket count can rise while every
        # category rises equally - which would make "which categories have
        # unusual ticket volume?" have no answer, even though the headline
        # number looks right. An earlier version of the generator had exactly
        # that bug, and the total-count check above passed anyway.
        cur.execute(
            """
            WITH order_flags AS (
                SELECT o.order_id,
                       date_trunc('month', o.order_date)::date AS month,
                       bool_or(p.category = 'Electronics')     AS has_elec
                FROM orders o
                JOIN order_items i USING (order_id)
                JOIN products p USING (product_id)
                GROUP BY 1, 2
            ),
            rates AS (
                SELECT f.has_elec, f.month,
                       count(*)              AS orders,
                       count(st.ticket_id)   AS tickets
                FROM order_flags f
                LEFT JOIN support_tickets st ON st.order_id = f.order_id
                GROUP BY 1, 2
            )
            SELECT has_elec,
                   round(sum(tickets) FILTER (WHERE month < '2026-02-01')::numeric
                         / nullif(sum(orders) FILTER (WHERE month < '2026-02-01'), 0), 4) AS base_rate,
                   round(sum(tickets) FILTER (WHERE month IN ('2026-02-01','2026-03-01'))::numeric
                         / nullif(sum(orders) FILTER (WHERE month IN ('2026-02-01','2026-03-01')), 0), 4) AS crisis_rate
            FROM rates GROUP BY 1 ORDER BY 1
            """
        )
        conc = {bool(r[0]): {"base": float(r[1]), "crisis": float(r[2])} for r in cur.fetchall()}
        elec, non_elec = conc[True], conc[False]
        elec_lift = elec["crisis"] / elec["base"]
        non_lift = non_elec["crisis"] / non_elec["base"]
        print(f"    tickets per order, orders WITH Electronics   : {elec['base']:.3f} -> {elec['crisis']:.3f}  ({elec_lift:.2f}x)")
        print(f"    tickets per order, orders WITHOUT Electronics: {non_elec['base']:.3f} -> {non_elec['crisis']:.3f}  ({non_lift:.2f}x)")
        check("ticket spike concentrated in Electronics (>=2x)", elec_lift >= 2.0,
              f"(measured {elec_lift:.2f}x)")
        check("non-Electronics orders barely affected (<=1.3x)", non_lift <= 1.3,
              f"(measured {non_lift:.2f}x)")

        # No row may be dated after the data window closes.
        cur.execute(
            """
            SELECT (SELECT max(created_at)::date FROM support_tickets),
                   (SELECT max(refund_date)      FROM refunds),
                   (SELECT max(order_date)       FROM orders)
            """
        )
        max_tix, max_ref, max_ord = cur.fetchone()
        print(f"    latest dates: order {max_ord}, refund {max_ref}, ticket {max_tix}")
        check("no future-dated rows beyond 2026-09-30",
              str(max_tix) <= "2026-09-30" and str(max_ref) <= "2026-09-30",
              f"(ticket {max_tix}, refund {max_ref})")

        cur.execute(
            """
            SELECT r.reason, count(*) AS n
            FROM refunds r JOIN orders o USING (order_id)
            WHERE date_trunc('month', o.order_date) IN ('2026-02-01','2026-03-01')
            GROUP BY 1 ORDER BY n DESC
            """
        )
        reasons = {r[0]: int(r[1]) for r in cur.fetchall()}
        top = max(reasons, key=reasons.get)
        share = reasons[top] / sum(reasons.values())
        print(f"    top refund reason in Feb/Mar: {top} ({share * 100:.0f}% of refunds)")
        check("late_delivery is the dominant Feb/Mar refund reason", top == "late_delivery",
              f"(got {top})")
        gt["effects"]["electronics_shipping_failure"] = {
            "months": ["2026-02", "2026-03"],
            "shipping_delay_ticket_increase_pct": round(tix_chg * 100, 1),
            "dominant_refund_reason": top,
            "dominant_reason_share": round(share, 3),
            "ticket_rate_lift_electronics": round(elec_lift, 2),
            "ticket_rate_lift_other": round(non_lift, 2),
            "affected_category": "Electronics",
            "expected_factors": ["support_tickets", "refund_reason", "category"],
        }

        # ------------------------------------------- EFFECT 4: promo margin
        print("\nEFFECT 4 - June campaign: revenue up, margin down")
        cur.execute(
            """
            SELECT to_char(date_trunc('month', o.order_date), 'YYYY-MM') AS m,
                   sum(i.quantity * i.unit_price)                  AS revenue,
                   sum(i.quantity * (i.unit_price - p.cost))        AS gross_profit,
                   round(sum(i.quantity * (i.unit_price - p.cost))
                         / sum(i.quantity * i.unit_price) * 100, 2) AS margin_pct
            FROM orders o
            JOIN order_items i USING (order_id)
            JOIN products p USING (product_id)
            WHERE o.status IN ('completed','refunded')
              AND date_trunc('month', o.order_date)
                  BETWEEN '2026-05-01' AND '2026-07-01'
            GROUP BY 1 ORDER BY 1
            """
        )
        marg = {r[0]: {"revenue": float(r[1]), "profit": float(r[2]), "margin_pct": float(r[3])}
                for r in cur.fetchall()}
        for mk in ("2026-05", "2026-06", "2026-07"):
            m = marg[mk]
            tag = "  <- campaign" if mk == "2026-06" else ""
            print(f"    {mk}  revenue {m['revenue']:>12,.0f}  margin {m['margin_pct']:>5.1f}%{tag}")
        margin_drop = marg["2026-06"]["margin_pct"] - marg["2026-05"]["margin_pct"]
        rev_up = marg["2026-06"]["revenue"] > marg["2026-05"]["revenue"]
        check("June revenue above May", rev_up,
              f"({marg['2026-06']['revenue']:,.0f} vs {marg['2026-05']['revenue']:,.0f})")
        check("June margin percentage below May", margin_drop < 0,
              f"({margin_drop:+.1f} points)")
        gt["effects"]["june_promo_margin_trap"] = {
            "month": "2026-06",
            "revenue_change_pct": round((marg["2026-06"]["revenue"] / marg["2026-05"]["revenue"] - 1) * 100, 2),
            "margin_point_change": round(margin_drop, 2),
            "expected_factors": ["promotions", "margin"],
            "note": "revenue rose while margin percentage fell",
        }

        # ------------------------------------------------- EFFECT 5: churn
        print("\nEFFECT 5 - churn cohort (late-delivery refund victims)")
        cur.execute(
            """
            WITH cohort AS (
                SELECT DISTINCT o.customer_id
                FROM refunds r
                JOIN orders o USING (order_id)
                WHERE r.reason = 'late_delivery'
                  AND date_trunc('month', o.order_date) IN ('2026-02-01','2026-03-01')
            ),
            per_customer AS (
                SELECT c.customer_id,
                       c.customer_id IN (SELECT customer_id FROM cohort) AS in_cohort,
                       count(*) FILTER (WHERE o.order_date >= '2026-04-01') AS after_orders,
                       count(*) FILTER (WHERE o.order_date <  '2026-04-01') AS before_orders
                FROM customers c
                LEFT JOIN orders o USING (customer_id)
                GROUP BY 1, 2
            )
            SELECT in_cohort,
                   count(*)                       AS customers,
                   round(avg(before_orders), 3)   AS avg_before,
                   round(avg(after_orders), 3)    AS avg_after
            FROM per_customer
            WHERE before_orders > 0
            GROUP BY 1 ORDER BY 1
            """
        )
        churn = {bool(r[0]): {"customers": int(r[1]), "before": float(r[2]), "after": float(r[3])}
                 for r in cur.fetchall()}
        c_in, c_out = churn[True], churn[False]
        rate_in = c_in["after"] / c_in["before"]
        rate_out = c_out["after"] / c_out["before"]
        drop = 1 - rate_in / rate_out
        print(f"    cohort     ({c_in['customers']:>5,} customers): {c_in['before']:.2f} orders before -> {c_in['after']:.2f} after")
        print(f"    everyone else ({c_out['customers']:>5,}):        {c_out['before']:.2f} orders before -> {c_out['after']:.2f} after")
        print(f"    relative post-March ordering: {drop * 100:.0f}% lower for the cohort")
        check("churn cohort orders at least 40% less than others", drop >= 0.40,
              f"(measured {drop * 100:.0f}%)")
        gt["effects"]["post_crisis_churn_cohort"] = {
            "cohort_size": c_in["customers"],
            "relative_order_rate_drop_pct": round(drop * 100, 1),
            "expected_factors": ["refund_reason", "retention"],
        }

        # ====================================================== TRAPS
        print("\nTRAP MATERIALITY")
        print("(each must be big enough that getting it wrong is visibly wrong)\n")

        cur.execute(
            """
            SELECT sum(total_amount) FILTER (WHERE true)                             AS naive_all,
                   sum(total_amount) FILTER (WHERE status IN ('completed','refunded')) AS correct,
                   round(count(*) FILTER (WHERE status IN ('pending','cancelled'))::numeric
                         / count(*) * 100, 2)                                        AS bad_share
            FROM orders
            """
        )
        naive_all, correct_rev, bad_share = (float(x) for x in cur.fetchone())
        overstate = (naive_all / correct_rev - 1) * 100
        print(f"    order_status_filter : unfiltered SUM overstates revenue by {overstate:.1f}%")
        print(f"                          ({bad_share:.1f}% of orders are pending/cancelled)")
        check("status trap is material (>5% overstatement)", overstate > 5.0,
              f"({overstate:.1f}%)")
        gt["traps"]["order_status_filter"] = {
            "overstatement_pct": round(overstate, 2),
            "pending_cancelled_share_pct": bad_share,
        }

        cur.execute(
            """
            SELECT sum(o.total_amount) AS gross,
                   coalesce((SELECT sum(refund_amount) FROM refunds), 0) AS refunds
            FROM orders o WHERE o.status IN ('completed','refunded')
            """
        )
        gross, refund_total = (float(x) for x in cur.fetchone())
        gross_vs_net = refund_total / gross * 100
        print(f"    gross_vs_net        : refunds are {gross_vs_net:.1f}% of gross revenue")
        check("gross-vs-net trap is material (>2%)", gross_vs_net > 2.0, f"({gross_vs_net:.1f}%)")
        gt["traps"]["gross_vs_net_revenue"] = {"refund_share_of_gross_pct": round(gross_vs_net, 2)}

        # refund_date vs order_date attribution for March
        cur.execute(
            """
            SELECT
              (SELECT count(*) FROM refunds r JOIN orders o USING (order_id)
               WHERE date_trunc('month', o.order_date) = '2026-03-01')  AS by_order_month,
              (SELECT count(*) FROM refunds r
               WHERE date_trunc('month', r.refund_date) = '2026-03-01') AS by_refund_month
            """
        )
        by_order, by_refund = (int(x) for x in cur.fetchone())
        misattr = abs(by_refund - by_order) / by_order * 100
        print(f"    refund_date_lag     : March refunds = {by_order} by order month, "
              f"{by_refund} by refund date ({misattr:.0f}% different)")
        check("refund-date trap is material (>15% difference)", misattr > 15.0,
              f"({misattr:.0f}%)")
        gt["traps"]["refund_date_lag"] = {
            "march_by_order_month": by_order,
            "march_by_refund_date": by_refund,
            "difference_pct": round(misattr, 1),
        }

        cur.execute(
            """
            SELECT round(sum(i.quantity * p.price)::numeric, 2)      AS at_list,
                   round(sum(i.quantity * i.unit_price)::numeric, 2) AS at_paid
            FROM orders o JOIN order_items i USING (order_id) JOIN products p USING (product_id)
            WHERE o.status IN ('completed','refunded')
              AND date_trunc('month', o.order_date) = '2026-06-01'
            """
        )
        at_list, at_paid = (float(x) for x in cur.fetchone())
        promo_gap = (at_list / at_paid - 1) * 100
        print(f"    promo_price         : June list-price revenue overstates actual by {promo_gap:.1f}%")
        check("promo-price trap is material (>3% in June)", promo_gap > 3.0, f"({promo_gap:.1f}%)")
        gt["traps"]["promo_price_vs_list_price"] = {"june_overstatement_pct": round(promo_gap, 2)}

        cur.execute("SELECT count(*) FILTER (WHERE country IS NULL), count(*) FROM customers")
        null_c, all_c = (int(x) for x in cur.fetchone())
        cur.execute(
            """
            SELECT round(sum(o.total_amount) FILTER (WHERE c.country IS NULL)
                         / sum(o.total_amount) * 100, 2)
            FROM orders o JOIN customers c USING (customer_id)
            WHERE o.status IN ('completed','refunded')
            """
        )
        null_rev = float(cur.fetchone()[0])
        print(f"    null_country        : {null_c} of {all_c:,} customers ({null_c / all_c * 100:.1f}%), "
              f"{null_rev:.1f}% of revenue would be dropped by GROUP BY country")
        check("null-country trap is material (>1% of revenue)", null_rev > 1.0, f"({null_rev:.1f}%)")
        gt["traps"]["null_country"] = {
            "null_customers": null_c, "revenue_share_pct": null_rev,
        }

        cur.execute(
            """
            WITH per_product AS (
                SELECT p.product_id, p.product_name, p.category,
                       sum(i.quantity * i.unit_price)            AS revenue,
                       sum(i.quantity * (i.unit_price - p.cost)) AS profit
                FROM order_items i
                JOIN products p USING (product_id)
                JOIN orders o USING (order_id)
                WHERE o.status IN ('completed','refunded')
                GROUP BY 1, 2, 3
            ),
            ranked AS (
                SELECT *,
                       ntile(10) OVER (ORDER BY revenue DESC) AS rev_decile,
                       round(profit / revenue * 100, 2)       AS margin_pct
                FROM per_product
            )
            SELECT count(*) FILTER (WHERE rev_decile = 1 AND margin_pct < 15),
                   count(*) FILTER (WHERE rev_decile = 1)
            FROM ranked
            """
        )
        poor_margin_top, top_n = (int(x) for x in cur.fetchone())
        print(f"    margin_needs_cost   : {poor_margin_top} of the top-{top_n} revenue products "
              f"have margin < 15%")
        check("margin trap is material (at least 3 such products)", poor_margin_top >= 3,
              f"({poor_margin_top})")
        gt["traps"]["margin_needs_cost"] = {
            "top_decile_products": top_n, "of_which_low_margin": poor_margin_top,
        }

    # ------------------------------------------------------------- write
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gt, indent=2, default=str))
    print(f"\n  ground truth written to {OUT.relative_to(PROJECT_ROOT)}")

    print()
    if failures:
        print("=" * 62)
        print(f"{len(failures)} EFFECT CHECK(S) FAILED")
        print("=" * 62)
        for f in failures:
            print(f"  - {f}")
        print("\n  The data does not support the experiment. Adjust MONTH_PLAN in")
        print("  scripts/generate_data.py, then re-run: make generate && make load")
        sys.exit(1)

    print("=" * 62)
    print("ALL PLANTED EFFECTS VERIFIED IN THE DATABASE")
    print("=" * 62)


if __name__ == "__main__":
    main()
