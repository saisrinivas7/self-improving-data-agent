"""
Generate the Lumen & Co. synthetic dataset.

Run:  make generate

Reads data/effects_manifest.yml and writes CSVs to data/generated/.
Everything is driven by DATA_SEED, so the same seed produces byte-identical
files and the benchmark is reproducible.

HOW IT WORKS, and why it is built this way

The hard requirement is that planted effects must actually appear in the
data. "Why did revenue decrease in March?" is unscoreable unless March
really does decline, for a reason we can state. So orders are generated
MONTH BY MONTH in chronological order, with each month's order count and
average basket scaled by multipliers derived from the manifest.

Chronological order is not a stylistic choice. The churn effect depends on
who received a late-delivery refund in Feb/March, and those customers must
then order less from April onward. That is only possible if February and
March have already been generated when April is being built.

Nothing here writes the "expected" answers anywhere. Ground truth is
measured back out of the seeded database by verify_effects.py, because
random noise shifts the realised numbers away from the targets.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "generated"
MANIFEST = PROJECT_ROOT / "data" / "effects_manifest.yml"

# ----------------------------------------------------------------------
# Monthly shape of the business.
#
# These multipliers ARE the story in data/effects_manifest.yml, expressed
# numerically. Each is relative to the October 2025 baseline.
#
#   orders : how many orders that month
#   aov    : average basket value multiplier (driven by basket size)
# ----------------------------------------------------------------------
#   orders  : how many orders that month, relative to the Oct 2025 baseline
#   aov     : average basket value multiplier (targeted directly, see below)
#   refund  : refund-rate multiplier - the knob that separates March from July
#   ticket  : support-ticket-rate multiplier
MONTH_PLAN: dict[str, dict[str, float]] = {
    "2025-10": {"orders": 1.00, "aov": 1.00, "refund": 1.00, "ticket": 1.0},  # baseline
    "2025-11": {"orders": 1.28, "aov": 1.02, "refund": 1.05, "ticket": 1.1},  # Black Friday
    "2025-12": {"orders": 1.34, "aov": 1.04, "refund": 1.10, "ticket": 1.2},  # Christmas
    "2026-01": {"orders": 0.78, "aov": 0.97, "refund": 1.00, "ticket": 1.0},  # post-holiday slump
    # Courier switched late Jan. TICKETS spike immediately as people chase
    # late deliveries, but REFUNDS lag: a customer complains first and asks
    # for money back weeks later. So February is loud in support and only
    # mildly elevated in refunds. Order volume has not reacted at all yet,
    # because reputation damage takes time to spread.
    #
    # This lag is deliberate and does double duty. It is realistic, and it
    # keeps February a usable comparison baseline for March - if February
    # were already at crisis refund levels, March's refund increase would
    # look small and the 78/22 attribution would collapse to 90/10.
    "2026-02": {"orders": 1.00, "aov": 1.00, "refund": 1.15, "ticket": 2.4},
    # THE DECLINE. -10% orders vs Feb, and the refund rate roughly doubles
    # as February's complaints convert into refunds. Together these give a
    # ~78% volume / ~22% refund split of the net revenue drop.
    "2026-03": {"orders": 0.90, "aov": 1.00, "refund": 1.80, "ticket": 2.2},
    "2026-04": {"orders": 0.94, "aov": 1.00, "refund": 1.10, "ticket": 1.2},  # fixed, churn drags
    "2026-05": {"orders": 0.98, "aov": 1.00, "refund": 1.00, "ticket": 1.0},
    "2026-06": {"orders": 1.10, "aov": 0.98, "refund": 1.00, "ticket": 1.0},  # win-back campaign
    # Campaign over. Orders +2% vs June but AOV -14%, so revenue FALLS while
    # refunds stay FLAT. This is the counterexample that refutes
    # "revenue declines are always caused by refunds".
    "2026-07": {"orders": 1.122, "aov": 0.843, "refund": 1.00, "ticket": 1.0},
    "2026-08": {"orders": 1.04, "aov": 1.00, "refund": 1.00, "ticket": 1.0},
    "2026-09": {"orders": 1.06, "aov": 1.01, "refund": 1.00, "ticket": 1.0},
}

# Baseline average basket value in currency units, before segment and
# monthly multipliers. Baskets are built to HIT a target value rather than
# emerging from random basket sizes, because the July effect needs average
# order value to move by a specific amount and an emergent AOV cannot be
# steered reliably.
BASE_BASKET = 250.0

CRISIS_MONTHS = {"2026-02", "2026-03"}
CRISIS_CATEGORY = "Electronics"
PROMO_MONTH = "2026-06"

CATEGORIES = [
    "Electronics",
    "Home & Kitchen",
    "Apparel",
    "Beauty",
    "Sports & Outdoors",
    "Office",
]
COUNTRIES = ["US", "UK", "Germany", "France", "Canada", "Australia", "Netherlands", "Sweden"]
COUNTRY_WEIGHTS = [0.34, 0.14, 0.12, 0.09, 0.09, 0.08, 0.08, 0.06]
SEGMENTS = ["Budget", "Standard", "Premium", "VIP"]
SEGMENT_WEIGHTS = [0.34, 0.40, 0.19, 0.07]
# Richer segments buy more per order.
SEGMENT_BASKET_MULT = {"Budget": 0.72, "Standard": 1.00, "Premium": 1.45, "VIP": 2.10}

ORDER_STATUSES = ["completed", "pending", "cancelled", "refunded"]

REFUND_REASONS = ["late_delivery", "damaged", "wrong_item", "not_as_described", "changed_mind"]
NORMAL_REFUND_MIX = [0.18, 0.22, 0.17, 0.21, 0.22]
# During the crisis, late_delivery dominates - this is the fingerprint the
# agent should find when it looks at refund REASONS rather than just counts.
CRISIS_REFUND_MIX = [0.55, 0.14, 0.11, 0.10, 0.10]

TICKET_CATEGORIES = [
    "shipping_delay",
    "damaged_item",
    "billing",
    "product_question",
    "return_request",
    "other",
]
NORMAL_TICKET_MIX = [0.16, 0.17, 0.15, 0.26, 0.16, 0.10]
CRISIS_TICKET_MIX = [0.62, 0.11, 0.06, 0.11, 0.07, 0.03]

TICKET_RESOLUTIONS = ["resolved", "escalated", "refunded", "closed_no_action", "open"]
TICKET_RES_WEIGHTS = [0.52, 0.14, 0.16, 0.13, 0.05]

# Price bands per category: (min, max, typical margin fraction)
CATEGORY_PRICING = {
    "Electronics": (39.0, 1299.0, 0.22),
    "Home & Kitchen": (14.0, 349.0, 0.38),
    "Apparel": (12.0, 189.0, 0.52),
    "Beauty": (6.0, 129.0, 0.58),
    "Sports & Outdoors": (18.0, 549.0, 0.34),
    "Office": (4.0, 299.0, 0.30),
}

PRODUCT_NOUNS = {
    "Electronics": ["Headphones", "Monitor", "Keyboard", "Webcam", "Speaker", "Router",
                    "Tablet", "Charger", "SSD", "Microphone", "Earbuds", "Hub"],
    "Home & Kitchen": ["Kettle", "Blender", "Cookware Set", "Knife Block", "Toaster",
                       "Air Fryer", "Mug Set", "Storage Jars", "Lamp", "Cushion"],
    "Apparel": ["T-Shirt", "Hoodie", "Jacket", "Trousers", "Socks", "Cap", "Scarf",
                "Trainers", "Shorts", "Jumper"],
    "Beauty": ["Serum", "Moisturiser", "Shampoo", "Lip Balm", "Cleanser", "Sunscreen",
               "Hair Oil", "Face Mask", "Toner"],
    "Sports & Outdoors": ["Yoga Mat", "Dumbbell Set", "Water Bottle", "Backpack", "Tent",
                          "Bike Lights", "Resistance Bands", "Running Belt"],
    "Office": ["Notebook", "Desk Organiser", "Pen Set", "Chair Mat", "Stapler",
               "Whiteboard", "Desk Lamp", "File Box", "Label Maker"],
}
PRODUCT_ADJECTIVES = ["Pro", "Lite", "Classic", "Compact", "Premium", "Essential",
                      "Studio", "Max", "Mini", "Everyday", "Nordic", "Urban"]


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_range(start: date, end: date) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def days_in_month(key: str) -> list[date]:
    y, m = int(key[:4]), int(key[5:7])
    nxt = date(y + (m == 12), 1 if m == 12 else m + 1, 1)
    first = date(y, m, 1)
    return [first + timedelta(days=i) for i in range((nxt - first).days)]


def main() -> None:
    import yaml

    from app.config import get_settings

    s = get_settings()
    manifest = yaml.safe_load(MANIFEST.read_text())

    period = manifest["period"]
    volume = manifest["volume"]

    start = period["start"] if isinstance(period["start"], date) else datetime.strptime(str(period["start"]), "%Y-%m-%d").date()
    end = period["end"] if isinstance(period["end"], date) else datetime.strptime(str(period["end"]), "%Y-%m-%d").date()

    n_customers = int(volume["customers"])
    n_products = int(volume["products"])
    n_orders_target = int(volume["orders"])
    refund_rate = float(volume["target_refund_rate"])
    ticket_rate = float(volume["target_ticket_rate"])

    rng = random.Random(s.data_seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    months = month_range(start, end)
    print(f"Generating Lumen & Co.  seed={s.data_seed}")
    print(f"  period   : {start} -> {end}  ({len(months)} months)")
    print(f"  target   : {n_customers:,} customers, {n_products} products, ~{n_orders_target:,} orders\n")

    # ------------------------------------------------------------------
    # 1. CUSTOMERS
    # Signup dates are spread from a year before the window to its end, so
    # there are both long-standing and brand-new customers. ~3% have a NULL
    # country, which is a planted trap for GROUP BY country.
    # ------------------------------------------------------------------
    # 72% signed up BEFORE the window opens, the rest during it. That matters:
    # if signups were spread evenly across the window, early months would have
    # far fewer eligible customers than late ones, and the resulting growth
    # artefact would dwarf every planted effect.
    customers: list[dict] = []
    signup_start = start - timedelta(days=730)
    for cid in range(1, n_customers + 1):
        seg = rng.choices(SEGMENTS, weights=SEGMENT_WEIGHTS)[0]
        country = rng.choices(COUNTRIES, weights=COUNTRY_WEIGHTS)[0]
        if rng.random() < 0.03:
            country = ""  # NULL on load
        if rng.random() < 0.72:
            signup = signup_start + timedelta(days=rng.randint(0, (start - signup_start).days - 1))
        else:
            signup = start + timedelta(days=rng.randint(0, (end - start).days))
        customers.append(
            {
                "customer_id": cid,
                "name": f"Customer {cid:05d}",
                "country": country,
                "signup_date": signup.isoformat(),
                "customer_segment": seg,
            }
        )

    # Sorted by signup date so we can sample only from customers who already
    # exist on a given day, via bisect. Previously an ineligible pick simply
    # dropped the order, which both lost ~26% of the target volume and
    # created the growth artefact described above.
    customers.sort(key=lambda c: c["signup_date"])
    signup_dates = [c["signup_date"] for c in customers]
    print(
        f"  customers: {len(customers):,}  "
        f"({sum(1 for c in customers if not c['country'])} with NULL country, "
        f"{sum(1 for c in customers if c['signup_date'] < start.isoformat()):,} pre-existing)"
    )

    # ------------------------------------------------------------------
    # 2. PRODUCTS
    # Cost is derived from price via a category margin, with noise - so some
    # products are genuinely low-margin despite high revenue. That is what
    # makes "high sales but poor margins" a real question.
    # ------------------------------------------------------------------
    products: list[dict] = []
    for pid in range(1, n_products + 1):
        cat = CATEGORIES[pid % len(CATEGORIES)]
        lo, hi, margin = CATEGORY_PRICING[cat]
        # Log-uniform so cheap items dominate, as in a real catalogue.
        price = round(lo * (hi / lo) ** rng.random(), 2)
        realised_margin = max(0.04, min(0.75, rng.gauss(margin, 0.12)))
        cost = round(price * (1 - realised_margin), 2)
        cost = max(0.5, min(cost, round(price * 0.96, 2)))
        noun = rng.choice(PRODUCT_NOUNS[cat])
        adj = rng.choice(PRODUCT_ADJECTIVES)
        products.append(
            {
                "product_id": pid,
                "product_name": f"{adj} {noun}",
                "category": cat,
                "price": f"{price:.2f}",
                "cost": f"{cost:.2f}",
            }
        )
    by_category: dict[str, list[dict]] = defaultdict(list)
    for p in products:
        by_category[p["category"]].append(p)
    print(f"  products : {len(products)} across {len(by_category)} categories")

    # ------------------------------------------------------------------
    # 3. PROMOTIONS
    # A background trickle all year, plus the big June win-back campaign.
    # ------------------------------------------------------------------
    promotions: list[dict] = []
    promo_id = 1
    # Background promos: a few per month on random products.
    for mk in months:
        dim = days_in_month(mk)
        for _ in range(rng.randint(3, 8)):
            p = rng.choice(products)
            s_day = rng.choice(dim[: max(1, len(dim) - 7)])
            promotions.append(
                {
                    "promotion_id": promo_id,
                    "product_id": p["product_id"],
                    "start_date": s_day.isoformat(),
                    "end_date": (s_day + timedelta(days=rng.randint(5, 14))).isoformat(),
                    "discount_percent": f"{rng.uniform(5, 20):.2f}",
                }
            )
            promo_id += 1
    # The June campaign: 22% of the catalogue, 15-35% off, whole month.
    june_days = days_in_month(PROMO_MONTH)
    promoted = rng.sample(products, int(len(products) * 0.22))
    for p in promoted:
        promotions.append(
            {
                "promotion_id": promo_id,
                "product_id": p["product_id"],
                "start_date": june_days[0].isoformat(),
                "end_date": june_days[-1].isoformat(),
                "discount_percent": f"{rng.uniform(15, 35):.2f}",
            }
        )
        promo_id += 1
    print(f"  promos   : {len(promotions)} ({len(promoted)} in the June campaign)")

    # Fast lookup: product_id -> list of (start, end, discount)
    promo_index: dict[int, list[tuple[date, date, float]]] = defaultdict(list)
    for pr in promotions:
        promo_index[pr["product_id"]].append(
            (
                date.fromisoformat(pr["start_date"]),
                date.fromisoformat(pr["end_date"]),
                float(pr["discount_percent"]),
            )
        )

    def effective_price(product: dict, when: date) -> float:
        """Price actually paid, applying any promotion live on that date."""
        base = float(product["price"])
        for s_d, e_d, disc in promo_index.get(product["product_id"], ()):
            if s_d <= when <= e_d:
                return round(base * (1 - disc / 100.0), 2)
        return base

    # Products sorted by list price, so a basket can be assembled to hit a
    # target value: bisect to roughly the right price, then pick from a
    # window around it. This is what makes average order value controllable,
    # which the July effect depends on.
    import bisect

    price_sorted = sorted(products, key=lambda p: float(p["price"]))
    price_keys = [float(p["price"]) for p in price_sorted]

    def pick_product_near(target_price: float, avoid: str | None = None) -> dict:
        """A product whose list price is near target_price."""
        i = bisect.bisect_left(price_keys, target_price)
        lo = max(0, i - 25)
        hi = min(len(price_sorted), i + 25)
        for _ in range(4):
            cand = price_sorted[rng.randrange(lo, hi)]
            if avoid is None or cand["category"] != avoid:
                return cand
        return price_sorted[rng.randrange(lo, hi)]

    # ------------------------------------------------------------------
    # 4. ORDERS, ITEMS, REFUNDS, TICKETS  - month by month, in order
    # ------------------------------------------------------------------
    total_weight = sum(MONTH_PLAN[m]["orders"] for m in months)
    base_per_month = n_orders_target / total_weight

    orders: list[dict] = []
    order_items: list[dict] = []
    refunds: list[dict] = []
    tickets: list[dict] = []

    order_id = 1
    refund_id = 1
    ticket_id = 1

    # Customers who received a late-delivery refund during the crisis. From
    # April onward they order far less. This is the churn cohort, and it can
    # only be built because months are generated in chronological order.
    churned: set[int] = set()
    monthly_summary: list[dict] = []

    for mk in months:
        plan = MONTH_PLAN[mk]
        dim = days_in_month(mk)
        n_month_orders = int(round(base_per_month * plan["orders"]))

        # Post-crisis churn: suppress the affected cohort's ordering.
        suppress = churned if mk >= "2026-04" else set()

        month_gross = 0.0
        month_orders = 0
        month_refund_amt = 0.0
        month_refund_cnt = 0

        for _ in range(n_month_orders):
            when = rng.choice(dim)
            # Only customers who had signed up by this date are eligible.
            eligible = bisect.bisect_right(signup_dates, when.isoformat())
            if eligible == 0:
                continue

            # Choose a customer, suppressing the churn cohort after March.
            cust = customers[rng.randrange(eligible)]
            for _attempt in range(5):
                if cust["customer_id"] in suppress and rng.random() < 0.64:
                    cust = customers[rng.randrange(eligible)]
                    continue
                break

            # --- build a basket to HIT a target value
            seg_mult = SEGMENT_BASKET_MULT[cust["customer_segment"]]
            target = BASE_BASKET * seg_mult * plan["aov"] * rng.lognormvariate(0, 0.45)
            n_lines = max(1, min(5, int(rng.triangular(1, 2, 4))))
            per_line = target / n_lines

            lines: list[tuple[dict, int, float]] = []
            chosen: set[int] = set()
            for _ in range(n_lines):
                # During the crisis, Electronics is under-represented in new
                # orders as reputation damage bites.
                avoid = (
                    CRISIS_CATEGORY
                    if (mk in CRISIS_MONTHS and rng.random() < 0.30)
                    else None
                )
                # Aim a bit below per_line so quantity can make up the rest.
                prod = pick_product_near(per_line * rng.uniform(0.45, 1.0), avoid)
                if prod["product_id"] in chosen:
                    continue
                chosen.add(prod["product_id"])
                unit = effective_price(prod, when)
                qty = max(1, min(8, int(round(per_line / max(unit, 1.0)))))
                lines.append((prod, qty, unit))

            if not lines:
                continue

            # Does this order contain the category that is actually failing?
            # Computed here, before the status branch, because BOTH refunds
            # and support tickets need it. Previously it was scoped inside
            # the refunds block, so the ticket spike was applied uniformly
            # across all categories - which made the claim "Electronics has
            # unusual ticket volume" false in the data even though the
            # overall ticket count rose.
            has_electronics = any(p["category"] == CRISIS_CATEGORY for p, _, _ in lines)

            gross = round(sum(q * u for _, q, u in lines), 2)

            # --- status. cancelled + pending are ~12% of rows: the planted
            #     trap for unfiltered SUM(total_amount).
            roll = rng.random()
            if roll < 0.075:
                status = "pending"
            elif roll < 0.12:
                status = "cancelled"
            else:
                status = "completed"

            orders.append(
                {
                    "order_id": order_id,
                    "customer_id": cust["customer_id"],
                    "order_date": when.isoformat(),
                    "status": status,
                    "total_amount": f"{gross:.2f}",
                }
            )
            for prod, qty, unit in lines:
                order_items.append(
                    {
                        "order_id": order_id,
                        "product_id": prod["product_id"],
                        "quantity": qty,
                        "unit_price": f"{unit:.2f}",
                    }
                )

            if status in ("completed",):
                month_gross += gross
                month_orders += 1

            # --- refunds, only on real sales
            if status == "completed":
                # The month's refund multiplier is the single knob that makes
                # March and July different stories: March's rate climbs while
                # July's stays flat, so July can refute "always refunds".
                # Within a crisis month the increase is concentrated in
                # Electronics, which is what makes the category visible when
                # the agent looks at refund reasons.
                rate = refund_rate * plan["refund"]
                if mk in CRISIS_MONTHS and has_electronics:
                    rate *= 1.8
                if rng.random() < rate:
                    mix = CRISIS_REFUND_MIX if (mk in CRISIS_MONTHS and has_electronics) else NORMAL_REFUND_MIX
                    reason = rng.choices(REFUND_REASONS, weights=mix)[0]
                    # Refund is PAID 12-45 days after the order: the customer
                    # raises a claim, ships the item back, it is inspected,
                    # then the money moves. The great majority therefore land
                    # in a LATER month than the order.
                    #
                    # That lag is the whole point of this trap. Grouping
                    # refunds by refund_date attributes them to the wrong
                    # month and smears the March spike into April, which is
                    # exactly the kind of error an analyst has to correct.
                    r_date = when + timedelta(days=rng.randint(12, 45))
                    # A refund dated past the end of the data window has not
                    # been processed yet, so it does not exist. Without this
                    # the table would hold future-dated rows.
                    if r_date <= end:
                        # Partial refunds are common; a full refund flips the
                        # order's status to 'refunded'.
                        full = rng.random() < 0.55
                        amount = gross if full else round(gross * rng.uniform(0.25, 0.8), 2)
                        amount = max(0.01, amount)
                        if full:
                            orders[-1]["status"] = "refunded"
                        refunds.append(
                            {
                                "refund_id": refund_id,
                                "order_id": order_id,
                                "refund_date": r_date.isoformat(),
                                "refund_amount": f"{amount:.2f}",
                                "reason": reason,
                            }
                        )
                        refund_id += 1
                        # Attributed to the ORDER's month, which is the
                        # correct attribution. refund_date is 12-45 days
                        # later and usually falls in a later month - that gap
                        # is the planted trap.
                        month_refund_amt += amount
                        month_refund_cnt += 1
                        if reason == "late_delivery" and mk in CRISIS_MONTHS:
                            churned.add(cust["customer_id"])

            # --- support tickets
            # The crisis multiplier applies ONLY to orders containing the
            # failing category. A courier problem with Electronics does not
            # make people complain about Beauty products. Gating on this is
            # what makes "which categories have unusual ticket volume?" have
            # Electronics as its answer, instead of a uniform rise everywhere.
            in_crisis = mk in CRISIS_MONTHS and has_electronics
            t_rate = ticket_rate * (plan["ticket"] if in_crisis else 1.0)
            if rng.random() < t_rate:
                mix = CRISIS_TICKET_MIX if in_crisis else NORMAL_TICKET_MIX
                # Same future-dating guard as refunds: a ticket raised after
                # the data window closes does not exist yet.
                t_date = when + timedelta(days=rng.randint(0, 9))
                if t_date <= end:
                    tickets.append(
                        {
                            "ticket_id": ticket_id,
                            "customer_id": cust["customer_id"],
                            "order_id": order_id,
                            "created_at": datetime.combine(t_date, datetime.min.time())
                            .replace(hour=rng.randint(7, 21), minute=rng.randint(0, 59))
                            .isoformat(sep=" "),
                            "category": rng.choices(TICKET_CATEGORIES, weights=mix)[0],
                            "resolution": rng.choices(TICKET_RESOLUTIONS, weights=TICKET_RES_WEIGHTS)[0],
                        }
                    )
                    ticket_id += 1

            order_id += 1

        monthly_summary.append(
            {
                "month": mk,
                "orders": month_orders,
                "gross": month_gross,
                "refund_amt": month_refund_amt,
                "refund_cnt": month_refund_cnt,
                "net": month_gross - month_refund_amt,
                "aov": month_gross / month_orders if month_orders else 0.0,
                "refund_share": month_refund_amt / month_gross if month_gross else 0.0,
            }
        )

    # A few tickets not tied to any order (billing questions and the like).
    for _ in range(int(len(tickets) * 0.12)):
        cust = rng.choice(customers)
        mk = rng.choice(months)
        when = rng.choice(days_in_month(mk))
        tickets.append(
            {
                "ticket_id": ticket_id,
                "customer_id": cust["customer_id"],
                "order_id": "",  # NULL
                "created_at": datetime.combine(when, datetime.min.time())
                .replace(hour=rng.randint(7, 21), minute=rng.randint(0, 59))
                .isoformat(sep=" "),
                "category": rng.choices(TICKET_CATEGORIES, weights=NORMAL_TICKET_MIX)[0],
                "resolution": rng.choices(TICKET_RESOLUTIONS, weights=TICKET_RES_WEIGHTS)[0],
            }
        )
        ticket_id += 1

    # ------------------------------------------------------------------
    # 5. WRITE CSVs
    # ------------------------------------------------------------------
    def write(name: str, rows: list[dict], cols: list[str]) -> None:
        path = OUT_DIR / f"{name}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {name + '.csv':<22} {len(rows):>8,} rows  ({path.stat().st_size / 1e6:.1f} MB)")

    print()
    write("customers", customers, ["customer_id", "name", "country", "signup_date", "customer_segment"])
    write("products", products, ["product_id", "product_name", "category", "price", "cost"])
    write("orders", orders, ["order_id", "customer_id", "order_date", "status", "total_amount"])
    write("order_items", order_items, ["order_id", "product_id", "quantity", "unit_price"])
    write("refunds", refunds, ["refund_id", "order_id", "refund_date", "refund_amount", "reason"])
    write("support_tickets", tickets, ["ticket_id", "customer_id", "order_id", "created_at", "category", "resolution"])
    write("promotions", promotions, ["promotion_id", "product_id", "start_date", "end_date", "discount_percent"])

    # ------------------------------------------------------------------
    # 6. QUICK SHAPE CHECK
    # Printed for eyeballing only. The authoritative check is
    # verify_effects.py, which measures against the loaded database.
    # ------------------------------------------------------------------
    print("\n  monthly shape (real sales, refunds attributed to the ORDER month):")
    print(
        f"    {'month':<9} {'orders':>7} {'AOV':>7} {'gross':>12} "
        f"{'refunds':>9} {'ref%':>6} {'net':>12} {'net Δ':>8}"
    )
    prev_net = None
    for row in monthly_summary:
        delta = ""
        if prev_net:
            delta = f"{(row['net'] - prev_net) / prev_net * 100:+6.1f}%"
        marker = ""
        if row["month"] == "2026-03":
            marker = "  <- THE DECLINE (volume + refunds)"
        elif row["month"] == "2026-07":
            marker = "  <- DECLINE, refunds flat"
        elif row["month"] == "2026-06":
            marker = "  <- promo campaign"
        print(
            f"    {row['month']:<9} {row['orders']:>7,} {row['aov']:>7.2f} "
            f"{row['gross']:>12,.0f} {row['refund_amt']:>9,.0f} "
            f"{row['refund_share'] * 100:>5.1f}% {row['net']:>12,.0f} {delta:>8}{marker}"
        )
        prev_net = row["net"]

    # ---- the two effects the whole experiment depends on ----
    by_month = {r["month"]: r for r in monthly_summary}
    feb, mar = by_month["2026-02"], by_month["2026-03"]
    jun, jul = by_month["2026-06"], by_month["2026-07"]

    print("\n  EFFECT 1 - March decline, and its attribution:")
    net_drop = feb["net"] - mar["net"]
    # Volume component: what the drop would have been at February's refund rate.
    vol_component = (feb["gross"] - mar["gross"]) * (1 - feb["refund_share"])
    ref_component = mar["gross"] * (mar["refund_share"] - feb["refund_share"])
    print(f"    net revenue      {feb['net']:>12,.0f} -> {mar['net']:,.0f}  ({-net_drop / feb['net'] * 100:+.1f}%)")
    print(f"    order volume     {feb['orders']:>12,} -> {mar['orders']:,}  ({(mar['orders'] / feb['orders'] - 1) * 100:+.1f}%)")
    print(f"    refund rate      {feb['refund_share'] * 100:>11.1f}% -> {mar['refund_share'] * 100:.1f}%")
    print(f"    refund count     {feb['refund_cnt']:>12,} -> {mar['refund_cnt']:,}  ({(mar['refund_cnt'] / feb['refund_cnt'] - 1) * 100:+.1f}%)")
    if net_drop > 0:
        print(f"    attribution      volume {vol_component / net_drop * 100:.0f}%  |  refunds {ref_component / net_drop * 100:.0f}%")
        print("      -> 'refunds increased substantially' is TRUE (verifiable)")
        print("      -> 'refunds CAUSED the decline' is FALSE (volume dominates)")
    else:
        print("    WARNING: March net revenue did NOT fall")

    print("\n  EFFECT 2 - July decline with flat refunds (refutes 'always refunds'):")
    print(f"    net revenue      {jun['net']:>12,.0f} -> {jul['net']:,.0f}  ({(jul['net'] / jun['net'] - 1) * 100:+.1f}%)")
    print(f"    order volume     {jun['orders']:>12,} -> {jul['orders']:,}  ({(jul['orders'] / jun['orders'] - 1) * 100:+.1f}%)")
    print(f"    AOV              {jun['aov']:>12.2f} -> {jul['aov']:.2f}  ({(jul['aov'] / jun['aov'] - 1) * 100:+.1f}%)")
    print(f"    refund count     {jun['refund_cnt']:>12,} -> {jul['refund_cnt']:,}  ({(jul['refund_cnt'] / jun['refund_cnt'] - 1) * 100:+.1f}%)")
    if jul["net"] >= jun["net"]:
        print("    WARNING: July net revenue did NOT fall - the counterexample is broken")

    print(f"\n  churn cohort: {len(churned):,} customers with a late-delivery refund in Feb/Mar")
    print(f"  refunds     : {len(refunds):,}  tickets: {len(tickets):,}")
    print("\n  Next: make load   (load into Postgres)")


if __name__ == "__main__":
    main()
