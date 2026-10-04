"""
Feedback verification (spec section 12).

Decides whether a piece of analyst feedback should be VERIFIED, REJECTED,
CONFLICTING or left PENDING.

THE DIVISION OF LABOUR, WHICH IS THE WHOLE DESIGN

    the model      maps a claim onto a metric from a fixed enum
    app/metrics    computes the number
    app/policy     decides what the number means

The model is never asked "is this feedback correct?". That question invites
it to agree with whatever it was shown, and its answer would be unfalsifiable.
It is also never asked to write the measuring SQL, because we watched it fall
into this dataset's planted traps - grouping refunds by `refund_date` instead
of the order's month, and dropping `refunded` orders from revenue. A verifier
that makes those mistakes rejects true feedback and accepts false feedback.

So its only job is translation: turn "refunds jumped in March" into
{metric: refund_count, month: 2026-03, compare_to: prior_month,
 direction: increase, magnitude_word: jumped}. That output is schema-
constrained, so it cannot invent a metric. Everything numeric is ours.

THREE ROUTES, ONE PER CLAIM KIND

  empirical   "refunds jumped in March"
              -> measure the metric, compare direction and magnitude

  causal      "refunds caused the decline"
              -> attribute the change; a cause must explain >50%
              "declines are ALWAYS caused by refunds"
              -> search every month for a counterexample; one is enough

  procedural  "rank customers by spend, not id"
              -> not true or false. Check it is executable against the
                 schema and that it does not contradict a stored lesson.

WHAT "COULD NOT CHECK" MEANS

If a claim cannot be mapped to the metric catalogue, the result is PENDING,
never REJECTED. Rejecting what we failed to understand would discard good
feedback whenever our own translation step had a bad day, and that failure
would be invisible.
"""

from __future__ import annotations

from typing import Any

from app.feedback.policy import (
    CAUSAL_DOMINANCE,
    CAUSAL_MINIMUM,
    FLAT_BAND_PCT,
    ClaimResult,
    LessonKind,
    Status,
    Verdict,
    VerificationOutcome,
    decide,
    required_magnitude,
)
from app.feedback.processor import Claim, ProcessedFeedback
from app.llm import LLMClient
from app.metrics import (
    Metric,
    all_months,
    attribute_net_change,
    category_month_tickets,
    find_counterexamples,
    measure_change,
    monthly_series,
)

# Factors a causal claim can name, mapped to what attribution calls them.
CAUSAL_FACTORS = {
    "refunds": "refunds",
    "refund": "refunds",
    "order_volume": "order_volume",
    "orders": "order_volume",
    "volume": "order_volume",
    "avg_order_value": "avg_order_value",
    "aov": "avg_order_value",
    "average_order_value": "avg_order_value",
    "price": "avg_order_value",
    "discounting": "avg_order_value",
}

VALID_TABLES = {
    "customers", "products", "orders", "order_items",
    "refunds", "support_tickets", "promotions",
}

# ------------------------------------------------------- claim -> metric spec

SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "checkable": {"type": "boolean"},
        "metric": {
            "type": "string",
            "enum": [m.value for m in Metric] + ["none"],
        },
        "month": {"type": "string"},
        "compare_to": {"type": "string"},
        "direction": {"type": "string", "enum": ["increase", "decrease", "flat", "none"]},
        "magnitude_word": {"type": "string"},
        "magnitude_pct": {"type": "number"},
        "category": {"type": "string"},
        "causal_factor": {
            "type": "string",
            "enum": ["refunds", "order_volume", "avg_order_value", "none"],
        },
        "causal_target_month": {"type": "string"},
    },
    # causal_factor MUST be required. Left optional, the model omitted it
    # every time, so every causal claim came back "no factor" and landed in
    # PENDING - including the poisoned universal claim the whole experiment
    # turns on. An optional field in a schema is a field the model will drop.
    "required": ["checkable", "metric", "direction", "causal_factor"],
}

# Deterministic fallback for the causal factor. The schema now requires the
# field, but the two claims this project depends on must not rest on the
# model filling one field correctly, so the claim text is also scanned
# directly. Belt and braces for the one check that cannot be allowed to fail.
FACTOR_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("refund", "refunds", "refunded", "return"), "refunds"),
    (("order volume", "order count", "fewer orders", "volume", "number of orders"),
     "order_volume"),
    (("average order value", "aov", "basket size", "order value", "discount",
      "discounting", "price"), "avg_order_value"),
)


def _infer_factor(text: str) -> str:
    low = text.lower()
    for words, factor in FACTOR_KEYWORDS:
        if any(w in low for w in words):
            return factor
    return "none"

SPEC_SYSTEM = """You translate a statement about business data into a \
machine-readable specification. You do NOT judge whether the statement is \
true - something else measures that. Translate only what the statement \
actually says."""

SPEC_USER = """Statement from an analyst:
"{claim}"

It was made while discussing this question:
{question}

Available metrics, and nothing else:
  net_revenue       revenue after refunds are subtracted
  gross_revenue     revenue before refunds
  order_count       number of real orders
  avg_order_value   mean value per order
  refund_count      number of refunds
  refund_amount     total refunded money
  refund_share      refunds as a fraction of gross revenue
  ticket_count      number of support tickets
  margin_pct        gross profit as a fraction of revenue

Months run from 2025-10 to 2026-09, written as YYYY-MM.

Fill in:
- checkable: true only if this statement is about a measurable quantity in \
a specific period. False for pure advice about how to analyse.
- metric: the single metric the statement is about, or "none".
- month: the month it refers to, as YYYY-MM. Empty if unspecified.
- compare_to: the month it is compared against, as YYYY-MM. Use \
"prior_month" if the statement implies a comparison to the previous month \
without naming it. Empty if there is no comparison.
- direction: increase, decrease, flat, or none.
- magnitude_word: the exact word describing the size of the change if there \
is one, such as substantially, jumped, slightly, doubled. Empty otherwise.
- magnitude_pct: a specific percentage if the statement names one, else 0.
- category: a product category if one is named, else empty.
- causal_factor: if the statement claims something CAUSED a revenue change, \
which factor it blames. Otherwise "none".
- causal_target_month: the month whose change is being explained, if the \
statement makes a causal claim."""


def _map_claim(claim: Claim, question: str, llm: LLMClient) -> dict:
    resp = llm.complete(
        SPEC_USER.format(claim=claim.text, question=question or "(not recorded)"),
        system=SPEC_SYSTEM,
        json_schema=SPEC_SCHEMA,
        max_tokens=500,
    )
    try:
        return resp.json()
    except ValueError:
        return {"checkable": False, "metric": "none", "direction": "none"}


def _resolve_month(spec: dict, key: str = "month") -> str | None:
    raw = str(spec.get(key) or "").strip()
    if not raw or raw.lower() in {"none", "prior_month", ""}:
        return None
    months = set(monthly_series())
    if raw in months:
        return raw
    # Tolerate "March 2026" and "2026-3".
    try:
        y, m = raw.replace("/", "-").split("-")[:2]
        cand = f"{int(y):04d}-{int(m):02d}"
        return cand if cand in months else None
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------- the checks

def _check_empirical(claim: Claim, spec: dict, question: str) -> ClaimResult:
    metric_name = str(spec.get("metric", "none"))
    if not spec.get("checkable") or metric_name == "none":
        return ClaimResult(
            claim.text, "empirical", Verdict.UNCHECKABLE, "no_metric",
            "the statement does not name a metric we can measure",
        )
    try:
        metric = Metric(metric_name)
    except ValueError:
        return ClaimResult(
            claim.text, "empirical", Verdict.UNCHECKABLE, "unknown_metric",
            f"'{metric_name}' is not in the metric catalogue",
        )

    month = _resolve_month(spec)
    if month is None:
        return ClaimResult(
            claim.text, "empirical", Verdict.UNCHECKABLE, "no_month",
            "the statement does not name a month that exists in the data",
        )

    # Category-specific ticket claims use a different series.
    category = str(spec.get("category") or "").strip()
    if category and metric is Metric.TICKET_COUNT:
        return _check_category_tickets(claim, month, category)

    compare_to = _resolve_month(spec, "compare_to")
    change = measure_change(metric, month, compare_to)
    if change is None:
        return ClaimResult(
            claim.text, "empirical", Verdict.UNCHECKABLE, "not_measurable",
            f"could not measure {metric.value} for {month}",
        )

    claimed_dir = str(spec.get("direction", "none"))
    need_pct = (
        float(spec.get("magnitude_pct") or 0)
        or required_magnitude(str(spec.get("magnitude_word") or ""))
    )
    measured = {
        "metric": metric.value,
        "month": change.month,
        "compare_to": change.compare_to,
        "before": round(change.before, 2),
        "after": round(change.after, 2),
        "pct_change": round(change.pct_change, 2),
        "measured_direction": change.direction,
        "required_pct": need_pct,
    }

    if claimed_dir in ("increase", "decrease"):
        if change.direction != claimed_dir:
            return ClaimResult(
                claim.text, "empirical", Verdict.REFUTED, "metric_comparison",
                f"{metric.value} in {month} actually {change.direction}d "
                f"{abs(change.pct_change):.1f}% vs {change.compare_to}, "
                f"but the statement claims it {claimed_dir}d",
                measured, claim.is_universal,
            )
        if abs(change.pct_change) >= need_pct:
            return ClaimResult(
                claim.text, "empirical", Verdict.CONFIRMED, "metric_comparison",
                f"confirmed: {metric.value} {claimed_dir}d "
                f"{abs(change.pct_change):.1f}% in {month} vs {change.compare_to} "
                f"(needed {need_pct:.0f}%)",
                measured, claim.is_universal,
            )
        return ClaimResult(
            claim.text, "empirical", Verdict.PARTIAL, "metric_comparison",
            f"direction is right but the size is overstated: {metric.value} "
            f"{claimed_dir}d only {abs(change.pct_change):.1f}% in {month}, "
            f"and the wording implies at least {need_pct:.0f}%",
            measured, claim.is_universal,
        )

    if claimed_dir == "flat":
        if abs(change.pct_change) <= FLAT_BAND_PCT:
            return ClaimResult(
                claim.text, "empirical", Verdict.CONFIRMED, "metric_comparison",
                f"confirmed: {metric.value} was flat in {month} "
                f"({change.pct_change:+.1f}%)",
                measured, claim.is_universal,
            )
        return ClaimResult(
            claim.text, "empirical", Verdict.REFUTED, "metric_comparison",
            f"{metric.value} moved {change.pct_change:+.1f}% in {month}, "
            "which is not flat",
            measured, claim.is_universal,
        )

    return ClaimResult(
        claim.text, "empirical", Verdict.UNCHECKABLE, "no_direction",
        "the statement does not claim a direction of change",
        measured,
    )


def _check_category_tickets(claim: Claim, month: str, category: str) -> ClaimResult:
    """Ticket rate for orders containing a category, vs orders without it."""
    data = category_month_tickets()
    key = (month, category)
    if key not in data:
        return ClaimResult(
            claim.text, "empirical", Verdict.UNCHECKABLE, "unknown_category",
            f"no data for category '{category}' in {month}",
        )
    this = data[key]["rate"]
    others = [v["rate"] for (m, c), v in data.items() if m == month and c != category]
    avg_other = sum(others) / len(others) if others else 0.0
    ratio = this / avg_other if avg_other else 0.0
    measured = {
        "category": category, "month": month,
        "ticket_rate": round(this, 4),
        "other_categories_avg": round(avg_other, 4),
        "ratio": round(ratio, 2),
    }
    if ratio >= 1.5:
        return ClaimResult(
            claim.text, "empirical", Verdict.CONFIRMED, "category_ticket_rate",
            f"confirmed: {category} orders in {month} had {ratio:.1f}x the "
            f"ticket rate of other categories",
            measured,
        )
    return ClaimResult(
        claim.text, "empirical", Verdict.REFUTED, "category_ticket_rate",
        f"{category} orders in {month} had {ratio:.1f}x the ticket rate of "
        "other categories, which is not unusual",
        measured,
    )


def _check_causal(claim: Claim, spec: dict, question: str) -> ClaimResult:
    """A cause must explain most of the change; 'always' needs no counterexample."""
    factor_raw = str(spec.get("causal_factor") or "none").lower()
    factor = CAUSAL_FACTORS.get(factor_raw, factor_raw)
    if factor not in {"refunds", "order_volume", "avg_order_value"}:
        # Fall back to reading the claim text directly.
        factor = _infer_factor(claim.text)
    if factor == "none" or factor not in {"refunds", "order_volume", "avg_order_value"}:
        return ClaimResult(
            claim.text, "causal", Verdict.UNCHECKABLE, "no_factor",
            "the statement does not name a factor we can attribute",
            is_universal=claim.is_universal,
        )

    # A universal claim is refuted by one counterexample, so it is checked
    # against every month rather than the one under discussion.
    if claim.is_universal:
        ce = find_counterexamples(factor)
        measured = {"factor": factor, "counterexamples": ce,
                    "months_examined": all_months(complete_refunds_only=True)}
        if ce:
            first = ce[0]
            return ClaimResult(
                claim.text, "causal", Verdict.REFUTED, "counterexample_search",
                f"refuted by {first['month']}: net revenue fell "
                f"{abs(first['net_change_pct']):.1f}% while {factor} explained only "
                f"{first[f'{factor}_share_of_decline'] * 100:.0f}% of it "
                f"(the main cause was {first['dominant_factor']}). "
                f"{len(ce)} such month(s) exist, and a claim that something "
                f"ALWAYS holds needs only one exception.",
                measured, True,
            )
        return ClaimResult(
            claim.text, "causal", Verdict.CONFIRMED, "counterexample_search",
            f"no counterexample found: {factor} dominated every decline examined",
            measured, True,
        )

    month = _resolve_month(spec, "causal_target_month") or _resolve_month(spec)
    if month is None:
        return ClaimResult(
            claim.text, "causal", Verdict.UNCHECKABLE, "no_month",
            "the statement does not name a month whose change we can attribute",
            is_universal=claim.is_universal,
        )

    attr = attribute_net_change(month)
    if attr is None:
        return ClaimResult(
            claim.text, "causal", Verdict.UNCHECKABLE, "not_attributable",
            f"could not attribute the {month} change",
        )

    share = attr.share_of(factor)
    measured = {
        "factor": factor, "month": month, "compare_to": attr.compare_to,
        "net_change_pct": round(attr.net_change_pct, 2),
        "share_of_change": round(share, 3),
        "dominant_factor": attr.dominant_factor,
        "threshold": CAUSAL_DOMINANCE,
    }

    if share > CAUSAL_DOMINANCE:
        return ClaimResult(
            claim.text, "causal", Verdict.CONFIRMED, "attribution",
            f"confirmed: {factor} explains {share * 100:.0f}% of the {month} "
            f"net revenue change",
            measured, False,
        )
    if share >= CAUSAL_MINIMUM:
        return ClaimResult(
            claim.text, "causal", Verdict.PARTIAL, "attribution",
            f"{factor} is a real contributor but not the cause: it explains "
            f"{share * 100:.0f}% of the {month} change, while "
            f"{attr.dominant_factor} explains "
            f"{attr.share_of(attr.dominant_factor) * 100:.0f}%",
            measured, False,
        )
    return ClaimResult(
        claim.text, "causal", Verdict.REFUTED, "attribution",
        f"refuted: {factor} explains only {share * 100:.0f}% of the {month} "
        f"net revenue change. The main cause was {attr.dominant_factor} "
        f"({attr.share_of(attr.dominant_factor) * 100:.0f}%).",
        measured, False,
    )


def _check_procedural(claim: Claim, tables: list[str]) -> ClaimResult:
    """Advice is not true or false - only executable or not.

    So this checks the one thing that CAN be wrong about a business rule:
    whether it refers to data that exists. A rule naming a column we do not
    have cannot be followed, however sensible it sounds.
    """
    named = [t for t in (tables or []) if t in VALID_TABLES]
    unknown = [t for t in (tables or []) if t not in VALID_TABLES]

    if unknown:
        return ClaimResult(
            claim.text, "procedural", Verdict.REFUTED, "schema_check",
            f"not executable: refers to {', '.join(unknown)}, which do not "
            f"exist. Available tables: {', '.join(sorted(VALID_TABLES))}",
            {"unknown_tables": unknown},
        )
    if not named:
        return ClaimResult(
            claim.text, "procedural", Verdict.UNCHECKABLE, "no_tables",
            "the advice does not reference any specific table, so there is "
            "nothing to check it against",
        )
    return ClaimResult(
        claim.text, "procedural", Verdict.CONFIRMED, "schema_check",
        f"executable: the tables it relies on ({', '.join(named)}) all exist. "
        "Note this is a rule, not a factual claim - nothing was fact-checked.",
        {"tables": named},
    )


# ------------------------------------------------------------------ entrypoint

def verify_feedback(
    p: ProcessedFeedback,
    *,
    conflicts_with: list[str] | None = None,
    llm: LLMClient | None = None,
) -> VerificationOutcome:
    """Check every claim in a piece of feedback and return one verdict."""
    own = llm is None
    llm = llm or LLMClient()
    results: list[ClaimResult] = []

    try:
        if not p.claims:
            return decide([], conflicts_with=conflicts_with,
                          clarity_confidence=p.confidence)

        for claim in p.claims:
            if claim.kind == "procedural":
                results.append(_check_procedural(claim, p.schema_context))
                continue

            spec = _map_claim(claim, p.original_question, llm)
            if claim.kind == "causal":
                results.append(_check_causal(claim, spec, p.original_question))
            else:
                results.append(_check_empirical(claim, spec, p.original_question))
    finally:
        if own:
            llm.close()

    return decide(results, conflicts_with=conflicts_with,
                  clarity_confidence=p.confidence)
