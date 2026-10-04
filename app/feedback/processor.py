"""
Feedback processor (spec section 11).

Turns free-text analyst feedback into a structured record.

    "You only looked at order volume. You should also check refunds,
     because refund volume increased substantially in March."

                              |
                              v

    feedback_type : MISSING_ANALYSIS
    mistake       : attributed the decline only to order volume
    correction    : refund trends should also be analysed
    lesson        : when investigating a revenue decline, check refund
                    trends before attributing it to order volume alone
    claims        : [{"text": "refund volume increased substantially in
                      March", "kind": "empirical"}]

THE IMPORTANT FIELD IS `lesson`, AND WHY

Storing the raw comment and retrieving it later does not work. "You only
looked at order volume" is about one specific answer to one specific
question; on a future question it is noise. The lesson has to be
generalised - phrased so it can fire on a question that has not been asked
yet - while staying narrow enough not to fire on everything.

That generalisation is a judgement call, which is why an LLM does it, and
why the raw text is kept alongside so a human can always audit what was
actually said.

`claims` EXISTS FOR THE VERIFIER

Feedback mixes three kinds of statement, and they are not checkable in the
same way:

  empirical  "refunds increased substantially in March"
             -> a SQL query can confirm or refute this
  causal     "refunds caused the decline"
             -> needs an attribution test, and "always"-type claims need
                a counterexample search
  procedural "check refunds when investigating declines"
             -> not true or false at all; only relevance and schema
                validity can be assessed

Splitting them out here is what lets the verifier (Phase 7) do the right
test for each, instead of asking the model "does this feedback seem right?"
and believing the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.llm import LLMClient

FEEDBACK_TYPES = [
    "SQL_ERROR",
    "MISSING_ANALYSIS",
    "WRONG_ASSUMPTION",
    "MISSING_DATA_SOURCE",
    "BUSINESS_LOGIC",
    "INTERPRETATION",
    "OTHER",
]

CLAIM_KINDS = ["empirical", "causal", "procedural"]

# The enum semantics have to be spelled out. Given only the bare names, the
# model picked OTHER for a textbook MISSING_ANALYSIS case during Phase 0
# verification.
TYPE_GUIDE = """Choose feedback_type from these, using the definitions:

- SQL_ERROR: the query was broken or wrong - bad join, wrong column, wrong
  filter, dropped rows, syntax error.
- MISSING_ANALYSIS: the query worked, but something that should also have
  been examined was not. Use this when the analyst says "you should also
  look at X" or "you didn't check Y".
- WRONG_ASSUMPTION: the agent assumed something untrue about the data, e.g.
  that a column means something it does not, or that a date range lines up.
- MISSING_DATA_SOURCE: a whole table or data source that was needed was
  never consulted.
- BUSINESS_LOGIC: the calculation disagrees with how the business defines
  the metric, e.g. which order statuses count as revenue.
- INTERPRETATION: the numbers were right but the conclusion drawn from them
  was wrong or overstated, e.g. claiming a cause from a correlation.
- OTHER: only when genuinely none of the above fit."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "feedback_type": {"type": "string", "enum": FEEDBACK_TYPES},
        "mistake": {"type": "string"},
        "correction": {"type": "string"},
        "lesson": {"type": "string"},
        "confidence": {"type": "number"},
        "schema_context": {"type": "array", "items": {"type": "string"}},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": CLAIM_KINDS},
                    "is_universal": {"type": "boolean"},
                },
                "required": ["text", "kind", "is_universal"],
            },
        },
    },
    "required": [
        "feedback_type", "mistake", "correction", "lesson",
        "confidence", "schema_context", "claims",
    ],
}

SYSTEM = """You convert an analyst's correction of a data agent into a \
structured record. Be precise and literal: capture what the analyst \
actually said, not what you think they should have said."""

USER = """A data agent was asked this question:
{question}

It ran this SQL:
{sql}

It gave this answer:
{answer}

An analyst reviewed it and said:
"{feedback}"

{type_guide}

Produce:

- feedback_type: from the list above.
- mistake: what the agent actually did wrong, in one sentence.
- correction: what the analyst says should have been done instead.
- lesson: a GENERAL instruction, reusable on future questions. It must not
  mention this specific question, month or number. Write it so it would
  still make sense applied to a different but similar question. One
  sentence, starting with "When".
- confidence: 0.0 to 1.0, how clear and specific the analyst's feedback is.
  Vague feedback gets a low score. This measures CLARITY, not correctness -
  whether the claim is true is checked separately.
- schema_context: database tables the lesson is about, lower case, from:
  customers, products, orders, order_items, refunds, support_tickets,
  promotions.
- claims: break the feedback into checkable statements. For each:
    text: the statement itself
    kind: "empirical" if data could confirm or refute it (e.g. "refunds
          rose 30%"); "causal" if it asserts one thing caused another;
          "procedural" if it is advice about how to analyse (e.g. "always
          check refunds") and so cannot be true or false
    is_universal: true if it claims something always or never holds
  Return an empty list only if the feedback contains no statement at all."""


@dataclass
class Claim:
    text: str
    kind: str
    is_universal: bool = False


@dataclass
class ProcessedFeedback:
    feedback_type: str
    mistake: str
    correction: str
    lesson: str
    confidence: float
    schema_context: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    raw_feedback: str = ""
    original_question: str = ""
    trace_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "feedback_type": self.feedback_type,
            "mistake": self.mistake,
            "correction": self.correction,
            "lesson": self.lesson,
            "confidence": self.confidence,
            "schema_context": self.schema_context,
            "claims": [
                {"text": c.text, "kind": c.kind, "is_universal": c.is_universal}
                for c in self.claims
            ],
        }


VALID_TABLES = {
    "customers", "products", "orders", "order_items",
    "refunds", "support_tickets", "promotions",
}


def process_feedback(
    *,
    feedback: str,
    question: str,
    sql: str = "",
    answer: str = "",
    trace_id: str | None = None,
    llm: LLMClient | None = None,
) -> ProcessedFeedback:
    """Classify and generalise one piece of analyst feedback."""
    own = llm is None
    llm = llm or LLMClient()
    try:
        resp = llm.complete(
            USER.format(
                question=question,
                sql=sql or "(none)",
                answer=answer or "(none)",
                feedback=feedback.strip(),
                type_guide=TYPE_GUIDE,
            ),
            system=SYSTEM,
            json_schema=SCHEMA,
            max_tokens=1200,
        )
        data = resp.json()
    finally:
        if own:
            llm.close()

    # Clamp and sanitise. A schema-constrained model still returns odd
    # values occasionally, and a confidence of 4.0 would quietly outrank
    # every verified lesson in the ranking step.
    conf = float(data.get("confidence", 0.5) or 0.5)
    conf = max(0.0, min(1.0, conf))

    ftype = str(data.get("feedback_type", "OTHER")).upper()
    if ftype not in FEEDBACK_TYPES:
        ftype = "OTHER"

    tables = [
        t.lower().strip()
        for t in (data.get("schema_context") or [])
        if isinstance(t, str) and t.lower().strip() in VALID_TABLES
    ]

    claims: list[Claim] = []
    for c in data.get("claims") or []:
        if not isinstance(c, dict) or not c.get("text"):
            continue
        kind = str(c.get("kind", "procedural")).lower()
        claims.append(
            Claim(
                text=str(c["text"]).strip(),
                kind=kind if kind in CLAIM_KINDS else "procedural",
                is_universal=bool(c.get("is_universal", False)),
            )
        )

    return ProcessedFeedback(
        feedback_type=ftype,
        mistake=str(data.get("mistake", "")).strip(),
        correction=str(data.get("correction", "")).strip(),
        lesson=str(data.get("lesson", "")).strip(),
        confidence=conf,
        schema_context=sorted(set(tables)),
        claims=claims,
        raw_feedback=feedback.strip(),
        original_question=question,
        trace_id=trace_id,
    )
