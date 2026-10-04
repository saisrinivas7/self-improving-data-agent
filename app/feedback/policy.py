"""
Every threshold and every status decision, in one file.

WHY THIS IS SEPARATE FROM THE VERIFIER

The verifier's job is to MEASURE. This file's job is to JUDGE. Keeping them
apart matters because the judging is where the project is most easily
fooled - by itself.

An analyst says "refunds jumped substantially". The data says refunds rose
35.4%. Is that "substantially"? The answer is a choice, not a fact. And the
choice has consequences in both directions:

  threshold too loose   -> almost everything is VERIFIED, so the verified
                           system behaves like the naive one and the
                           experiment shows no difference
  threshold too strict  -> true feedback is REJECTED, memory stays empty,
                           and the verified system behaves like the baseline

Either mistake produces a null result that LOOKS like a finding. So the
numbers live here, written down, in one place, where they can be seen,
cited in the write-up, and changed deliberately rather than by accident.

It also makes "why was my feedback rejected?" answerable with a number,
which is what the failure-analysis section of the project needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# ----------------------------------------------------------------- thresholds

# Vague magnitude words, mapped to a minimum percentage change. An analyst
# who says "jumped" and means 3% is overstating; one who says it and means
# 35% is not. 20% is the line.
MAGNITUDE_WORDS: dict[str, float] = {
    "slightly": 2.0,
    "somewhat": 5.0,
    "noticeably": 10.0,
    "significantly": 20.0,
    "substantially": 20.0,
    "sharply": 20.0,
    "jumped": 20.0,
    "spiked": 25.0,
    "surged": 25.0,
    "soared": 30.0,
    "collapsed": 30.0,
    "plummeted": 30.0,
    "doubled": 90.0,
    "halved": 45.0,
}
DEFAULT_MAGNITUDE = 10.0

# A claimed direction is confirmed only if the measured change exceeds this.
# Below it, the metric is treated as flat.
FLAT_BAND_PCT = 2.0

# "X caused Y" requires X to account for more than half the change. Below
# that X is a contributor, not the cause - which is the exact distinction
# the March data was built to test (refunds explain 13%).
CAUSAL_DOMINANCE = 0.50

# A contributing factor must explain at least this much to count as real.
CAUSAL_MINIMUM = 0.10

# A universal claim ("always", "never") is refuted by a single
# counterexample.
UNIVERSAL_COUNTEREXAMPLES_TO_REFUTE = 1

# Confidence assigned per verification outcome. Deliberately conservative:
# a rule that was only checked for being executable never scores as high as
# a numeric claim confirmed against the data.
CONFIDENCE = {
    "empirical_confirmed": 0.95,
    "empirical_direction_only": 0.70,   # right direction, weaker magnitude
    "causal_confirmed": 0.90,
    "causal_partial": 0.45,             # real contributor, not the cause
    "rule_validated": 0.75,             # executable, material, no conflict
    "unverifiable": 0.40,               # nothing checkable
    "refuted": 0.05,
}


class Status(StrEnum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    CONFLICTING = "CONFLICTING"


class LessonKind(StrEnum):
    """What sort of thing the lesson is.

    This distinction came out of a question the user asked, and it changes
    what a good benchmark result means. A domain convention improves answers
    trivially, because it hands the agent information it could not have
    derived. An error correction is the interesting case: it claims the
    agent can avoid repeating a mistake on a question it has not seen.

    Reporting them together would let "feedback helps" stand on the back of
    "telling it the answer helps".
    """

    ERROR_CORRECTION = "error_correction"
    DOMAIN_CONVENTION = "domain_convention"
    UNKNOWN = "unknown"


class Verdict(StrEnum):
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    PARTIAL = "partial"
    UNCHECKABLE = "uncheckable"


@dataclass
class ClaimResult:
    """The outcome of checking one claim."""

    claim: str
    kind: str                  # empirical | causal | procedural
    verdict: Verdict
    method: str                # how it was checked
    detail: str                # human-readable, cites the numbers
    measured: dict = field(default_factory=dict)
    is_universal: bool = False


@dataclass
class VerificationOutcome:
    status: Status
    verified: bool
    confidence: float
    lesson_kind: LessonKind
    reason: str
    claim_results: list[ClaimResult] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)

    def as_notes(self) -> dict:
        """What gets written to feedback_memory.verification_notes.

        This is the audit trail. It records the method and the measured
        numbers, so a stored lesson can be questioned later rather than
        taken on trust.
        """
        return {
            "status": str(self.status),
            "confidence": self.confidence,
            "lesson_kind": str(self.lesson_kind),
            "reason": self.reason,
            "conflicts_with": self.conflicts_with,
            "claims": [
                {
                    "claim": c.claim,
                    "kind": c.kind,
                    "verdict": str(c.verdict),
                    "method": c.method,
                    "detail": c.detail,
                    "measured": c.measured,
                    "is_universal": c.is_universal,
                }
                for c in self.claim_results
            ],
            "thresholds": {
                "causal_dominance": CAUSAL_DOMINANCE,
                "flat_band_pct": FLAT_BAND_PCT,
                "default_magnitude_pct": DEFAULT_MAGNITUDE,
            },
        }


# ------------------------------------------------------------------ the policy

def required_magnitude(magnitude_word: str | None) -> float:
    """Minimum percentage change implied by a vague word."""
    if not magnitude_word:
        return DEFAULT_MAGNITUDE
    return MAGNITUDE_WORDS.get(magnitude_word.strip().lower(), DEFAULT_MAGNITUDE)


def decide(
    claim_results: list[ClaimResult],
    *,
    conflicts_with: list[str] | None = None,
    clarity_confidence: float = 0.5,
) -> VerificationOutcome:
    """Turn claim-level verdicts into one status for the lesson.

    The rules, in order of precedence:

      1. A conflict with an existing lesson -> CONFLICTING. Two lessons that
         contradict each other must not both be applied, and which one is
         right is a human decision, not a measurement.
      2. Any REFUTED claim -> REJECTED. One false statement poisons the
         lesson, even if other parts of it check out, because the agent
         would receive the whole lesson as a unit.
      3. At least one CONFIRMED claim and nothing refuted -> VERIFIED.
      4. Only procedural claims, all passing -> VERIFIED as a domain
         convention. Note that nothing about it was fact-checked; the
         confidence and the notes say so explicitly.
      5. Anything else -> PENDING, meaning "could not check", which is NOT
         the same as "false". Keeping these apart is important: a verifier
         that rejects what it cannot check would throw away good feedback
         whenever its own check query failed.
    """
    conflicts_with = conflicts_with or []

    if conflicts_with:
        return VerificationOutcome(
            status=Status.CONFLICTING,
            verified=False,
            confidence=min(clarity_confidence, 0.40),
            lesson_kind=LessonKind.UNKNOWN,
            reason=(
                f"contradicts {len(conflicts_with)} stored lesson(s); "
                "neither will be applied until a human resolves it"
            ),
            claim_results=claim_results,
            conflicts_with=conflicts_with,
        )

    refuted = [c for c in claim_results if c.verdict is Verdict.REFUTED]
    confirmed = [c for c in claim_results if c.verdict is Verdict.CONFIRMED]
    partial = [c for c in claim_results if c.verdict is Verdict.PARTIAL]
    procedural = [c for c in claim_results if c.kind == "procedural"]
    checkable = [c for c in claim_results if c.kind in ("empirical", "causal")]

    if refuted:
        first = refuted[0]
        return VerificationOutcome(
            status=Status.REJECTED,
            verified=False,
            confidence=CONFIDENCE["refuted"],
            lesson_kind=LessonKind.UNKNOWN,
            reason=f"the data refutes this: {first.detail}",
            claim_results=claim_results,
        )

    # Checked BEFORE the generic confirmed branch. A lesson whose only claims
    # are procedural is a domain convention, even though its schema check
    # "confirmed". Getting this order wrong labelled business rules as error
    # corrections and gave them a fact-checked confidence of 0.95, which is
    # precisely the rubber-stamping this file exists to prevent.
    if procedural and not checkable:
        if all(c.verdict is not Verdict.REFUTED for c in procedural):
            return VerificationOutcome(
                status=Status.VERIFIED,
                verified=True,
                confidence=CONFIDENCE["rule_validated"],
                lesson_kind=LessonKind.DOMAIN_CONVENTION,
                reason=(
                    "accepted as a business rule: it is executable against the "
                    "schema and does not conflict with a stored lesson. Nothing "
                    "about it was fact-checked, because it states no fact."
                ),
                claim_results=claim_results,
            )

    if confirmed:
        kinds = {c.kind for c in confirmed}
        conf = (
            CONFIDENCE["causal_confirmed"]
            if "causal" in kinds
            else CONFIDENCE["empirical_confirmed"]
        )
        return VerificationOutcome(
            status=Status.VERIFIED,
            verified=True,
            confidence=round(min(conf, 0.5 + clarity_confidence / 2), 3),
            lesson_kind=LessonKind.ERROR_CORRECTION,
            reason="; ".join(c.detail for c in confirmed[:2]),
            claim_results=claim_results,
        )

    if partial:
        # PARTIAL means different things for the two checkable kinds, and
        # conflating them would let an unproven causal claim through.
        #
        #   empirical partial  the fact is right, the wording oversells it
        #                      ("refunds doubled" when they rose 35%). The
        #                      underlying observation is usable, so VERIFIED
        #                      with reduced confidence.
        #
        #   causal partial     the named factor is a contributor but NOT the
        #                      cause ("refunds caused the decline" when they
        #                      explain 13%). The causal assertion is simply
        #                      not established, and spec section 12 is
        #                      explicit that such a claim must not be
        #                      accepted on the analyst's word. PENDING.
        causal_partial = [c for c in partial if c.kind == "causal"]
        if causal_partial:
            return VerificationOutcome(
                status=Status.PENDING,
                verified=False,
                confidence=CONFIDENCE["causal_partial"],
                lesson_kind=LessonKind.UNKNOWN,
                reason=(
                    "the causal claim is not established: "
                    + "; ".join(c.detail for c in causal_partial[:2])
                ),
                claim_results=claim_results,
            )
        return VerificationOutcome(
            status=Status.VERIFIED,
            verified=True,
            confidence=CONFIDENCE["empirical_direction_only"],
            lesson_kind=LessonKind.ERROR_CORRECTION,
            reason="; ".join(c.detail for c in partial[:2]),
            claim_results=claim_results,
        )

    return VerificationOutcome(
        status=Status.PENDING,
        verified=False,
        confidence=CONFIDENCE["unverifiable"],
        lesson_kind=LessonKind.UNKNOWN,
        reason=(
            "could not be checked against the data - stored unverified. "
            "This is not the same as being false."
        ),
        claim_results=claim_results,
    )
