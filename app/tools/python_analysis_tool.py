"""
Tool 3 - Python Analysis Tool (spec section 10).

Turns a SQL result into described numbers: period-over-period change, share
of total, trend, outliers, correlation.

WHY THIS IS A FIXED LIBRARY AND NOT GENERATED CODE

The obvious design is to let the model write pandas and exec() it. We do not,
for two reasons:

  - exec() on model output is arbitrary code execution. There is no sandbox
    here worth trusting, and the agent would be one prompt injection away
    from running anything.
  - It would make results irreproducible. The benchmark compares three
    systems; if each run invents its own arithmetic, a difference in scores
    could come from the analysis code rather than from the feedback memory
    being tested. Fixed functions keep the measurement honest.

So the model's only choice is WHICH analysis to run on which columns. The
arithmetic is ours, it is tested, and it is identical across all three
systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Finding:
    """One described number, in words the answer step can quote."""

    kind: str
    text: str
    values: dict[str, Any] = field(default_factory=dict)


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame, converting Postgres Decimals to float.

    psycopg returns numeric columns as Decimal, which pandas keeps as
    dtype=object - so .mean() and .pct_change() silently fail or return
    objects. Converting up front avoids a class of confusing bugs.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, Decimal)).any():
            df[col] = df[col].astype(float)
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


# ----------------------------------------------------------------- analyses

def period_change(df: pd.DataFrame, label_col: str, value_col: str) -> list[Finding]:
    """Change from each row to the next, plus first-to-last overall."""
    if len(df) < 2:
        return []
    out: list[Finding] = []
    s = df[value_col].astype(float).reset_index(drop=True)
    labels = df[label_col].astype(str).reset_index(drop=True)

    for i in range(1, len(s)):
        prev, cur = s[i - 1], s[i]
        if prev == 0:
            continue
        pct = (cur - prev) / abs(prev) * 100
        out.append(
            Finding(
                "period_change",
                f"{labels[i]} vs {labels[i - 1]}: {value_col} "
                f"{'rose' if pct >= 0 else 'fell'} {abs(pct):.1f}% "
                f"({prev:,.0f} -> {cur:,.0f})",
                {"from": labels[i - 1], "to": labels[i], "pct": round(pct, 2)},
            )
        )

    first, last = s.iloc[0], s.iloc[-1]
    if first:
        pct = (last - first) / abs(first) * 100
        out.append(
            Finding(
                "overall_change",
                f"Across the whole period, {value_col} "
                f"{'rose' if pct >= 0 else 'fell'} {abs(pct):.1f}% "
                f"({labels.iloc[0]} {first:,.0f} -> {labels.iloc[-1]} {last:,.0f})",
                {"pct": round(pct, 2)},
            )
        )
    return out


def biggest_movers(df: pd.DataFrame, label_col: str, value_col: str, n: int = 3) -> list[Finding]:
    """The largest single-step rises and falls."""
    if len(df) < 3:
        return []
    s = df[value_col].astype(float).reset_index(drop=True)
    labels = df[label_col].astype(str).reset_index(drop=True)
    deltas = []
    for i in range(1, len(s)):
        if s[i - 1]:
            deltas.append((labels[i], (s[i] - s[i - 1]) / abs(s[i - 1]) * 100, s[i] - s[i - 1]))
    if not deltas:
        return []
    worst = sorted(deltas, key=lambda d: d[1])[:n]
    best = sorted(deltas, key=lambda d: -d[1])[:n]
    out = []
    if worst:
        out.append(
            Finding(
                "largest_declines",
                "Largest declines in "
                + value_col
                + ": "
                + ", ".join(f"{lbl} {pct:+.1f}%" for lbl, pct, _ in worst),
                {"items": [{"label": l, "pct": round(p, 2)} for l, p, _ in worst]},
            )
        )
    if best:
        out.append(
            Finding(
                "largest_increases",
                "Largest increases in "
                + value_col
                + ": "
                + ", ".join(f"{lbl} {pct:+.1f}%" for lbl, pct, _ in best),
                {"items": [{"label": l, "pct": round(p, 2)} for l, p, _ in best]},
            )
        )
    return out


def contribution(df: pd.DataFrame, label_col: str, value_col: str, n: int = 5) -> list[Finding]:
    """Each category's share of the total."""
    if df.empty:
        return []
    s = df[value_col].astype(float)
    total = s.sum()
    if total == 0:
        return []
    top = df.assign(_pct=s / total * 100).nlargest(n, "_pct")
    parts = [
        f"{row[label_col]} {row['_pct']:.1f}% ({row[value_col]:,.0f})"
        for _, row in top.iterrows()
    ]
    return [
        Finding(
            "contribution",
            f"Share of total {value_col} ({total:,.0f}): " + ", ".join(parts),
            {"total": round(total, 2)},
        )
    ]


def outliers(df: pd.DataFrame, label_col: str, value_col: str) -> list[Finding]:
    """Values more than 2 standard deviations from the mean."""
    if len(df) < 4:
        return []
    s = df[value_col].astype(float)
    mu, sd = s.mean(), s.std()
    if not sd or np.isnan(sd):
        return []
    flagged = [
        (str(df[label_col].iloc[i]), s.iloc[i], (s.iloc[i] - mu) / sd)
        for i in range(len(s))
        if abs((s.iloc[i] - mu) / sd) > 2
    ]
    if not flagged:
        return []
    return [
        Finding(
            "outliers",
            f"Unusual {value_col} values (>2 sd from mean {mu:,.0f}): "
            + ", ".join(f"{lbl} {val:,.0f} ({z:+.1f} sd)" for lbl, val, z in flagged),
            {"mean": round(mu, 2), "sd": round(sd, 2)},
        )
    ]


def correlate(df: pd.DataFrame, a: str, b: str) -> list[Finding]:
    """Pearson correlation between two numeric columns."""
    if len(df) < 4:
        return []
    x, y = df[a].astype(float), df[b].astype(float)
    if x.std() == 0 or y.std() == 0:
        return []
    r = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(r):
        return []
    strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
    return [
        Finding(
            "correlation",
            f"{a} and {b} show a {strength} "
            f"{'positive' if r > 0 else 'negative'} correlation (r = {r:.2f}). "
            "Correlation is not causation.",
            {"r": round(r, 3)},
        )
    ]


def describe(df: pd.DataFrame, value_col: str) -> list[Finding]:
    s = df[value_col].astype(float)
    return [
        Finding(
            "summary",
            f"{value_col}: total {s.sum():,.0f}, mean {s.mean():,.1f}, "
            f"min {s.min():,.0f}, max {s.max():,.0f} across {len(s)} rows",
            {"total": round(float(s.sum()), 2), "mean": round(float(s.mean()), 2)},
        )
    ]


# ---------------------------------------------------------------- entry point

def analyse(rows: list[dict], *, question: str = "") -> tuple[str, list[Finding]]:
    """Run whichever analyses fit the shape of the result.

    Selection is by data shape, not by asking the model: a label column plus
    a numeric column over several rows means period change and contribution
    are meaningful; two numeric columns make correlation meaningful. This
    keeps the step deterministic.
    """
    df = to_dataframe(rows)
    if df.empty:
        return "The query returned no rows.", []

    nums = numeric_columns(df)
    labels = label_columns(df)
    findings: list[Finding] = []

    if not nums:
        return (
            f"The result has {len(df)} rows but no numeric column to analyse.",
            [],
        )

    primary = nums[0]
    # Prefer a column that looks like a measure over an id column.
    for c in nums:
        if any(k in c.lower() for k in ("revenue", "amount", "total", "net", "sum", "count", "sales")):
            primary = c
            break

    findings += describe(df, primary)

    if labels:
        lab = labels[0]
        # An ordered label (month, date) makes period-over-period meaningful.
        ordered = any(k in lab.lower() for k in ("month", "date", "day", "week", "year", "period"))
        if ordered and len(df) >= 2:
            findings += period_change(df, lab, primary)
            findings += biggest_movers(df, lab, primary)
            findings += outliers(df, lab, primary)
        elif len(df) >= 2:
            findings += contribution(df, lab, primary)
            findings += outliers(df, lab, primary)

    if len(nums) >= 2:
        findings += correlate(df, nums[0], nums[1])

    text = "\n".join(f"- {f.text}" for f in findings) or "No further analysis available."
    return text, findings
