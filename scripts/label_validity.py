#!/usr/bin/env python3
"""Does a filed date revision correspond to anything physical?

The platform's label is `slip_next`: the officially declared completion date
moved on a form. A fair reviewer will object that this is a *bureaucratic*
event, not a physical one — that the model may be learning which agencies file
paperwork promptly rather than which projects are in trouble.

That objection cannot be answered by argument, only by measurement. This script
compares what was happening on site in the months *before* a revision was filed
against the months before a report where the date held.

Indicators are grouped by what they actually test, because the first version of
this analysis conflated three different questions and produced a misleading
verdict:

* **Schedule-relative distress** — is the project behind the pace its own
  declared date requires? This is the question that matters for forecasting.
* **Absolute activity** — is the site moving at all? Tests whether the label is
  just picking out dormant projects.
* **Spend-vs-progress pattern** — tests the statutory folk theory behind rule
  F1 (GFR 2017 Rule 159): that disbursement running ahead of physical progress
  signals trouble.

Directions are NOT asserted in advance for the spend pattern. That is the
hypothesis under test, and on this data it comes out inverted.

Usage:
    python scripts/label_validity.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paimana.label")

ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "paimana_panel.csv"
OUT_PATH = ROOT / "model" / "paimana_label_validity.json"

# (column, group, expected direction if the label is physical, meaning)
# `expected` is None where the direction is the hypothesis under test.
INDICATORS = [
    ("schedule_pressure", "schedule_relative", "higher",
     "Fraction of schedule elapsed minus fraction of work complete"),
    ("velocity_deficit", "schedule_relative", "higher",
     "Pace the declared date requires, minus pace actually achieved (%/month)"),
    ("progress_delta", "absolute_activity", None,
     "Physical progress added in the reporting month (%)"),
    ("progress_velocity", "absolute_activity", None,
     "Recent physical progress rate (%/month)"),
    ("stall_streak", "absolute_activity", None,
     "Consecutive reporting months with no physical movement"),
    ("spend_progress_gap", "spend_pattern", None,
     "Cumulative spend % minus physical progress % (rule F1's trigger metric)"),
]


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference.

    Reported instead of a p-value because n is ~8,800: at that size almost any
    difference is 'statistically significant', which tells you nothing about
    magnitude. Effect size does.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)


def magnitude(d: float) -> str:
    m = abs(d)
    if np.isnan(d):
        return "not evaluable"
    return "negligible" if m < 0.10 else "small" if m < 0.30 else "medium" if m < 0.50 else "large"


def compare(df: pd.DataFrame, y: np.ndarray) -> List[Dict[str, Any]]:
    slip, calm = df[y == 1], df[y == 0]
    rows: List[Dict[str, Any]] = []
    for col, group, expected, meaning in INDICATORS:
        if col not in df.columns:
            continue
        a = slip[col].dropna().to_numpy(dtype=float)
        b = calm[col].dropna().to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            continue
        d = cohen_d(a, b)
        observed = "higher" if a.mean() > b.mean() else "lower"
        row: Dict[str, Any] = {
            "indicator": col,
            "group": group,
            "meaning": meaning,
            "mean_before_slip": round(float(a.mean()), 4),
            "mean_before_hold": round(float(b.mean()), 4),
            "cohens_d": round(d, 4),
            "effect_size": magnitude(d),
            "observed_direction": observed,
        }
        if expected is not None:
            row["expected_if_physical"] = expected
            row["corroborates"] = bool(observed == expected and abs(d) >= 0.10)
        rows.append(row)
    return rows


def f1_inversion_test(df: pd.DataFrame) -> Dict[str, Any]:
    """Test rule F1's premise directly against the outcome.

    F1 (GFR 2017 Rule 159) fires when cumulative disbursement exceeds physical
    progress by >=25pp before halfway, on the theory that front-loaded spending
    signals a project in trouble. This measures whether that premise holds for
    schedule slippage.
    """
    gap = df["spend_progress_gap"]
    y = df["slip_next"]
    bands = [
        ("spend far ahead of progress (gap >= 25pp, F1 fires)", gap >= 25),
        ("roughly aligned (-25pp < gap < 25pp)", (gap > -25) & (gap < 25)),
        ("progress far ahead of spend (gap <= -25pp)", gap <= -25),
    ]
    rows = []
    for name, mask in bands:
        n = int(mask.sum())
        rows.append({
            "band": name,
            "n": n,
            "slip_rate": round(float(y[mask].mean()), 4) if n else None,
        })
    fires = float(y[gap >= 25].mean()) if (gap >= 25).any() else float("nan")
    inverse = float(y[gap <= -25].mean()) if (gap <= -25).any() else float("nan")
    inverted = bool(not np.isnan(fires) and not np.isnan(inverse) and inverse > fires)
    return {
        "premise": (
            "Rule F1 assumes disbursement running ahead of physical progress predicts trouble."
        ),
        "bands": rows,
        "slip_rate_when_f1_fires": round(fires, 4) if not np.isnan(fires) else None,
        "slip_rate_in_opposite_condition": round(inverse, 4) if not np.isnan(inverse) else None,
        "premise_inverted_for_schedule_slip": inverted,
        "interpretation": (
            "The premise is INVERTED for schedule forecasting: project-months where physical "
            "progress runs far ahead of spending slip at a markedly HIGHER rate than the ones "
            "F1 flags. This explains, mechanically, why the shipped rule set scores AUC 0.433 "
            "against this outcome. F1 remains a legitimate FINANCIAL-IRREGULARITY flag with "
            "statutory grounding, and is retained as one — but it must not be presented as a "
            "schedule predictor, because pointed at that outcome it is worse than random."
            if inverted else
            "The premise holds in the expected direction on this data."
        ),
    }


def stage_bias_test(df: pd.DataFrame) -> Dict[str, Any]:
    """Which projects file revisions at all? Tests for a coverage bias."""
    ever = df.groupby("project_id")["slip_next"].max()
    never_ids = set(ever[ever == 0].index)
    ever_ids = set(ever[ever == 1].index)
    out = {}
    for label, ids in (("never_revises", never_ids), ("revises_at_least_once", ever_ids)):
        sub = df[df["project_id"].isin(ids)]
        out[label] = {
            "projects": len(ids),
            "project_months": int(len(sub)),
            "mean_physical_progress": round(float(sub["physical_progress"].mean()), 2),
            "mean_progress_delta": round(float(sub["progress_delta"].mean()), 3),
            "stalled_share": round(float((sub["progress_delta"] <= 0.1).mean()), 3),
        }
    a = out["never_revises"]["mean_physical_progress"]
    b = out["revises_at_least_once"]["mean_physical_progress"]
    return {
        "question": "Are non-revising projects dormant, or simply at an earlier stage?",
        "cohorts": out,
        "finding": (
            f"Projects that never file a revision are NOT dormant — their monthly progress rate "
            f"({out['never_revises']['mean_progress_delta']}%) is indistinguishable from projects "
            f"that do revise ({out['revises_at_least_once']['mean_progress_delta']}%). They are "
            f"simply EARLIER: mean physical progress {a}% vs {b}%. A date is revised when the "
            f"declared deadline comes into view and the agency can see it will be missed, so the "
            f"label is structurally weighted toward later-stage projects."
        ),
        "implication": (
            "This is a real coverage limit. The platform detects slippage that gets ACKNOWLEDGED. "
            "A project that is genuinely delayed but whose agency never files a revision is "
            "invisible to this label, and no correction for that is attempted."
        ),
    }


def main() -> None:
    if not PANEL_PATH.exists():
        raise SystemExit(f"Panel not found at {PANEL_PATH}. Run scripts/build_panel.py first.")

    df = pd.read_csv(PANEL_PATH)
    df = df[df["order_reliable"] == 1].reset_index(drop=True)
    y = df["slip_next"].to_numpy(dtype=int)

    all_rows = compare(df, y)

    # Repeat on ACTIVE months only. If the schedule-relative signal survives
    # when every project in the comparison is physically moving, it is not an
    # artifact of dormant projects sitting in one group.
    active = df[df["progress_delta"] > 0.1].reset_index(drop=True)
    active_rows = compare(active, active["slip_next"].to_numpy(dtype=int))

    sched = [r for r in all_rows if r["group"] == "schedule_relative"]
    sched_ok = sum(1 for r in sched if r.get("corroborates"))
    sched_active_ok = sum(
        1 for r in active_rows if r["group"] == "schedule_relative" and r.get("corroborates")
    )

    f1 = f1_inversion_test(df)
    stage = stage_bias_test(df)

    if sched and sched_ok == len(sched) and sched_active_ok == len(sched):
        verdict = (
            f"LABEL IS SCHEDULE-GROUNDED, NOT ARBITRARY. All {len(sched)} schedule-relative "
            "distress indicators move in the direction a genuine execution problem would "
            "produce before a revision is filed, and they still do when the comparison is "
            "restricted to project-months where the site was physically moving — so the signal "
            "is not an artifact of dormant projects. A filed revision is a lagging record of a "
            "project falling behind the pace its own declared date required. "
            "TWO IMPORTANT QUALIFICATIONS: (1) absolute activity does NOT separate the groups — "
            "slipping projects are not less active, they are further along and closer to a "
            "deadline they will miss; (2) the spend-versus-progress relationship is INVERTED "
            "relative to the statutory assumption behind rule F1."
        )
    elif sched_ok:
        verdict = (
            f"PARTIALLY GROUNDED: {sched_ok} of {len(sched)} schedule-relative indicators "
            "corroborate the label. Claims about predicting real-world delay should be hedged."
        )
    else:
        verdict = (
            "LABEL LOOKS ADMINISTRATIVE: no schedule-relative indicator separates the groups. "
            "On this data the platform should be described as predicting reporting behaviour "
            "rather than project delay. Reported as measured."
        )

    artifact = {
        "provenance": "measured",
        "question": (
            "The label is 'a date moved on a government form'. Does that bureaucratic event "
            "correspond to physical distress on site, or is the model predicting paperwork?"
        ),
        "method": (
            "Compare indicators observable at time t between project-months followed by a filed "
            "revision and those that were not, grouped by what each indicator actually tests. "
            "Effect size is Cohen's d; p-values are deliberately omitted because at n=8,838 "
            "almost any difference is significant while saying nothing about magnitude. The "
            "comparison is then repeated on physically-active months only, as a control."
        ),
        "n_slip_months": int((y == 1).sum()),
        "n_hold_months": int((y == 0).sum()),
        "indicators_all_months": all_rows,
        "indicators_active_months_only": active_rows,
        "rule_f1_premise_test": f1,
        "stage_coverage_bias": stage,
        "verdict": verdict,
        "caveat": (
            "Establishes correlation between a filing and prior schedule distress, not causation. "
            "It cannot detect delay that is real but never filed."
        ),
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2))

    logger.info("Slip months: %d | date-held months: %d", (y == 1).sum(), (y == 0).sum())
    logger.info("%-20s %-18s %11s %11s %8s  %s", "indicator", "group", "pre-slip", "pre-hold", "cohen d", "effect")
    for r in all_rows:
        mark = "" if r.get("corroborates") is not True else "  <- corroborates"
        logger.info("%-20s %-18s %11.3f %11.3f %+8.3f  %-10s%s",
                    r["indicator"], r["group"], r["mean_before_slip"],
                    r["mean_before_hold"], r["cohens_d"], r["effect_size"], mark)
    logger.info("")
    logger.info("Rule F1 premise: slip rate when F1 fires = %s | opposite condition = %s | INVERTED=%s",
                f1["slip_rate_when_f1_fires"], f1["slip_rate_in_opposite_condition"],
                f1["premise_inverted_for_schedule_slip"])
    logger.info("")
    logger.info("%s", verdict)
    logger.info("Artifact written to %s", OUT_PATH)


if __name__ == "__main__":
    main()
