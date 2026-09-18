#!/usr/bin/env python3
"""Measure how early the model warns, relative to the ministry's own paperwork.

Implements contracts/lead_time_backtest.md.

The platform's entire value proposition is "we know before the revised date is
filed." This script replaces that assertion with a measured distribution on
real MoSPI data -- including the cases where lead time is zero or negative, and
always paired with the false-alarm rate that makes it interpretable.

Every project is scored with out-of-fold predictions, i.e. by a model fitted on
folds that never contained that project.

Usage:
    python scripts/lead_time_backtest.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paimana.leadtime")

ROOT = Path(__file__).resolve().parent.parent
OOF_PATH = ROOT / "model" / "paimana_oof_predictions.csv"
PANEL_META_PATH = ROOT / "data" / "paimana_panel_meta.json"
OUT_PATH = ROOT / "model" / "paimana_lead_time.json"

DEFAULT_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


def analyse_threshold(groups: List[pd.DataFrame], threshold: float) -> Dict[str, Any]:
    """Walk every project's timeline once at a fixed alert threshold."""
    lead_times: List[int] = []
    n_slip = 0
    n_alerted_before = 0
    n_never_alerted = 0
    n_left_censored = 0
    n_truncated = 0
    # Confusion at project level: did we ever alert / did it ever slip?
    tp = fp = tn = fn = 0

    for g in groups:
        slips = g.index[g["slip_next"] == 1].tolist()
        alerts = g.index[g["p_xgboost"] >= threshold].tolist()
        ever_slipped = bool(slips)
        ever_alerted = bool(alerts)

        if ever_slipped and ever_alerted:
            tp += 1
        elif ever_slipped and not ever_alerted:
            fn += 1
        elif not ever_slipped and ever_alerted:
            fp += 1
        else:
            tn += 1

        if not ever_slipped:
            continue

        n_slip += 1
        m_slip = int(g.loc[slips[0], "seq_index"])

        # Left censoring: the project's very first observed transition is
        # already a slip, so any earlier warning was outside the window.
        if m_slip == int(g["seq_index"].min()):
            n_left_censored += 1

        if not ever_alerted:
            n_never_alerted += 1
            continue

        m_alert = int(g.loc[alerts[0], "seq_index"])
        # Off-by-one matters here and is easy to get wrong. A transition at
        # seq_index i is scored from the report *at* i and describes what the
        # report at i+1 will say. So an alert raised at transition m_alert
        # reaches the desk at report m_alert, while the revision first appears
        # on the record at report m_slip + 1. Operational lead time is the gap
        # between those two reports.
        lead = (m_slip + 1) - m_alert
        lead_times.append(lead)
        if lead > 0:
            n_alerted_before += 1
        if int(g["n_snapshots"].iloc[0]) <= 3:
            n_truncated += 1

    arr = np.array(lead_times, dtype=float)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": round(threshold, 2),
        "n_slip_projects": n_slip,
        "n_alerted_before_slip": n_alerted_before,
        "n_never_alerted": n_never_alerted,
        "n_left_censored": n_left_censored,
        "n_window_truncated": n_truncated,
        "median_lead_months": round(float(np.median(arr)), 2) if arr.size else None,
        "mean_lead_months": round(float(arr.mean()), 2) if arr.size else None,
        "p25_lead_months": round(float(np.percentile(arr, 25)), 2) if arr.size else None,
        "p75_lead_months": round(float(np.percentile(arr, 75)), 2) if arr.size else None,
        "share_lead_positive": round(float((arr > 0).mean()), 4) if arr.size else None,
        "share_lead_zero": round(float((arr == 0).mean()), 4) if arr.size else None,
        "share_lead_negative": round(float((arr < 0).mean()), 4) if arr.size else None,
        "alert_precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_alarm_rate": round(false_alarm, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall)
        else 0.0,
        "projects_flagged": tp + fp,
        "projects_total": tp + fp + tn + fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure model lead time vs. filed revisions.")
    parser.add_argument("--oof", type=Path, default=OOF_PATH)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    args = parser.parse_args()

    if not args.oof.exists():
        raise SystemExit(
            f"Out-of-fold predictions not found at {args.oof}. Run scripts/train_model.py first."
        )

    df = pd.read_csv(args.oof).sort_values(["project_id", "seq_index"])
    groups = [g for _, g in df.groupby("project_id", sort=True)]
    logger.info("Backtesting %d projects / %d transitions", len(groups), len(df))

    per_threshold = [analyse_threshold(groups, t) for t in args.thresholds]

    scored = [r for r in per_threshold if r["n_slip_projects"] > 0]
    if not scored:
        OUT_PATH.write_text(
            json.dumps({"status": "insufficient_events", "provenance": "measured"}, indent=2)
        )
        logger.warning("No slip events in cohort; wrote insufficient_events artifact.")
        return

    best = max(scored, key=lambda r: r["f1"])
    max_window = int(df.groupby("project_id")["n_snapshots"].first().max()) - 1

    panel_meta = json.loads(PANEL_META_PATH.read_text()) if PANEL_META_PATH.exists() else {}

    artifact = {
        "provenance": "measured",
        "cohort": "out-of-fold (GroupKFold by project_id; no project scored by a model that saw it)",
        "definition": {
            "slip_event": "the first monthly report at which the official revised completion date moves out by more than 15 days",
            "alert": "the first monthly report at which out-of-fold p_model reaches the threshold",
            "lead_time_months": "reports between the alert landing on the desk (report m_alert) and the revision first appearing on the record (report m_slip + 1); positive means the model warned first",
        },
        "per_threshold": per_threshold,
        "recommended_threshold": {
            "threshold": best["threshold"],
            "basis": "maximises project-level F1 on the out-of-fold cohort",
            "median_lead_months": best["median_lead_months"],
            "alert_precision": best["alert_precision"],
            "recall": best["recall"],
            "false_alarm_rate": best["false_alarm_rate"],
            "note": (
                "This is a recommendation with an explicit trade-off, not an optimum. "
                "A ministry prioritising coverage should move the threshold down and "
                "accept the higher false-alarm rate shown in per_threshold."
            ),
        },
        "caveats": {
            "observation_window": (
                f"The public PAIMANA feed exposes at most {max_window + 1} monthly snapshots per "
                f"project, so a lead time longer than {max_window} months cannot be observed. "
                "Every figure here is therefore a LOWER BOUND on achievable lead time, and any "
                "claim exceeding the observation window is unsupportable from this data."
            ),
            "never_alerted_excluded": (
                "Projects that slip without ever crossing the threshold are counted in "
                "n_never_alerted and excluded from the median rather than imputed, which would "
                "otherwise flatter the statistic."
            ),
            "left_censoring": (
                "Projects whose first observed transition is already a slip are counted in "
                "n_left_censored: the warning opportunity predates the data window."
            ),
            "reconstructed_months": panel_meta.get("assumptions", {}),
        },
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }

    OUT_PATH.write_text(json.dumps(artifact, indent=2))

    logger.info("%-6s %-8s %-9s %-10s %-10s %-9s", "thr", "median", "precision", "recall", "false-alarm", "n_slip")
    for r in per_threshold:
        logger.info(
            "%-6.2f %-8s %-9.3f %-10.3f %-10.3f %-9d",
            r["threshold"],
            r["median_lead_months"],
            r["alert_precision"],
            r["recall"],
            r["false_alarm_rate"],
            r["n_slip_projects"],
        )
    logger.info(
        "Recommended threshold %.2f: median lead %s months, precision %.3f, false-alarm %.3f",
        best["threshold"],
        best["median_lead_months"],
        best["alert_precision"],
        best["false_alarm_rate"],
    )
    logger.info("Artifact written to %s", OUT_PATH)


if __name__ == "__main__":
    main()
