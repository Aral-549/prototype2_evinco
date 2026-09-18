"""FROZEN golden cases for the early-warning lead-time backtest.

Contract: contracts/lead_time_backtest.md

Lead time is the platform's headline product claim, and it already shipped
wrong once: the original implementation was off by exactly one reporting cycle
and reported a median of 0 months (see BUGLOG.md). These cases pin the
arithmetic, including the `+1`, and pin the refusals — a project that slips
without ever being flagged must never be quietly dropped from the median.

Add cases; do not edit or delete.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from lead_time_backtest import analyse_threshold  # noqa: E402


def project(pid, scores, slips, n_snapshots=None):
    """Build one project's out-of-fold timeline.

    scores: per-transition p_model. slips: per-transition 0/1 label.
    """
    assert len(scores) == len(slips)
    return pd.DataFrame(
        {
            "project_id": [pid] * len(scores),
            "seq_index": list(range(len(scores))),
            "n_snapshots": [n_snapshots or (len(scores) + 1)] * len(scores),
            "p_xgboost": scores,
            "slip_next": slips,
        }
    )


def run(groups, threshold=0.5):
    # analyse_threshold indexes by DataFrame index, so give each group a
    # distinct index range exactly as the real groupby does.
    out, offset = [], 0
    for g in groups:
        g = g.copy()
        g.index = range(offset, offset + len(g))
        offset += len(g)
        out.append(g)
    return analyse_threshold(out, threshold)


# ------------------------------------------------------------ core arithmetic

def test_golden_case1_alert_well_before_the_filing():
    """Case 1: slip first labelled at transition 8, first alert at transition 3.

    The alert lands at report 3; the revision first appears on the record at
    report 9. Lead time is 6 reporting cycles, not 5.
    """
    scores = [0.1] * 3 + [0.9] * 6
    slips = [0] * 8 + [1]
    r = run([project("P1", scores, slips)])
    assert r["n_slip_projects"] == 1
    assert r["median_lead_months"] == 6.0
    assert r["n_alerted_before_slip"] == 1


def test_golden_case2_same_cycle_detection_is_one_month_of_warning():
    """Case 2: alert and slip label at the same transition still warns a cycle ahead.

    The model said "the next report will move the date" and it did. That is one
    month of usable warning, not zero.
    """
    r = run([project("P1", [0.1, 0.1, 0.1, 0.9], [0, 0, 0, 1])])
    assert r["median_lead_months"] == 1.0


def test_golden_case3_negative_lead_is_reported_not_clipped():
    """Case 3: the model only woke up after the revision was already filed."""
    scores = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
    slips = [0, 0, 1, 0, 0, 0, 0]
    r = run([project("P1", scores, slips)])
    # slip at transition 2 -> on record at report 3; alert at report 6.
    assert r["median_lead_months"] == -3.0
    assert r["n_alerted_before_slip"] == 0
    assert r["share_lead_negative"] == 1.0


def test_golden_case4_slipped_but_never_alerted_is_excluded_not_imputed():
    """Case 4: silently dropping these would flatter the median; counting them
    as a huge lead would flatter it more. They are excluded AND counted."""
    slipping_but_missed = project("MISS", [0.1, 0.1, 0.1], [0, 0, 1])
    caught = project("HIT", [0.9, 0.1, 0.1], [0, 0, 1])
    r = run([slipping_but_missed, caught])
    assert r["n_slip_projects"] == 2
    assert r["n_never_alerted"] == 1
    # Median is computed from the one measurable project only.
    assert r["median_lead_months"] == 3.0
    assert r["recall"] == pytest.approx(0.5)


def test_golden_case5_quiet_clean_project_is_a_true_negative():
    r = run([project("CLEAN", [0.1, 0.1, 0.1], [0, 0, 0])])
    assert r["n_slip_projects"] == 0
    assert r["false_alarm_rate"] == 0.0
    assert r["projects_flagged"] == 0


def test_golden_case6_alert_on_a_project_that_never_slips_is_a_false_alarm():
    """Case 6: without this denominator a lead-time figure means nothing."""
    r = run([project("NOISY", [0.1, 0.9, 0.1], [0, 0, 0])])
    assert r["n_slip_projects"] == 0
    assert r["false_alarm_rate"] == 1.0
    assert r["alert_precision"] == 0.0


# ----------------------------------------------------------------- censoring

def test_golden_left_censored_project_is_counted_separately():
    """A project whose very first transition is already a slip had its warning
    window open before the data did."""
    r = run([project("EARLY", [0.9, 0.1], [1, 0])])
    assert r["n_left_censored"] == 1


def test_golden_short_window_is_flagged_as_truncated():
    """Case 7: a 2-snapshot project can never demonstrate a long lead, so its
    contribution biases the median downward and must be visible."""
    r = run([project("SHORT", [0.9], [1], n_snapshots=2)])
    assert r["n_window_truncated"] == 1
    assert r["median_lead_months"] == 1.0


# ------------------------------------------------- threshold sweep behaviour

def test_golden_case8_degenerate_low_threshold_is_obvious_in_the_artifact():
    """Case 8: a threshold that flags everything buys lead time at the cost of
    a ~1.0 false-alarm rate. Both numbers must be visible together."""
    groups = [
        project("SLIP", [0.4, 0.4, 0.4], [0, 0, 1]),
        project("CLEAN", [0.4, 0.4, 0.4], [0, 0, 0]),
    ]
    r = run(groups, threshold=0.05)
    assert r["median_lead_months"] == 3.0
    assert r["false_alarm_rate"] == 1.0
    assert r["alert_precision"] == pytest.approx(0.5)


def test_golden_raising_the_threshold_trades_recall_for_precision():
    """The monotone trade-off a ministry chooses its operating point along."""
    groups = [
        project("SLIP", [0.3, 0.6, 0.9], [0, 0, 1]),
        project("CLEAN", [0.3, 0.4, 0.45], [0, 0, 0]),
    ]
    loose = run(groups, threshold=0.25)
    tight = run(groups, threshold=0.55)
    assert loose["recall"] >= tight["recall"]
    assert tight["alert_precision"] >= loose["alert_precision"]
    assert tight["false_alarm_rate"] <= loose["false_alarm_rate"]


def test_golden_threshold_boundary_is_inclusive():
    """A score exactly at the threshold counts as an alert (>=, not >)."""
    r = run([project("EDGE", [0.5, 0.1], [0, 1])], threshold=0.5)
    assert r["projects_flagged"] == 1
    assert r["median_lead_months"] == 2.0


def test_golden_mixed_portfolio_confusion_matrix_is_consistent():
    """Project-level confusion counts must partition the portfolio exactly."""
    groups = [
        project("TP", [0.9, 0.1], [0, 1]),
        project("FN", [0.1, 0.1], [0, 1]),
        project("FP", [0.9, 0.1], [0, 0]),
        project("TN", [0.1, 0.1], [0, 0]),
    ]
    r = run(groups)
    assert r["projects_total"] == 4
    assert r["projects_flagged"] == 2
    assert r["alert_precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["false_alarm_rate"] == pytest.approx(0.5)
