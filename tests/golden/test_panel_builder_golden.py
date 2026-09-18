"""FROZEN golden cases for panel reconstruction (contracts/panel_builder.md).

Hand-verified ground truth for the label definition. If the panel builder's
notion of "a slip" drifts, every metric in the platform silently changes
meaning -- so these cases are frozen. Add cases; do not edit or delete.

Each expected value below is derived from the contract's behaviour table by
hand. They are written adversarially: the aim is to catch the builder counting
a slip where there is none, inventing history from duplicate query slices, or
coding an unknown outcome as a negative.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_panel import build_panel  # noqa: E402


def record(expenditure, progress, revised, *, cost=1000, sanction="01/01/2020",
           orig_end="01/01/2024", pid=9001, name="Golden Panel Project",
           sector="Railways"):
    """One raw PAIMANA record in the portal's exact wire shape."""
    return {
        "ProjectId": pid,
        "ProjectName": name,
        "SectorName": sector,
        "StateName": None,
        "OriginalCost": cost,
        "RevisedCost": str(cost),
        "Expenditure": str(expenditure),
        "PhysicalProgress": progress,
        "SanctionDate": sanction,
        "OriginalEndDate": orig_end,
        "RevisedDate": revised,
        "Remarks": None,
    }


def build(records, tmp_path, **kwargs):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(records))
    return build_panel(raw, **kwargs)


def labels(rows):
    return [r["slip_next"] for r in sorted(rows, key=lambda r: r["seq_index"])]


# ------------------------------------------------------------ label semantics

def test_golden_case1_slip_detected_on_date_push(tmp_path):
    """Case 1: dates [Jan31, Jan31, Jun30] -> labels [0, 1]."""
    rows, _ = build(
        [
            record(10, 10, "31/01/2026"),
            record(20, 20, "31/01/2026"),
            record(30, 30, "30/06/2026"),
        ],
        tmp_path,
    )
    assert len(rows) == 2
    assert labels(rows) == [0, 1]


def test_golden_case2_single_snapshot_yields_no_rows(tmp_path):
    """Case 2: a project with one snapshot has no transition to label."""
    rows, meta = build([record(10, 10, "31/01/2026")], tmp_path)
    assert rows == []
    assert meta["counts"].get("projects_dropped_single_snapshot") == 1


def test_golden_case3_missing_next_date_drops_the_row(tmp_path):
    """Case 3: an unlabelable transition is dropped, never guessed."""
    rows, meta = build(
        [record(10, 10, "31/01/2026"), record(20, 20, None)],
        tmp_path,
    )
    assert rows == []
    assert meta["counts"].get("transitions_dropped_no_label") == 1


def test_golden_case4_ten_day_move_is_not_a_slip(tmp_path):
    """Case 4: 10 days <= the 15-day threshold, absorbing month-end jitter."""
    rows, _ = build(
        [record(10, 10, "31/03/2026"), record(20, 20, "10/04/2026")],
        tmp_path,
    )
    assert labels(rows) == [0]


def test_golden_case4b_sixteen_day_move_is_a_slip(tmp_path):
    """Boundary: strictly greater than 15 days counts."""
    rows, _ = build(
        [record(10, 10, "31/03/2026"), record(20, 20, "16/04/2026")],
        tmp_path,
    )
    assert labels(rows) == [1]


def test_golden_case5_date_pulled_earlier_is_not_a_slip(tmp_path):
    """Case 5: an accelerated project must never be labelled a slip."""
    rows, _ = build(
        [record(10, 10, "30/06/2026"), record(20, 20, "31/01/2026")],
        tmp_path,
    )
    assert labels(rows) == [0]


# ---------------------------------------------------------------- ordering

def test_golden_case7_duplicate_query_slices_are_not_extra_history(tmp_path):
    """Case 7: the harvester returns a project in several query slices.

    Those repeats are byte-identical and must collapse. Counting them as extra
    months would fabricate history that never happened.
    """
    a = record(10, 10, "31/01/2026")
    b = record(20, 20, "30/06/2026")
    rows, meta = build([a, b, dict(a), dict(b), dict(a)], tmp_path)
    assert len(rows) == 1
    assert rows[0]["n_snapshots"] == 2
    assert meta["counts"]["duplicate_snapshots_dropped"] == 3


def test_golden_case6_unreliable_ordering_is_flagged_not_silently_used(tmp_path):
    """Case 6: when progress contradicts expenditure order, say so.

    Expenditure ascending gives progress [90, 70, 50, 30] -- monotonically
    wrong on every adjacent pair, so assumption A1 is falsified here.
    """
    rows, meta = build(
        [
            record(10, 90, "31/01/2026"),
            record(20, 70, "31/01/2026"),
            record(30, 50, "31/01/2026"),
            record(40, 30, "31/01/2026"),
        ],
        tmp_path,
    )
    assert rows, "rows are retained for diagnostics"
    assert all(r["order_reliable"] == 0 for r in rows)
    assert rows[0]["order_confidence"] == 0.0
    assert meta["counts"]["projects_order_unreliable"] == 1


def test_golden_snapshots_are_ordered_oldest_first(tmp_path):
    """seq_index 0 must be the earliest snapshot, not the newest."""
    rows, _ = build(
        [
            record(90, 90, "30/06/2026"),
            record(10, 10, "31/01/2026"),
            record(50, 50, "31/03/2026"),
        ],
        tmp_path,
    )
    ordered = sorted(rows, key=lambda r: r["seq_index"])
    assert [r["physical_progress"] for r in ordered] == [10.0, 50.0]
    assert labels(rows) == [1, 1]


# ------------------------------------------------------------- robustness

def test_golden_case8_zero_cost_project_is_dropped(tmp_path):
    """Case 8: cost-normalised features are undefined without a cost."""
    rows, meta = build(
        [record(10, 10, "31/01/2026", cost=0), record(20, 20, "30/06/2026", cost=0)],
        tmp_path,
    )
    assert rows == []
    assert meta["counts"].get("projects_dropped_no_cost") == 1


def test_golden_case9_zero_progress_does_not_divide_by_zero(tmp_path):
    rows, _ = build(
        [record(0, 0, "31/01/2026"), record(0, 0, "30/06/2026")],
        tmp_path,
    )
    assert len(rows) == 1
    assert rows[0]["burn_to_progress_ratio"] == 0.0


def test_golden_case10_missing_sanction_date_is_marked_imputed(tmp_path):
    """Case 10: never silently default a missing sanction date."""
    rows, _ = build(
        [
            record(10, 10, "31/01/2026", sanction=None),
            record(20, 20, "30/06/2026", sanction=None),
        ],
        tmp_path,
    )
    assert rows[0]["sanction_imputed"] == 1


def test_golden_expenditure_reported_in_rupees_is_rescaled(tmp_path):
    """Some agencies file rupees where crore is expected (> 20x sanction)."""
    rows, meta = build(
        [
            record(100_000, 10, "31/01/2026", cost=1000),
            record(200_000, 20, "30/06/2026", cost=1000),
        ],
        tmp_path,
    )
    assert meta["counts"]["expenditure_unit_rescaled"] == 2
    assert rows[0]["expenditure"] == pytest.approx(1000.0)


def test_golden_project_that_never_slips_is_kept_as_negatives(tmp_path):
    """The majority class must not be discarded: it defines the base rate."""
    rows, _ = build(
        [
            record(10, 10, "31/12/2027"),
            record(20, 20, "31/12/2027"),
            record(30, 30, "31/12/2027"),
        ],
        tmp_path,
    )
    assert labels(rows) == [0, 0]


# ------------------------------------------------------- horizon censoring

def test_golden_horizon_label_counts_a_slip_anywhere_in_the_window(tmp_path):
    """3-month horizon: a slip at t+3 counts for the row at t."""
    rows, _ = build(
        [
            record(10, 10, "31/01/2026"),
            record(20, 20, "31/01/2026"),
            record(30, 30, "31/01/2026"),
            record(40, 40, "30/09/2026"),
        ],
        tmp_path,
    )
    first = min(rows, key=lambda r: r["seq_index"])
    assert first["slip_next"] == 0
    assert first["slip_within_3m"] == 1


def test_golden_short_window_is_censored_not_counted_as_no_slip(tmp_path):
    """An exhausted observation window is UNKNOWN, never a negative.

    This is the single most dangerous silent failure in the pipeline: coding
    "we stopped looking" as "it did not happen" manufactures negatives out of
    missing follow-up and inflates every metric downstream.
    """
    rows, _ = build(
        [record(10, 10, "31/12/2027"), record(20, 20, "31/12/2027")],
        tmp_path,
    )
    row = rows[0]
    assert row["slip_next"] == 0          # the 1-month answer IS observed
    assert row["slip_within_3m"] == ""    # the 3-month answer is not
    assert row["censored_3m"] == 1
    assert row["slip_within_6m"] == ""
    assert row["censored_6m"] == 1


def test_golden_early_slip_is_not_censored_even_in_a_short_window(tmp_path):
    """If the slip already happened, the horizon answer is known: 1."""
    rows, _ = build(
        [record(10, 10, "31/01/2026"), record(20, 20, "30/09/2026")],
        tmp_path,
    )
    row = rows[0]
    assert row["slip_next"] == 1
    assert row["slip_within_3m"] == 1
    assert row["censored_3m"] == 0


# ------------------------------------------------------------ determinism

def test_golden_build_is_deterministic(tmp_path):
    """Same input must give byte-identical output, or nothing downstream is
    reproducible."""
    records = [
        record(10, 10, "31/01/2026"),
        record(20, 20, "30/06/2026"),
        record(30, 30, "30/06/2026"),
    ]
    first, _ = build(records, tmp_path)
    second, _ = build(list(reversed(records)), tmp_path)
    assert first == second


def test_golden_absent_feed_fields_are_reported_not_imputed(tmp_path):
    """The public feed publishes no Remarks/StateName; the panel must say so
    rather than let downstream code assume the NLP pillar has input."""
    _rows, meta = build(
        [record(10, 10, "31/01/2026"), record(20, 20, "30/06/2026")],
        tmp_path,
    )
    report = meta["fields_absent_from_live_feed"]
    assert "0.0%" in report["Remarks"]
    assert "0.0%" in report["StateName"]
