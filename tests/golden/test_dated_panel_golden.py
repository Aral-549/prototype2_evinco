"""FROZEN golden cases for the OBSERVED (dated) time axis.

Contract: contracts/monthly_harvest.md, contracts/panel_builder.md

`scripts/harvest_monthly.py` attaches the portal's own freeze month to every
record, so snapshot order is read from the calendar instead of reconstructed
under assumptions A1/A2. These cases pin that mode, and in particular pin the
trap that would have silently destroyed data:

    The legacy dedupe key did not include the reporting month. Two genuinely
    distinct months in which nothing changed on site are byte-identical on the
    measurement fields, so they would have collapsed into one snapshot --
    deleting a real observation and corrupting every month-denominated feature
    downstream. `test_golden_two_quiet_months_are_not_collapsed` fails if that
    is ever reintroduced.

Add cases; do not edit or delete.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_panel import build_panel  # noqa: E402


def rec(month, expenditure, progress, revised, *, cost=1000, pid=8800,
        sanction="01/01/2020", orig_end="01/01/2024"):
    """One raw record carrying a real freeze month."""
    return {
        "ProjectId": pid,
        "ProjectName": "Dated Golden Project",
        "SectorName": "Railways",
        "OriginalCost": cost,
        "RevisedCost": str(cost),
        "Expenditure": str(expenditure),
        "PhysicalProgress": progress,
        "SanctionDate": sanction,
        "OriginalEndDate": orig_end,
        "RevisedDate": revised,
        "Remarks": None,
        "freeze_month": month,
        "harvest_sector_id": "108",
    }


def build(records, tmp_path, **kw):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(records))
    return build_panel(raw, **kw)


def ordered(rows):
    return sorted(rows, key=lambda r: r["seq_index"])


# ------------------------------------------------------------- mode detection

def test_golden_dated_records_use_the_observed_axis(tmp_path):
    rows, meta = build(
        [
            rec("2025-07", 10, 10, "31/12/2026"),
            rec("2025-08", 20, 20, "31/12/2026"),
            rec("2025-09", 30, 30, "30/06/2027"),
        ],
        tmp_path,
    )
    assert meta["time_axis"]["mode"] == "observed"
    assert meta["assumptions_apply"] is False
    assert meta["time_axis"]["dated_projects"] == 1
    assert meta["time_axis"]["reconstructed_projects"] == 0
    assert all(r["time_axis"] == "observed" for r in rows)
    assert [r["as_of_month"] for r in ordered(rows)] == ["2025-07", "2025-08"]


def test_golden_undated_records_still_use_the_reconstructed_axis(tmp_path):
    """Backward compatibility: the legacy harvest must keep working."""
    legacy = rec("2025-07", 10, 10, "31/12/2026")
    legacy.pop("freeze_month")
    legacy2 = rec("2025-08", 20, 20, "30/06/2027")
    legacy2.pop("freeze_month")
    rows, meta = build([legacy, legacy2], tmp_path)
    assert meta["time_axis"]["mode"] == "reconstructed"
    assert meta["assumptions_apply"] is True
    assert all(r["time_axis"] == "reconstructed" for r in rows)


# ------------------------------------------------ the data-destroying trap

def test_golden_two_quiet_months_are_not_collapsed(tmp_path):
    """THE regression guard for this feature.

    Nothing moved on site between July and August: expenditure, progress and
    the declared date are all identical. Those two records are byte-identical
    apart from the month. If the reporting month is dropped from the dedupe
    key they collapse to one snapshot and a real month of history vanishes.
    """
    rows, _meta = build(
        [
            rec("2025-07", 50, 40, "31/12/2026"),
            rec("2025-08", 50, 40, "31/12/2026"),
            rec("2025-09", 50, 40, "30/06/2027"),
        ],
        tmp_path,
    )
    assert len(rows) == 2, "a quiet month was collapsed away"
    assert [r["as_of_month"] for r in ordered(rows)] == ["2025-07", "2025-08"]
    assert rows[0]["n_snapshots"] == 3
    # The slip is filed in September, so only the August transition is positive.
    assert [r["slip_next"] for r in ordered(rows)] == [0, 1]


def test_golden_ordering_follows_the_calendar_not_expenditure(tmp_path):
    """Expenditure is NOT always monotone in the real feed (observed: a probe
    project fell from 160.48 to 155.72 between two freezes). On a dated panel
    the calendar wins, so a dip must not reorder the project."""
    rows, _meta = build(
        [
            rec("2025-07", 160.48, 41, "31/12/2026"),
            rec("2025-08", 155.72, 41, "31/12/2026"),   # spend went DOWN
            rec("2025-09", 161.30, 43, "31/12/2026"),
        ],
        tmp_path,
    )
    seq = ordered(rows)
    assert [r["as_of_month"] for r in seq] == ["2025-07", "2025-08"]
    assert seq[0]["expenditure"] == pytest.approx(160.48)
    assert seq[1]["expenditure"] == pytest.approx(155.72)
    # A negative month-on-month spend delta is floored at 0, not made negative.
    assert seq[1]["expenditure_delta"] >= 0


def test_golden_backward_progress_is_reported_not_excluded(tmp_path):
    """On an observed axis there is no A1 left to falsify, so a project whose
    reported progress goes backwards is a FEED QUALITY signal and must still
    be trained on -- not silently dropped as it was in reconstructed mode."""
    rows, meta = build(
        [
            rec("2025-07", 10, 90, "31/12/2026"),
            rec("2025-08", 20, 70, "31/12/2026"),
            rec("2025-09", 30, 50, "31/12/2026"),
            rec("2025-10", 40, 30, "30/06/2027"),
        ],
        tmp_path,
    )
    assert all(r["order_reliable"] == 1 for r in rows), "dated rows must not be excluded"
    assert meta["counts"].get("dated_projects_with_backward_progress") == 1
    assert meta["counts"].get("projects_order_unreliable", 0) == 0


# ------------------------------------------------------------- calendar gaps

def test_golden_gap_months_records_a_missing_report(tmp_path):
    """A project absent from a freeze month yields a 2-month transition."""
    rows, _meta = build(
        [
            rec("2025-07", 10, 10, "31/12/2026"),
            rec("2025-09", 30, 30, "31/12/2026"),   # August missing
        ],
        tmp_path,
    )
    assert len(rows) == 1
    assert rows[0]["gap_months"] == 2


def test_golden_horizon_window_is_calendar_months_not_reports(tmp_path):
    """With a gap, "within 3 months" must mean the calendar, not 3 reports.

    July -> the slip lands in December, which is 5 calendar months later. It
    must NOT count inside the 3-month window even though it is only the 2nd
    subsequent report.
    """
    rows, _meta = build(
        [
            rec("2025-07", 10, 10, "31/12/2026"),
            rec("2025-12", 20, 20, "30/06/2027"),   # slip, 5 months later
            rec("2026-01", 30, 30, "30/06/2027"),
            rec("2026-02", 40, 40, "30/06/2027"),
            rec("2026-03", 50, 50, "30/06/2027"),
            rec("2026-04", 60, 60, "30/06/2027"),
            rec("2026-05", 70, 70, "30/06/2027"),
        ],
        tmp_path,
    )
    july = min(rows, key=lambda r: r["seq_index"])
    assert july["as_of_month"] == "2025-07"
    assert july["slip_within_3m"] == 0, "a slip 5 months out must not count as 3-month"
    assert july["slip_within_6m"] == 1


def test_golden_incomplete_calendar_window_is_censored(tmp_path):
    """The window is censored on the calendar, not on report count."""
    rows, _meta = build(
        [
            rec("2026-05", 10, 10, "31/12/2027"),
            rec("2026-06", 20, 20, "31/12/2027"),
        ],
        tmp_path,
    )
    row = rows[0]
    assert row["slip_next"] == 0          # observed
    assert row["slip_within_3m"] == ""    # only 1 month of follow-up exists
    assert row["censored_3m"] == 1


# --------------------------------------------------------------- determinism

def test_golden_dated_build_is_order_independent(tmp_path):
    """Input record order must not affect output: the calendar decides."""
    records = [
        rec("2025-09", 30, 30, "30/06/2027"),
        rec("2025-07", 10, 10, "31/12/2026"),
        rec("2025-08", 20, 20, "31/12/2026"),
    ]
    first, _ = build(records, tmp_path)
    second, _ = build(list(reversed(records)), tmp_path)
    assert first == second
    assert [r["as_of_month"] for r in ordered(first)] == ["2025-07", "2025-08"]
