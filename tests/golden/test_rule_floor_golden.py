"""FROZEN golden cases for the deterministic RuleFloor (contracts/rule_floor.md).

These are hand-verified ground truth for safety-critical statutory logic.
Per the project workflow rules: cases may be ADDED, never modified or deleted
without explicit human approval. If a change to `paimana_rules.py` breaks one
of these, the change is wrong until a human says otherwise.

Every expected value below was computed by hand from the contract's flag table,
not by running the implementation and recording whatever it printed.
"""

import pytest

from app.paimana_contracts import ProjectInput
from app.paimana_rules import calculate_rule_floor

# Flag weights from contracts/rule_floor.md. Duplicated deliberately: if
# someone edits the weights in the implementation, these cases must fail.
W_F1, W_F2, W_F3, W_F4, W_F5, W_F6 = 30.0, 25.0, 25.0, 20.0, 30.0, 25.0


def project(**overrides) -> ProjectInput:
    """A deliberately clean baseline: no flag fires.

    original_duration 60, months_elapsed 12, delay 0  -> 48 months still to run
    expenditure 20% vs progress 20%                   -> no decoupling
    progress_change 2.0/month                         -> not stalled
    """
    base = dict(
        project_id="GOLDEN-000",
        project_name="Golden Baseline Project",
        sector="Railways",
        implementing_agency="RVNL",
        original_cost=1000.0,
        expenditure=200.0,
        physical_progress=20.0,
        progress_change_recent=2.0,
        original_duration_months=60.0,
        months_elapsed=12.0,
        current_delay_months=0.0,
        overdue_milestones=0,
        total_scheduled_milestones=10,
        clearance_pending_days=0,
        days_since_last_update=15,
        dispute_status="NONE",
    )
    base.update(overrides)
    return ProjectInput(**base)


def active_ids(evaluation) -> set:
    return {s.flag_id for s in evaluation.signals if s.is_active}


# ---------------------------------------------------------------- baseline

def test_golden_clean_project_scores_zero():
    """Case 6: a clean project floors at 0 and still reports all six flags."""
    ev = calculate_rule_floor(project())
    assert ev.rule_floor == 0.0
    assert ev.active_signals_count == 0
    assert ev.critical_override is False
    assert len(ev.signals) == 6
    assert {s.flag_id for s in ev.signals} == {"F1", "F2", "F3", "F4", "F5", "F6"}


# ------------------------------------------------------------ F6 behaviour

def test_golden_f6_declared_date_passed():
    """Case 1: duration 36, elapsed 40, delay 0 -> 4 months overdue."""
    ev = calculate_rule_floor(
        project(original_duration_months=36.0, months_elapsed=40.0, physical_progress=60.0)
    )
    assert active_ids(ev) == {"F6"}
    assert ev.rule_floor == W_F6
    assert ev.critical_override is False
    f6 = next(s for s in ev.signals if s.flag_id == "F6")
    assert f6.metric_value == pytest.approx(4.0, abs=0.01)


def test_golden_f6_inactive_when_delay_extends_the_date():
    """Case 2: delay 12 pushes the declared date to month 48; elapsed 40 < 48."""
    ev = calculate_rule_floor(
        project(
            original_duration_months=36.0,
            months_elapsed=40.0,
            current_delay_months=12.0,
            physical_progress=60.0,
        )
    )
    assert "F6" not in active_ids(ev)
    assert ev.rule_floor == 0.0


def test_golden_f6_boundary_exactly_at_declared_date():
    """Case 3: elapsed == duration is 'elapsed' (condition is <= 0)."""
    ev = calculate_rule_floor(
        project(original_duration_months=36.0, months_elapsed=36.0, physical_progress=60.0)
    )
    assert "F6" in active_ids(ev)
    assert ev.rule_floor == W_F6


def test_golden_f6_inactive_when_work_essentially_complete():
    """Case 4: progress 97 >= 95, so a trailing date is administrative only."""
    ev = calculate_rule_floor(
        project(original_duration_months=36.0, months_elapsed=40.0, physical_progress=97.0)
    )
    assert "F6" not in active_ids(ev)


def test_golden_f6_boundary_progress_exactly_95_is_inactive():
    """Edge case: the threshold is strict (< 95), matching F2."""
    ev = calculate_rule_floor(
        project(original_duration_months=36.0, months_elapsed=40.0, physical_progress=95.0)
    )
    assert "F6" not in active_ids(ev)


def test_golden_f6_critical_escalation_at_twelve_months_overdue():
    """Case 5: duration 36, elapsed 60 -> 24 months overdue, critical."""
    ev = calculate_rule_floor(
        project(original_duration_months=36.0, months_elapsed=60.0, physical_progress=20.0)
    )
    assert "F6" in active_ids(ev)
    assert ev.critical_override is True
    f6 = next(s for s in ev.signals if s.flag_id == "F6")
    assert f6.severity == "CRITICAL"
    assert f6.metric_value == pytest.approx(24.0, abs=0.01)


def test_golden_f6_severity_high_just_under_escalation():
    """11 months overdue is HIGH, not CRITICAL: the boundary is 12."""
    ev = calculate_rule_floor(
        project(original_duration_months=36.0, months_elapsed=47.0, physical_progress=20.0)
    )
    f6 = next(s for s in ev.signals if s.flag_id == "F6")
    assert f6.is_active is True
    assert f6.severity == "HIGH"
    assert ev.critical_override is False


# ------------------------------------------------- pre-existing flags F1..F5

def test_golden_f1_spend_progress_decoupling():
    """Spend 60% against 20% progress = 40pp gap, below halfway -> F1 critical."""
    ev = calculate_rule_floor(project(expenditure=600.0, physical_progress=20.0))
    assert "F1" in active_ids(ev)
    assert ev.critical_override is True  # gap >= 40pp


def test_golden_f1_boundary_exactly_25pp_gap_fires():
    """45% spend - 20% progress = exactly 25pp: condition is >=, so it fires."""
    ev = calculate_rule_floor(project(expenditure=450.0, physical_progress=20.0))
    assert "F1" in active_ids(ev)
    assert ev.critical_override is False  # 25pp < 40pp escalation


def test_golden_f2_progress_stall():
    """Flatlined progress after 20% of the schedule has elapsed."""
    ev = calculate_rule_floor(
        project(progress_change_recent=0.0, months_elapsed=30.0, original_duration_months=60.0)
    )
    assert "F2" in active_ids(ev)


def test_golden_f4_stale_reporting():
    ev = calculate_rule_floor(project(days_since_last_update=61))
    assert active_ids(ev) == {"F4"}
    assert ev.rule_floor == W_F4


def test_golden_f4_boundary_sixty_days_does_not_fire():
    """Threshold is strictly greater than 60."""
    ev = calculate_rule_floor(project(days_since_last_update=60))
    assert "F4" not in active_ids(ev)


def test_golden_f5_litigation_is_always_critical():
    ev = calculate_rule_floor(project(dispute_status="HIGH_COURT_STAY"))
    assert active_ids(ev) == {"F5"}
    assert ev.rule_floor == W_F5
    assert ev.critical_override is True


def test_golden_f5_conciliation_does_not_fire():
    """CONCILIATION is deliberately outside the F5 trigger set."""
    ev = calculate_rule_floor(project(dispute_status="CONCILIATION"))
    assert "F5" not in active_ids(ev)


# ------------------------------------------------------------ accumulation

def test_golden_additive_accumulation_f4_plus_f6():
    """Case 8 extended: independent flags add exactly once each."""
    ev = calculate_rule_floor(
        project(
            days_since_last_update=90,
            original_duration_months=36.0,
            months_elapsed=40.0,
            physical_progress=60.0,
        )
    )
    assert active_ids(ev) == {"F4", "F6"}
    assert ev.rule_floor == W_F4 + W_F6  # 45.0


def test_golden_floor_clamps_at_one_hundred():
    """Case 7: F1+F2+F3+F5+F6 = 135 raw, clamped to 100."""
    ev = calculate_rule_floor(
        project(
            expenditure=600.0,          # F1 (30)
            physical_progress=20.0,
            progress_change_recent=0.0,  # F2 (25)
            months_elapsed=60.0,
            original_duration_months=36.0,  # also F6 (25), 24m overdue
            clearance_pending_days=400,  # F3 (25)
            dispute_status="ARBITRATION",  # F5 (30)
        )
    )
    assert ev.rule_floor == 100.0
    assert ev.active_signals_count == 5
    assert ev.critical_override is True


# ------------------------------------------------- regression: F6 must not
# ------------------------------------------------- disturb existing cases

@pytest.mark.parametrize(
    "overrides,expected_floor",
    [
        ({}, 0.0),
        ({"days_since_last_update": 61}, W_F4),
        ({"dispute_status": "ARBITRATION"}, W_F5),
        ({"clearance_pending_days": 200}, W_F3),
        ({"overdue_milestones": 5, "total_scheduled_milestones": 10}, W_F3),
    ],
)
def test_golden_f6_does_not_change_floors_where_it_is_inactive(overrides, expected_floor):
    """Adding F6 must be inert for every project whose deadline has not passed.

    The baseline has 48 months still to run, so F6 cannot fire in any of these
    rows. This is the regression guard the contract calls for.
    """
    ev = calculate_rule_floor(project(**overrides))
    assert "F6" not in active_ids(ev)
    assert ev.rule_floor == expected_floor
