"""Unit tests for the deterministic RuleFloor engine (Spec Section 3.2.1)."""

import pytest

from app.paimana_contracts import ProjectInput
from app.paimana_rules import calculate_rule_floor


def _project(**overrides):
    payload = dict(
        project_id="MOSPI-TST-0001",
        project_name="RuleFloor Test Project",
        sector="Railways",
        implementing_agency="RVNL",
        original_cost=1000.0,
        expenditure=100.0,          # spend 10%, progress 20% -> gap -10 (F1 off)
        physical_progress=20.0,
        progress_change_recent=1.0,  # active velocity (F2 off)
        original_duration_months=36.0,
        months_elapsed=6.0,          # elapsed ratio 0.167 (< 0.20, F2 off anyway)
        current_delay_months=0.0,
        overdue_milestones=0,
        total_scheduled_milestones=10,  # density 0.0 (F3 off)
        clearance_pending_days=0,
        days_since_last_update=15,      # (F4 off)
        dispute_status="NONE",          # (F5 off)
    )
    payload.update(overrides)
    return ProjectInput(**payload)


class TestCleanProject:
    def test_zero_flags_zero_floor(self):
        ev = calculate_rule_floor(_project())
        assert ev.rule_floor == 0.0
        assert ev.active_signals_count == 0
        assert ev.critical_override is False
        assert all(not s.is_active for s in ev.signals)
        assert len(ev.signals) == 5  # F1..F5 always evaluated

    def test_summary_normal_message(self):
        ev = calculate_rule_floor(_project())
        assert "normal tolerance" in ev.summary


class TestFlagF1SpendDecoupling:
    def test_gap_25_with_low_progress_triggers(self):
        # spend 30% vs progress 4% -> gap 26 >= 25, progress < 50
        ev = calculate_rule_floor(_project(expenditure=300.0, physical_progress=4.0))
        f1 = next(s for s in ev.signals if s.flag_id == "F1")
        assert f1.is_active is True
        assert f1.weight == 30.0
        assert f1.severity == "HIGH"  # gap < 40
        assert ev.rule_floor == 30.0

    def test_gap_40_is_critical_override(self):
        # spend 45% vs progress 4% -> gap 41 >= 40
        ev = calculate_rule_floor(_project(expenditure=450.0, physical_progress=4.0))
        f1 = next(s for s in ev.signals if s.flag_id == "F1")
        assert f1.severity == "CRITICAL"
        assert ev.critical_override is True

    def test_no_trigger_when_progress_at_50(self):
        # spend 80% vs progress 50% -> gap 30 but progress NOT < 50
        ev = calculate_rule_floor(_project(expenditure=800.0, physical_progress=50.0))
        f1 = next(s for s in ev.signals if s.flag_id == "F1")
        assert f1.is_active is False

    def test_negative_gap_no_trigger(self):
        ev = calculate_rule_floor(_project(expenditure=50.0, physical_progress=40.0))
        assert not ev.signals[0].is_active


class TestFlagF2ProgressStall:
    def test_stall_after_20pct_elapsed(self):
        ev = calculate_rule_floor(
            _project(progress_change_recent=0.0, months_elapsed=8.0, original_duration_months=36.0)
        )
        f2 = next(s for s in ev.signals if s.flag_id == "F2")
        assert f2.is_active is True
        assert f2.weight == 25.0

    def test_no_stall_before_20pct_elapsed(self):
        ev = calculate_rule_floor(
            _project(progress_change_recent=0.0, months_elapsed=6.0, original_duration_months=60.0)
        )
        f2 = next(s for s in ev.signals if s.flag_id == "F2")
        assert f2.is_active is False

    def test_no_stall_when_progress_95_plus(self):
        ev = calculate_rule_floor(
            _project(progress_change_recent=0.0, months_elapsed=8.0, physical_progress=96.0)
        )
        f2 = next(s for s in ev.signals if s.flag_id == "F2")
        assert f2.is_active is False

    def test_severity_high_at_halfway_elapsed(self):
        ev = calculate_rule_floor(
            _project(progress_change_recent=0.0, months_elapsed=30.0, original_duration_months=50.0)
        )
        f2 = next(s for s in ev.signals if s.flag_id == "F2")
        assert f2.severity == "HIGH"


class TestFlagF3MilestonesClearances:
    def test_milestone_density_triggers(self):
        ev = calculate_rule_floor(_project(overdue_milestones=5, total_scheduled_milestones=10))
        f3 = next(s for s in ev.signals if s.flag_id == "F3")
        assert f3.is_active is True
        assert f3.metric_name == "milestone_density"
        assert f3.metric_value == 0.5

    def test_clearance_over_180_triggers(self):
        ev = calculate_rule_floor(_project(clearance_pending_days=200))
        f3 = next(s for s in ev.signals if s.flag_id == "F3")
        assert f3.is_active is True
        assert f3.metric_name == "clearance_pending_days"

    def test_clearance_over_365_is_critical(self):
        ev = calculate_rule_floor(_project(clearance_pending_days=400))
        f3 = next(s for s in ev.signals if s.flag_id == "F3")
        assert f3.severity == "CRITICAL"
        assert ev.critical_override is True

    def test_neither_condition_no_trigger(self):
        ev = calculate_rule_floor(_project(overdue_milestones=4, total_scheduled_milestones=10, clearance_pending_days=100))
        f3 = next(s for s in ev.signals if s.flag_id == "F3")
        assert f3.is_active is False


class TestFlagF4Reporting:
    def test_stale_registry_triggers(self):
        ev = calculate_rule_floor(_project(days_since_last_update=61))
        f4 = next(s for s in ev.signals if s.flag_id == "F4")
        assert f4.is_active is True
        assert f4.weight == 20.0

    def test_60_days_exactly_no_trigger(self):
        ev = calculate_rule_floor(_project(days_since_last_update=60))
        f4 = next(s for s in ev.signals if s.flag_id == "F4")
        assert f4.is_active is False


class TestFlagF5Disputes:
    @pytest.mark.parametrize("status", ["ARBITRATION", "HIGH_COURT_STAY", "TERMINATION_NOTICE"])
    def test_litigation_statuses_trigger(self, status):
        ev = calculate_rule_floor(_project(dispute_status=status))
        f5 = next(s for s in ev.signals if s.flag_id == "F5")
        assert f5.is_active is True
        assert f5.severity == "CRITICAL"
        assert ev.critical_override is True

    def test_conciliation_does_not_trigger(self):
        ev = calculate_rule_floor(_project(dispute_status="CONCILIATION"))
        f5 = next(s for s in ev.signals if s.flag_id == "F5")
        assert f5.is_active is False


class TestClamping:
    def test_additive_clamp_at_100(self):
        # F1 (30) + F2 (25) + F3 (25) + F4 (20) + F5 (30) = 130 -> clamped to 100
        ev = calculate_rule_floor(
            _project(
                expenditure=300.0,
                physical_progress=4.0,
                progress_change_recent=0.0,
                months_elapsed=8.0,
                overdue_milestones=5,
                total_scheduled_milestones=10,
                days_since_last_update=90,
                dispute_status="ARBITRATION",
            )
        )
        assert ev.rule_floor == 100.0
        assert ev.active_signals_count == 5

    def test_custom_weights_override(self):
        ev = calculate_rule_floor(_project(dispute_status="ARBITRATION"), custom_weights={"F5": 15.0})
        assert ev.rule_floor == 15.0

    def test_sub_additive_combination(self):
        # F1 (30) + F4 (20) = 50
        ev = calculate_rule_floor(
            _project(expenditure=300.0, physical_progress=4.0, days_since_last_update=90)
        )
        assert ev.rule_floor == 50.0
        assert ev.active_signals_count == 2
