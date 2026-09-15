"""Unit tests for the GovScore engine (Spec Sections 2.1, 2.2)."""

import pytest

from app.paimana_contracts import ProjectInput
from app.paimana_engine import (
    assess_project,
    classify_risk_tier,
    compute_gov_score,
    formulate_interventions,
)
from app.paimana_car import calculate_capital_at_risk
from app.paimana_rules import calculate_rule_floor


class TestGovScoreOverride:
    def test_max_operator_basic(self):
        gov, source = compute_gov_score(p_model=0.85, rule_floor=60.0)
        assert gov == 85.0
        assert source == "MACHINE_LEARNING"

    def test_rule_floor_wins(self):
        gov, source = compute_gov_score(p_model=0.08, rule_floor=85.0)
        assert gov == 85.0
        assert source == "RULE_FLOOR_OVERRIDE"

    def test_tie_prefers_rule_floor_override_label(self):
        # At an exact tie the supremum subgradient is set-valued; the tie is
        # credited to the statutory branch so attribution is never ambiguous
        # (Spec 2.2 item 4: RuleFloor-governed scores are 100% explained by
        # the tripped flags {F_k}).
        gov, source = compute_gov_score(p_model=0.5, rule_floor=50.0)
        assert gov == 50.0
        assert source == "RULE_FLOOR_OVERRIDE"

    def test_tie_detection_robust_to_floating_point_noise(self):
        # 0.55 * 100 == 55.00000000000001 in IEEE-754; the tie against a
        # rule floor of 55.0 must still resolve to RULE_FLOOR_OVERRIDE.
        gov, source = compute_gov_score(p_model=0.55, rule_floor=55.0)
        assert gov == pytest.approx(55.0)
        assert source == "RULE_FLOOR_OVERRIDE"

    def test_inputs_clamped_to_valid_ranges(self):
        gov, _ = compute_gov_score(p_model=1.5, rule_floor=250.0)
        assert gov == 100.0
        gov, _ = compute_gov_score(p_model=-0.5, rule_floor=-10.0)
        assert gov == 0.0

    def test_zero_scores(self):
        gov, source = compute_gov_score(p_model=0.0, rule_floor=0.0)
        assert gov == 0.0
        # Exact tie at zero is credited to the statutory branch (see
        # test_tie_prefers_rule_floor_override_label).
        assert source == "RULE_FLOOR_OVERRIDE"


class TestNonMaskingInvariant:
    def test_statutory_violation_cannot_be_diluted(self):
        # Spec 2.2 Theorem 1: RuleFloor >= tau implies GovScore >= tau for ANY P_model
        for p_model in (0.0, 0.08, 0.15, 0.5):
            gov, source = compute_gov_score(p_model=p_model, rule_floor=85.0)
            assert gov >= 75.0, "critical statutory floor must survive any ML probability"
            assert source == "RULE_FLOOR_OVERRIDE"

    def test_predictive_alert_cannot_be_suppressed(self):
        # Spec 2.2 item 2: 100*P_model >= tau implies GovScore >= tau for ANY RuleFloor
        for rule_floor in (0.0, 20.0, 45.0):
            gov, source = compute_gov_score(p_model=0.85, rule_floor=rule_floor)
            assert gov >= 75.0
            assert source == "MACHINE_LEARNING"

    def test_convex_blend_would_mask_but_max_does_not(self):
        # The exact pathology from Spec 2.2: RuleFloor=85, P_model=0.08
        p_scaled, rule_floor = 8.0, 85.0
        convex_blend = 0.5 * p_scaled + 0.5 * rule_floor
        assert convex_blend < 50.0  # alpha=0.5 blend masks below High threshold
        gov, _ = compute_gov_score(p_model=0.08, rule_floor=85.0)
        assert gov == 85.0  # supremum never masks


class TestRiskBanding:
    @pytest.mark.parametrize(
        "score,tier",
        [
            (0.0, "LOW"),
            (24.99, "LOW"),
            (25.0, "MODERATE"),
            (49.99, "MODERATE"),
            (50.0, "HIGH"),
            (74.99, "HIGH"),
            (75.0, "CRITICAL"),
            (100.0, "CRITICAL"),
        ],
    )
    def test_band_boundaries(self, score, tier):
        assert classify_risk_tier(score) == tier


class TestFullAssessment:
    def _distressed(self) -> ProjectInput:
        return ProjectInput(
            project_id="MOSPI-TST-9999",
            project_name="Distressed Test Bridge",
            sector="Railways",
            implementing_agency="RVNL",
            original_cost=2000.0,
            revised_cost=3800.0,
            expenditure=2600.0,
            physical_progress=42.0,
            progress_change_recent=0.0,
            original_duration_months=36.0,
            months_elapsed=12.0,
            current_delay_months=36.0,
            days_since_last_update=70,
            dispute_status="ARBITRATION",
        )

    def test_distressed_project_full_pipeline(self):
        p = self._distressed()
        assessment = assess_project(p, p_model=0.05)
        rf = calculate_rule_floor(p)
        # F1: spend 130% - progress 42% = 88 gap -> active + critical
        # F2: 0.0 change, elapsed ratio 0.33 -> active
        # F4: 70 days stale -> active
        # F5: arbitration -> active + critical
        assert rf.rule_floor == 100.0  # 30+25+20+30 = 105 clamped
        assert assessment.gov_score == 100.0
        assert assessment.risk_tier == "CRITICAL"
        assert assessment.dominant_source == "RULE_FLOOR_OVERRIDE"
        # CaR uses P_model, not GovScore
        expected_car = 2000.0 * 0.05 * max(90.0, 25.0) / 100.0  # overrun 90% > median 25%
        assert assessment.capital_at_risk_crores == pytest.approx(expected_car, abs=0.01)
        assert assessment.effective_overrun_pct == 90.0
        assert len(assessment.rule_signals) == 5
        assert any("PRAGATI" in i for i in assessment.prescriptive_interventions)

    def test_healthy_project_ml_dominant(self):
        p = ProjectInput(
            project_id="MOSPI-TST-0002",
            project_name="Healthy Test Bypass",
            sector="Road Transport & Highways",
            implementing_agency="NHAI",
            original_cost=1500.0,
            revised_cost=1600.0,
            expenditure=500.0,
            physical_progress=45.0,
            progress_change_recent=2.5,
            original_duration_months=36.0,
            months_elapsed=18.0,
            current_delay_months=0.0,
            days_since_last_update=10,
            dispute_status="NONE",
        )
        assessment = assess_project(p, p_model=0.9)
        rf = calculate_rule_floor(p)
        assert rf.rule_floor == 0.0
        assert assessment.gov_score == pytest.approx(90.0)
        assert assessment.dominant_source == "MACHINE_LEARNING"
        assert assessment.risk_tier == "CRITICAL"
        # Sector median 12% vs current overrun ~6.67% -> fallback used
        assert assessment.effective_overrun_pct == 12.0
        assert assessment.sector_median_overrun_pct == 12.0

    def test_gov_score_never_below_rule_floor(self):
        # Property: for any inputs, GovScore == max(100*P, RuleFloor)
        for p_model in (0.0, 0.2, 0.6, 1.0):
            for floor in (0.0, 30.0, 55.0, 100.0):
                a = assess_project(self._distressed(), p_model=p_model)
                # recompute via engine directly (assessment uses the project's own floor)
                pass
        # Direct invariant check with the floor from the distressed project
        p = self._distressed()
        rf = calculate_rule_floor(p).rule_floor
        for p_model in (0.0, 0.2, 0.6, 1.0):
            assessment = assess_project(p, p_model=p_model)
            assert assessment.gov_score >= rf
            assert assessment.gov_score >= 100.0 * p_model - 1e-9

    def test_car_is_finance_engine_p_model_not_govscore(self):
        p = self._distressed()
        low_p = assess_project(p, p_model=0.01)
        high_p = assess_project(p, p_model=0.60)
        # CaR scales with P_model even when GovScore is pinned at the floor
        assert low_p.capital_at_risk_crores < high_p.capital_at_risk_crores
        assert low_p.gov_score == high_p.gov_score == 100.0


class TestInterventions:
    def test_flag_specific_directives(self):
        p = ProjectInput(
            project_id="MOSPI-TST-0003",
            project_name="Intervention Test Project",
            sector="Power",
            implementing_agency="NTPC",
            original_cost=5000.0,
            expenditure=2000.0,  # spend 40% vs progress 5% -> F1 gap 35 (active)
            physical_progress=5.0,
            progress_change_recent=0.0,
            original_duration_months=48.0,
            months_elapsed=20.0,
            days_since_last_update=100,
            clearance_pending_days=400,
            dispute_status="HIGH_COURT_STAY",
        )
        rf = calculate_rule_floor(p)
        car = calculate_capital_at_risk(5000.0, 0.4, 0.0, "Power")
        interventions = formulate_interventions(p, 100.0, "CRITICAL", rf, car)
        text = " ".join(interventions)
        assert "GFR 2017 Rule 159" in text          # F1
        assert "recovery schedule" in text           # F2
        assert "Project Monitoring Group" in text    # F3
        assert "non-compliance notice" in text       # F4
        assert "Arbitration Act" in text             # F5
        assert "PRAGATI" in text                     # CRITICAL escalation

    def test_low_risk_gets_monitoring_directive(self):
        p = ProjectInput(
            project_id="MOSPI-TST-0004",
            project_name="Quiet Test Project",
            sector="Coal",
            implementing_agency="CCL",
            original_cost=800.0,
            expenditure=300.0,
            physical_progress=50.0,
            progress_change_recent=1.5,
        )
        rf = calculate_rule_floor(p)
        car = calculate_capital_at_risk(800.0, 0.1, 0.0, "Coal")
        interventions = formulate_interventions(p, 10.0, "LOW", rf, car)
        assert any("monthly PAIMANA reporting" in i for i in interventions)
