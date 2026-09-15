"""Unit tests for the Day-1 contracts (Spec Section 3.1)."""

import pytest
from pydantic import ValidationError

from app.paimana_contracts import (
    BatchCsvResponse,
    BatchProjectRequest,
    ProjectGovernanceAssessment,
    ProjectInput,
)


def _base_payload(**overrides):
    payload = {
        "project_id": "MOSPI-RLW-0194",
        "project_name": "Western Dedicated Freight Corridor",
        "sector": "Railways",
        "implementing_agency": "DFCCIL",
        "original_cost": 28181.0,
        "expenditure": 44800.0,
        "physical_progress": 88.2,
    }
    payload.update(overrides)
    return payload


class TestProjectInputValidation:
    def test_valid_minimal_payload(self):
        p = ProjectInput(**_base_payload())
        assert p.project_id == "MOSPI-RLW-0194"
        assert p.expenditure_pct == pytest.approx(158.9729, abs=1e-3)

    def test_sector_alias_normalization(self):
        assert ProjectInput(**_base_payload(sector="railway")).sector == "Railways"
        assert ProjectInput(**_base_payload(sector="roads")).sector == "Road Transport & Highways"
        assert ProjectInput(**_base_payload(sector="highways")).sector == "Road Transport & Highways"
        assert ProjectInput(**_base_payload(sector="power")).sector == "Power"

    def test_empty_sector_rejected(self):
        with pytest.raises(ValidationError):
            ProjectInput(**_base_payload(sector="  "))

    def test_negative_original_cost_rejected(self):
        with pytest.raises(ValidationError):
            ProjectInput(**_base_payload(original_cost=-5.0))

    def test_physical_progress_bounds(self):
        with pytest.raises(ValidationError):
            ProjectInput(**_base_payload(physical_progress=100.5))
        with pytest.raises(ValidationError):
            ProjectInput(**_base_payload(physical_progress=-0.1))

    def test_expenditure_unit_check_rejects_50x(self):
        with pytest.raises(ValidationError, match="50x"):
            ProjectInput(**_base_payload(expenditure=28181.0 * 51))

    def test_dispute_status_literal(self):
        with pytest.raises(ValidationError):
            ProjectInput(**_base_payload(dispute_status="SUPREME_COURT"))

    def test_min_project_id_length(self):
        with pytest.raises(ValidationError):
            ProjectInput(**_base_payload(project_id="X"))


class TestComputedFields:
    def test_cost_overrun_pct_current(self):
        p = ProjectInput(**_base_payload(revised_cost=51101.0))
        expected = (51101.0 - 28181.0) / 28181.0 * 100.0
        assert p.cost_overrun_pct_current == pytest.approx(expected, abs=1e-3)

    def test_cost_overrun_none_when_unrevised(self):
        p = ProjectInput(**_base_payload())
        assert p.cost_overrun_pct_current == 0.0

    def test_spend_progress_gap(self):
        p = ProjectInput(**_base_payload())
        assert p.spend_progress_gap == pytest.approx(p.expenditure_pct - 88.2, abs=1e-3)

    def test_remaining_progress_pct(self):
        p = ProjectInput(**_base_payload())
        assert p.remaining_progress_pct == pytest.approx(11.8, abs=1e-6)

    def test_execution_velocity_completed_project(self):
        p = ProjectInput(**_base_payload(physical_progress=100.0))
        assert p.execution_velocity == 1.0

    def test_execution_velocity_slow_project(self):
        # 12 months elapsed of 36-month duration; 10% done -> realized 0.833%/mo,
        # required 90/24 = 3.75%/mo -> velocity 0.2222
        p = ProjectInput(
            **_base_payload(
                physical_progress=10.0,
                original_duration_months=36.0,
                months_elapsed=12.0,
            )
        )
        assert p.execution_velocity == pytest.approx(0.2222, abs=1e-3)


class TestLeakageQuarantine:
    def test_feature_dict_zeroes_has_revised_doc(self):
        p = ProjectInput(**_base_payload())
        columns = [
            "physical_progress", "progress_change", "expenditure",
            "expenditure_change_robust", "expenditure_pct_robust", "original_cost",
            "revised_cost", "cost_overrun_pct_robust", "cost_change_robust",
            "remaining_progress_pct", "current_delay_months_robust", "has_revised_doc",
        ]
        fd = p.to_feature_dict(columns)
        assert fd["has_revised_doc"] == 0.0
        # Spec Sections 1.2 / 4.1: both post-facto outcome records are
        # quarantined -- the frame must carry zeros, not raw payload values.
        assert fd["current_delay_months_robust"] == 0.0
        assert fd["cost_overrun_pct_robust"] == pytest.approx(p.cost_overrun_pct_current, abs=1e-3)

    def test_feature_dict_fills_missing_columns_with_zero(self):
        p = ProjectInput(**_base_payload())
        fd = p.to_feature_dict(["some_unknown_future_feature", "original_cost"])
        assert fd["some_unknown_future_feature"] == 0.0
        assert fd["original_cost"] == 28181.0


class TestBatchSchemas:
    def test_batch_request_rejects_empty(self):
        with pytest.raises(ValidationError):
            BatchProjectRequest(projects=[])

    def test_batch_request_max_5000(self):
        with pytest.raises(ValidationError):
            BatchProjectRequest(projects=[ProjectInput(**_base_payload())] * 5001)


class TestAssessmentBounds:
    def test_gov_score_bounds_enforced(self):
        with pytest.raises(ValidationError):
            ProjectGovernanceAssessment(
                project_id="X-1",
                project_name="Bounds Test Project",
                sector="Railways",
                implementing_agency="RVNL",
                original_cost_crores=1000.0,
                p_model=0.5,
                p_model_score=50.0,
                rule_floor=101.0,  # out of bounds
                gov_score=75.0,
                dominant_source="RULE_FLOOR_OVERRIDE",
                risk_tier="CRITICAL",
                sector_median_overrun_pct=25.0,
                effective_overrun_pct=25.0,
                capital_at_risk_crores=100.0,
                base_rate_probability=0.5,
            )

    def test_csv_response_inherits_columns_ignored(self):
        # Construct minimal valid assessment
        a = ProjectGovernanceAssessment(
            project_id="X-1",
            project_name="CSV Test Project Alpha",
            sector="Railways",
            implementing_agency="RVNL",
            original_cost_crores=1000.0,
            p_model=0.5,
            p_model_score=50.0,
            rule_floor=0.0,
            gov_score=50.0,
            dominant_source="MACHINE_LEARNING",
            risk_tier="HIGH",
            sector_median_overrun_pct=25.0,
            effective_overrun_pct=25.0,
            capital_at_risk_crores=125.0,
            base_rate_probability=0.5,
        )
        r = BatchCsvResponse(
            total_projects=1,
            total_monitored_capex_crores=1000.0,
            total_capital_at_risk_crores=125.0,
            risk_tier_counts={"LOW": 0, "MODERATE": 0, "HIGH": 1, "CRITICAL": 0},
            dominant_source_counts={"MACHINE_LEARNING": 1, "RULE_FLOOR_OVERRIDE": 0},
            ranked_projects=[a],
            columns_ignored=["Remarks"],
        )
        assert r.columns_ignored == ["Remarks"]
