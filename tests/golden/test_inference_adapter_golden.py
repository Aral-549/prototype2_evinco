"""FROZEN golden cases for the CUF -> design-matrix adapter.

Contract: contracts/inference_adapter.md

These exist because the v1 platform died of a train/serve mismatch. The most
valuable test in this file is `test_golden_train_serve_parity`: it builds the
same project twice — once through the training-time panel builder and once
through the serving-time adapter — and requires the shared features to agree.
A silent divergence between those two paths is exactly the failure that drove
p_model below 0.035 on all 2,155 live projects and made the ML branch inert.

Add cases; do not edit or delete.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.paimana_contracts import ProjectInput  # noqa: E402
from app.services.feature_adapter import build_feature_row  # noqa: E402

FEATURES = [
    "original_cost",
    "log_original_cost",
    "expenditure",
    "expenditure_pct",
    "expenditure_delta",
    "expenditure_delta_pct",
    "physical_progress",
    "progress_delta",
    "progress_delta_3m",
    "remaining_progress_pct",
    "spend_progress_gap",
    "burn_to_progress_ratio",
    "months_elapsed",
    "original_duration_months",
    "elapsed_ratio",
    "schedule_pressure",
    "months_to_revised_date",
    "revised_date_already_slipped_months",
    "stall_streak",
    "progress_velocity",
    "required_velocity",
    "velocity_deficit",
]

MEDIANS = {name: 0.0 for name in FEATURES}


def adapt(**overrides):
    base = dict(
        project_id="ADAPT-000",
        project_name="Adapter Golden Project",
        sector="Railways",
        implementing_agency="RVNL",
        original_cost=1000.0,
        expenditure=200.0,
        physical_progress=20.0,
        progress_change_recent=1.0,
        original_duration_months=36.0,
        months_elapsed=12.0,
        current_delay_months=0.0,
    )
    base.update(overrides)
    frame, approximations = build_feature_row(
        ProjectInput(**base), FEATURES, {}, MEDIANS
    )
    return frame.iloc[0], approximations


def test_golden_case1_spend_progress_decoupling():
    """Case 1: 60% spent against 30% progress is a 30pp decoupling gap."""
    row, _ = adapt(original_cost=1000.0, expenditure=600.0, physical_progress=30.0)
    assert row["expenditure_pct"] == pytest.approx(60.0)
    assert row["spend_progress_gap"] == pytest.approx(30.0)


def test_golden_case2_zero_progress_does_not_divide_by_zero():
    row, _ = adapt(physical_progress=0.0)
    assert row["burn_to_progress_ratio"] == 0.0
    assert row["remaining_progress_pct"] == pytest.approx(100.0)


def test_golden_case3_passed_deadline_is_negative_not_clipped():
    """Case 3: elapsed 60 against a 36-month schedule is 24 months overdue.

    The sign carries meaning — clipping it to zero would erase the single
    strongest signal in the dataset.
    """
    row, _ = adapt(original_duration_months=36.0, months_elapsed=60.0, current_delay_months=0.0)
    assert row["months_to_revised_date"] == pytest.approx(-24.0)


def test_golden_case3b_delay_extends_the_declared_date():
    """Accumulated slippage moves the declared date out; 36 + 12 - 40 = 8."""
    row, _ = adapt(
        original_duration_months=36.0, months_elapsed=40.0, current_delay_months=12.0
    )
    assert row["months_to_revised_date"] == pytest.approx(8.0)
    assert row["revised_date_already_slipped_months"] == pytest.approx(12.0)


def test_golden_case4_required_velocity_when_deadline_passed():
    """Case 4: with no time left, everything remaining is required immediately."""
    row, _ = adapt(
        original_duration_months=36.0, months_elapsed=60.0, physical_progress=40.0
    )
    assert row["months_to_revised_date"] < 0
    assert row["required_velocity"] == pytest.approx(60.0)


def test_golden_case5_unseen_sector_does_not_raise():
    """Case 5: an unseen sector must map to a neutral value, not an exception."""
    row, _ = adapt(sector="Quantum Teleportation Corridor")
    assert row is not None


def test_golden_case6_stall_is_detected_and_declared_an_approximation():
    """Case 6: a single snapshot can only say stalled / not stalled."""
    row, approximations = adapt(progress_change_recent=0.0)
    assert row["stall_streak"] == 1.0
    assert "stall_streak" in approximations

    row2, _ = adapt(progress_change_recent=2.0)
    assert row2["stall_streak"] == 0.0


def test_golden_case7_column_order_matches_exactly_and_no_nan():
    """Case 7: XGBoost consumes positionally — a reordered frame is silent corruption."""
    frame, _ = build_feature_row(
        ProjectInput(
            project_id="ADAPT-ORDER",
            project_name="Order Check",
            original_cost=500.0,
            expenditure=100.0,
            physical_progress=10.0,
        ),
        FEATURES,
        {},
        MEDIANS,
    )
    assert list(frame.columns) == FEATURES
    assert not frame.isna().any().any()


def test_golden_case8_unknown_column_is_median_filled_and_reported():
    """Case 8: a column the adapter cannot derive is never silently zero-filled."""
    columns = FEATURES + ["some_future_feature"]
    medians = {**MEDIANS, "some_future_feature": 7.5}
    frame, approximations = build_feature_row(
        ProjectInput(
            project_id="ADAPT-NEW",
            project_name="Unknown Column",
            original_cost=500.0,
            expenditure=100.0,
            physical_progress=10.0,
        ),
        columns,
        {},
        medians,
    )
    assert frame.iloc[0]["some_future_feature"] == pytest.approx(7.5)
    assert "some_future_feature" in approximations


def test_golden_complete_project_has_no_remaining_work():
    """Edge case: progress 100 must not produce NaN anywhere."""
    row, _ = adapt(physical_progress=100.0)
    assert row["remaining_progress_pct"] == 0.0
    assert row["required_velocity"] == 0.0
    assert all(math.isfinite(float(v)) for v in row.values)


def test_golden_the_three_history_approximations_are_always_declared():
    """The caller must always be told which inputs were estimated."""
    _row, approximations = adapt()
    for name in ("progress_delta_3m", "progress_velocity", "stall_streak"):
        assert name in approximations
        assert len(approximations[name]) > 20


# ---------------------------------------------------------------- parity

def test_golden_train_serve_parity():
    """The training path and the serving path must agree on shared features.

    This is the regression guard for the defect that killed v1. A project is
    constructed as raw PAIMANA snapshots, run through the panel builder, and
    then the same underlying facts are passed through the serving adapter. The
    features that do not depend on project history must match.

    The three history-dependent features (progress_delta_3m, progress_velocity,
    stall_streak) are excluded here because the contract documents them as
    approximations at serve time — that divergence is declared, not silent.
    """
    from build_panel import build_panel

    def record(expenditure, progress, revised):
        return {
            "ProjectId": 7700,
            "ProjectName": "Parity Project",
            "SectorName": "Railways",
            "OriginalCost": 1000,
            "RevisedCost": "1000",
            "Expenditure": str(expenditure),
            "PhysicalProgress": progress,
            "SanctionDate": "01/01/2020",
            "OriginalEndDate": "01/01/2024",
            "RevisedDate": revised,
            "Remarks": None,
        }

    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.json"
        raw.write_text(
            json.dumps(
                [
                    record(100, 10, "01/01/2024"),
                    record(200, 20, "01/01/2024"),
                    record(300, 30, "01/07/2024"),
                ]
            )
        )
        rows, _meta = build_panel(raw)

    panel_row = max(rows, key=lambda r: r["seq_index"])

    # Rebuild the same facts as a CUF snapshot. months_to_revised_date is the
    # bridge: the panel measures it from the declared date, the adapter derives
    # it as duration + delay - elapsed, so we supply the delay that reproduces
    # the panel's declared-date position.
    served, _approx = adapt(
        original_cost=panel_row["original_cost"],
        expenditure=panel_row["expenditure"],
        physical_progress=panel_row["physical_progress"],
        progress_change_recent=panel_row["progress_delta"],
        expenditure_change_recent=panel_row["expenditure_delta"],
        original_duration_months=panel_row["original_duration_months"],
        months_elapsed=panel_row["months_elapsed"],
        current_delay_months=panel_row["revised_date_already_slipped_months"],
    )

    shared = [
        "original_cost",
        "log_original_cost",
        "expenditure",
        "expenditure_pct",
        "expenditure_delta",
        "expenditure_delta_pct",
        "physical_progress",
        "progress_delta",
        "remaining_progress_pct",
        "spend_progress_gap",
        "burn_to_progress_ratio",
        "months_elapsed",
        "original_duration_months",
        "elapsed_ratio",
        "schedule_pressure",
        "revised_date_already_slipped_months",
    ]
    for name in shared:
        assert float(served[name]) == pytest.approx(
            float(panel_row[name]), rel=1e-3, abs=1e-3
        ), f"train/serve divergence on '{name}'"

    # The declared-date position must agree to within a month of rounding.
    assert float(served["months_to_revised_date"]) == pytest.approx(
        float(panel_row["months_to_revised_date"]), abs=1.0
    )
