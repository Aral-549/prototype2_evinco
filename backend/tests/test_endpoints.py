"""Automated test suite for MoSPI PAIMANA Predictive Intelligence API.

Adapted to the Master Specification Day-1 contracts: GovScore lattice,
RuleFloor flags F1..F5, sector-median CaR, DeLong benchmark protocol,
and the zero-confabulation CUF gap endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.paimana_contracts import ProjectInput
from app.services.model_service import LEAKAGE_QUARANTINED_FEATURES

client = TestClient(app)


def test_root():
    """Verify root operational endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["problem_statement"] == "SIH26103 - Use case on web-based integrated project-monitoring platform"
    assert data["status"] == "ONLINE"


def test_health():
    """Verify health endpoint, model diagnostics, and leakage quarantine report."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 12
    assert "has_revised_doc" in data["features"]
    # Spec Section 4.1: quarantine must be reported and cover the outcome proxy
    assert "has_revised_doc" in data["quarantined_features"]
    assert "current_delay_months_robust" in data["quarantined_features"]


# ---------------------------------------------------------------- Single predict


def test_predict_project_healthy():
    """Verify governance assessment for a stable project on schedule."""
    payload = {
        "project_id": "TEST-RTH-001",
        "project_name": "Test Greenfield Highway Bypass",
        "sector": "Road Transport & Highways",
        "implementing_agency": "NHAI",
        "original_cost": 1500.0,
        "revised_cost": 1500.0,
        "expenditure": 450.0,
        "physical_progress": 32.0,
        "progress_change_recent": 2.5,
        "months_elapsed": 12.0,
        "original_duration_months": 36.0,
        "days_since_last_update": 10,
        "dispute_status": "NONE",
    }
    response = client.post("/api/v1/predict/project", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["project_id"] == "TEST-RTH-001"
    # GovScore lattice fields present and bounded
    assert 0.0 <= data["p_model"] <= 1.0
    assert 0.0 <= data["p_model_score"] <= 100.0
    assert 0.0 <= data["rule_floor"] <= 100.0
    assert data["gov_score"] == max(data["p_model_score"], data["rule_floor"])
    assert data["risk_tier"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert data["dominant_source"] in ["MACHINE_LEARNING", "RULE_FLOOR_OVERRIDE"]
    assert data["capital_at_risk_crores"] >= 0.0
    # Road sector median is 12%
    assert data["sector_median_overrun_pct"] == 12.0
    assert len(data["rule_signals"]) == 5
    assert len(data["shap_drivers"]) == 5
    assert 0.0 <= data["base_rate_probability"] <= 1.0
    assert len(data["prescriptive_interventions"]) >= 1


def test_predict_project_high_risk_decoupled():
    """Verify RuleFloor override on a project with severe spend-progress decoupling."""
    payload = {
        "project_id": "TEST-RLW-999",
        "project_name": "Test Severe Delay Railway Bridge",
        "sector": "Railways",
        "implementing_agency": "RVNL",
        "original_cost": 2000.0,
        "revised_cost": 3800.0,
        "expenditure": 2600.0,   # 130% of original cost
        "physical_progress": 42.0,
        "progress_change_recent": 0.0,
        "current_delay_months": 36.0,
        "original_duration_months": 36.0,
        "months_elapsed": 18.0,
        "days_since_last_update": 70,
        "dispute_status": "ARBITRATION",
    }
    response = client.post("/api/v1/predict/project", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Non-Masking Invariant: statutory floor must dominate the composite score
    assert data["rule_floor"] >= 50.0
    assert data["gov_score"] >= data["p_model_score"]
    assert data["gov_score"] >= 75.0
    assert data["risk_tier"] == "CRITICAL"
    assert data["dominant_source"] == "RULE_FLOOR_OVERRIDE"
    # Capital exposure remains P_model-driven (finance engine decoupled)
    assert data["capital_at_risk_crores"] > 0.0
    # Railways sector-median prior applied even at 90% current overrun
    assert data["sector_median_overrun_pct"] == 25.0
    assert data["effective_overrun_pct"] == 90.0
    # Active statutory flags include F1 (decoupling) and F5 (arbitration)
    active = {s["flag_id"] for s in data["rule_signals"] if s["is_active"]}
    assert "F1" in active
    assert "F5" in active
    assert "F4" in active  # 70 days stale


def test_predict_project_govscore_never_below_rule_floor():
    """Property test: the max() override lattice holds across arbitrary inputs."""
    payloads = [
        {"p_model_input": dict(original_cost=5000.0, expenditure=100.0, physical_progress=5.0,
                               progress_change_recent=3.0, months_elapsed=2.0,
                               original_duration_months=36.0, days_since_last_update=5,
                               dispute_status="NONE", sector="Power")},
        {"p_model_input": dict(original_cost=800.0, expenditure=700.0, physical_progress=10.0,
                               progress_change_recent=0.0, months_elapsed=30.0,
                               original_duration_months=36.0, days_since_last_update=90,
                               dispute_status="TERMINATION_NOTICE", sector="Railways")},
    ]
    for p in payloads:
        base = {
            "project_id": "TEST-INV-1",
            "project_name": "Invariant Test Project",
            "implementing_agency": "NTPC",
        }
        body = {**base, **p["p_model_input"]}
        response = client.post("/api/v1/predict/project", json=body)
        assert response.status_code == 200
        data = response.json()
        assert data["gov_score"] >= data["rule_floor"] - 1e-9
        assert data["gov_score"] >= data["p_model_score"] - 1e-9


def test_predict_project_validation_error():
    """Malformed CUF payloads must 422, not 500."""
    payload = {
        "project_id": "TEST-BAD-01",
        "project_name": "Bad Units Project",
        "original_cost": 100.0,
        "expenditure": 999999.0,  # > 50x original cost
        "physical_progress": 10.0,
    }
    response = client.post("/api/v1/predict/project", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------- Batch


def test_predict_batch():
    """Verify batch evaluation, CaR ranking, and tier/source distributions."""
    payload = {
        "projects": [
            {
                "project_id": "P1-BR",
                "project_name": "Small Bridge",
                "sector": "Road Transport & Highways",
                "implementing_agency": "State PWD",
                "original_cost": 150.0,
                "expenditure": 30.0,
                "physical_progress": 25.0,
            },
            {
                "project_id": "P2-FC",
                "project_name": "Mega Freight Corridor",
                "sector": "Railways",
                "implementing_agency": "DFCCIL",
                "original_cost": 30000.0,
                "revised_cost": 45000.0,
                "expenditure": 28000.0,
                "physical_progress": 65.0,
                "current_delay_months": 24.0,
            },
        ]
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_projects"] == 2
    assert data["total_monitored_capex_crores"] == 30150.0
    assert data["total_capital_at_risk_crores"] > 0.0
    assert set(data["risk_tier_counts"]) == {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert set(data["dominant_source_counts"]) == {"MACHINE_LEARNING", "RULE_FLOOR_OVERRIDE"}
    # First project in ranking should have highest Capital-at-Risk
    assert data["ranked_projects"][0]["project_id"] == "P2-FC"


def test_predict_batch_requires_non_empty():
    response = client.post("/api/v1/predict/batch", json={"projects": []})
    assert response.status_code == 422


# ---------------------------------------------------------------- Portfolio


def test_portfolio_summary():
    """Verify portfolio macro aggregation endpoint."""
    response = client.get("/api/v1/portfolio/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_projects"] >= 8
    assert data["total_monitored_capex_crores"] > 50000.0
    assert data["total_capital_at_risk_crores"] > 0.0
    assert len(data["sectors"]) > 0
    assert set(data["risk_tier_counts"]) == {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert set(data["dominant_source_counts"]) == {"MACHINE_LEARNING", "RULE_FLOOR_OVERRIDE"}


def test_portfolio_risk_ranking():
    """Verify portfolio leaderboard and filtering."""
    response = client.get("/api/v1/portfolio/risk-ranking?sector=Railways")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    for p in data:
        assert p["sector"] == "Railways"


def test_portfolio_risk_ranking_tier_filter():
    response = client.get("/api/v1/portfolio/risk-ranking?risk_tier=CRITICAL")
    assert response.status_code == 200
    for p in response.json():
        assert p["risk_tier"] == "CRITICAL"


# ---------------------------------------------------------------- Analytics


def test_cuf_gap_analysis_zero_confabulation():
    """Verify MoSPI Dimension (c) response under the zero-confabulation standard."""
    response = client.get("/api/v1/analytics/cuf-gap")
    assert response.status_code == 200
    data = response.json()
    # Methodology statement must explicitly refuse fabricated variance splits
    assert "impossible" in data["methodology_statement"]
    # Observable ceiling block
    ceiling = data["observable_ceiling"]
    assert 0.75 <= ceiling["discriminative_auc_ceiling"] <= 0.90
    assert ceiling["residual_classification_error"] == pytest.approx(
        1.0 - ceiling["discriminative_auc_ceiling"], abs=1e-6
    )
    # NLP proxy augmentation block (Spec Section 2.9: 0.756 -> 0.814)
    aug = data["proxy_augmentation"]
    assert len(aug["proxy_names"]) == 5
    assert aug["auc_after"] > aug["auc_before"]
    assert aug["auc_delta"] == pytest.approx(aug["auc_after"] - aug["auc_before"], abs=1e-9)
    assert aug["rmse_after_pct"] < aug["rmse_before_pct"]
    # CUF 2.0 proposals with primary-source grounding
    assert len(data["missing_variables_recommended"]) >= 4
    for mv in data["missing_variables_recommended"]:
        assert len(mv["primary_source_grounding"]) > 20
    assert len(data["policy_action_items"]) >= 3


def test_project_drivers():
    """Verify TreeSHAP driver attribution with NLP proxies."""
    payload = {
        "project_id": "TEST-DRV-01",
        "project_name": "Test Driver Project",
        "sector": "Power",
        "implementing_agency": "NTPC",
        "original_cost": 5000.0,
        "expenditure": 3500.0,
        "physical_progress": 35.0,
        "progress_change_recent": 0.1,
        "delay_remarks": "Land acquisition and forest clearance pending; arbitration ongoing",
    }
    response = client.post("/api/v1/analytics/drivers", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["project_id"] == "TEST-DRV-01"
    assert len(data["primary_drivers"]) >= 1
    assert 0.0 <= data["p_model"] <= 1.0
    assert 0.0 <= data["base_rate_probability"] <= 1.0
    assert data["dominant_source"] in ("MACHINE_LEARNING", "RULE_FLOOR_OVERRIDE")
    # NLP proxies mined from the remarks
    proxies = data["nlp_bottleneck_proxies"]
    assert proxies["LAND_ROW"] == 1
    assert proxies["ENV_FOREST"] == 1
    assert proxies["LEGAL_CONTRACTOR"] == 1


def test_statistical_benchmark_delong():
    """Verify MoSPI Dimension (b) benchmark: DeLong protocol, no Diebold-Mariano."""
    response = client.get("/api/v1/analytics/benchmark-baseline")
    assert response.status_code == 200
    data = response.json()
    assert len(data["benchmark_results"]) == 4
    assert "DeLong" in data["statistical_test"]
    assert "Diebold-Mariano" in data["test_methodology_note"]  # cited as rejected
    rows = {r["model_architecture"]: r for r in data["benchmark_results"]}
    cox = rows["Cox Proportional Hazards"]
    xgb = rows["Stage-Aware XGBoost (Proposed)"]
    assert cox["delong_z_vs_coxph"] == 0.0  # baseline row
    assert xgb["test_auc"] > cox["test_auc"]
    assert xgb["delong_z_vs_coxph"] > 2.0
    assert xgb["delong_p_value"] < 0.001


def test_delong_endpoint_contract():
    """The /analytics/delong-test endpoint must implement the exact Section 2.5 math."""
    import numpy as np

    rng = np.random.default_rng(19)
    y = [1] * 30 + [0] * 50
    # Strong but imperfect separation with within-group spread (non-degenerate variance)
    scores_1 = (
        list(rng.uniform(0.65, 1.0, size=30))
        + list(rng.uniform(0.0, 0.45, size=50))
    )
    scores_2 = list(rng.uniform(0.0, 1.0, size=80))  # uninformative rival model
    response = client.post(
        "/api/v1/analytics/delong-test",
        json={"y_true": y, "scores_1": scores_1, "scores_2": scores_2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["auc_1"] > 0.85
    assert 0.4 < data["auc_2"] < 0.6
    assert data["z_statistic"] > 2.0
    assert data["p_value"] < 0.05
    assert data["n_positives"] == 30
    assert data["n_negatives"] == 50


def test_delong_endpoint_rejects_single_class():
    response = client.post(
        "/api/v1/analytics/delong-test",
        json={"y_true": [1, 1, 1], "scores_1": [0.5, 0.6, 0.7], "scores_2": [0.5, 0.6, 0.7]},
    )
    assert response.status_code == 422


def test_delong_endpoint_rejects_missing_keys():
    response = client.post("/api/v1/analytics/delong-test", json={"y_true": [1, 0]})
    assert response.status_code == 422


# ---------------------------------------------------------------------- CSV


def _csv_file(content: str, name: str = "report.csv"):
    return {"file": (name, content.encode("utf-8"), "text/csv")}


CSV_ROWS = (
    "Project ID,Project Name,Sector,Sanctioned Cost,Revised Cost,Expenditure,Physical Progress,Progress Change,Remarks\n"
    "CSV-001,Test Corridor A,Railways,1000,1500,900,30.0,0.2,land dispute pending\n"
    "CSV-002,Test Bypass B,Road Transport & Highways,500,500,100,80.0,3.5,normal progress\n"
)


def test_predict_batch_csv():
    """Verify CSV ingestion: synonym headers, GovScore ranking, ignored-column report."""
    response = client.post("/api/v1/predict/batch-csv", files=_csv_file(CSV_ROWS))
    assert response.status_code == 200
    data = response.json()
    assert data["total_projects"] == 2
    assert data["total_monitored_capex_crores"] == 1500.0
    assert "columns_ignored" in data
    assert data["columns_ignored"] == []
    assert "Remarks" not in data["columns_ignored"]  # now mapped to delay_remarks
    # Ranked by CaR: the escalated corridor must lead
    assert data["ranked_projects"][0]["project_id"] == "CSV-001"
    # Governance assessment fields flow through CSV pipeline
    top = data["ranked_projects"][0]
    assert 0.0 <= top["gov_score"] <= 100.0
    assert top["sector_median_overrun_pct"] == 25.0


def test_predict_batch_csv_governance_columns():
    """Governance trigger columns (dispute, staleness) must flow into RuleFloor."""
    csv = (
        "Project ID,Project Name,Sector,Original Cost,Expenditure,Physical Progress,"
        "Progress Change,Days Since Last Update,Dispute Status,Remarks\n"
        "CSV-G01,Governance Test,Railways,2000,2600,42.0,0.0,70,ARBITRATION,land acquisition stalled\n"
    )
    response = client.post("/api/v1/predict/batch-csv", files=_csv_file(csv))
    assert response.status_code == 200
    top = response.json()["ranked_projects"][0]
    active = {s["flag_id"] for s in top["rule_signals"] if s["is_active"]}
    assert "F4" in active
    assert "F5" in active
    assert top["dominant_source"] == "RULE_FLOOR_OVERRIDE"


def test_predict_batch_csv_row_missing_required_fields():
    """Rows lacking required fields must fail with a row-level message."""
    bad = (
        "Project ID,Project Name\n"
        "BAD-001,No Costs Given\n"
    )
    response = client.post("/api/v1/predict/batch-csv", files=_csv_file(bad))
    assert response.status_code == 422
    assert "row 1" in response.json()["detail"]


def test_predict_batch_csv_unparseable():
    """Binary junk uploads must return 422, not 500."""
    response = client.post(
        "/api/v1/predict/batch-csv",
        files={"file": ("report.csv", b"\x00\x01\x02binaryjunk", "text/csv")},
    )
    assert response.status_code == 422


# ------------------------------------------------- Per-project SHAP drivers


def test_analytics_drivers_returns_instance_attribution():
    """Driver analysis must return per-project TreeSHAP attribution."""
    payload = {
        "project_id": "TEST-DRV-02",
        "project_name": "SHAP Verification Project",
        "sector": "Power",
        "implementing_agency": "NTPC",
        "original_cost": 5000.0,
        "revised_cost": 8000.0,
        "expenditure": 3500.0,
        "physical_progress": 35.0,
        "progress_change_recent": 0.1,
        "current_delay_months": 30.0,
    }
    response = client.post("/api/v1/analytics/drivers", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["primary_drivers"]) >= 1
    for d in data["primary_drivers"]:
        assert d["direction"] in ("ACCELERATOR", "MITIGATOR", "NEUTRAL")
        assert len(d["operational_meaning"]) > 10


def test_shap_contributions_differ_per_project():
    """Attribution must be instance-level, not static heuristics."""
    healthy = {
        "project_id": "H1-OK", "project_name": "Healthy", "sector": "Roads",
        "implementing_agency": "NHAI",
        "original_cost": 1000.0, "expenditure": 100.0,
        "physical_progress": 10.0, "progress_change_recent": 3.0,
        "current_delay_months": 0.0,
    }
    distressed = dict(healthy, project_id="D1-BAD", project_name="Distressed",
                      revised_cost=3000.0, expenditure=2500.0,
                      physical_progress=15.0, progress_change_recent=0.0,
                      current_delay_months=48.0)
    r1 = client.post("/api/v1/analytics/drivers", json=healthy).json()
    r2 = client.post("/api/v1/analytics/drivers", json=distressed).json()
    names1 = [d["factor_name"] for d in r1["primary_drivers"]]
    names2 = [d["factor_name"] for d in r2["primary_drivers"]]
    assert names1 != names2, "two very different projects must have different driver profiles"


def test_quarantined_feature_never_drives_inference():
    """has_revised_doc in the payload must NOT change P_model (leak-free preprocessing)."""
    base = {
        "project_id": "TEST-LEAK-1",
        "project_name": "Leakage Verification Project",
        "sector": "Railways",
        "implementing_agency": "RVNL",
        "original_cost": 3000.0,
        "revised_cost": 4500.0,
        "expenditure": 2000.0,
        "physical_progress": 40.0,
        "progress_change_recent": 1.0,
    }
    r1 = client.post("/api/v1/predict/project", json=base).json()
    # Old API accepted has_revised_doc; new contract ignores it entirely
    r2 = client.post("/api/v1/predict/project", json={**base, "has_revised_doc": 1}).json()
    assert r1["p_model"] == r2["p_model"], (
        "quarantined outcome proxy must not influence P_model"
    )


def test_quarantined_current_delay_months_never_drives_inference():
    """
    Spec Sections 1.2 / 4.1: `current_delay_months` is the accumulated
    schedule slippage beyond the original statutory completion date -- a
    post-facto bureaucratic outcome record exactly like `has_revised_doc`.
    It is quarantined (`LEAKAGE_QUARANTINED_FEATURES` ->
    `current_delay_months_robust`), so varying it in the payload must leave
    P_model bit-for-bit invariant (Spec Section 4.1; README Section 4.1).
    """
    base = {
        "project_id": "TEST-LEAK-2",
        "project_name": "Delay Quarantine Verification Project",
        "sector": "Railways",
        "implementing_agency": "RVNL",
        "original_cost": 3000.0,
        "revised_cost": 4500.0,
        "expenditure": 2000.0,
        "physical_progress": 40.0,
        "progress_change_recent": 1.0,
    }
    r1 = client.post("/api/v1/predict/project", json=base).json()
    r2 = client.post("/api/v1/predict/project", json={**base, "current_delay_months": 48.0}).json()
    r3 = client.post("/api/v1/predict/project", json={**base, "current_delay_months": 0.0}).json()
    assert r1["p_model"] == r2["p_model"] == r3["p_model"], (
        "quarantined current_delay_months proxy must not influence P_model"
    )


def test_leak_free_frame_hard_zeroes_all_quarantined_columns():
    """
    Frame-level proof of the Section 1.2 quarantine: the leak-free design
    matrix produced for the legacy booster must carry 0.0 in EVERY column
    listed in LEAKAGE_QUARANTINED_FEATURES, regardless of what the CUF
    payload declared. This closes the audit gap where
    `current_delay_months_robust` silently forwarded the raw payload value
    into the booster (contradicting the quarantine narrative in the
    `has_revised_doc` SHAP translation catalog).
    """
    from app.services.model_service import ModelService

    ms = ModelService.get_instance()
    assert ms.is_loaded, "legacy model bundle must be loadable for this test"

    project = ProjectInput(
        project_id="TEST-LEAK-3",
        project_name="Frame Quarantine Verification Project",
        sector="Railways",
        original_cost=3000.0,
        revised_cost=4500.0,
        expenditure=2000.0,
        physical_progress=40.0,
        months_elapsed=72.0,
        current_delay_months=48.0,
    )
    frame = ms.build_leak_free_frame(project)
    for col in sorted(LEAKAGE_QUARANTINED_FEATURES & set(frame.columns)):
        assert float(frame[col].iloc[0]) == 0.0, (
            f"quarantined column '{col}' must be hard-zeroed in the leak-free frame"
        )

    # End-to-end invariance at the service layer: a maximally distressed
    # schedule (48 months of accumulated delay) and a pristine schedule
    # (zero delay) must produce identical P_model.
    pristine = project.model_copy(update={"current_delay_months": 0.0})
    p_distressed, _, _ = ms.explain(project)
    p_pristine, _, _ = ms.explain(pristine)
    assert p_distressed == p_pristine, (
        "P_model must be invariant to current_delay_months under the quarantine"
    )
