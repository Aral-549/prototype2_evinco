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
    # v2 serves the 23-column panel design matrix. Leakage is handled by
    # EXCLUSION at training time, so the outcome proxies must be absent from
    # the served feature list entirely -- not present-but-zeroed as in v1.
    assert data["feature_count"] == 22
    assert "has_revised_doc" not in data["features"]
    assert "current_delay_months" not in data["features"]
    assert data["quarantined_features"] == []


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
    assert len(data["rule_signals"]) == 6
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
    assert 0.70 <= ceiling["discriminative_auc_ceiling"] <= 0.95
    assert ceiling["residual_classification_error"] == pytest.approx(
        1.0 - ceiling["discriminative_auc_ceiling"], abs=1e-6
    )
    # The ceiling must be MEASURED, not a declared constant.
    assert data["observable_ceiling_provenance"].startswith("MEASURED")

    # NLP proxy augmentation: the public PAIMANA feed publishes no delay
    # narratives (Remarks null on 14,917/14,917 records), so no augmentation
    # delta can exist. A previous revision reported a 0.756 -> 0.814 gain from
    # text that is not in the data. Every numeric field must now be the -1.0
    # "not measurable" sentinel, with the reason stated.
    aug = data["proxy_augmentation"]
    assert len(aug["proxy_names"]) == 5
    assert data["proxy_augmentation_provenance"].startswith("NOT MEASURABLE")
    for field in (
        "auc_before",
        "auc_after",
        "auc_delta",
        "rmse_before_pct",
        "rmse_after_pct",
        "delayed_projects_citing_top_three_factors_pct",
    ):
        assert aug[field] == -1.0, f"{field} must be the not-measurable sentinel"
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

    # Everything reported must come from the measured training artifact. The
    # previous revision of this endpoint returned a hardcoded table that had
    # never been computed; that is what this assertion exists to prevent.
    assert data["provenance"] == "measured"
    assert "DeLong" in data["statistical_test"]
    assert "Diebold-Mariano" in data["statistical_test"]  # cited as inapplicable

    primary = data["horizons"]["1m"]["generalisation_to_unseen_projects"]
    rows = {r["model_architecture"]: r for r in primary["results"] if "auc" in r}
    xgb = rows["Stage-Aware XGBoost (Proposed)"]
    logit = rows["Logistic Regression (ElasticNet)"]
    dtph = rows["Discrete-Time Proportional Hazards (cloglog)"]

    # The proposed model must beat both classical baselines, and the margin
    # must be certified by a real paired DeLong test on the same cohort.
    assert xgb["auc"] > logit["auc"]
    assert xgb["auc"] > dtph["auc"]
    tests = {t["comparison"]: t for t in primary["delong_tests"] if "z_statistic" in t}
    for key in ("xgboost vs logit", "xgboost vs dtph"):
        assert tests[key]["z_statistic"] > 2.0
        assert tests[key]["p_value"] < 0.001
        assert tests[key]["significant_at_0.05"] is True


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


def test_schedule_state_is_permitted_but_outcome_proxies_are_not():
    """
    v2 draws the leakage line in a different place than v1, deliberately.

    v1 quarantined `current_delay_months` and asserted P_model must be
    bit-identical as it varied. That was the wrong line. `current_delay_months`
    is the slippage ALREADY on the record at time t -- the gap between the
    original completion date and the date the ministry is currently declaring.
    A desk officer reading the file on day t can see it. It is state, not
    outcome, and suppressing it threw away the strongest legitimate signal in
    the data.

    What must never be used is the OUTCOME: whether the date moves *again* at
    the next report. That is the label. `has_revised_doc` and
    `current_delay_months_robust` encode it and are excluded from the design
    matrix entirely (see contracts/model_training.md).

    So this test asserts the opposite of its v1 predecessor: schedule state
    MUST move the prediction, and it must move it in the correct direction.
    """
    from app.services.model_service import ModelService

    ms = ModelService.get_instance()
    for banned in ("has_revised_doc", "current_delay_months", "current_delay_months_robust"):
        assert banned not in ms.feature_columns

    base = {
        "project_id": "TEST-LEAK-2",
        "project_name": "Schedule State Verification Project",
        "sector": "Railways",
        "implementing_agency": "RVNL",
        "original_cost": 3000.0,
        "revised_cost": 4500.0,
        "expenditure": 2000.0,
        "physical_progress": 40.0,
        "progress_change_recent": 1.0,
        "original_duration_months": 60.0,
    }

    def p_of(**over):
        r = client.post("/api/v1/predict/project", json={**base, **over})
        assert r.status_code == 200
        return r.json()["p_model"]

    badly_overdue = p_of(months_elapsed=84.0, current_delay_months=0.0)
    mildly_overdue = p_of(months_elapsed=66.0, current_delay_months=0.0)
    on_schedule = p_of(months_elapsed=40.0, current_delay_months=0.0)
    date_extended = p_of(months_elapsed=66.0, current_delay_months=24.0)

    # Further past the declared completion date => higher slip risk.
    assert badly_overdue > mildly_overdue > on_schedule

    # Same elapsed time, but the declared date was already pushed out so the
    # project is no longer past it: risk must fall, not rise.
    assert date_extended < mildly_overdue


def test_v2_excludes_leakage_proxies_from_design_matrix_entirely():
    """
    v2's leakage guarantee is stronger than v1's and is asserted differently.

    v1 trained ON `has_revised_doc` (34% of booster gain) and then zeroed it at
    inference. That is train/serve skew, not a leakage fix: it pushed every
    live project into leaf regions whose training base rate was ~0, and
    p_model collapsed below 0.035 across all 2,155 real projects.

    v2 excludes those fields from the design matrix at TRAINING time, so there
    is nothing to zero. This test asserts the columns are absent rather than
    zeroed, and that the served probabilities actually spread.
    """
    from app.services.model_service import ModelService

    ms = ModelService.get_instance()
    assert ms.is_loaded
    assert ms.bundle_generation == "v2-panel-trained", (
        "run scripts/build_panel.py && scripts/train_model.py to build the v2 bundle"
    )

    for banned in ("has_revised_doc", "current_delay_months", "current_delay_months_robust"):
        assert banned not in ms.feature_columns

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
    frame, _approximations = ms.build_frame(project)
    assert list(frame.columns) == ms.feature_columns
    assert not frame.isna().any().any()
    assert LEAKAGE_QUARANTINED_FEATURES.isdisjoint(set(frame.columns))


def test_v2_probabilities_are_not_degenerate():
    """Regression guard for the v1 collapse: p_model must actually spread.

    The v1 bundle returned p_model in [0.0001, 0.0345] for every one of the
    2,155 real projects, which made Capital-at-Risk meaningless and left the
    ML branch unable to ever dominate the RuleFloor. If a future change
    reintroduces train/serve skew, this test fails.
    """
    from app.services.real_data import get_evaluated_portfolio

    scores = [e.p_model for e in get_evaluated_portfolio()]
    assert len(scores) > 100
    assert max(scores) > 0.5, "no project scores above 0.5 -- model is collapsed"
    assert max(scores) - min(scores) > 0.5, "p_model range is degenerate"
    tiers = {e.risk_tier for e in get_evaluated_portfolio()}
    assert "CRITICAL" in tiers or "HIGH" in tiers


def test_risk_ranking_slim_mode_drops_only_unused_payload():
    """The leaderboard must not have to download 7.7 MB to render a table.

    `rule_signals` alone is ~82% of each assessment (2,918 of 3,563 bytes) --
    six statutory signals each carrying a paragraph of rationale, none of which
    a leaderboard row displays. Serving the whole portfolio unslimmed took 7.9s
    and 7.7 MB, which is enough to make the dashboard look broken on venue wifi.
    """
    full = client.get("/api/v1/portfolio/risk-ranking?limit=50")
    slim = client.get("/api/v1/portfolio/risk-ranking?limit=50&slim=true")
    assert full.status_code == 200 and slim.status_code == 200

    assert len(slim.json()) == len(full.json())
    assert len(slim.content) < len(full.content) / 3, "slim must be a large saving"

    row = slim.json()[0]
    for dropped in ("rule_signals", "shap_drivers", "prescriptive_interventions"):
        assert dropped not in row
    # Everything the leaderboard actually renders must survive.
    for kept in (
        "project_id", "project_name", "sector", "implementing_agency",
        "original_cost_crores", "physical_progress", "gov_score",
        "risk_tier", "dominant_source", "capital_at_risk_crores",
    ):
        assert kept in row, f"slim mode dropped a field the table renders: {kept}"

    # Ordering must be identical -- slim is a projection, not a different query.
    assert [r["project_id"] for r in slim.json()] == [r["project_id"] for r in full.json()]


def test_assessment_carries_physical_progress():
    """Regression: the dashboard's Progress column rendered "-" for every row.

    `ProjectGovernanceAssessment` never carried `physical_progress`, so the
    leaderboard's guard (`p.physical_progress != null ? ... : '-'`) silently
    took the null branch for all 2,155 projects. The column was dead.
    """
    payload = {
        "project_id": "TEST-PROG-1",
        "project_name": "Progress Field Regression",
        "sector": "Railways",
        "implementing_agency": "RVNL",
        "original_cost": 1000.0,
        "expenditure": 400.0,
        "physical_progress": 37.5,
    }
    body = client.post("/api/v1/predict/project", json=payload).json()
    assert body["physical_progress"] == pytest.approx(37.5)

    ranked = client.get("/api/v1/portfolio/risk-ranking?limit=25").json()
    assert all("physical_progress" in r for r in ranked)
    # Not every project is at 0% -- i.e. the field is really populated, not defaulted.
    assert any(r["physical_progress"] > 0 for r in ranked)


def test_dashboard_styling_is_served_locally_not_from_a_cdn():
    """The demo must not depend on venue wifi.

    The dashboard takes its ENTIRE layout from Tailwind utility classes. When
    those came from cdn.tailwindcss.com, a blocked or absent connection left an
    unstyled wall of text -- a total demo failure with no error message.
    Tailwind is now vendored under frontend/vendor/ and served from /static.
    """
    asset = client.get("/static/vendor/tailwind.play.js")
    assert asset.status_code == 200
    assert len(asset.content) > 100_000, "vendored Tailwind looks truncated"

    html = client.get("/dashboard").text
    assert "/static/vendor/tailwind.play.js" in html
    # The CDN may remain only as a fallback behind a window.tailwind check.
    cdn_at = html.find("cdn.tailwindcss.com")
    if cdn_at != -1:
        assert "window.tailwind" in html[max(0, cdn_at - 400):cdn_at], (
            "the CDN must only be reachable as a guarded fallback"
        )


def test_dashboard_kpi_captions_are_not_hardcoded_sample_text():
    """Regression: the KPI sub-captions contradicted the numbers above them.

    The headline tiles were populated live from /portfolio/summary (2,155
    projects, Rs 41.8 lakh crore), while the small print beneath them was
    static HTML reading "8 Sample Megaprojects", "3 of 8 Projects" and
    "1 Crit / 2 Mod / 5 Low". A reviewer reading the caption would conclude the
    platform ran on eight rows.
    """
    html = client.get("/dashboard").text
    for stale in ("8 Sample Megaprojects", "3 of 8 Projects", ">1 Crit<", ">2 Mod<", ">5 Low<"):
        assert stale not in html, f"stale hardcoded KPI caption still present: {stale}"
    # The captions must exist as live-populated elements instead.
    for element_id in ("kpi-project-count", "kpi-override-detail", "kpi-tier-breakdown"):
        assert f'id="{element_id}"' in html


def test_project_detail_matches_the_leaderboard_row_exactly():
    """A detail view must never contradict the list it was opened from.

    The front end originally rebuilt a ProjectInput from a slim leaderboard row
    and re-scored it. The slim payload has no schedule fields, so
    `months_elapsed`, `original_duration_months` and `current_delay_months`
    fell back to ProjectInput defaults — a generic early-stage project — and
    p_model collapsed to ~0.01 for EVERY project while the row it came from
    said 0.15-0.51. The forecast curve shown in that sheet was meaningless.

    This endpoint scores the project's own stored record, so the two views
    agree by construction rather than by luck.
    """
    rows = client.get("/api/v1/portfolio/risk-ranking?slim=true&limit=5").json()
    assert rows, "portfolio should not be empty"

    for row in rows:
        detail = client.get(f"/api/v1/portfolio/project/{row['project_id']}")
        assert detail.status_code == 200
        body = detail.json()
        a = body["assessment"]

        assert a["project_id"] == row["project_id"]
        assert a["p_model"] == pytest.approx(row["p_model"], abs=1e-9)
        assert a["gov_score"] == pytest.approx(row["gov_score"], abs=1e-9)
        assert a["rule_floor"] == pytest.approx(row["rule_floor"], abs=1e-9)
        assert a["capital_at_risk_crores"] == pytest.approx(
            row["capital_at_risk_crores"], abs=1e-9
        )

        # The detail view carries what the leaderboard deliberately omits.
        assert len(a["rule_signals"]) == 6
        assert a["shap_drivers"]

        # The 1-month horizon is the same quantity as p_model, so a drift
        # between them would mean the sheet's headline and its forecast
        # disagree with each other.
        if body["horizons"]:
            assert body["horizons"]["1m"]["probability"] == pytest.approx(
                row["p_model"], abs=5e-3
            )


def test_project_detail_404s_on_unknown_id():
    assert client.get("/api/v1/portfolio/project/NOT_A_PROJECT").status_code == 404
