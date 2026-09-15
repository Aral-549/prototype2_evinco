"""Advanced governance analytics, CUF gap analysis, and statistical benchmark endpoints.

MoSPI Dimension (b): paired DeLong AUC comparison protocol (Spec Section 2.5).
MoSPI Dimension (c): zero-confabulation CUF information-ceiling reporting
(Spec Sections 2.8-2.10). This module deliberately does NOT reproduce the
legacy fabricated claims ("CUF explains 64.2%", "Diebold-Mariano p = 0.0018").
"""

from typing import Dict, List

from fastapi import APIRouter, HTTPException

from app.paimana_nlp import extract_bottleneck_proxies
from app.paimana_statistics import DeLongResult
from app.schemas.analytics import (
    BenchmarkModelRow,
    BenchmarkResponse,
    CUFGapAnalysisResponse,
    ProjectDriversResponse,
)
from app.paimana_contracts import ProjectInput
from app.services.cuf_analytics import CUFAnalyticsService
from app.services.governance_service import ModelNotLoadedError
from app.services.model_service import ModelService

router = APIRouter()

# ---------------------------------------------------------------------------
# Dimension (b) benchmark values (Spec Section 2.7, Table A).
# PROTOCOL REFERENCE VALUES, NOT A LIVE EVALUATION: these rows are the
# specification's registered benchmark constants. They are reported as-is
# with an explicit provenance statement (`benchmark_values_provenance` in
# BenchmarkResponse) distinguishing them from live inference, so the
# zero-confabulation standard (Spec Section 2.10) is never breached. Exact
# DeLong statistics for arbitrary paired score vectors are computed live by
# the POST /analytics/delong-test endpoint (Spec Section 2.5 protocol).
# ---------------------------------------------------------------------------
_BENCHMARK_ROWS: List[Dict] = [
    {
        "model_architecture": "Cox Proportional Hazards",
        "model_class": "Conventional Survival Baseline",
        "test_auc": 0.732,
        "delong_z_vs_coxph": 0.0,
        "delong_p_value": 1.0,
        "brier_score": 0.168,
        "log_loss": 0.512,
    },
    {
        "model_architecture": "Logistic Regression (ElasticNet)",
        "model_class": "Classical Econometric Baseline",
        "test_auc": 0.704,
        "delong_z_vs_coxph": -1.84,
        "delong_p_value": 0.0657,
        "brier_score": 0.179,
        "log_loss": 0.548,
    },
    {
        "model_architecture": "Random Survival Forest",
        "model_class": "Non-Linear Survival Ensemble",
        "test_auc": 0.781,
        "delong_z_vs_coxph": 2.89,
        "delong_p_value": 0.0039,
        "brier_score": 0.145,
        "log_loss": 0.441,
    },
    {
        "model_architecture": "Stage-Aware XGBoost (Proposed)",
        "model_class": "Gradient Boosted Trees (Day-1)",
        "test_auc": 0.814,
        "delong_z_vs_coxph": 4.12,
        "delong_p_value": 0.00001,
        "brier_score": 0.128,
        "log_loss": 0.395,
    },
]


@router.get(
    "/analytics/cuf-gap",
    response_model=CUFGapAnalysisResponse,
    summary="MoSPI Dimension (c): CUF Information Ceiling (Zero-Confabulation)",
)
def get_cuf_gap_analysis() -> CUFGapAnalysisResponse:
    """
    Directly answers MoSPI Problem Statement Dimension (c) under the
    zero-confabulation standard (Spec Sections 2.8-2.10, 4.5):
    - Reports the empirical asymptotic performance ceiling of the observable CUF feature set
    - Reports the empirical NLP-proxy augmentation deltas mined from 'Reasons for Delay'
    - Explicitly refuses to fabricate an exact variance decomposition of unrecorded variables
    - Delivers the actionable CUF 2.0 schema proposal with primary-source grounding
    """
    return CUFAnalyticsService.get_gap_analysis()


@router.post(
    "/analytics/drivers",
    response_model=ProjectDriversResponse,
    summary="TreeSHAP Driver Analysis for a Specific Project",
)
def analyze_project_drivers(project: ProjectInput) -> ProjectDriversResponse:
    """
    Identifies the top-5 TreeSHAP risk accelerators and mitigators for a
    specific project via native C++ TreeSHAP (`pred_contribs=True`),
    complemented by NLP bottleneck proxies mined from the delay remarks.
    Attribution is instance-level: it explains why THIS project scores as it does.
    """
    model_service = ModelService.get_instance()
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Predictive model bundle is not loaded.")

    try:
        p_model, base_rate, drivers = model_service.explain(project, top_k=5)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Attribution error: {str(exc)}") from exc

    # GovScore lattice with a neutral RuleFloor pass for display purposes
    from app.paimana_engine import compute_gov_score
    from app.paimana_rules import calculate_rule_floor

    rule_floor_eval = calculate_rule_floor(project)
    gov_score, dominant_source = compute_gov_score(p_model, rule_floor_eval.rule_floor)

    factors = [
        {
            "factor_name": d.feature_name,
            "direction": "ACCELERATOR" if d.direction == "RISK_INCREASING" else (
                "MITIGATOR" if d.direction == "RISK_DECREASING" else "NEUTRAL"
            ),
            "impact_score": abs(d.shap_value),
            "operational_meaning": d.administrative_interpretation,
        }
        for d in drivers
    ]

    return ProjectDriversResponse(
        project_id=project.project_id,
        project_name=project.project_name,
        p_model=round(p_model, 4),
        base_rate_probability=base_rate,
        gov_score=round(gov_score, 2),
        dominant_source=dominant_source,
        primary_drivers=factors,
        nlp_bottleneck_proxies=extract_bottleneck_proxies(project.delay_remarks),
    )


@router.get(
    "/analytics/benchmark-baseline",
    response_model=BenchmarkResponse,
    summary="MoSPI Dimension (b): ML vs. Conventional Statistics (DeLong Protocol)",
)
def get_statistical_benchmark() -> BenchmarkResponse:
    """
    Directly answers MoSPI Problem Statement Dimension (b) using DeLong's
    paired AUC test (DeLong et al., 1988) between the Stage-Aware XGBoost
    classifier and the 6-month horizon-binarized Cox Proportional Hazards
    baseline. Diebold-Mariano is methodologically invalid for this panel
    (Spec Section 2.6) and is deliberately not used.
    """
    return BenchmarkResponse(
        dimension="Dimension (b): ML vs Conventional Statistical Methods",
        evaluation_protocol="GroupKFold (grouped by project_id) + Out-of-Time cohort (train <= 2022; test 2023-2024)",
        statistical_test="DeLong Paired AUC Test (DeLong, DeLong & Clarke-Pearson, Biometrics 1988)",
        test_methodology_note=(
            "AUC is a combinatorial rank-concordance metric over all case-control pairs and cannot be "
            "decomposed into an additive point-wise loss differential d_t; the panel has no single 1D "
            "time axis. Diebold-Mariano is therefore mathematically undefined here, while DeLong's "
            "generalized U-statistic test requires zero distributional assumptions and operates on "
            "paired placement values V10/V01 from the same evaluation cohort (Spec Section 2.6)."
        ),
        benchmark_results=_BENCHMARK_ROWS,
        benchmark_values_provenance=BenchmarkResponse.model_fields["benchmark_values_provenance"].default,
    )


@router.post(
    "/analytics/delong-test",
    response_model=Dict,
    summary="Run DeLong's Paired AUC Test on Supplied Score Vectors",
)
def run_delong_test(payload: Dict) -> Dict:
    """
    Executes the exact Section 2.5 DeLong protocol on two paired score vectors
    evaluated over the same cohort. Expected JSON body:
        {"y_true": [0,1,...], "scores_1": [...], "scores_2": [...]}
    Returns AUCs, asymptotic variance-covariance components, Z statistic,
    and the two-sided p-value.
    """
    try:
        y_true = payload["y_true"]
        scores_1 = payload["scores_1"]
        scores_2 = payload["scores_2"]
    except KeyError as exc:
        raise HTTPException(
            status_code=422, detail=f"Missing required key: {exc.args[0]}"
        ) from exc

    import numpy as np

    try:
        result: DeLongResult = delong_test_paired_auc(
            np.asarray(y_true), np.asarray(scores_1, dtype=float), np.asarray(scores_2, dtype=float)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "test_name": result.test_name,
        "auc_1": result.auc_1,
        "auc_2": result.auc_2,
        "variance_auc_1": result.var_1,
        "variance_auc_2": result.var_2,
        "covariance_12": result.cov_12,
        "z_statistic": result.z_statistic,
        "p_value": result.p_value,
        "n_positives": result.n_positives,
        "n_negatives": result.n_negatives,
    }


def delong_test_paired_auc(y_true, scores_1, scores_2) -> DeLongResult:
    """Local import indirection to keep module import time low."""
    from app.paimana_statistics import delong_test_paired_auc as _impl

    return _impl(y_true, scores_1, scores_2)
