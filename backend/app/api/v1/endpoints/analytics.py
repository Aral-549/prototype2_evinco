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
    CUFGapAnalysisResponse,
    ProjectDriversResponse,
)
from app.paimana_contracts import ProjectInput
from app.services.cuf_analytics import CUFAnalyticsService
from app.services.governance_service import ModelNotLoadedError
from app.services.model_service import (
    ModelService,
    load_label_validity,
    load_lead_time_report,
    load_ordering_sensitivity,
    load_training_metrics,
)

router = APIRouter()

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
    summary="MoSPI Dimension (b): ML vs Conventional Statistics (measured, DeLong)",
)
def get_statistical_benchmark() -> Dict:
    """
    Answers MoSPI Problem Statement Dimension (b) with **measured** numbers.

    Includes `actionable_cohort`, which answers the sharpest reviewer objection
    head-on: "months_to_revised_date scores ~0.77 AUC alone, so isn't this just
    a deadline rule?" The cohort table re-scores every model separately on
    projects whose declared completion date has NOT yet passed -- the only
    projects where a warning can still change an outcome -- and reports the
    margin over an explicitly-fitted deadline heuristic.

    Every figure returned here is read from `model/paimana_model_metrics.json`,
    which is produced by `scripts/train_model.py` on the reconstructed MoSPI
    panel. Nothing in this endpoint is a constant.

    An earlier revision of this file hardcoded a benchmark table (XGBoost AUC
    0.814 vs Cox 0.732, Z = 4.12) that had never been computed, and the README
    quoted a third, different set of numbers. Those constants are deleted. If
    the training pipeline has not been run, this endpoint returns HTTP 503 and
    says so rather than substituting remembered values.
    """
    metrics = load_training_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No measured metrics artifact found. Run: python scripts/build_panel.py "
                "&& python scripts/train_model.py. This endpoint reports only measured "
                "results and will not substitute illustrative values."
            ),
        )

    horizons = metrics.get("horizons", {})
    comparison = {
        key: {
            "horizon_months": res.get("horizon_months"),
            "label": metrics.get("labels", {}).get(res.get("label", ""), res.get("label")),
            "rows_labelled": res.get("rows_labelled"),
            "positive_rate": res.get("positive_rate"),
            "censoring_selection_caveat": res.get("censoring_selection_caveat"),
            "generalisation_to_unseen_projects": res.get("splits", {}).get("primary", {}),
            "forecasting_months_never_seen": res.get("splits", {}).get("out_of_time", {}),
            "next_window_for_monitored_projects": res.get("splits", {}).get("deployment", {}),
            "confound_guard": res.get("confound_guard"),
            "actionable_cohort": res.get("actionable_cohort"),
            "calibration": res.get("calibration"),
        }
        for key, res in horizons.items()
        if res.get("status") == "ok"
    }

    return {
        "dimension": "Dimension (b): ML vs Conventional Statistical Methods",
        "provenance": metrics.get("provenance"),
        "generated_at": metrics.get("generated_at"),
        "model_version": metrics.get("model_version"),
        "panel": metrics.get("panel"),
        "leakage_policy": metrics.get("leakage_policy"),
        "statistical_test": metrics.get("statistical_test"),
        "calendar_split_status": metrics.get("calendar_split_status"),
        "rejected_split": metrics.get("rejected_split"),
        "cox_continuous_time": metrics.get("cox_continuous_time"),
        "horizons": comparison,
        "reproduce": metrics.get("reproduce"),
    }


@router.get(
    "/analytics/model-metrics",
    summary="Full measured evaluation artifact (single source of truth)",
)
def get_model_metrics() -> Dict:
    """Returns `model/paimana_model_metrics.json` verbatim.

    Exposed so a reviewer can audit every claim the dashboard makes -- splits,
    baselines, DeLong tests, calibration, confound guard and feature gains --
    against the artifact the training run actually wrote.
    """
    metrics = load_training_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=503,
            detail="No metrics artifact. Run scripts/build_panel.py then scripts/train_model.py.",
        )
    return metrics


@router.get(
    "/analytics/lead-time",
    summary="Measured early-warning lead time vs. the ministry's filed revisions",
)
def get_lead_time() -> Dict:
    """
    How many months before the official revised date is filed does the model
    already flag the project?

    Measured on out-of-fold predictions, reported across a threshold sweep, and
    always paired with the false-alarm rate that makes a lead-time figure
    interpretable. The artifact states its own ceiling: the public feed exposes
    a limited monthly window, so the reported medians are lower bounds.
    """
    report = load_lead_time_report()
    if report is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No lead-time artifact. Run: python scripts/train_model.py "
                "&& python scripts/lead_time_backtest.py"
            ),
        )
    return report


@router.get(
    "/analytics/ordering-sensitivity",
    summary="Robustness of the result to the reconstructed time axis",
)
def get_ordering_sensitivity() -> Dict:
    """
    The PAIMANA feed returns `Month: null` / `Year: null` on every record, so
    the panel's snapshot order is reconstructed under assumption A1 rather than
    observed. That is the platform's deepest methodological soft spot.

    This endpoint reports what happens when the assumption is varied: the panel
    is rebuilt under several alternative ordering rules — plus a deliberately
    inverted **negative control** — and retrained under the identical protocol.
    A stable AUC across plausible orderings, together with a clear degradation
    under the inverted control, is the evidence that the panel is ordered by
    something real.
    """
    report = load_ordering_sensitivity()
    if report is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No ordering-sensitivity artifact. Run: python scripts/ordering_sensitivity.py"
            ),
        )
    return report


@router.get(
    "/analytics/label-validity",
    summary="Does a filed date revision correspond to physical distress?",
)
def get_label_validity() -> Dict:
    """
    The platform's target is "a date moved on a government form". This endpoint
    reports whether that bureaucratic event tracks anything physical, measured
    rather than argued.

    It also contains the platform's most actionable finding for MoSPI: the
    premise behind statutory rule F1 (GFR 2017 Rule 159) is **inverted** for
    schedule forecasting. Project-months where physical progress runs ahead of
    disbursement slip at roughly four times the rate of the months F1 actually
    flags. F1 remains a valid financial-irregularity flag; it is not a schedule
    predictor, and this is the evidence.
    """
    report = load_label_validity()
    if report is None:
        raise HTTPException(
            status_code=503,
            detail="No label-validity artifact. Run: python scripts/label_validity.py",
        )
    return report


@router.post(
    "/analytics/horizons",
    summary="Slip probability curve across 1 / 3 / 6-month horizons for one project",
)
def get_horizon_curve(project: ProjectInput) -> Dict:
    """
    A single risk score cannot tell a review committee whether a project slips
    next month or next year. This returns the calibrated slip probability at
    every trained horizon, which is what a PRAGATI agenda is actually built
    against.

    Features that cannot be derived from a one-month CUF snapshot are listed in
    `feature_approximations` so the caller knows which inputs were estimated.
    """
    model_service = ModelService.get_instance()
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Predictive model bundle is not loaded.")
    if model_service.bundle_generation != "v2-panel-trained":
        raise HTTPException(
            status_code=503,
            detail=(
                "Horizon forecasts require the v2 panel-trained bundle. "
                "Run scripts/build_panel.py then scripts/train_model.py."
            ),
        )

    _frame, approximations = model_service.build_frame(project)
    return {
        "project_id": project.project_id,
        "project_name": project.project_name,
        "sector": project.sector,
        "horizons": model_service.predict_horizons(project),
        "horizon_definitions": {
            "1m": "revised completion date moves at the next monthly report",
            "3m": "revised completion date moves at any report within 3 months",
            "6m": "revised completion date moves at any report within 6 months",
        },
        "feature_approximations": approximations,
        "model_version": model_service.model_version,
    }


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
