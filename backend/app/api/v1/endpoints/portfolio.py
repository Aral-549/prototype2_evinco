"""Portfolio-level monitoring and risk ranking endpoints."""

from collections import defaultdict
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.paimana_contracts import ProjectGovernanceAssessment
from app.schemas.analytics import PortfolioSummaryResponse, SectorRiskAggregation
from app.services.governance_service import ModelNotLoadedError, evaluate_project
from app.services.model_service import ModelService
from app.services.real_data import get_evaluated_portfolio, load_real_projects

router = APIRouter()


def _get_all_evaluated_projects() -> List[ProjectGovernanceAssessment]:
    """Helper to evaluate all registered infrastructure projects from real dataset."""
    try:
        return get_evaluated_portfolio()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/portfolio/summary",
    response_model=PortfolioSummaryResponse,
    summary="National Portfolio Risk Overview",
)
def get_portfolio_summary() -> PortfolioSummaryResponse:
    """
    Returns an aggregated macro overview of the monitored infrastructure portfolio:
    - Total monitored capital commitments
    - Aggregate Capital-at-Risk (CaR)
    - Sectoral distribution and Critical/High-risk concentrations
    """
    evaluated = _get_all_evaluated_projects()
    raw_projects = load_real_projects()

    total_capex = sum(p.original_cost_crores for p in evaluated)
    total_expenditure = 0.0
    # NOTE: expenditure is carried on the CUF input, not on the assessment;
    # recompute from the seed inputs for the expenditure aggregate.
    input_by_id = {p.project_id: p for p in raw_projects}
    for p in evaluated:
        src = input_by_id.get(p.project_id)
        if src is not None:
            total_expenditure += src.expenditure
    total_car = sum(p.capital_at_risk_crores for p in evaluated)

    tier_counts: dict[str, int] = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    source_counts: dict[str, int] = {"MACHINE_LEARNING": 0, "RULE_FLOOR_OVERRIDE": 0}
    sector_groups = defaultdict(list)

    for p in evaluated:
        tier_counts[p.risk_tier] = tier_counts.get(p.risk_tier, 0) + 1
        source_counts[p.dominant_source] = source_counts.get(p.dominant_source, 0) + 1
        sector_groups[p.sector].append(p)

    sectors: List[SectorRiskAggregation] = []
    for sector_name, proj_list in sector_groups.items():
        sec_capex = sum(x.original_cost_crores for x in proj_list)
        sec_exp = sum(input_by_id[x.project_id].expenditure for x in proj_list)
        sec_car = sum(x.capital_at_risk_crores for x in proj_list)
        avg_gov = sum(x.gov_score for x in proj_list) / len(proj_list)
        crit_high = sum(1 for x in proj_list if x.risk_tier in ("HIGH", "CRITICAL"))

        sectors.append(
            SectorRiskAggregation(
                sector=sector_name,
                project_count=len(proj_list),
                total_original_cost_crores=round(sec_capex, 2),
                total_expenditure_crores=round(sec_exp, 2),
                total_capital_at_risk_crores=round(sec_car, 2),
                avg_gov_score=round(avg_gov, 2),
                critical_high_projects=crit_high,
            )
        )

    # Sort sectors by highest Capital-at-Risk
    sectors.sort(key=lambda s: s.total_capital_at_risk_crores, reverse=True)

    car_percentage = (total_car / total_capex * 100.0) if total_capex > 0 else 0.0

    return PortfolioSummaryResponse(
        total_projects=len(evaluated),
        total_monitored_capex_crores=round(total_capex, 2),
        total_expenditure_crores=round(total_expenditure, 2),
        total_capital_at_risk_crores=round(total_car, 2),
        capital_at_risk_percentage=round(car_percentage, 2),
        risk_tier_counts=tier_counts,
        dominant_source_counts=source_counts,
        sectors=sectors,
    )


@router.get(
    "/portfolio/risk-ranking",
    response_model=List[ProjectGovernanceAssessment],
    summary="Portfolio Capital-at-Risk Leaderboard",
)
def get_risk_ranking(
    sector: Optional[str] = Query(default=None, description="Optional sector filter, e.g. 'Railways'"),
    risk_tier: Optional[str] = Query(
        default=None,
        description="Optional risk tier filter ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')",
    ),
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        le=5000,
        description="Optional maximum number of projects to return",
    ),
    slim: bool = Query(
        default=False,
        description=(
            "Omit per-project rule_signals, shap_drivers and prescriptive_interventions. "
            "Those make up ~82% of the payload and are not used by leaderboard views."
        ),
    ),
) -> List[ProjectGovernanceAssessment]:
    """
    Returns individual monitored infrastructure projects sorted by Capital-at-Risk (CaR).
    Directly serves the executive early-warning priority view for administrators.
    """
    evaluated = list(_get_all_evaluated_projects())

    if sector:
        evaluated = [p for p in evaluated if p.sector.lower() == sector.lower()]
    if risk_tier:
        evaluated = [p for p in evaluated if p.risk_tier.upper() == risk_tier.upper()]

    # Sort descending by CaR
    evaluated.sort(key=lambda p: p.capital_at_risk_crores, reverse=True)
    if limit is not None:
        evaluated = evaluated[:limit]

    if slim:
        # The full assessment carries six RuleActivationSignal objects per
        # project, each with a paragraph of statutory rationale -- 82% of the
        # payload (2,918 of 3,563 bytes) for fields a leaderboard never renders.
        # Returning the whole portfolio unslimmed is 7.7 MB, which is enough to
        # make the dashboard look broken on a slow connection.
        #
        # A JSONResponse is returned directly so FastAPI skips `response_model`
        # validation, which would otherwise re-populate the omitted fields.
        heavy = {"rule_signals", "shap_drivers", "prescriptive_interventions"}
        return JSONResponse(
            content=[p.model_dump(mode="json", exclude=heavy) for p in evaluated]
        )
    return evaluated


@router.get(
    "/portfolio/project/{project_id}",
    summary="Full assessment + horizon curve for one monitored project",
)
def get_project_detail(project_id: str) -> dict:
    """
    Everything a detail view needs for a monitored project, scored from the
    project's own stored CUF record.

    This endpoint exists because the alternative is worse. A client holding a
    slim leaderboard row does not have the schedule fields (`months_elapsed`,
    `original_duration_months`, `current_delay_months`), and those drive the
    model's strongest features. Rebuilding an input from the row and re-scoring
    it silently produced p_model ~= 0.01 for every project while the row it was
    opened from said 0.15-0.51 -- a detail view that disagreed with the list
    that launched it. Serving the real record removes that whole class of
    error rather than patching one symptom of it.
    """
    project = next(
        (p for p in load_real_projects() if p.project_id == project_id), None
    )
    if project is None:
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'.")

    try:
        assessment = evaluate_project(project, include_drivers=True)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    model_service = ModelService.get_instance()
    horizons: dict = {}
    approximations: dict = {}
    if model_service.bundle_generation == "v2-panel-trained":
        horizons = model_service.predict_horizons(project)
        _frame, approximations = model_service.build_frame(project)

    return {
        "assessment": assessment.model_dump(mode="json"),
        "horizons": horizons,
        "horizon_definitions": {
            "1m": "revised completion date moves at the next monthly report",
            "3m": "revised completion date moves at any report within 3 months",
            "6m": "revised completion date moves at any report within 6 months",
        },
        # The served portfolio holds ONE snapshot per project, so the three
        # history-dependent features are still approximated here exactly as
        # they are for an ad-hoc CUF submission. Reported rather than hidden.
        "feature_approximations": approximations,
    }
