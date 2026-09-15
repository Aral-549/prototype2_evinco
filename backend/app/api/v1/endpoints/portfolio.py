"""Portfolio-level monitoring and risk ranking endpoints."""

from collections import defaultdict
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.paimana_contracts import ProjectGovernanceAssessment
from app.schemas.analytics import PortfolioSummaryResponse, SectorRiskAggregation
from app.services.governance_service import ModelNotLoadedError, evaluate_project
from app.services.mock_data import MOCK_PAIMANA_PROJECTS

router = APIRouter()


def _evaluate_all_mock_projects() -> List[ProjectGovernanceAssessment]:
    """Helper to evaluate all registered infrastructure projects."""
    try:
        return [evaluate_project(p, include_drivers=False) for p in MOCK_PAIMANA_PROJECTS]
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
    evaluated = _evaluate_all_mock_projects()

    total_capex = sum(p.original_cost_crores for p in evaluated)
    total_expenditure = 0.0
    # NOTE: expenditure is carried on the CUF input, not on the assessment;
    # recompute from the seed inputs for the expenditure aggregate.
    input_by_id = {p.project_id: p for p in MOCK_PAIMANA_PROJECTS}
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
) -> List[ProjectGovernanceAssessment]:
    """
    Returns individual monitored infrastructure projects sorted by Capital-at-Risk (CaR).
    Directly serves the executive early-warning priority view for administrators.
    """
    evaluated = _evaluate_all_mock_projects()

    if sector:
        evaluated = [p for p in evaluated if p.sector.lower() == sector.lower()]
    if risk_tier:
        evaluated = [p for p in evaluated if p.risk_tier.upper() == risk_tier.upper()]

    # Sort descending by CaR
    evaluated.sort(key=lambda p: p.capital_at_risk_crores, reverse=True)
    return evaluated
