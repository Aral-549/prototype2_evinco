"""Project prediction and inference endpoints (Spec Section 3.3 Day-1 scope)."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from typing import Dict, List

from app.paimana_contracts import (
    BatchCsvResponse,
    BatchProjectRequest,
    BatchProjectResponse,
    ProjectGovernanceAssessment,
    ProjectInput,
)
from app.services.csv_ingest import CsvIngestError, csv_to_project_inputs
from app.services.governance_service import ModelNotLoadedError, evaluate_project

router = APIRouter()


@router.post(
    "/predict/project",
    response_model=ProjectGovernanceAssessment,
    summary="Predict Governance Risk for a Single Project",
)
def predict_project(project: ProjectInput) -> ProjectGovernanceAssessment:
    """
    Evaluates an infrastructure project's Common Upload Form (CUF) parameters:
    - Runs leak-free Stage-Aware XGBoost inference (quarantined outcome proxies)
    - Evaluates the deterministic statutory RuleFloor (Flags F1..F5)
    - Applies the Master GovScore override: max(100 * P_model, RuleFloor)
    - Quantifies Capital-at-Risk with the empirical sector-median overrun prior
    - Attaches native TreeSHAP top-5 driver waterfall with administrative translations
    - Formulates prescriptive administrative recommendations
    """
    try:
        return evaluate_project(project)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}") from exc


def _aggregate(results: List[ProjectGovernanceAssessment]) -> Dict[str, int]:
    tiers: Dict[str, int] = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    sources: Dict[str, int] = {"MACHINE_LEARNING": 0, "RULE_FLOOR_OVERRIDE": 0}
    for r in results:
        tiers[r.risk_tier] += 1
        sources[r.dominant_source] += 1
    tiers.update(sources)
    return tiers


@router.post(
    "/predict/batch",
    response_model=BatchProjectResponse,
    summary="Batch Portfolio Governance Inference",
)
def predict_batch(payload: BatchProjectRequest) -> BatchProjectResponse:
    """
    Evaluates a list of infrastructure projects and returns a portfolio-level
    assessment rank-ordered by Capital-at-Risk (CaR), with risk-tier and
    dominant-source distributions.
    """
    results: List[ProjectGovernanceAssessment] = []
    try:
        for p in payload.projects:
            results.append(evaluate_project(p, include_drivers=False))
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}") from exc

    results.sort(key=lambda r: r.capital_at_risk_crores, reverse=True)
    counts = _aggregate(results)

    return BatchProjectResponse(
        total_projects=len(results),
        total_monitored_capex_crores=round(sum(r.original_cost_crores for r in results), 2),
        total_capital_at_risk_crores=round(sum(r.capital_at_risk_crores for r in results), 2),
        risk_tier_counts={k: v for k, v in counts.items() if k in {"LOW", "MODERATE", "HIGH", "CRITICAL"}},
        dominant_source_counts={k: v for k, v in counts.items() if k in {"MACHINE_LEARNING", "RULE_FLOOR_OVERRIDE"}},
        ranked_projects=results,
    )


def _evaluate_project(p: ProjectInput) -> ProjectGovernanceAssessment:
    """Shared single-project inference + GovScore pipeline (no drivers for CSV scale)."""
    return evaluate_project(p, include_drivers=False)


@router.post(
    "/predict/batch-csv",
    response_model=BatchCsvResponse,
    summary="Batch Inference from PAIMANA Report CSV",
)
async def predict_batch_csv(file: UploadFile = File(...)) -> BatchCsvResponse:
    """
    Accepts a monthly PAIMANA report export (CSV) and returns the full
    portfolio evaluation sorted by Capital-at-Risk.

    Header matching is case/space-insensitive with a synonym table for
    common PAIMANA column names (e.g. 'Sanctioned Cost', 'OriginalCost',
    'original_cost' all map to original_cost). Unrecognized columns are
    ignored and reported back in `columns_ignored`. Rows missing required
    fields return a row-level validation error.
    """
    try:
        items, diagnostics = csv_to_project_inputs(await file.read())
    except CsvIngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not items:
        raise HTTPException(status_code=422, detail="CSV contains no data rows")

    results: List[ProjectGovernanceAssessment] = []
    try:
        for item in items:
            results.append(_evaluate_project(item))
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}") from exc

    results.sort(key=lambda r: r.capital_at_risk_crores, reverse=True)
    counts = _aggregate(results)

    return BatchCsvResponse(
        total_projects=len(results),
        total_monitored_capex_crores=round(sum(r.original_cost_crores for r in results), 2),
        total_capital_at_risk_crores=round(sum(r.capital_at_risk_crores for r in results), 2),
        risk_tier_counts={k: v for k, v in counts.items() if k in {"LOW", "MODERATE", "HIGH", "CRITICAL"}},
        dominant_source_counts={k: v for k, v in counts.items() if k in {"MACHINE_LEARNING", "RULE_FLOOR_OVERRIDE"}},
        ranked_projects=results,
        columns_ignored=diagnostics.get("columns_ignored", []),
    )
