"""Schema package export (canonical Day-1 contracts + analytics)."""

from app.paimana_contracts import (
    BatchCsvResponse,
    BatchProjectRequest,
    BatchProjectResponse,
    ProjectGovernanceAssessment,
    ProjectInput,
    RuleActivationSignal,
    RuleFloorEvaluation,
    SHAPDriver,
)
from app.schemas.project import EarlyWarningSignal, ProjectPredictionResponse, ProjectRiskAnalysis
from app.schemas.analytics import (
    BenchmarkModelRow,
    BenchmarkResponse,
    CUFMissingVariable,
    CUFProxyAugmentation,
    CUFObservableCeiling,
    CUFGapAnalysisResponse,
    DriverFactor,
    ProjectDriversResponse,
    HealthResponse,
    PortfolioSummaryResponse,
    SectorRiskAggregation,
)

__all__ = [
    # Day-1 contracts (Spec Section 3.1)
    "BatchCsvResponse",
    "BatchProjectRequest",
    "BatchProjectResponse",
    "ProjectGovernanceAssessment",
    "ProjectInput",
    "RuleActivationSignal",
    "RuleFloorEvaluation",
    "SHAPDriver",
    # Legacy project schemas retained for compatibility
    "EarlyWarningSignal",
    "ProjectRiskAnalysis",
    "ProjectPredictionResponse",
    # Analytics schemas
    "BenchmarkModelRow",
    "BenchmarkResponse",
    "CUFMissingVariable",
    "CUFProxyAugmentation",
    "CUFObservableCeiling",
    "CUFGapAnalysisResponse",
    "DriverFactor",
    "ProjectDriversResponse",
    "HealthResponse",
    "PortfolioSummaryResponse",
    "SectorRiskAggregation",
]
