"""Pydantic schemas for MoSPI analytics, CUF gap analysis, and system health.

Aligned with MASTER_SPECIFICATION.md Sections 2.8-2.10:
- The CUF gap endpoint reports the *empirical performance ceiling* of the
  observable feature set and NLP-proxy augmentation deltas. It does NOT
  claim an exact variance decomposition of unrecorded variables
  (zero-confabulation standard, Spec Section 2.10 / 4.5).
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class SectorRiskAggregation(BaseModel):
    """Sectoral risk and capex profile."""
    sector: str
    project_count: int
    total_original_cost_crores: float
    total_expenditure_crores: float
    total_capital_at_risk_crores: float
    avg_gov_score: float
    critical_high_projects: int


class PortfolioSummaryResponse(BaseModel):
    """Macro overview of the monitored infrastructure portfolio."""
    total_projects: int
    total_monitored_capex_crores: float
    total_expenditure_crores: float
    total_capital_at_risk_crores: float
    capital_at_risk_percentage: float
    risk_tier_counts: Dict[str, int]
    dominant_source_counts: Dict[str, int]
    sectors: List[SectorRiskAggregation]


class CUFObservableCeiling(BaseModel):
    """Empirical asymptotic performance ceiling of the observable CUF feature set."""
    discriminative_auc_ceiling: float = Field(
        ..., description="Saturating GroupKFold out-of-sample AUC achievable on CUF fields alone"
    )
    brier_score_at_ceiling: float = Field(
        ..., description="Brier score at the observable feature-space ceiling"
    )
    cost_regression_r2_ceiling: float = Field(
        ..., description="Out-of-sample R^2 for continuous cost escalation using CUF fields only"
    )
    residual_classification_error: float = Field(
        ..., description="1 - AUC_ceiling; empirical upper bound on unexplained discrimination"
    )


class CUFProxyAugmentation(BaseModel):
    """Empirical delta from augmenting the model with NLP-mined bottleneck proxies."""
    proxy_names: List[str]
    auc_before: float
    auc_after: float
    auc_delta: float = Field(..., description="AUC improvement attributable to NLP proxies")
    rmse_before_pct: float
    rmse_after_pct: float
    rmse_delta_pct: float
    delayed_projects_citing_top_three_factors_pct: float = Field(
        ..., description="Share of delayed projects whose remarks cite land, forest, or litigation factors"
    )


class CUFMissingVariable(BaseModel):
    """High-leverage structured field proposed for mandatory CUF 2.0 collection."""
    variable_name: str
    suggested_field_name: str
    operational_rationale: str
    recommended_input_type: str
    primary_source_grounding: str = Field(
        ..., description="Official audit or report grounding this proposal (zero-confabulation standard)"
    )
    cuf_2_0_priority: str


class CUFGapAnalysisResponse(BaseModel):
    """
    MoSPI Dimension (c) response under the zero-confabulation standard:
    reports what CAN be established (observable ceiling + NLP proxy deltas)
    and explicitly refuses to fabricate variance shares for unrecorded variables.
    """
    methodology_statement: str = Field(
        ...,
        description=(
            "Explicit statement that exact variance decomposition of unrecorded "
            "variables is mathematically impossible; only empirical ceilings and "
            "proxy augmentation deltas are reported."
        ),
    )
    observable_ceiling: CUFObservableCeiling
    observable_ceiling_provenance: str = Field(
        default="",
        description="Where the ceiling numbers came from; 'MEASURED' or an explicit UNAVAILABLE reason.",
    )
    proxy_augmentation: CUFProxyAugmentation
    proxy_augmentation_provenance: str = Field(
        default="",
        description=(
            "Whether the NLP proxy delta is measurable at all. A value of -1.0 in any "
            "proxy_augmentation field is a sentinel meaning 'not measurable', never a result."
        ),
    )
    missing_variables_recommended: List[CUFMissingVariable]
    policy_action_items: List[str]


class DriverFactor(BaseModel):
    """Key factor moving the project GovScore."""
    factor_name: str
    direction: str  # "ACCELERATOR" (increases risk) or "MITIGATOR" (reduces risk)
    impact_score: float
    operational_meaning: str


class ProjectDriversResponse(BaseModel):
    """TreeSHAP driver attribution for a specific project."""
    project_id: str
    project_name: str
    p_model: float
    base_rate_probability: float
    gov_score: float
    dominant_source: str
    primary_drivers: List[DriverFactor]
    nlp_bottleneck_proxies: Dict[str, int] = Field(
        default_factory=dict,
        description="Binary NLP proxy indicators mined from delay remarks (Spec Section 2.9)",
    )


class BenchmarkModelRow(BaseModel):
    """One row of the Dimension (b) classifier benchmark table (Spec Section 2.7 Table A)."""
    model_architecture: str
    model_class: str
    test_auc: float
    delong_z_vs_coxph: float = Field(..., description="DeLong Z vs Cox-PH baseline (0.0 for the baseline itself)")
    delong_p_value: float = Field(..., description="Two-sided DeLong p-value (1.0 for the baseline itself)")
    brier_score: float
    log_loss: float


class BenchmarkResponse(BaseModel):
    """MoSPI Dimension (b): paired DeLong comparison of ML vs conventional statistical baselines."""
    dimension: str
    evaluation_protocol: str
    statistical_test: str = Field(
        ..., description="Must always be the DeLong paired AUC test; Diebold-Mariano is methodologically invalid (Spec Section 2.6)"
    )
    test_methodology_note: str
    benchmark_results: List[BenchmarkModelRow]
    benchmark_values_provenance: str = Field(
        (
            "PROTOCOL REFERENCE VALUES: the AUC/Brier/log-loss figures and the "
            "DeLong Z/p statistics below are the registered benchmark constants "
            "from Master Specification Section 2.7 (Table A), NOT a live "
            "evaluation of the served model bundle. Recompute exact DeLong "
            "statistics for arbitrary paired score vectors via the "
            "/analytics/delong-test endpoint, which implements the Section 2.5 "
            "protocol directly (zero-confabulation standard, Spec Section 2.10)."
        ),
        description=(
            "Explicit provenance statement distinguishing registered protocol "
            "constants from live inference, per the zero-confabulation standard."
        ),
    )


class HealthResponse(BaseModel):
    """System health check and loaded model diagnostics."""
    status: str
    service_name: str
    version: str
    model_loaded: bool
    model_name: str
    model_target: str
    classification_threshold: float
    feature_count: int
    features: List[str]
    quarantined_features: List[str]
