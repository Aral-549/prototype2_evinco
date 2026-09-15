"""Pydantic schemas for Project input, prediction responses, and risk profiles.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, computed_field


class ProjectInput(BaseModel):
    """Common Upload Form (CUF) input for a MoSPI central infrastructure project."""
    
    project_id: str = Field(..., description="Unique MoSPI / PAIMANA Project ID, e.g. 'MOSPI-RLW-0194'")
    project_name: str = Field(..., description="Official title of the project")
    sector: str = Field(default="Railways", description="Infrastructure sector, e.g. 'Railways', 'Road Transport & Highways', 'Power'")
    implementing_agency: str = Field(default="RVNL", description="Implementing agency, e.g. 'NHAI', 'RVNL', 'NTPC'")
    
    # Financial metrics in ₹ Crore
    original_cost: float = Field(..., gt=0, description="Sanctioned original cost in ₹ Crore")
    revised_cost: Optional[float] = Field(default=None, ge=0, description="Current revised cost in ₹ Crore (if revised)")
    expenditure: float = Field(..., ge=0, description="Cumulative expenditure to date in ₹ Crore")
    expenditure_change_robust: Optional[float] = Field(default=0.0, description="Recent monthly expenditure increment in ₹ Crore")
    
    # Progress metrics (0 - 100%)
    physical_progress: float = Field(..., ge=0.0, le=100.0, description="Reported physical progress percentage (0-100)")
    progress_change: Optional[float] = Field(default=0.0, description="Recent monthly progress delta (percentage points)")
    
    # Schedule metrics
    current_delay_months_robust: Optional[float] = Field(default=0.0, ge=0.0, description="Current delay elapsed in months against original schedule")
    has_revised_doc: Optional[int] = Field(default=0, ge=0, le=1, description="Binary flag (1 if a revised Date of Completion has been registered, 0 otherwise)")

    @computed_field
    @property
    def remaining_progress_pct(self) -> float:
        """Remaining physical work percentage."""
        return max(0.0, 100.0 - self.physical_progress)

    @computed_field
    @property
    def expenditure_pct_robust(self) -> float:
        """Expenditure as a percentage of original sanctioned cost."""
        if self.original_cost <= 0:
            return 0.0
        return (self.expenditure / self.original_cost) * 100.0

    @computed_field
    @property
    def cost_overrun_pct_robust(self) -> float:
        """Cost overrun percentage against original cost."""
        if not self.revised_cost or self.original_cost <= 0:
            return 0.0
        return max(0.0, ((self.revised_cost - self.original_cost) / self.original_cost) * 100.0)

    @computed_field
    @property
    def cost_change_robust(self) -> float:
        """Absolute cost change in ₹ Crore."""
        if not self.revised_cost:
            return 0.0
        return max(0.0, self.revised_cost - self.original_cost)

    def to_feature_dict(self, feature_columns: List[str]) -> Dict[str, float]:
        """Convert input to the exact feature vector expected by the XGBoost pipeline."""
        computed_dict = {
            "physical_progress": float(self.physical_progress),
            "progress_change": float(self.progress_change or 0.0),
            "expenditure": float(self.expenditure),
            "expenditure_change_robust": float(self.expenditure_change_robust or 0.0),
            "expenditure_pct_robust": float(self.expenditure_pct_robust),
            "original_cost": float(self.original_cost),
            "revised_cost": float(self.revised_cost if self.revised_cost is not None else self.original_cost),
            "cost_overrun_pct_robust": float(self.cost_overrun_pct_robust),
            "cost_change_robust": float(self.cost_change_robust),
            "remaining_progress_pct": float(self.remaining_progress_pct),
            "current_delay_months_robust": float(self.current_delay_months_robust or 0.0),
            "has_revised_doc": int(self.has_revised_doc or 0),
        }
        return {col: computed_dict.get(col, 0.0) for col in feature_columns}


class EarlyWarningSignal(BaseModel):
    """Specific governance red flag or lead indicator."""
    code: str = Field(..., description="Identifier code, e.g. 'SPEND_PROGRESS_DECOUPLING'")
    severity: str = Field(..., description="'HIGH', 'MEDIUM', 'INFO'")
    title: str = Field(..., description="Short summary of the trigger")
    description: str = Field(..., description="Detailed operational rationale for administrators")
    metric_value: float = Field(..., description="Observed quantitative value")
    threshold_value: float = Field(..., description="Governing threshold")


class ProjectRiskAnalysis(BaseModel):
    """Detailed risk scoring and early warning intelligence."""
    risk_score: float = Field(..., description="Calibrated risk index (0 - 100)")
    risk_probability: float = Field(..., description="Raw model probability of schedule slippage (0.0 - 1.0)")
    risk_tier: str = Field(..., description="'HIGH' (>=70), 'MEDIUM' (40-69), 'LOW' (<40)")
    classification_decision: int = Field(..., description="Binary threshold decision (1 = Overrun risk, 0 = On schedule)")
    
    # Public finance metric
    capital_at_risk_crores: float = Field(..., description="Public funds exposed: Original Cost * Risk Probability * Overrun Factor")
    
    # Lead indicators and flags
    early_warning_signals: List[EarlyWarningSignal] = Field(default_factory=list)
    leakage_proxy_warning: bool = Field(..., description="True if risk is largely driven by retrospective revision flag rather than velocity")
    
    # Prescriptive recommendations
    recommended_interventions: List[str] = Field(default_factory=list)
    feature_contributions: Dict[str, float] = Field(default_factory=dict, description="Normalized feature contribution weights")


class ProjectPredictionResponse(BaseModel):
    """Complete prediction envelope for a single infrastructure project."""
    project_id: str
    project_name: str
    sector: str
    implementing_agency: str
    original_cost: float
    revised_cost: Optional[float]
    expenditure: float
    physical_progress: float
    analysis: ProjectRiskAnalysis


class BatchProjectRequest(BaseModel):
    """Payload for evaluating multiple projects in a single API call."""
    projects: List[ProjectInput]


class BatchProjectResponse(BaseModel):
    """Portfolio-level aggregation sorted by Capital-at-Risk."""
    total_projects: int
    total_monitored_capex_crores: float
    total_capital_at_risk_crores: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    ranked_projects: List[ProjectPredictionResponse]


class BatchCsvResponse(BatchProjectResponse):
    """Batch result from CSV ingestion, with column-mapping diagnostics."""
    columns_ignored: List[str] = Field(default_factory=list, description="CSV columns present in the header but not mapped to any model field")
