"""
MoSPI PAIMANA Predictive Intelligence Platform - Day-1 Code Contracts
Module: app/paimana_contracts.py
Specification: MASTER_SPECIFICATION.md Section 3.1 (Pydantic v2.10+ compatible,
production-ready schemas). Zero placeholders ('...') in core validation logic.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
    ConfigDict,
)

# Canonical mapping for common MoSPI / CPSE sector aliases to official 22-sector taxonomy
SECTOR_SYNONYMS: Dict[str, str] = {
    "road transport": "Road Transport & Highways",
    "roads": "Road Transport & Highways",
    "highways": "Road Transport & Highways",
    "railway": "Railways",
    "power": "Power",
}


# ============================================================================
# 1. Project Input Schema (Common Upload Form - CUF Normalized)
# ============================================================================

class ProjectInput(BaseModel):
    """
    Standardized project input snapshot reflecting the MoSPI PAIMANA / OCMS
    Common Upload Form (CUF) monthly submission.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Identifiers
    project_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique MoSPI / PAIMANA Project Identifier, e.g. 'MOSPI-RLW-0194'",
        examples=["MOSPI-RLW-0194"],
    )
    project_name: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Official sanctioned title of the infrastructure project",
        examples=["Western Dedicated Freight Corridor (Dadri to JNPT)"],
    )
    sector: str = Field(
        default="Railways",
        description="Infrastructure sector (one of 22 central sectors)",
        examples=["Railways"],
    )
    implementing_agency: str = Field(
        default="RVNL",
        description="Central Public Sector Enterprise (CPSE) or Departmental Undertaking",
        examples=["DFCCIL"],
    )
    state: Optional[str] = Field(
        default=None,
        description="Primary geographical State / Union Territory of execution",
        examples=["Maharashtra"],
    )

    # Financial Metrics (Sanctioned & Incurred in ₹ Crore)
    original_cost: float = Field(
        ...,
        gt=0.0,
        description="Original sanctioned capital outlay (sanctioned capex) in ₹ Crore",
        examples=[28181.0],
    )
    revised_cost: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Current revised/anticipated completion cost in ₹ Crore (None if unrevised)",
        examples=[51101.0],
    )
    expenditure: float = Field(
        ...,
        ge=0.0,
        description="Cumulative financial expenditure incurred to date in ₹ Crore",
        examples=[44800.0],
    )
    expenditure_change_recent: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        description="Financial expenditure incurred in the most recent reporting month in ₹ Crore",
        examples=[410.0],
    )

    # Physical Progress Metrics (0.0% to 100.0%)
    physical_progress: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Cumulative physical milestone progress percentage (0.0 to 100.0)",
        examples=[88.2],
    )
    progress_change_recent: Optional[float] = Field(
        default=0.0,
        ge=-10.0,
        le=100.0,
        description="Physical progress percentage delta achieved in the most recent month",
        examples=[1.4],
    )

    # Schedule & Timeline Metrics (in Months)
    original_duration_months: float = Field(
        default=36.0,
        gt=0.0,
        description="Original sanctioned project construction duration in months",
        examples=[60.0],
    )
    months_elapsed: float = Field(
        default=12.0,
        ge=0.0,
        description="Total duration elapsed since statutory project sanction date in months",
        examples=[72.0],
    )
    current_delay_months: float = Field(
        default=0.0,
        ge=0.0,
        description="Accumulated schedule slippage beyond original statutory completion date in months",
        examples=[72.0],
    )

    # Operational & Milestone Governance Triggers
    overdue_milestones: int = Field(
        default=0,
        ge=0,
        description="Count of critical path milestones currently breached or overdue",
        examples=[3],
    )
    total_scheduled_milestones: int = Field(
        default=10,
        ge=1,
        description="Total milestones scheduled for completion up to the current reporting period",
        examples=[12],
    )
    clearance_pending_days: int = Field(
        default=0,
        ge=0,
        description="Maximum pendency (in calendar days) for pending Stage-II Forest, NBWL, or CRS clearances",
        examples=[195],
    )
    days_since_last_update: int = Field(
        default=15,
        ge=0,
        description="Calendar days elapsed since implementing agency uploaded monthly progress data",
        examples=[22],
    )
    dispute_status: Literal[
        "NONE",
        "CONCILIATION",
        "ARBITRATION",
        "HIGH_COURT_STAY",
        "TERMINATION_NOTICE",
    ] = Field(
        default="NONE",
        description="Formal statutory contractual dispute status under GFR / Arbitration Act",
        examples=["ARBITRATION"],
    )

    # Narrative delay remarks (raw CUF free-text, consumed by NLP proxy mining)
    delay_remarks: Optional[str] = Field(
        default=None,
        description="Free-text 'Reasons for Delay' remarks from the monthly CUF submission",
        examples=["Land acquisition compensation pending with SLAO; forest Stage-II clearance awaited"],
    )

    # ------------------------------------------------------------------------
    # Pydantic v2 Computed Properties (Derived Operational Indicators)
    # ------------------------------------------------------------------------

    @computed_field
    @property
    def remaining_progress_pct(self) -> float:
        """Remaining physical work percentage required for completion."""
        return max(0.0, round(100.0 - self.physical_progress, 4))

    @computed_field
    @property
    def expenditure_pct(self) -> float:
        """Cumulative expenditure expressed as a percentage of original sanctioned cost."""
        if self.original_cost <= 0.0:
            return 0.0
        return round((self.expenditure / self.original_cost) * 100.0, 4)

    @computed_field
    @property
    def cost_overrun_pct_current(self) -> float:
        """Current approved or reported cost overrun percentage against original sanction."""
        if self.revised_cost is None or self.original_cost <= 0.0:
            return 0.0
        return max(0.0, round(((self.revised_cost - self.original_cost) / self.original_cost) * 100.0, 4))

    @computed_field
    @property
    def spend_progress_gap(self) -> float:
        """Spend-to-physical decoupling gap (Expenditure % - Physical Progress %)."""
        return round(self.expenditure_pct - self.physical_progress, 4)

    @computed_field
    @property
    def execution_velocity(self) -> float:
        """
        Ratio of realized historical monthly progress run-rate to the required
        forward monthly run-rate needed to complete on original schedule.
        Values < 1.0 indicate sub-critical velocity requiring timeline extension.
        """
        if self.months_elapsed <= 0.0:
            return 1.0
        if self.physical_progress >= 100.0:
            return 1.0  # Completed works have stable nominal velocity
        realized_rate = self.physical_progress / self.months_elapsed
        remaining_work = max(0.01, 100.0 - self.physical_progress)
        remaining_months = max(1.0, self.original_duration_months - self.months_elapsed)
        required_rate = remaining_work / remaining_months
        if required_rate <= 0.0:
            return 1.0
        return round(realized_rate / required_rate, 4)

    # ------------------------------------------------------------------------
    # Field & Model Validators
    # ------------------------------------------------------------------------

    SECTOR_SYNONYMS: ClassVar[Dict[str, str]] = {
        "road transport": "Road Transport & Highways",
        "roads": "Road Transport & Highways",
        "highways": "Road Transport & Highways",
        "railway": "Railways",
        "power": "Power",
    }

    @field_validator("sector")
    @classmethod
    def normalize_sector_string(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("sector cannot be an empty string")
        normalized = cls.SECTOR_SYNONYMS.get(s.lower(), s)
        return normalized

    @model_validator(mode="after")
    def validate_expenditure_coherence(self) -> ProjectInput:
        if self.expenditure > (self.original_cost * 50.0):
            raise ValueError(
                f"Expenditure (₹{self.expenditure} Cr) exceeds 50x sanctioned cost (₹{self.original_cost} Cr). Check units."
            )
        return self

    def to_feature_dict(self, feature_columns: List[str]) -> Dict[str, float]:
        """
        Convert input to the exact feature vector expected by a trained model,
        filling any unknown feature column with 0.0. Columns matching the
        quarantine list (leakage proxies) are emitted as 0.0 so that legacy
        boosters never receive post-facto outcome information.
        """
        computed_dict = {
            "physical_progress": float(self.physical_progress),
            "progress_change": float(self.progress_change_recent or 0.0),
            "expenditure": float(self.expenditure),
            "expenditure_change_robust": float(self.expenditure_change_recent or 0.0),
            "expenditure_pct_robust": float(self.expenditure_pct),
            "original_cost": float(self.original_cost),
            "revised_cost": float(self.revised_cost if self.revised_cost is not None else self.original_cost),
            "cost_overrun_pct_robust": float(self.cost_overrun_pct_current),
            "cost_change_robust": float(
                max(0.0, (self.revised_cost - self.original_cost)) if self.revised_cost is not None else 0.0
            ),
            "remaining_progress_pct": float(self.remaining_progress_pct),
            # Quarantined leakage proxies (Spec Sections 1.2 / 4.1): always
            # zeroed at inference time. `current_delay_months` is a post-facto
            # bureaucratic outcome record exactly like `has_revised_doc` -- a
            # revised DOC and accumulated delay are filed only AFTER chronic
            # failure has materialized, so feeding either to the booster lets
            # the model read an already-acknowledged outcome.
            "has_revised_doc": 0.0,
            "current_delay_months_robust": 0.0,
        }
        return {col: computed_dict.get(col, 0.0) for col in feature_columns}


# ============================================================================
# 2. Rule Activation Signal Schema
# ============================================================================

class RuleActivationSignal(BaseModel):
    """
    Detailed audit payload for a single tripped statutory or operational rule.
    """
    model_config = ConfigDict(extra="ignore")

    flag_id: str = Field(
        ...,
        description="Canonical identifier of the statutory flag (F1 through F5)",
        examples=["F1"],
    )
    signal_code: str = Field(
        ...,
        description="Machine-readable symbolic code",
        examples=["SPEND_PROGRESS_DECOUPLING"],
    )
    rule_name: str = Field(
        ...,
        description="Official administrative title of the governance rule",
        examples=["Expenditure Pace Decoupled from Physical Milestone Delivery"],
    )
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "INFO"] = Field(
        ...,
        description="Administrative urgency classification",
        examples=["CRITICAL"],
    )
    weight: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Clamped penalty weight contributed to RuleFloor",
        examples=[30.0],
    )
    is_active: bool = Field(
        ...,
        description="True if the project breached the rule threshold",
        examples=[True],
    )
    metric_name: str = Field(
        ...,
        description="Name of the evaluated operational metric",
        examples=["spend_progress_gap"],
    )
    metric_value: float = Field(
        ...,
        description="Observed quantitative value on the project",
        examples=[70.77],
    )
    threshold_value: float = Field(
        ...,
        description="Governing statutory threshold",
        examples=[25.0],
    )
    statutory_rationale: str = Field(
        ...,
        description="CVC / GFR / MoSPI administrative rationale underpinning the rule",
        examples=[
            "GFR 2017 Rule 159 violation: Advance disbursement without verified physical progress signals contractor hoarding."
        ],
    )


class RuleFloorEvaluation(BaseModel):
    """
    Result container for the complete deterministic RuleFloor evaluation.
    """
    model_config = ConfigDict(extra="ignore")

    rule_floor: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Total clamped deterministic rule floor score [0.0 - 100.0]",
        examples=[85.0],
    )
    active_signals_count: int = Field(
        ...,
        ge=0,
        description="Number of statutory flags active for this project",
        examples=[2],
    )
    signals: List[RuleActivationSignal] = Field(
        default_factory=list,
        description="List of evaluated governance rule signals",
    )
    critical_override: bool = Field(
        ...,
        description="True if a fatal statutory flag (F1 severe, F3 severe, F5 dispute) triggered mandatory floor clamping",
        examples=[True],
    )
    summary: str = Field(
        ...,
        description="Executive summary of rule evaluation",
        examples=["Critical statutory risk detected: severe spend decoupling and active legal dispute."],
    )


# ============================================================================
# 3. Local TreeSHAP Explainability Schema
# ============================================================================

class SHAPDriver(BaseModel):
    """
    Individual feature contribution derived via exact TreeSHAP.
    """
    model_config = ConfigDict(extra="ignore")

    feature_name: str = Field(
        ...,
        description="Name of the input feature",
        examples=["spend_progress_gap"],
    )
    feature_value: Optional[float] = Field(
        default=None,
        description="Raw numerical value of the feature for this project snapshot",
        examples=[70.77],
    )
    shap_value: float = Field(
        ...,
        description="Marginal log-odds contribution (phi_i) from TreeSHAP",
        examples=[0.4852],
    )
    direction: Literal["RISK_INCREASING", "RISK_DECREASING", "NEUTRAL"] = Field(
        ...,
        description="Whether this factor accelerates, mitigates, or has neutral impact on slippage hazard",
        examples=["RISK_INCREASING"],
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Importance rank (1 = most influential driver)",
        examples=[1],
    )
    administrative_interpretation: str = Field(
        ...,
        description="Non-technical operational meaning of this driver for project directors",
        examples=[
            "Excess capital absorption relative to physical progress is the primary statistical risk accelerator."
        ],
    )


# ============================================================================
# 4. Project Governance Assessment / Prediction Output Schema
# ============================================================================

class ProjectGovernanceAssessment(BaseModel):
    """
    The unified executive prediction envelope produced by the PAIMANA platform.
    Integrates ML hazard probability, deterministic RuleFloor, GovScore override,
    Capital-at-Risk, and TreeSHAP waterfall drivers.
    """
    model_config = ConfigDict(extra="ignore")

    # Project Identifiers
    project_id: str = Field(..., description="Unique Project Identifier")
    project_name: str = Field(..., description="Official Project Name")
    sector: str = Field(..., description="Infrastructure Sector")
    implementing_agency: str = Field(..., description="Implementing Agency")

    # Capital Scale
    original_cost_crores: float = Field(..., description="Sanctioned Capital Cost (₹ Cr)")
    revised_cost_crores: Optional[float] = Field(None, description="Current Revised Cost (₹ Cr)")

    # Core Multi-Source Scores (The GovScore Architecture)
    p_model: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Stage-Aware XGBoost calibrated schedule slippage probability [0.0 - 1.0]",
        examples=[0.8524],
    )
    p_model_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Scaled ML probability (100 * P_model) [0.0 - 100.0]",
        examples=[85.24],
    )
    rule_floor: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Deterministic administrative safety floor [0.0 - 100.0]",
        examples=[60.0],
    )
    gov_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite Governance Risk Score = max(100 * P_model, RuleFloor) [0.0 - 100.0]",
        examples=[85.24],
    )
    dominant_source: Literal["MACHINE_LEARNING", "RULE_FLOOR_OVERRIDE"] = Field(
        ...,
        description="Identifies which evaluation pathway determined the final GovScore",
        examples=["MACHINE_LEARNING"],
    )
    risk_tier: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"] = Field(
        ...,
        description="Operational risk classification tier",
        examples=["CRITICAL"],
    )

    # Public Finance & Capital Exposure Metrics
    sector_median_overrun_pct: float = Field(
        ...,
        description="Empirical historical median cost overrun percentage for this sector",
        examples=[42.3],
    )
    effective_overrun_pct: float = Field(
        ...,
        description="Effective overrun factor used in CaR = max(current_overrun, sector_median)",
        examples=[81.33],
    )
    capital_at_risk_crores: float = Field(
        ...,
        ge=0.0,
        description="Capital-at-Risk in ₹ Crore: Original_Cost * P_model * (Effective_Overrun / 100)",
        examples=[19481.67],
    )

    # Local Explainability & Statutory Triggers
    base_rate_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Portfolio expected base rate probability (sigmoid of TreeSHAP bias)",
        examples=[0.544],
    )
    shap_drivers: List[SHAPDriver] = Field(
        default_factory=list,
        description="Top 5 local TreeSHAP risk accelerators and mitigators",
    )
    rule_signals: List[RuleActivationSignal] = Field(
        default_factory=list,
        description="Activated statutory and operational governance flags",
    )
    prescriptive_interventions: List[str] = Field(
        default_factory=list,
        description="Tailored administrative intervention directives for Cabinet Secretariat / PRAGATI",
    )


# ============================================================================
# 5. Portfolio Batch Request & Response Schemas
# ============================================================================

class BatchProjectRequest(BaseModel):
    """Batch scoring request payload for monitoring portfolios."""
    model_config = ConfigDict(extra="ignore")
    projects: List[ProjectInput] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of project snapshots to evaluate",
    )


class BatchProjectResponse(BaseModel):
    """
    Portfolio aggregation response rank-ordered by Capital-at-Risk (CaR).
    """
    model_config = ConfigDict(extra="ignore")

    total_projects: int = Field(..., description="Total project count evaluated")
    total_monitored_capex_crores: float = Field(..., description="Sum of sanctioned original cost in ₹ Cr")
    total_capital_at_risk_crores: float = Field(..., description="Sum of Capital-at-Risk across portfolio in ₹ Cr")
    risk_tier_counts: Dict[str, int] = Field(..., description="Distribution across risk tiers")
    dominant_source_counts: Dict[str, int] = Field(..., description="Counts of ML vs RuleFloor governed projects")
    ranked_projects: List[ProjectGovernanceAssessment] = Field(
        ...,
        description="Evaluated projects sorted in descending order of Capital-at-Risk (CaR)",
    )


class BatchCsvResponse(BatchProjectResponse):
    """Batch result from CSV ingestion, with column-mapping diagnostics."""

    columns_ignored: List[str] = Field(
        default_factory=list,
        description="CSV columns present in the header but not mapped to any model field",
    )
