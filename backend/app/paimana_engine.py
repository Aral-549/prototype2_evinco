"""
MoSPI PAIMANA - GovScore Orchestration Engine

Implements the Master GovScore Lattice (Spec Sections 1.2, 2.1, 2.2):

    GovScore = max(100 * P_model, RuleFloor)

The supremum operator is the unique binary operator on ([0,100], <=)
satisfying the Non-Masking Invariant:
    RuleFloor >= tau  =>  GovScore >= tau
    100*P_model >= tau => GovScore >= tau
i.e. neither statistical uncertainty can dilute an established statutory
violation, nor administrative reporting lag can suppress a predictive alert.

Risk banding (Spec 2.1):
    [0, 25) Low | [25, 50) Moderate | [50, 75) High | [75, 100] Critical
"""

from __future__ import annotations

from typing import List, Literal, Optional

from app.paimana_car import (
    CaRResult,
    calculate_capital_at_risk,
)
from app.paimana_contracts import (
    ProjectGovernanceAssessment,
    ProjectInput,
    RuleFloorEvaluation,
)
from app.paimana_rules import calculate_rule_floor

RiskTier = Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]

# Operational risk band thresholds (Spec 2.1)
BAND_MODERATE: float = 25.0
BAND_HIGH: float = 50.0
BAND_CRITICAL: float = 75.0


def classify_risk_tier(gov_score: float) -> RiskTier:
    """Maps a GovScore to its operational risk band (Spec 2.1)."""
    if gov_score >= BAND_CRITICAL:
        return "CRITICAL"
    if gov_score >= BAND_HIGH:
        return "HIGH"
    if gov_score >= BAND_MODERATE:
        return "MODERATE"
    return "LOW"


def compute_gov_score(
    p_model: float,
    rule_floor: float,
) -> tuple[float, Literal["MACHINE_LEARNING", "RULE_FLOOR_OVERRIDE"]]:
    """
    Master GovScore override (Spec 2.1/2.2):

        GovScore = max(100 * P_model, RuleFloor)

    Returns the composite score in [0, 100] and the dominant evaluation
    pathway (explainability decoupling, Spec 2.2 item 4).

    Dominant-source convention:
      - 100*P_model >  RuleFloor -> MACHINE_LEARNING (TreeSHAP waterfall
        fully explains the score).
      - 100*P_model <= RuleFloor -> RULE_FLOOR_OVERRIDE. Exact ties are
        credited to the statutory branch: at a tie the subgradient of the
        supremum is set-valued ({0} ∪ {1} scaled), and assigning the tie to
        the deterministic rule pathway preserves the Section 2.2 guarantee
        that attribution is never ambiguous between the two branches -- the
        tripped flags {F_k} fully explain a tie, so administrative auditing
        always has unambiguous legal standing.
    """
    p_model = max(0.0, min(1.0, float(p_model)))
    rule_floor = max(0.0, min(100.0, float(rule_floor)))
    p_scaled = 100.0 * p_model
    gov_score = max(p_scaled, rule_floor)
    # Round away IEEE-754 noise (e.g. 0.55 * 100 == 55.00000000000001) so
    # exact ties are detected as ties, not as razor-thin ML wins.
    dominant_source: Literal["MACHINE_LEARNING", "RULE_FLOOR_OVERRIDE"] = (
        "MACHINE_LEARNING" if round(p_scaled, 9) > round(rule_floor, 9) else "RULE_FLOOR_OVERRIDE"
    )
    return gov_score, dominant_source


def formulate_interventions(
    project: ProjectInput,
    gov_score: float,
    risk_tier: RiskTier,
    rule_floor_eval: RuleFloorEvaluation,
    car_result: CaRResult,
) -> List[str]:
    """Tailored administrative intervention directives for PRAGATI / Cabinet Secretariat."""
    interventions: List[str] = []
    active_flags = {s.flag_id for s in rule_floor_eval.signals if s.is_active}

    if risk_tier in ("HIGH", "CRITICAL"):
        interventions.append(
            "Escalate project to MoSPI Project Review Committee / Cabinet Secretariat PRAGATI agenda."
        )
    if "F1" in active_flags:
        interventions.append(
            "Conduct forensic financial audit of contractor mobilization advances and on-site material inventories (GFR 2017 Rule 159)."
        )
    if "F2" in active_flags:
        interventions.append(
            "Commission immediate site inspection and demand a 90-day critical path recovery schedule from the implementing agency."
        )
    if "F3" in active_flags:
        interventions.append(
            "Refer pending statutory clearances (Forest Stage-II / NBWL / CRS) to the Project Monitoring Group for inter-ministerial resolution."
        )
    if "F4" in active_flags:
        interventions.append(
            "Issue statutory reporting non-compliance notice; mandate submission of overdue monthly CUF data within 7 days."
        )
    if "F5" in active_flags:
        interventions.append(
            "Initiate structured conciliation under the Arbitration Act and assess contractor liquidity before further escrow drawdowns."
        )
    if "F6" in active_flags:
        interventions.append(
            "Completion date on record has elapsed with work incomplete: demand a formal Revised Cost/Date Estimate "
            "filing within 30 days and freeze further milestone-linked disbursement until it is received."
        )
    if car_result.is_sector_fallback_used and risk_tier in ("HIGH", "CRITICAL"):
        interventions.append(
            "Apply the empirical sector-median overrun prior in budget re-projection until a Revised Cost Estimate is formally filed."
        )
    if risk_tier == "LOW":
        interventions.append(
            "Project execution within acceptable tolerance. Maintain standard monthly PAIMANA reporting."
        )
    if not interventions:
        interventions.append(
            "Monitor composite GovScore trend; no statutory triggers or elevated ML hazard at this reporting cycle."
        )
    return interventions


def assess_project(
    project: ProjectInput,
    p_model: float,
    base_rate_probability: float = 0.5,
    shap_drivers: Optional[list] = None,
) -> ProjectGovernanceAssessment:
    """
    End-to-end single-project governance assessment combining:
      1. Deterministic RuleFloor evaluation (Flags F1..F5)
      2. Master GovScore override: max(100 * P_model, RuleFloor)
      3. Capital-at-Risk with sector-median overrun prior
    SHAP drivers and the base rate are passed through from the ML engine.

    Note: shap_drivers is typed loosely (list of SHAPDriver) to avoid a
    circular import; the FastAPI layer serializes the pydantic models.
    """
    # Branch B: statutory administrative rules
    rule_floor_eval = calculate_rule_floor(project)

    # Master GovScore Lattice
    gov_score, dominant_source = compute_gov_score(p_model, rule_floor_eval.rule_floor)
    risk_tier = classify_risk_tier(gov_score)

    # Branch: public finance exposure
    car_result = calculate_capital_at_risk(
        original_cost=project.original_cost,
        p_model=p_model,
        cost_overrun_pct_current=project.cost_overrun_pct_current,
        sector=project.sector,
    )

    interventions = formulate_interventions(
        project, gov_score, risk_tier, rule_floor_eval, car_result
    )

    return ProjectGovernanceAssessment(
        project_id=project.project_id,
        project_name=project.project_name,
        sector=project.sector,
        implementing_agency=project.implementing_agency,
        physical_progress=project.physical_progress,
        original_cost_crores=project.original_cost,
        revised_cost_crores=project.revised_cost,
        p_model=round(p_model, 4),
        p_model_score=round(100.0 * p_model, 2),
        rule_floor=rule_floor_eval.rule_floor,
        gov_score=round(gov_score, 2),
        dominant_source=dominant_source,
        risk_tier=risk_tier,
        sector_median_overrun_pct=car_result.sector_median_overrun_pct,
        effective_overrun_pct=car_result.effective_overrun_pct,
        capital_at_risk_crores=car_result.capital_at_risk_crores,
        base_rate_probability=round(max(0.0, min(1.0, base_rate_probability)), 4),
        shap_drivers=list(shap_drivers or []),
        rule_signals=rule_floor_eval.signals,
        prescriptive_interventions=interventions,
    )
