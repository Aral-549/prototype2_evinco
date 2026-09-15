"""
MoSPI PAIMANA - Deterministic RuleFloor Evaluation (Spec Section 3.2.1)

Evaluates 5 canonical administrative and statutory governance flags (F1..F5)
to compute the deterministic safety floor for GovScore.

    RuleFloor = min(100.0, sum_k w_k * 1_{F_k})

This is a pure, deterministic, side-effect-free function of a validated
ProjectInput. It never touches the ML model.
"""

from typing import Dict, List, Optional

from app.paimana_contracts import (
    ProjectInput,
    RuleActivationSignal,
    RuleFloorEvaluation,
)


def calculate_rule_floor(
    project: ProjectInput,
    custom_weights: Optional[Dict[str, float]] = None,
) -> RuleFloorEvaluation:
    """
    Evaluates 5 canonical administrative and statutory governance flags to compute
    the deterministic safety floor for GovScore.

    Args:
        project: Fully validated ProjectInput model instance.
        custom_weights: Optional dictionary overriding default clamped weights.

    Returns:
        RuleFloorEvaluation containing total score, active signals, and metadata.
    """
    weights: Dict[str, float] = {
        "F1": 30.0,  # Spend-to-Physical Decoupling
        "F2": 25.0,  # Physical Progress Stall
        "F3": 25.0,  # Milestone Slippage Density / Clearance Stall
        "F4": 20.0,  # Reporting Non-Compliance / Stale Registry
        "F5": 30.0,  # Contractual Litigation / Arbitration Notice
    }
    if custom_weights:
        weights.update(custom_weights)

    evaluated_signals: List[RuleActivationSignal] = []
    total_weight = 0.0
    has_critical_override = False

    # ------------------------------------------------------------------------
    # Flag F1: Spend-to-Physical Decoupling (Front-Loading Advance Check)
    # Condition: (Expenditure / C0 - Progress / 100) >= 0.25 AND Progress < 50%
    # ------------------------------------------------------------------------
    spend_fraction = project.expenditure / max(1.0, project.original_cost)
    progress_fraction = project.physical_progress / 100.0
    decoupling_gap_pct = (spend_fraction - progress_fraction) * 100.0
    f1_active = bool(decoupling_gap_pct >= 25.0 and project.physical_progress < 50.0)

    f1_weight = weights["F1"] if f1_active else 0.0
    if f1_active:
        total_weight += f1_weight
        if decoupling_gap_pct >= 40.0:
            has_critical_override = True

    evaluated_signals.append(
        RuleActivationSignal(
            flag_id="F1",
            signal_code="SPEND_PROGRESS_DECOUPLING",
            rule_name="Expenditure Pace Decoupled from Physical Milestone Delivery",
            severity="CRITICAL" if decoupling_gap_pct >= 40.0 else "HIGH",
            weight=f1_weight,
            is_active=f1_active,
            metric_name="spend_progress_gap",
            metric_value=round(decoupling_gap_pct, 2),
            threshold_value=25.0,
            statutory_rationale=(
                "GFR 2017 Rule 159 & CVC guidelines: Cumulative disbursement exceeds physical execution by >25% "
                "prior to reaching 50% completion, indicating unearned contractor advances or procurement hoarding."
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Flag F2: Physical Progress Stall (Chronic Site Stagnation)
    # Condition: Monthly Progress Change <= 0.1% AND Elapsed Ratio >= 0.20 AND Progress < 95%
    # ------------------------------------------------------------------------
    elapsed_ratio = project.months_elapsed / max(1.0, project.original_duration_months)
    recent_prog = float(project.progress_change_recent or 0.0)
    f2_active = bool(recent_prog <= 0.1 and elapsed_ratio >= 0.20 and project.physical_progress < 95.0)

    f2_weight = weights["F2"] if f2_active else 0.0
    if f2_active:
        total_weight += f2_weight

    evaluated_signals.append(
        RuleActivationSignal(
            flag_id="F2",
            signal_code="PROGRESS_STALL",
            rule_name="Chronic Physical Execution Stagnation",
            severity="HIGH" if elapsed_ratio >= 0.50 else "MEDIUM",
            weight=f2_weight,
            is_active=f2_active,
            metric_name="progress_change_recent",
            metric_value=round(recent_prog, 2),
            threshold_value=0.1,
            statutory_rationale=(
                "Site delivery has flatlined (<= 0.1% progress) after >20% duration elapsed, "
                "signaling contractor insolvency, design freeze, or unresolved site encumbrances."
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Flag F3: Milestone Slippage Density & Statutory Clearance Stall
    # Condition: Overdue Milestones / Scheduled Milestones >= 0.50 OR Clearance Days > 180
    # ------------------------------------------------------------------------
    milestone_density = project.overdue_milestones / max(1, project.total_scheduled_milestones)
    clearance_days = project.clearance_pending_days
    f3_active = bool(milestone_density >= 0.50 or clearance_days > 180)

    f3_weight = weights["F3"] if f3_active else 0.0
    if f3_active:
        total_weight += f3_weight
        if clearance_days > 365:
            has_critical_override = True

    trigger_metric = "clearance_pending_days" if clearance_days > 180 else "milestone_density"
    observed_val = float(clearance_days) if clearance_days > 180 else round(milestone_density, 2)
    threshold_val = 180.0 if clearance_days > 180 else 0.50

    evaluated_signals.append(
        RuleActivationSignal(
            flag_id="F3",
            signal_code="MILESTONE_CLEARANCE_STALL",
            rule_name="Critical Path Milestone Breakdown / Regulatory Clearance Stall",
            severity="CRITICAL" if clearance_days > 365 else "HIGH",
            weight=f3_weight,
            is_active=f3_active,
            metric_name=trigger_metric,
            metric_value=observed_val,
            threshold_value=threshold_val,
            statutory_rationale=(
                "Over 50% of scheduled project milestones are overdue or statutory environmental/railway clearances "
                "have been pending for >180 days, breaking the critical path."
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Flag F4: Reporting Non-Compliance / Stale Registry
    # Condition: Days since last monthly update > 60
    # ------------------------------------------------------------------------
    days_stale = project.days_since_last_update
    f4_active = bool(days_stale > 60)

    f4_weight = weights["F4"] if f4_active else 0.0
    if f4_active:
        total_weight += f4_weight

    evaluated_signals.append(
        RuleActivationSignal(
            flag_id="F4",
            signal_code="REPORTING_NON_COMPLIANCE",
            rule_name="Statutory Reporting Non-Compliance / Stale Progress Registry",
            severity="MEDIUM",
            weight=f4_weight,
            is_active=f4_active,
            metric_name="days_since_last_update",
            metric_value=float(days_stale),
            threshold_value=60.0,
            statutory_rationale=(
                "MoSPI Project Monitoring Division (IPMD) mandate: Implementing agencies failing to update monthly "
                "progress records for >60 days are penalized for administrative non-transparency."
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Flag F5: Contractual Litigation / Arbitration Notice
    # Condition: dispute_status in ('ARBITRATION', 'HIGH_COURT_STAY', 'TERMINATION_NOTICE')
    # ------------------------------------------------------------------------
    f5_active = bool(
        project.dispute_status in ("ARBITRATION", "HIGH_COURT_STAY", "TERMINATION_NOTICE")
    )
    f5_weight = weights["F5"] if f5_active else 0.0
    if f5_active:
        total_weight += f5_weight
        has_critical_override = True

    evaluated_signals.append(
        RuleActivationSignal(
            flag_id="F5",
            signal_code="CONTRACTUAL_LITIGATION",
            rule_name="Active Legal Injunction / Contractual Arbitration Dispute",
            severity="CRITICAL",
            weight=f5_weight,
            is_active=f5_active,
            metric_name="dispute_status_indicator",
            metric_value=1.0 if f5_active else 0.0,
            threshold_value=1.0,
            statutory_rationale=(
                "Arbitration Act / High Court stay injunction actively freezes site access, escrow drawdowns, "
                "or structural works. Multiplies median completion timeline by 2.4x."
            ),
        )
    )

    # Additive Clamped Formulation: RuleFloor = min(100.0, sum(w_k * 1_{F_k}))
    calculated_floor = min(100.0, total_weight)
    active_count = sum(1 for s in evaluated_signals if s.is_active)

    summary_msg = (
        f"RuleFloor evaluated at {calculated_floor:.1f}/100 with {active_count} active statutory triggers."
        if active_count > 0
        else "All statutory and operational parameters within normal tolerance windows."
    )

    return RuleFloorEvaluation(
        rule_floor=round(calculated_floor, 2),
        active_signals_count=active_count,
        signals=evaluated_signals,
        critical_override=has_critical_override,
        summary=summary_msg,
    )
