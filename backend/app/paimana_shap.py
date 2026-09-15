"""
MoSPI PAIMANA - Native TreeSHAP Local Explainability (Spec Section 3.2.3)

Computes exact local TreeSHAP feature attributions using XGBoost's built-in
C++ TreeSHAP implementation (Lundberg et al., 2020) via `pred_contribs=True`,
mapping marginal log-odds impacts to actionable administrative narratives.
"""

from typing import Any, Dict, List, Literal, Tuple

import numpy as np

from app.paimana_contracts import SHAPDriver

# Administrative translation catalog mapping raw feature margins to policy narratives
FEATURE_POLICY_TRANSLATION: Dict[str, Dict[str, str]] = {
    "spend_progress_gap": {
        "RISK_INCREASING": "Capex disbursement outpaces physical delivery (front-loading advance risk).",
        "RISK_DECREASING": "Expenditure tightly synchronized with physical milestone completion.",
    },
    "execution_velocity": {
        "RISK_INCREASING": "Current physical progress run-rate is below critical threshold to meet schedule.",
        "RISK_DECREASING": "Execution velocity exceeds required forward run-rate, building schedule buffer.",
    },
    "physical_progress": {
        "RISK_INCREASING": "Early construction stage exposes project to high geotechnical and environmental variance.",
        "RISK_DECREASING": "Advanced structural completion (>80%) significantly bounds remaining risk envelope.",
    },
    "progress_change": {
        "RISK_INCREASING": "Monthly execution velocity decelerated, signaling on-site operational friction.",
        "RISK_DECREASING": "Healthy monthly progress increment indicates active contractor mobilization.",
    },
    "progress_change_recent": {
        "RISK_INCREASING": "Monthly execution velocity decelerated, signaling on-site operational friction.",
        "RISK_DECREASING": "Healthy monthly progress increment indicates active contractor mobilization.",
    },
    "remaining_progress_pct": {
        "RISK_INCREASING": "Substantial quantum of remaining work exposed to forward supply chain delays.",
        "RISK_DECREASING": "Terminal punch-list stage; core civil engineering risks retired.",
    },
    "expenditure_pct": {
        "RISK_INCREASING": "High capital absorption rate rapidly consuming contingency reserves.",
        "RISK_DECREASING": "Low capital burn rate indicates remaining contingency cushion.",
    },
    "expenditure_pct_robust": {
        "RISK_INCREASING": "High capital absorption rate rapidly consuming contingency reserves.",
        "RISK_DECREASING": "Low capital burn rate indicates remaining contingency cushion.",
    },
    "current_delay_months": {
        "RISK_INCREASING": "Historical timeline slippage inertia compounding forward critical path.",
        "RISK_DECREASING": "Zero historical schedule delay maintains original float allocation.",
    },
    "current_delay_months_robust": {
        "RISK_INCREASING": "Historical timeline slippage inertia compounding forward critical path.",
        "RISK_DECREASING": "Zero historical schedule delay maintains original float allocation.",
    },
    "cost_overrun_pct_robust": {
        "RISK_INCREASING": "Realized cost escalation over sanction signals fiscal stress and re-approval friction.",
        "RISK_DECREASING": "Cost discipline within sanctioned envelope preserves budgetary headroom.",
    },
    "expenditure_change_robust": {
        "RISK_INCREASING": "Surging monthly capital burn without matching milestone delivery.",
        "RISK_DECREASING": "Moderate, steady monthly disbursement consistent with planned cash flow.",
    },
    "original_cost": {
        "RISK_INCREASING": "Megaproject scale embeds coordination and oversight complexity penalties.",
        "RISK_DECREASING": "Smaller sanction reduces multi-agency coordination overhead.",
    },
    "revised_cost": {
        "RISK_INCREASING": "Revised sanction above original cost indicates acknowledged fiscal stress.",
        "RISK_DECREASING": "Revised sanction aligned with original estimate indicates stable scope.",
    },
    "cost_change_robust": {
        "RISK_INCREASING": "Absolute rupee escalation beyond the sanctioned envelope compounds exposure.",
        "RISK_DECREASING": "No absolute cost growth beyond sanction.",
    },
    "expenditure": {
        "RISK_INCREASING": "Cumulative capital deployment concentrated ahead of physical delivery.",
        "RISK_DECREASING": "Cumulative deployment tracks physical execution progress.",
    },
    "has_revised_doc": {
        "RISK_INCREASING": "Post-facto bureaucratic record of an already-acknowledged schedule failure (leakage-prone, quarantined at inference).",
        "RISK_DECREASING": "No revised Date of Completion filed.",
    },
}


def explain_prediction_shap(
    booster: Any,
    feature_row: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
) -> Tuple[float, List[SHAPDriver]]:
    """
    Computes exact local TreeSHAP feature attributions using XGBoost's built-in C++
    TreeSHAP implementation (Lundberg et al., 2020), mapping marginal log-odds impacts
    to actionable administrative narratives.

    Args:
        booster: Trained xgboost.Booster or XGBClassifier instance.
        feature_row: 1D or 2D numpy array containing normalized feature values in canonical order.
        feature_names: List of feature names corresponding to feature_row columns.
        top_k: Number of most influential drivers to return.

    Returns:
        Tuple of (base_rate_probability, list of top_k SHAPDriver models).
    """
    import xgboost as xgb

    raw_booster = booster.get_booster() if hasattr(booster, "get_booster") else booster

    # Ensure 2D float32 design matrix
    X = np.asarray(feature_row, dtype=np.float32).reshape(1, -1)
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    # Native TreeSHAP call: returns shape (1, n_features + 1)
    # The last element is the expected log-odds bias (base value)
    contribs = raw_booster.predict(dmatrix, pred_contribs=True)[0]

    feature_contribs = contribs[: len(feature_names)]
    base_logit = float(contribs[-1])
    base_rate_prob = 1.0 / (1.0 + float(np.exp(-base_logit)))

    # Sort feature indices by descending absolute contribution magnitude
    ranked_indices = sorted(
        range(len(feature_names)),
        key=lambda i: abs(feature_contribs[i]),
        reverse=True,
    )

    drivers: List[SHAPDriver] = []
    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        feat = feature_names[idx]
        val = float(X[0, idx]) if not np.isnan(X[0, idx]) else None
        phi = float(feature_contribs[idx])
        direction: Literal["RISK_INCREASING", "RISK_DECREASING", "NEUTRAL"]
        if phi > 0.0:
            direction = "RISK_INCREASING"
        elif phi < 0.0:
            direction = "RISK_DECREASING"
        else:
            direction = "NEUTRAL"

        translation_map = FEATURE_POLICY_TRANSLATION.get(feat, {})
        interpretation = translation_map.get(
            direction,
            (
                f"Feature {feat} had neutral impact (0.000 log-odds) on schedule risk."
                if direction == "NEUTRAL"
                else f"Feature {feat} contributed {phi:+.3f} log-odds to schedule risk."
            ),
        )

        drivers.append(
            SHAPDriver(
                feature_name=feat,
                feature_value=val,
                shap_value=round(phi, 4),
                direction=direction,
                rank=rank,
                administrative_interpretation=interpretation,
            )
        )

    return round(base_rate_prob, 4), drivers
