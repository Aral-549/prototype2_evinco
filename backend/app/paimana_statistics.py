"""
MoSPI PAIMANA - Statistical Testing Protocol (Spec Sections 2.5 and 2.7)

DeLong's paired AUC comparison (DeLong, DeLong & Clarke-Pearson, Biometrics
1988) for MoSPI Dimension (b), plus the independent continuous cost-overrun
evaluation metrics (RMSE / MAE / R^2 / Quantile Pinball Loss).

Diebold-Mariano is deliberately NOT implemented: it requires an additive
point-wise loss differential on a univariate sequential time series
(spec Section 2.6), none of which holds for a paired cross-sectional
panel of ~1,870 projects evaluated under a combinatorial rank-concordance
metric (AUC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class DeLongResult:
    """Result container for DeLong's paired AUC significance test."""
    auc_1: float
    auc_2: float
    var_1: float
    var_2: float
    cov_12: float
    z_statistic: float
    p_value: float
    n_positives: int
    n_negatives: int
    test_name: str = "DeLong Paired AUC Test (DeLong et al., 1988)"


def _auc_and_placement_values(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the empirical AUC (Mann-Whitney kernel psi) and the structural
    placement value vectors V10 (per positive case) and V01 (per control).

    psi(x, y) = 1 if x > y, 0.5 if x == y, 0 if x < y
    """
    X = scores[y_true == 1]  # cases (m)
    Y = scores[y_true == 0]  # controls (n)
    m, n = len(X), len(Y)
    if m == 0 or n == 0:
        raise ValueError("DeLong test requires at least one positive and one negative sample.")

    # Pairwise comparison matrix (m x n)
    diff = X[:, None] - Y[None, :]
    psi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))

    auc = float(psi.mean())
    v10 = psi.mean(axis=1)  # E_Y[psi(X_i, .)] per positive case i
    v01 = psi.mean(axis=0)  # E_X[psi(., Y_j)] per control j
    return auc, v10, v01


def delong_test_paired_auc(
    y_true: np.ndarray,
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    alpha: float = 0.05,
) -> DeLongResult:
    """
    DeLong's test for two correlated (paired) ROC curves evaluated on the
    same cohort (spec Section 2.5).

    H0: theta_1 - theta_2 = 0
    Z = (AUC1 - AUC2) / sqrt(V1 + V2 - 2*Cov12)  ~  N(0, 1) under H0
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores_1 = np.asarray(scores_1, dtype=np.float64)
    scores_2 = np.asarray(scores_2, dtype=np.float64)

    if not (y_true.shape == scores_1.shape == scores_2.shape):
        raise ValueError("y_true and both score vectors must share identical shapes.")

    auc_1, v10_1, v01_1 = _auc_and_placement_values(y_true, scores_1)
    auc_2, v10_2, v01_2 = _auc_and_placement_values(y_true, scores_2)

    m = int((y_true == 1).sum())
    n = int((y_true == 0).sum())

    # Covariance components S10 (over cases) and S01 (over controls).
    # With fewer than two cases or controls, the ddof=1 variance is undefined;
    # those components contribute zero and the test degenerates gracefully.
    s10_11 = float(np.var(v10_1, ddof=1)) if m > 1 else 0.0
    s10_22 = float(np.var(v10_2, ddof=1)) if m > 1 else 0.0
    s10_12 = float(np.cov(v10_1, v10_2, ddof=1)[0, 1]) if m > 1 else 0.0
    s01_11 = float(np.var(v01_1, ddof=1)) if n > 1 else 0.0
    s01_22 = float(np.var(v01_2, ddof=1)) if n > 1 else 0.0
    s01_12 = float(np.cov(v01_1, v01_2, ddof=1)[0, 1]) if n > 1 else 0.0

    var_1 = s10_11 / m + s01_11 / n
    var_2 = s10_22 / m + s01_22 / n
    cov_12 = s10_12 / m + s01_12 / n

    sigma2_delta = var_1 + var_2 - 2.0 * cov_12
    delta = auc_1 - auc_2
    if sigma2_delta <= 0.0:
        # Identical score vectors (or degenerate variance): Z is undefined;
        # a zero-variance difference means identical discriminative placement.
        z = 0.0 if delta == 0.0 else float(np.sign(delta) * np.inf)
        p = 1.0 if delta == 0.0 else 0.0
    else:
        z = float(delta / np.sqrt(sigma2_delta))
        p = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(z))))

    return DeLongResult(
        auc_1=auc_1,
        auc_2=auc_2,
        var_1=var_1,
        var_2=var_2,
        cov_12=cov_12,
        z_statistic=z,
        p_value=p,
        n_positives=m,
        n_negatives=n,
    )


# ---------------------------------------------------------------------------
# Continuous Cost Overrun Evaluation Protocol (Spec Section 2.7)
# ---------------------------------------------------------------------------

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def quantile_pinball_loss(
    y_true: np.ndarray,
    y_quantile_pred: np.ndarray,
    tau: float,
) -> float:
    """
    Pinball loss for quantile regression at level tau:
        L_tau(y, q) = mean( max( tau*(y - q), (tau - 1)*(y - q) ) )
    """
    if not (0.0 < tau < 1.0):
        raise ValueError("Pinball loss requires quantile level tau in (0, 1).")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_quantile_pred = np.asarray(y_quantile_pred, dtype=np.float64)
    diff = y_true - y_quantile_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def continuous_cost_overrun_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile_predictions: Optional[Dict[float, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Full evaluation bundle for a continuous cost-escalation regressor.
    Quantile pinball losses at tau in {0.10, 0.50, 0.90} are appended when
    corresponding quantile forecasts are supplied.
    """
    report: Dict[str, float] = {
        "rmse_pct": root_mean_squared_error(y_true, y_pred),
        "mae_pct": mean_absolute_error(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
    }
    if quantile_predictions:
        for tau, preds in quantile_predictions.items():
            report[f"pinball_loss_tau_{tau:.2f}"] = quantile_pinball_loss(y_true, preds, tau)
    return report
