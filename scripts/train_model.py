#!/usr/bin/env python3
"""Fit, calibrate and honestly evaluate the PAIMANA early-warning classifier.

Implements contracts/model_training.md.

Every performance number the API reports must originate from this script's
`model/paimana_model_metrics.json` artifact. Hardcoded performance constants in
the serving layer are a contract violation.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --panel data/paimana_panel.csv --seed 20260917
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paimana.train")

ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "paimana_panel.csv"
PANEL_META_PATH = ROOT / "data" / "paimana_panel_meta.json"
MODEL_DIR = ROOT / "model"
BUNDLE_PATH = MODEL_DIR / "paimana_schedule_risk_v2.pkl"
METRICS_PATH = MODEL_DIR / "paimana_model_metrics.json"
OOF_PATH = MODEL_DIR / "paimana_oof_predictions.csv"

DEFAULT_SEED = 20260917
MIN_PANEL_ROWS = 200
MODEL_VERSION = "2.0.0"

# Contract: excluded from the design matrix entirely, not zeroed at inference.
# Zeroing a feature the booster was trained to split on produces train/serve
# skew -- the v1 bundle did exactly that with `has_revised_doc` (34% of gain)
# and collapsed to p_model < 0.035 on every one of the 2,155 live projects.
LEAKAGE_EXCLUDED_FEATURES: List[str] = [
    "has_revised_doc",
    "current_delay_months",
    "current_delay_months_robust",
    "revised_date_next",
    "slip_next",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Binned |confidence - accuracy| gap, weighted by bin mass.

    Reported because Capital-at-Risk multiplies this probability by rupees: a
    miscalibrated probability is a miscalibrated rupee figure.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def score_report(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Cannot score a cohort containing a single class.")
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "n": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
    }


# ---------------------------------------------------------------------------
# Baseline: discrete-time proportional hazards (complementary log-log)
# ---------------------------------------------------------------------------

class DiscreteTimePH:
    """Prentice-Gloeckler (1978) discrete-time proportional hazards model.

        h(t | x) = 1 - exp(-exp(alpha(t) + x'beta))

    This is the correct classical survival specification for a monthly panel
    with heavy ties, where continuous-time Cox's partial likelihood degrades.
    The complementary log-log link makes it the exact discrete analogue of the
    Cox proportional-hazards assumption, so it is a fair "conventional
    statistics" comparator for MoSPI Dimension (b).

    Baseline hazard alpha(t) is modelled as a linear spline in elapsed_ratio
    rather than a free parameter per period, which keeps it identifiable on
    projects with short observation windows.
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 500):
        self.l2 = l2
        self.max_iter = max_iter
        self.beta_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    @staticmethod
    def _cloglog(eta: np.ndarray) -> np.ndarray:
        # Clip eta so exp(eta) cannot overflow before the outer exp.
        return 1.0 - np.exp(-np.exp(np.clip(eta, -30.0, 30.0)))

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean_) / self.scale_
        return np.hstack([np.ones((Z.shape[0], 1)), Z])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiscreteTimePH":
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        D = self._design(X)

        def nll(params: np.ndarray) -> float:
            h = np.clip(self._cloglog(D @ params), 1e-9, 1 - 1e-9)
            ll = np.sum(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))
            # L2 on slopes only; the intercept carries the baseline hazard.
            return float(-ll + self.l2 * np.sum(params[1:] ** 2))

        init = np.zeros(D.shape[1])
        init[0] = np.log(-np.log(1.0 - np.clip(y.mean(), 1e-4, 0.99)))
        res = minimize(nll, init, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.beta_ = res.x
        return self

    def predict_proba_1d(self, X: np.ndarray) -> np.ndarray:
        return self._cloglog(self._design(np.asarray(X, dtype=float)) @ self.beta_)


class DeadlineRuleBaseline:
    """The trivial rule: has the officially declared completion date passed?

    This baseline exists to answer the sharpest question a reviewer can ask:
    *"your strongest single feature is 'the deadline passed' -- that is a SQL
    query, not machine learning. Why do I need a gradient-boosted ensemble?"*

    Making the objection a first-class competitor is the only honest way to
    answer it. `months_to_revised_date` scores AUC ~0.77 on its own, so any
    claim the model makes has to be measured against this line, not against
    a straw-man prior.

    `binary=True` reproduces the flag exactly as a rules engine would fire it
    (deadline passed, work incomplete). `binary=False` ranks by how far past
    (or short of) the declared date the project is, which is the strongest
    form of the objection.
    """

    def __init__(self, binary: bool = True):
        self.binary = binary

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "DeadlineRuleBaseline":
        return self

    def predict_proba_1d(self, df: pd.DataFrame) -> np.ndarray:
        months_left = df["months_to_revised_date"].to_numpy(dtype=float)
        if self.binary:
            passed = (months_left <= 0) & (df["physical_progress"].to_numpy(dtype=float) < 95.0)
            return passed.astype(float)
        # Continuous form: more overdue => higher risk. Squashed to [0,1] so it
        # is scored on the same footing as a probability; AUC is rank-based so
        # the squashing is monotone and does not change discrimination.
        return 1.0 / (1.0 + np.exp(np.clip(months_left / 6.0, -30.0, 30.0)))


class RuleFloorBaseline:
    """The deterministic statutory rules the platform already ships, scored as
    a probability so they can be compared against the model on equal terms.

    Only F1 (spend/progress decoupling) and F2 (progress stall) are computable
    from the public PAIMANA feed. F3 (clearance pendency), F4 (reporting
    staleness) and F5 (litigation) require CUF fields the portal does not
    publish, so they are structurally inert here -- which is itself the
    argument for the CUF 2.0 schema proposal.

    This baseline answers the question a reviewer will ask first: does the ML
    add anything over the rules we already have?
    """

    COMPUTABLE_FLAGS = ["F1", "F2"]
    INERT_FLAGS = {
        "F3": "clearance_pending_days absent from public feed",
        "F4": "days_since_last_cuf_update absent from public feed",
        "F5": "litigation / dispute status absent from public feed",
    }
    MAX_SCORE = 55.0  # F1 (30) + F2 (25)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "RuleFloorBaseline":
        return self

    def predict_proba_1d(self, df: pd.DataFrame) -> np.ndarray:
        f1 = ((df["spend_progress_gap"] >= 25.0) & (df["physical_progress"] < 50.0)) * 30.0
        f2 = (
            (df["progress_delta"] <= 0.1)
            & (df["elapsed_ratio"] >= 0.20)
            & (df["physical_progress"] < 95.0)
        ) * 25.0
        return np.asarray((f1 + f2) / self.MAX_SCORE, dtype=float)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    key: str
    display_name: str
    model_class: str
    needs_frame: bool          # True if the estimator consumes the DataFrame
    build: Callable[[int, float], Any]
    fit: Callable[[Any, Any, np.ndarray], Any]
    predict: Callable[[Any, Any], np.ndarray]


def _xgb_build(seed: int, pos_weight: float):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=8,
        gamma=0.4,
        reg_lambda=2.0,
        reg_alpha=0.2,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )


def _logit_build(seed: int, pos_weight: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=0.3,
                    C=0.5,
                    solver="saga",
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


class PriorBaseline:
    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba_1d(self, X):
        n = len(X) if not hasattr(X, "shape") else X.shape[0]
        return np.full(n, self.p_, dtype=float)


def candidates(seed: int, pos_weight: float) -> List[Candidate]:
    return [
        Candidate(
            key="prior",
            display_name="Stratified Class Prior",
            model_class="Sanity Floor",
            needs_frame=False,
            build=lambda s, w: PriorBaseline(),
            fit=lambda m, X, y: m.fit(X, y),
            predict=lambda m, X: m.predict_proba_1d(X),
        ),
        Candidate(
            key="rulefloor",
            display_name="Deterministic RuleFloor (F1-F2, shipped rules)",
            model_class="Existing Rule Engine",
            needs_frame=True,
            build=lambda s, w: RuleFloorBaseline(),
            fit=lambda m, X, y: m.fit(X, y),
            predict=lambda m, X: m.predict_proba_1d(X),
        ),
        Candidate(
            key="deadline_binary",
            display_name="Deadline Rule (declared date passed, binary)",
            model_class="Trivial Heuristic",
            needs_frame=True,
            build=lambda s, w: DeadlineRuleBaseline(binary=True),
            fit=lambda m, X, y: m.fit(X, y),
            predict=lambda m, X: m.predict_proba_1d(X),
        ),
        Candidate(
            key="deadline_continuous",
            display_name="Deadline Proximity (months to declared date)",
            model_class="Trivial Heuristic",
            needs_frame=True,
            build=lambda s, w: DeadlineRuleBaseline(binary=False),
            fit=lambda m, X, y: m.fit(X, y),
            predict=lambda m, X: m.predict_proba_1d(X),
        ),
        Candidate(
            key="logit",
            display_name="Logistic Regression (ElasticNet)",
            model_class="Classical Econometric Baseline",
            needs_frame=False,
            build=_logit_build,
            fit=lambda m, X, y: m.fit(X, y),
            predict=lambda m, X: m.predict_proba(X)[:, 1],
        ),
        Candidate(
            key="dtph",
            display_name="Discrete-Time Proportional Hazards (cloglog)",
            model_class="Classical Survival Baseline",
            needs_frame=False,
            build=lambda s, w: DiscreteTimePH(l2=1.0),
            fit=lambda m, X, y: m.fit(X, y),
            predict=lambda m, X: m.predict_proba_1d(X),
        ),
        Candidate(
            key="xgboost",
            display_name="Stage-Aware XGBoost (Proposed)",
            model_class="Gradient Boosted Trees",
            needs_frame=False,
            build=_xgb_build,
            fit=lambda m, X, y: m.fit(X, y),
            predict=lambda m, X: m.predict_proba(X)[:, 1],
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def assert_no_leakage(feature_cols: List[str]) -> None:
    """Contract case #1: the exclusion list is enforced in code, not convention."""
    violations = sorted(set(feature_cols) & set(LEAKAGE_EXCLUDED_FEATURES))
    if violations:
        raise ValueError(
            f"Leakage policy violation: {violations} present in the design matrix. "
            "These fields are the label's own administrative footprint and must be "
            "excluded from training, not zeroed at inference."
        )


def impute_median(train: pd.DataFrame, other: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Median imputation fitted on the training fold only."""
    med = train.median(numeric_only=True)
    return other.fillna(med), med


def run_grouped_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    n_splits: int = 5,
    label_col: str = "slip_next",
) -> Dict[str, np.ndarray]:
    """Split A: GroupKFold by project_id. Returns out-of-fold predictions."""
    from sklearn.model_selection import GroupKFold

    y = df[label_col].to_numpy(dtype=int)
    groups = df["project_id"].to_numpy()
    oof: Dict[str, np.ndarray] = {c.key: np.zeros(len(df)) for c in candidates(seed, 1.0)}

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(df, y, groups), start=1):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        y_tr = y[tr]
        pos_weight = float((len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))

        X_tr_raw, X_te_raw = tr_df[feature_cols], te_df[feature_cols]
        X_tr, med = impute_median(X_tr_raw, X_tr_raw)
        X_te = X_te_raw.fillna(med)

        for cand in candidates(seed, pos_weight):
            model = cand.build(seed, pos_weight)
            if cand.needs_frame:
                cand.fit(model, tr_df, y_tr)
                oof[cand.key][te] = cand.predict(model, te_df)
            else:
                cand.fit(model, X_tr.to_numpy(), y_tr)
                oof[cand.key][te] = cand.predict(model, X_te.to_numpy())
        logger.info("  GroupKFold fold %d/%d complete (%d test rows)", fold, n_splits, len(te))

    return oof


def run_last_transition_holdout(
    df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    label_col: str = "slip_next",
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Split B: hold out each project's most recent transition."""
    last_idx = df.groupby("project_id")["seq_index"].transform("max")
    is_test = df["seq_index"] == last_idx
    tr_df, te_df = df[~is_test], df[is_test]

    y_tr = tr_df[label_col].to_numpy(dtype=int)
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))

    X_tr_raw = tr_df[feature_cols]
    X_tr, med = impute_median(X_tr_raw, X_tr_raw)
    X_te = te_df[feature_cols].fillna(med)

    preds: Dict[str, np.ndarray] = {}
    for cand in candidates(seed, pos_weight):
        model = cand.build(seed, pos_weight)
        if cand.needs_frame:
            cand.fit(model, tr_df, y_tr)
            preds[cand.key] = cand.predict(model, te_df)
        else:
            cand.fit(model, X_tr.to_numpy(), y_tr)
            preds[cand.key] = cand.predict(model, X_te.to_numpy())
    return te_df, preds


def run_out_of_time(
    df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    label_col: str,
    test_months: int = 3,
) -> "str | Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]":
    """Split C: train on the earlier calendar months, test on the latest ones.

    This is the split a forecasting system should really be judged on, and it
    was **deliberately rejected** while the panel's time axis was reconstructed:
    because each project's newest snapshot was stamped as the harvest month,
    history length determined how far back a project's rows reached, so a
    calendar split measured that artifact rather than forecasting skill.

    With the portal's own freeze months now attached to every record
    (`scripts/harvest_monthly.py`), the axis is observed and the split is
    finally meaningful. It is still not a free lunch: the portal onboarded
    projects over the window, so later months contain more projects than
    earlier ones. That is a real property of the programme rather than an
    artifact of our reconstruction, and it is reported alongside the result.

    Returns a Tuple on success, or a string naming the exact reason it could
    not be run. It never returns a bare None: a split that silently vanishes
    with a misleading status is how a limitation gets mistaken for a result.
    """
    if "time_axis" not in df.columns or not (df["time_axis"] == "observed").all():
        return "unavailable_reconstructed_axis"

    months = sorted(df["as_of_month"].unique())
    if len(months) < test_months + 3:
        return f"unavailable_only_{len(months)}_months_in_panel"
    cutoff = months[-test_months]
    train_mask = df["as_of_month"] < cutoff
    test_mask = ~train_mask

    tr_df, te_df = df[train_mask], df[test_mask]
    y_tr = tr_df[label_col].to_numpy(dtype=int)
    y_te = te_df[label_col].to_numpy(dtype=int)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        # Degenerate cohort. At long horizons this is structural rather than a
        # fluke: a recent row can only SURVIVE censoring if a slip was already
        # observed inside its window, so every uncensored recent row is a
        # positive and AUC is undefined. Say exactly that instead of blaming
        # the time axis.
        which = "train" if len(np.unique(y_tr)) < 2 else "test"
        rate = float(y_te.mean()) if which == "test" else float(y_tr.mean())
        return (
            f"unavailable_single_class_{which}_cohort (positive rate {rate:.3f}); at this "
            f"horizon a recent row survives censoring only if it already slipped, so the "
            f"out-of-time cohort is all-positive and AUC is undefined"
        )

    pos_weight = float((len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
    X_tr_raw = tr_df[feature_cols]
    X_tr, med = impute_median(X_tr_raw, X_tr_raw)
    X_te = te_df[feature_cols].fillna(med)

    preds: Dict[str, np.ndarray] = {}
    for cand in candidates(seed, pos_weight):
        model = cand.build(seed, pos_weight)
        if cand.needs_frame:
            cand.fit(model, tr_df, y_tr)
            preds[cand.key] = cand.predict(model, te_df)
        else:
            cand.fit(model, X_tr.to_numpy(), y_tr)
            preds[cand.key] = cand.predict(model, X_te.to_numpy())

    info = {
        "train_months": [m for m in months if m < cutoff],
        "test_months": [m for m in months if m >= cutoff],
        "cutoff": cutoff,
        "n_train": int(len(tr_df)),
        "n_test": int(len(te_df)),
        "train_projects": int(tr_df["project_id"].nunique()),
        "test_projects": int(te_df["project_id"].nunique()),
        "train_positive_rate": round(float(y_tr.mean()), 4),
        "test_positive_rate": round(float(y_te.mean()), 4),
        "caveat": (
            "The portal onboarded projects across the window, so later months contain more "
            "projects than earlier ones. This is a real property of the monitoring programme, "
            "not an artifact of the panel; positive rates also move month to month (a mass "
            "revision wave is visible in 2026-02), so train and test base rates differ."
        ),
    }
    return te_df, preds, info


def delong_vs_reference(
    y: np.ndarray,
    preds: Dict[str, np.ndarray],
    reference: str = "xgboost",
) -> List[Dict[str, Any]]:
    """Paired DeLong AUC tests of every baseline against the proposed model."""
    import sys

    sys.path.insert(0, str(ROOT / "backend"))
    from app.paimana_statistics import delong_test_paired_auc

    rows: List[Dict[str, Any]] = []
    for key, scores in preds.items():
        if key == reference:
            continue
        # Degenerate constant predictors have zero AUC variance; DeLong's Z is
        # undefined there, so report the AUCs and say so rather than emit NaN.
        if np.allclose(scores, scores[0]):
            rows.append(
                {
                    "comparison": f"{reference} vs {key}",
                    "status": "z_undefined_constant_predictor",
                    "note": "Baseline emits a constant score; DeLong variance is zero.",
                }
            )
            continue
        res = delong_test_paired_auc(y, preds[reference].astype(float), scores.astype(float))
        rows.append(
            {
                "comparison": f"{reference} vs {key}",
                "auc_proposed": round(res.auc_1, 4),
                "auc_baseline": round(res.auc_2, 4),
                "auc_delta": round(res.auc_1 - res.auc_2, 4),
                "z_statistic": round(res.z_statistic, 4),
                "p_value": float(f"{res.p_value:.6g}"),
                "significant_at_0.05": bool(res.p_value < 0.05),
                "n_positives": res.n_positives,
                "n_negatives": res.n_negatives,
            }
        )
    return rows


def stratified_confound_check(
    df: pd.DataFrame, p: np.ndarray, y: np.ndarray
) -> List[Dict[str, Any]]:
    """Contract guard: does discrimination survive within history-length strata?

    History length correlates strongly with the label on this panel (short
    histories ~33% positive, long histories ~5-16%). If AUC only exists across
    strata and vanishes within them, the headline number was reading that
    artifact rather than project dynamics.
    """
    from sklearn.metrics import roc_auc_score

    buckets = [(2, 5), (6, 8), (9, 11), (12, 20)]
    out: List[Dict[str, Any]] = []
    for lo, hi in buckets:
        mask = (df["n_snapshots"] >= lo) & (df["n_snapshots"] <= hi)
        yy, pp = y[mask.to_numpy()], p[mask.to_numpy()]
        row: Dict[str, Any] = {
            "history_length": f"{lo}-{hi} snapshots",
            "n": int(mask.sum()),
            "positive_rate": round(float(yy.mean()), 4) if len(yy) else None,
        }
        row["auc"] = (
            round(float(roc_auc_score(yy, pp)), 4) if len(np.unique(yy)) == 2 else None
        )
        out.append(row)
    return out


def _new_calibrator(method: str, seed: int):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    return LogisticRegression(C=1e6, solver="lbfgs", random_state=seed)


def _cal_fit(cal, p: np.ndarray, y: np.ndarray, method: str):
    cal.fit(p, y) if method == "isotonic" else cal.fit(p.reshape(-1, 1), y)
    return cal


def _cal_apply(cal, p: np.ndarray, method: str) -> np.ndarray:
    if method == "isotonic":
        return np.asarray(cal.predict(p), dtype=float)
    return np.asarray(cal.predict_proba(p.reshape(-1, 1))[:, 1], dtype=float)


def actionable_cohort_analysis(
    df: pd.DataFrame,
    oof: Dict[str, np.ndarray],
    y: np.ndarray,
) -> Dict[str, Any]:
    """Does the model beat the deadline rule where early warning is still possible?

    This is the platform's most important self-check, and it exists to answer
    one objection directly: `months_to_revised_date` scores ~0.77 AUC alone, so
    a reviewer can fairly ask whether the model is anything more than "the
    deadline passed" dressed up in 400 trees.

    Splitting the panel by deadline status separates two different questions:

    * **Already overdue** — the declared date has passed. A revision is
      statutorily inevitable; the rule already knows this and there is little
      for a model to add. Flagging these is bookkeeping, not forecasting.
    * **Not yet overdue** — the declared date is still in the future. This is
      the *actionable* cohort: the only projects where a warning can still
      change an outcome. If the model's advantage survives here, it is reading
      execution dynamics. If it collapses here, the headline AUC was carried by
      projects that were already lost, and the honest conclusion is that a rule
      would do.

    The deeper strata (>3, >6 months of declared runway) are progressively
    harder and progressively more valuable.
    """
    from sklearn.metrics import roc_auc_score

    months_left = df["months_to_revised_date"].to_numpy(dtype=float)
    cohorts = [
        ("already_overdue", months_left <= 0,
         "Declared completion date has passed; a revision is effectively inevitable."),
        ("not_yet_overdue", months_left > 0,
         "ACTIONABLE COHORT: the declared date is still in the future, so a warning can still change the outcome."),
        ("runway_over_3_months", months_left > 3,
         "At least a quarter of declared runway remains."),
        ("runway_over_6_months", months_left > 6,
         "At least two quarters of declared runway remain; hardest and most valuable."),
    ]

    compared = ["xgboost", "logit", "dtph", "deadline_continuous", "deadline_binary", "rulefloor"]
    rows: List[Dict[str, Any]] = []
    for name, mask, description in cohorts:
        n = int(mask.sum())
        entry: Dict[str, Any] = {
            "cohort": name,
            "description": description,
            "n": n,
            "positive_rate": round(float(y[mask].mean()), 4) if n else None,
        }
        if n < 50 or len(np.unique(y[mask])) < 2:
            entry["status"] = "insufficient_or_single_class"
            rows.append(entry)
            continue
        for key in compared:
            if key not in oof:
                continue
            scores = oof[key][mask]
            # A constant predictor inside a cohort has no rank information.
            entry[f"auc_{key}"] = (
                round(float(roc_auc_score(y[mask], scores)), 4)
                if not np.allclose(scores, scores[0])
                else None
            )
        if entry.get("auc_xgboost") is not None and entry.get("auc_deadline_continuous") is not None:
            entry["xgboost_minus_deadline_rule"] = round(
                entry["auc_xgboost"] - entry["auc_deadline_continuous"], 4
            )
        rows.append(entry)

    verdict = _cohort_verdict(rows)
    return {
        "question": (
            "Is the model more than a deadline rule? Measured where early warning is "
            "still actionable, i.e. the declared completion date has not yet passed."
        ),
        "cohorts": rows,
        "verdict": verdict,
    }


def _cohort_verdict(rows: List[Dict[str, Any]]) -> str:
    """State plainly what the cohort table implies, including a bad answer."""
    actionable = next((r for r in rows if r["cohort"] == "not_yet_overdue"), None)
    if not actionable or actionable.get("auc_xgboost") is None:
        return "Not evaluable: the actionable cohort was too small or single-class."

    model = actionable["auc_xgboost"]
    rule = actionable.get("auc_deadline_continuous")
    if rule is None:
        return f"Model AUC {model} on the actionable cohort; deadline rule was constant there."

    delta = round(model - rule, 4)
    if delta >= 0.05:
        return (
            f"Model beats the deadline rule by {delta} AUC ({model} vs {rule}) on the "
            f"{actionable['n']} project-months where the declared date has NOT yet passed. "
            "The model is reading execution dynamics, not just restating the calendar."
        )
    if delta > 0.0:
        return (
            f"Model beats the deadline rule by only {delta} AUC ({model} vs {rule}) on the "
            "actionable cohort. The advantage is real but small; a rules engine would "
            "capture most of the available value here."
        )
    return (
        f"Model does NOT beat the deadline rule on the actionable cohort "
        f"({model} vs {rule}). On this data the headline AUC is carried by projects that "
        "are already overdue, and a deterministic rule would serve as well. Reported "
        "as measured."
    )


def fit_calibrator(
    y: np.ndarray,
    p: np.ndarray,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Fit isotonic (or sigmoid when positives are scarce) on out-of-fold scores.

    The post-calibration ECE is measured **cross-fitted**: the calibrator is fit
    on 4 folds and scored on the 5th, grouped by project so a project never
    calibrates itself. Isotonic regression is flexible enough to drive in-sample
    ECE to exactly 0.0, which would be a meaningless number to report.

    Contract case #6: a calibrator that makes held-out ECE worse is discarded.
    """
    from sklearn.model_selection import GroupKFold, KFold

    pre = expected_calibration_error(y, p)
    n_pos = int(y.sum())
    method = "isotonic" if n_pos >= 500 else "sigmoid"

    # Cross-fitted estimate of what calibration buys on unseen rows.
    held_out = np.zeros_like(p, dtype=float)
    if groups is not None:
        splitter = GroupKFold(n_splits=5).split(p.reshape(-1, 1), y, groups)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed).split(p.reshape(-1, 1))
    for tr, te in splitter:
        cal_fold = _cal_fit(_new_calibrator(method, seed), p[tr], y[tr], method)
        held_out[te] = _cal_apply(cal_fold, p[te], method)

    post = expected_calibration_error(y, held_out)
    report = {
        "method": method,
        "ece_before": round(pre, 4),
        "ece_after_cross_fitted": round(post, 4),
        "estimation": (
            "Post-calibration ECE is cross-fitted over 5 project-grouped folds. "
            "An in-sample isotonic ECE would read ~0.0 and mean nothing."
        ),
        "kept": bool(post <= pre),
    }
    if post > pre:
        logger.warning(
            "Calibrator worsened held-out ECE (%.4f -> %.4f); discarding per contract.",
            pre,
            post,
        )
        return None, report

    # Ship a calibrator refit on all out-of-fold scores.
    final = _cal_fit(_new_calibrator(method, seed), p, y, method)
    return {"calibrator": final, "method": method}, report


def pick_threshold(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    """Operating point maximising F1 on out-of-fold scores, with its curve."""
    from sklearn.metrics import precision_recall_fscore_support

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    curve: List[Dict[str, Any]] = []
    best = {"threshold": 0.5, "f1": -1.0}
    for t in grid:
        pred = (p >= t).astype(int)
        if pred.sum() == 0:
            continue
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        entry = {
            "threshold": float(t),
            "precision": round(float(pr), 4),
            "recall": round(float(rc), 4),
            "f1": round(float(f1), 4),
            "flagged_share": round(float(pred.mean()), 4),
        }
        curve.append(entry)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": round(float(f1), 4), **entry}
    return {"recommended": best, "curve": curve}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_horizon(
    df_all_reliable: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    horizon_months: int,
    seed: int,
    folds: int,
) -> Dict[str, Any]:
    """Full evaluation + production fit for one forecast horizon.

    Rows censored at this horizon (window shorter than the horizon, no slip
    observed inside it) carry a null label and are dropped here. They are never
    coerced to 0: that would manufacture negatives out of missing follow-up and
    inflate every metric on this table.
    """
    df = df_all_reliable[df_all_reliable[label_col].notna()].reset_index(drop=True)
    n_censored = len(df_all_reliable) - len(df)

    if len(df) < MIN_PANEL_ROWS:
        return {
            "horizon_months": horizon_months,
            "status": "insufficient_rows",
            "rows_available": int(len(df)),
        }

    y = df[label_col].to_numpy(dtype=int)
    logger.info(
        "── Horizon %dm (%s): %d labelled rows, %d censored dropped, positive rate %.2f%%",
        horizon_months,
        label_col,
        len(df),
        n_censored,
        100 * y.mean(),
    )

    # Split A -- GroupKFold by project
    oof = run_grouped_cv(df, feature_cols, seed, folds, label_col=label_col)
    cv_rows: List[Dict[str, Any]] = []
    for cand in candidates(seed, 1.0):
        try:
            rep = score_report(y, oof[cand.key])
        except ValueError as exc:
            cv_rows.append({"key": cand.key, "status": f"unscoreable: {exc}"})
            continue
        cv_rows.append(
            {
                "key": cand.key,
                "model_architecture": cand.display_name,
                "model_class": cand.model_class,
                **rep,
            }
        )
        logger.info(
            "   %-52s AUC=%.4f PR-AUC=%.4f Brier=%.4f",
            cand.display_name,
            rep["auc"],
            rep["pr_auc"],
            rep["brier"],
        )
    cv_delong = delong_vs_reference(y, oof, reference="xgboost")

    # Split B -- last transition per project
    te_df, holdout_preds = run_last_transition_holdout(
        df, feature_cols, seed, label_col=label_col
    )
    y_te = te_df[label_col].to_numpy(dtype=int)
    holdout_rows: List[Dict[str, Any]] = []
    for cand in candidates(seed, 1.0):
        try:
            rep = score_report(y_te, holdout_preds[cand.key])
        except ValueError as exc:
            holdout_rows.append({"key": cand.key, "status": f"unscoreable: {exc}"})
            continue
        holdout_rows.append(
            {
                "key": cand.key,
                "model_architecture": cand.display_name,
                "model_class": cand.model_class,
                **rep,
            }
        )
    holdout_delong = delong_vs_reference(y_te, holdout_preds, reference="xgboost")

    # Split C -- genuine calendar out-of-time (observed axis only)
    oot = run_out_of_time(df, feature_cols, seed, label_col)
    oot_block: Dict[str, Any] = (
        {"status": oot} if isinstance(oot, str) else {"status": "pending"}
    )
    if not isinstance(oot, str):
        oot_te, oot_preds, oot_info = oot
        y_oot = oot_te[label_col].to_numpy(dtype=int)
        oot_rows: List[Dict[str, Any]] = []
        for cand in candidates(seed, 1.0):
            try:
                rep = score_report(y_oot, oot_preds[cand.key])
            except ValueError as exc:
                oot_rows.append({"key": cand.key, "status": f"unscoreable: {exc}"})
                continue
            oot_rows.append({
                "key": cand.key,
                "model_architecture": cand.display_name,
                "model_class": cand.model_class,
                **rep,
            })
        oot_block = {
            "status": "ok",
            "name": f"Calendar out-of-time (train < {oot_info['cutoff']}, test >=)",
            "question": "Trained on the past, does it forecast months it has never seen?",
            **oot_info,
            "results": oot_rows,
            "delong_tests": delong_vs_reference(y_oot, oot_preds, reference="xgboost"),
        }
        xgb_oot = next((r for r in oot_rows if r.get("key") == "xgboost"), None)
        if xgb_oot:
            logger.info(
                "   out-of-time (train<%s): XGBoost AUC=%.4f on %d rows / %d projects",
                oot_info["cutoff"], xgb_oot["auc"], oot_info["n_test"], oot_info["test_projects"],
            )

    confound = stratified_confound_check(df, oof["xgboost"], y)
    cohorts = actionable_cohort_analysis(df, oof, y)
    logger.info("   actionable-cohort verdict: %s", cohorts["verdict"])
    calibrator, calib_report = fit_calibrator(
        y, oof["xgboost"], seed, groups=df["project_id"].to_numpy()
    )
    logger.info(
        "   calibration (%s): ECE %.4f -> %.4f cross-fitted (%s)",
        calib_report["method"],
        calib_report["ece_before"],
        calib_report["ece_after_cross_fitted"],
        "kept" if calib_report["kept"] else "discarded",
    )

    threshold_info = pick_threshold(y, oof["xgboost"])
    logger.info(
        "   operating point %.2f: precision %.3f recall %.3f (flags %.1f%%)",
        threshold_info["recommended"]["threshold"],
        threshold_info["recommended"]["precision"],
        threshold_info["recommended"]["recall"],
        100 * threshold_info["recommended"]["flagged_share"],
    )

    # Production fit on every labelled row for this horizon
    X_raw = df[feature_cols]
    med = X_raw.median(numeric_only=True)
    X = X_raw.fillna(med).to_numpy()
    pos_weight = float((len(y) - y.sum()) / max(1, y.sum()))
    model = _xgb_build(seed, pos_weight)
    model.fit(X, y)

    importances = sorted(
        (
            {"feature": f, "gain_share": round(float(v), 4)}
            for f, v in zip(feature_cols, model.feature_importances_)
        ),
        key=lambda d: -d["gain_share"],
    )

    return {
        "horizon_months": horizon_months,
        "label": label_col,
        "status": "ok",
        "rows_labelled": int(len(df)),
        "rows_censored_dropped": int(n_censored),
        "positive_rate": round(float(y.mean()), 4),
        "censoring_selection_caveat": (
            f"{n_censored} rows ({100 * n_censored / max(1, len(df_all_reliable)):.1f}%) are "
            f"right-censored at this horizon and dropped. Censoring is informative, not "
            f"random: the dropped rows are systematically the *latest* transition of each "
            f"project, where fewer than {horizon_months} months of follow-up remain inside "
            f"the observation window. The surviving cohort therefore skews toward earlier "
            f"transitions of longer-history projects, and metrics at this horizon are "
            f"conditional on that cohort. The 1-month horizon has no censoring and is the "
            f"cleanest comparison."
            if n_censored
            else "No censoring at this horizon; every labelled transition is used."
        ),
        "splits": {
            "primary": {
                "name": f"GroupKFold(k={folds}) grouped by project_id",
                "question": "Can the model generalise to a project it has never seen?",
                "results": cv_rows,
                "delong_tests": cv_delong,
            },
            "out_of_time": oot_block,
            "deployment": {
                "name": "Last-transition-per-project holdout",
                "question": "For projects already monitored, can we call the next window?",
                "results": holdout_rows,
                "delong_tests": holdout_delong,
            },
        },
        "confound_guard": {
            "description": (
                "AUC within history-length strata. Discrimination that survives "
                "within strata is project dynamics, not the history-length artifact."
            ),
            "strata": confound,
        },
        "actionable_cohort": cohorts,
        "calibration": calib_report,
        "operating_point": threshold_info,
        "feature_importance": importances,
        "_artifacts": {
            "model": model,
            "calibrator": calibrator,
            "medians": med,
            "oof": oof["xgboost"],
            "oof_all": oof,
            "index": df[["project_id", "seq_index"]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PAIMANA early-warning model.")
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    if not args.panel.exists():
        raise SystemExit(f"Panel not found at {args.panel}. Run scripts/build_panel.py first.")

    np.random.seed(args.seed)
    panel_meta = json.loads(PANEL_META_PATH.read_text()) if PANEL_META_PATH.exists() else {}
    feature_cols: List[str] = panel_meta.get("feature_columns") or []
    if not feature_cols:
        raise SystemExit("Panel meta is missing feature_columns; rebuild the panel.")

    assert_no_leakage(feature_cols)

    df_all = pd.read_csv(args.panel)
    df = df_all[df_all["order_reliable"] == 1].reset_index(drop=True)
    if len(df) < MIN_PANEL_ROWS:
        raise SystemExit(
            f"Panel has only {len(df)} usable rows (minimum {MIN_PANEL_ROWS}). "
            "Refusing to emit a metrics artifact that would look authoritative."
        )

    logger.info(
        "Panel: %d transitions, %d projects, %d features",
        len(df),
        df["project_id"].nunique(),
        len(feature_cols),
    )

    # Horizon 1 is `slip_next`; longer horizons come from the panel meta.
    horizon_specs: List[Tuple[int, str]] = [(1, "slip_next")]
    for k in panel_meta.get("horizon_months", []):
        col = f"slip_within_{k}m"
        if col in df.columns:
            horizon_specs.append((k, col))

    horizon_results: Dict[str, Any] = {}
    for months, col in horizon_specs:
        horizon_results[f"{months}m"] = evaluate_horizon(
            df, feature_cols, col, months, args.seed, args.folds
        )

    primary = horizon_results["1m"]
    if primary.get("status") != "ok":
        raise SystemExit("Primary 1-month horizon failed to train; aborting.")

    # Out-of-fold scores for the 1-month model drive the lead-time backtest.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    oof_frame = df[
        [
            "project_id",
            "project_name",
            "sector",
            "seq_index",
            "n_snapshots",
            "as_of_month",
            "revised_date_t",
            "revised_date_next",
            "slip_next",
        ]
    ].copy()
    for _key, _scores in primary["_artifacts"]["oof_all"].items():
        oof_frame[f"p_{_key}"] = np.round(_scores, 6)
    oof_frame.to_csv(OOF_PATH, index=False)
    logger.info("Out-of-fold predictions written to %s", OOF_PATH)

    import joblib

    bundle = {
        "models": {
            key: res["_artifacts"]["model"]
            for key, res in horizon_results.items()
            if res.get("status") == "ok"
        },
        "calibrators": {
            key: res["_artifacts"]["calibrator"]
            for key, res in horizon_results.items()
            if res.get("status") == "ok"
        },
        "imputer_medians": primary["_artifacts"]["medians"].to_dict(),
        "feature_columns": feature_cols,
        "sector_frequency_encoding": panel_meta.get("sector_frequency_encoding", {}),
        "primary_horizon": "1m",
        "horizons": [k for k, r in horizon_results.items() if r.get("status") == "ok"],
        "classification_threshold": primary["operating_point"]["recommended"]["threshold"],
        "risk_thresholds": {"high": 75.0, "medium": 50.0},
        "model_name": "PAIMANA Stage-Aware Schedule-Slip Early Warning",
        "model_version": MODEL_VERSION,
        "target": "slip_next",
        "target_description": (
            "Probability that the project's officially reported revised completion "
            "date is pushed further out within the stated horizon."
        ),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "panel_fingerprint": panel_meta.get("source_sha256", "unknown"),
        "class_prior": primary["positive_rate"],
        "leakage_excluded_features": LEAKAGE_EXCLUDED_FEATURES,
        "seed": args.seed,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    logger.info("Model bundle written to %s", BUNDLE_PATH)

    # Strip unserialisable estimator handles before writing the metrics file.
    for res in horizon_results.values():
        res.pop("_artifacts", None)

    metrics = {
        "provenance": "measured",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "panel": {
            "rows_total": int(len(df_all)),
            "rows_used": int(len(df)),
            "rows_excluded_unreliable_order": int(len(df_all) - len(df)),
            "projects": int(df["project_id"].nunique()),
            "source_sha256": panel_meta.get("source_sha256", "unknown"),
            "censoring_policy": panel_meta.get("censoring_policy", ""),
            "time_axis": panel_meta.get("time_axis", {}),
        },
        "labels": {
            "slip_next": "revised completion date pushed > 15 days at the next monthly report",
            "slip_within_3m": "revised completion date pushed > 15 days at any report in the next 3 months",
            "slip_within_6m": "revised completion date pushed > 15 days at any report in the next 6 months",
        },
        "leakage_policy": {
            "excluded_from_design_matrix": LEAKAGE_EXCLUDED_FEATURES,
            "rationale": (
                "Excluded entirely rather than zeroed at inference. The v1 bundle "
                "trained on has_revised_doc (34% of gain) and zeroed it at serve "
                "time, producing train/serve skew that collapsed p_model below "
                "0.035 on all 2,155 live projects."
            ),
            "permitted_history_feature": (
                "revised_date_already_slipped_months is retained: slippage already "
                "on the record at time t is observable before the next revision is "
                "filed, and is a legitimate early-warning signal."
            ),
        },
        "horizons": horizon_results,
        "calendar_split_status": (
            "ENABLED: the panel now carries the portal's own freeze months "
            "(scripts/harvest_monthly.py), so a calendar out-of-time split is meaningful and "
            "is reported per horizon under splits.out_of_time."
            if (df_all.get("time_axis") is not None and (df_all["time_axis"] == "observed").all())
            else "REJECTED: see rejected_split."
        ),
        "rejected_split": {
            "name": "Calendar out-of-time split (historical note)",
            "applies_to": "panels built from the legacy undated harvest only",
            "reason": (
                "Confounded on this panel. Assumption A2 stamps each project's newest "
                "snapshot as the harvest month, so history length determines how far "
                "back a project's rows reach. Short-history projects (~33% positive) "
                "populate only recent months while long-history projects (~5-16% "
                "positive) populate the early ones. A calendar split would measure "
                "that artifact, not forecasting skill."
            ),
        },
        "statistical_test": (
            "DeLong paired AUC test (DeLong, DeLong & Clarke-Pearson, Biometrics 1988). "
            "Diebold-Mariano is not applicable: AUC is a combinatorial rank-concordance "
            "metric over case-control pairs and does not decompose into an additive "
            "point-wise loss differential on a single time axis."
        ),
        "cox_continuous_time": _cox_status(),
        "reproduce": "python scripts/build_panel.py && python scripts/train_model.py && python scripts/lead_time_backtest.py",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("Metrics artifact written to %s", METRICS_PATH)


def _cox_status() -> Dict[str, str]:
    """Contract case #3: absent dependency is reported, never fabricated."""
    try:
        import lifelines  # noqa: F401
    except ImportError:
        return {
            "status": "unavailable",
            "note": (
                "lifelines is not installed, so no continuous-time Cox row is reported. "
                "The discrete-time proportional hazards baseline (complementary log-log, "
                "Prentice & Gloeckler 1978) is the correct classical survival "
                "specification for a monthly panel with heavy ties and is reported in "
                "its place. No placeholder value is substituted."
            ),
        }
    return {"status": "available", "note": "lifelines present; see baseline rows."}


if __name__ == "__main__":
    main()
