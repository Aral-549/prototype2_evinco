"""Model loading and inference for the MoSPI PAIMANA early-warning predictor.

Supports two bundle generations:

* **v2 (preferred)** -- trained by `scripts/train_model.py` on the reconstructed
  longitudinal panel. Leakage is handled by *excluding* post-facto fields from
  the design matrix at training time, so there is nothing to zero at inference
  and no train/serve skew. Carries per-horizon models (1m / 3m / 6m), an
  isotonic calibrator, training-fold medians, and the sector encoding.

* **v1 (legacy fallback)** -- the original bundle, kept only so a fresh checkout
  that has not yet run the training pipeline still starts. It trained on
  `has_revised_doc` (34% of gain) and zeroed that column at serve time; the
  resulting skew collapsed `p_model` below 0.035 on all 2,155 live projects.
  When this path is taken the service flags itself as degraded so the API can
  say so rather than quietly serving a dead model.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.paimana_contracts import ProjectInput
from app.paimana_shap import explain_prediction_shap
from app.services.feature_adapter import build_feature_row

logger = logging.getLogger("paimana.model_service")


def _horizon_months(key: str) -> int:
    """Parse a horizon key like "3m" into its month count for ordering."""
    digits = "".join(ch for ch in key if ch.isdigit())
    return int(digits) if digits else 0

# v1-only: fields the legacy booster trained on that must be zeroed at serve
# time. v2 never populates this because those fields are not in its matrix.
LEAKAGE_QUARANTINED_FEATURES: frozenset[str] = frozenset(
    {"has_revised_doc", "current_delay_months_robust"}
)


class ModelService:
    """Lifecycle, feature adaptation and inference for the schedule-slip model."""

    _instance: Optional["ModelService"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.models: Dict[str, Any] = {}
        self.calibrators: Dict[str, Any] = {}
        self.imputer = None
        self.medians: Dict[str, float] = {}
        self.sector_frequency: Dict[str, float] = {}
        self.feature_columns: List[str] = []
        self.classification_threshold: float = 0.45
        self.risk_thresholds: Dict[str, float] = {"high": 75.0, "medium": 50.0}
        self.model_name: str = "PAIMANA Project Schedule Risk"
        self.model_version: str = "unknown"
        self.bundle_generation: str = "unknown"
        self.primary_horizon: str = "1m"
        self.horizons: List[str] = []
        self.target: str = "slip_next"
        self.target_description: str = ""
        self.trained_at: Optional[str] = None
        self.class_prior: Optional[float] = None
        self.is_loaded: bool = False
        self.is_degraded: bool = False
        self.degraded_reason: Optional[str] = None

        self.load_model()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the cached singleton so a retrained bundle is picked up."""
        cls._instance = None

    # ------------------------------------------------------------------ load

    def load_model(self) -> None:
        if not self.model_path.exists():
            logger.error("Model file not found at: %s", self.model_path)
            self.is_loaded = False
            return

        try:
            bundle = joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load model bundle: %s", exc)
            self.is_loaded = False
            return

        self.feature_columns = list(bundle.get("feature_columns", []))
        self.classification_threshold = float(bundle.get("classification_threshold", 0.45))
        self.risk_thresholds = bundle.get("risk_thresholds", {"high": 75.0, "medium": 50.0})
        self.model_name = bundle.get("model_name", "PAIMANA Schedule Risk Predictor")
        self.target = bundle.get("target", "slip_next")
        self.target_description = bundle.get("target_description", "")
        self.model_version = str(bundle.get("model_version", "1.x-legacy"))
        self.trained_at = bundle.get("trained_at")
        self.class_prior = bundle.get("class_prior")

        if "models" in bundle:
            self._load_v2(bundle)
        else:
            self._load_v1_legacy(bundle)

        self.is_loaded = True
        logger.info(
            "Loaded '%s' (%s, %d features, horizons=%s)%s",
            self.model_name,
            self.bundle_generation,
            len(self.feature_columns),
            self.horizons or [self.primary_horizon],
            f" DEGRADED: {self.degraded_reason}" if self.is_degraded else "",
        )

    def _load_v2(self, bundle: Dict[str, Any]) -> None:
        self.bundle_generation = "v2-panel-trained"
        self.models = bundle["models"]
        raw_cals = bundle.get("calibrators", {}) or {}
        # Training stores each calibrator as {"calibrator": est, "method": str},
        # or None when the cross-fitted ECE check rejected it.
        self.calibrators = {k: v for k, v in raw_cals.items() if v}
        self.medians = bundle.get("imputer_medians", {}) or {}
        self.sector_frequency = bundle.get("sector_frequency_encoding", {}) or {}
        self.primary_horizon = bundle.get("primary_horizon", "1m")
        self.horizons = list(bundle.get("horizons", list(self.models.keys())))
        self.model = self.models.get(self.primary_horizon)

    def _load_v1_legacy(self, bundle: Dict[str, Any]) -> None:
        self.bundle_generation = "v1-legacy"
        self.model = bundle.get("model")
        self.models = {"1m": self.model}
        self.imputer = bundle.get("imputer")
        self.horizons = ["1m"]

        quarantined = sorted(LEAKAGE_QUARANTINED_FEATURES & set(self.feature_columns))
        if quarantined:
            self.is_degraded = True
            self.degraded_reason = (
                f"Legacy v1 bundle: {quarantined} were trained on but must be zeroed at "
                "inference, producing train/serve skew that drives p_model toward 0 on "
                "real projects. Run `python scripts/build_panel.py && python "
                "scripts/train_model.py` to produce the leak-free v2 bundle."
            )
            logger.warning(self.degraded_reason)

        # scikit-learn 1.6 -> 1.9 SimpleImputer attribute rename.
        if self.imputer is not None:
            if not hasattr(self.imputer, "_fill_dtype") and hasattr(self.imputer, "_fit_dtype"):
                self.imputer._fill_dtype = self.imputer._fit_dtype
            elif not hasattr(self.imputer, "_fill_dtype"):
                self.imputer._fill_dtype = np.float64

    # --------------------------------------------------------------- features

    def build_frame(self, project: ProjectInput) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Build the model's input frame for one project."""
        if self.bundle_generation == "v2-panel-trained":
            return build_feature_row(
                project, self.feature_columns, self.sector_frequency, self.medians
            )

        # Legacy path: the v1 schema, with trained-on leakage proxies zeroed.
        feature_dict = project.to_feature_dict(self.feature_columns)
        for col in self.feature_columns:
            if col in LEAKAGE_QUARANTINED_FEATURES:
                feature_dict[col] = 0.0
        return pd.DataFrame([feature_dict])[self.feature_columns], {}

    def _to_array(self, frame: pd.DataFrame) -> np.ndarray:
        if self.imputer is not None:
            return self.imputer.transform(frame)
        return frame.to_numpy()

    def _calibrate(self, horizon: str, raw: float) -> float:
        entry = self.calibrators.get(horizon)
        if not entry:
            return raw
        est, method = entry.get("calibrator"), entry.get("method")
        if est is None:
            return raw
        try:
            arr = np.asarray([raw], dtype=float)
            out = (
                est.predict(arr) if method == "isotonic"
                else est.predict_proba(arr.reshape(-1, 1))[:, 1]
            )
            return float(np.clip(out[0], 0.0, 1.0))
        except Exception:  # noqa: BLE001 - calibration must never break inference
            logger.exception("Calibration failed for horizon %s; returning raw score.", horizon)
            return raw

    # -------------------------------------------------------------- inference

    def predict(self, project: ProjectInput) -> Tuple[float, int, Dict[str, float]]:
        """Calibrated probability, threshold decision, and TreeSHAP shares."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Cannot execute prediction.")

        frame, _ = self.build_frame(project)
        array = self._to_array(frame)

        raw = float(self.model.predict_proba(array)[0][1])
        probability = self._calibrate(self.primary_horizon, raw)
        decision = 1 if probability >= self.classification_threshold else 0
        return probability, decision, self._shap_contributions(array)

    def predict_horizons(self, project: ProjectInput) -> Dict[str, Dict[str, float]]:
        """Calibrated slip probability at every trained horizon.

        A single "risk score" cannot tell a ministry whether a project slips
        next month or next year. The horizon curve is what a review committee
        actually schedules against.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        frame, _ = self.build_frame(project)
        array = self._to_array(frame)

        scored: Dict[int, Dict[str, Any]] = {}
        for horizon, model in self.models.items():
            if model is None:
                continue
            raw = float(model.predict_proba(array)[0][1])
            scored[_horizon_months(horizon)] = {
                "horizon": horizon,
                "independent_probability": round(self._calibrate(horizon, raw), 4),
                "probability_uncalibrated": round(raw, 4),
                "calibrated": horizon in self.calibrators,
            }

        # Enforce the nesting constraint. "Slips within 6 months" strictly
        # contains "slips within 3 months", so P(6m) >= P(3m) >= P(1m) is a
        # mathematical necessity, not a modelling preference. The horizon
        # models are fitted independently on differently-censored cohorts, so
        # nothing in the fitting guarantees it, and an inverted curve on a
        # dashboard is indefensible. A running maximum is the minimal
        # projection back onto the feasible set; where it bites, we say so.
        out: Dict[str, Dict[str, Any]] = {}
        running = 0.0
        for months in sorted(scored):
            entry = scored[months]
            independent = entry["independent_probability"]
            enforced = max(running, independent)
            running = enforced
            entry["probability"] = round(enforced, 4)
            entry["monotonicity_enforced"] = bool(enforced > independent + 1e-9)
            out[entry.pop("horizon")] = entry
        return out

    def explain(self, project: ProjectInput, top_k: int = 5) -> Tuple[float, float, list]:
        """(calibrated p_model, base rate, top-k SHAP drivers) for one project."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Cannot explain prediction.")

        frame, _ = self.build_frame(project)
        array = self._to_array(frame)
        raw = float(self.model.predict_proba(array)[0][1])
        base_rate, drivers = explain_prediction_shap(
            booster=self.model,
            feature_row=array,
            feature_names=self.feature_columns,
            top_k=top_k,
        )
        return self._calibrate(self.primary_horizon, raw), base_rate, drivers

    def _shap_contributions(self, input_array: np.ndarray) -> Dict[str, float]:
        """Per-project TreeSHAP contributions as absolute-weight shares.

        Falls back to global `feature_importances_` only if the booster's
        `pred_contribs` path fails -- that answers "why the model in general",
        not "why THIS project", so it is a last resort.
        """
        try:
            import xgboost as xgb

            booster = self.model.get_booster()
            feature_names = None
            if hasattr(self.model, "feature_names_in_"):
                feature_names = list(self.model.feature_names_in_)
            elif booster.feature_names:
                feature_names = booster.feature_names

            dmatrix = xgb.DMatrix(input_array, feature_names=feature_names)
            contribs = booster.predict(dmatrix, pred_contribs=True)
            row = np.asarray(contribs[0, : len(self.feature_columns)], dtype=float)

            total = float(np.sum(np.abs(row))) or 1.0
            return {
                col: round(float(val) / total, 4)
                for col, val in zip(self.feature_columns, row)
            }
        except Exception:  # noqa: BLE001 - attribution must never break inference
            logger.exception("TreeSHAP contributions failed; falling back to importances")
            contributions: Dict[str, float] = {}
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                total_imp = float(np.sum(importances)) or 1.0
                for col, imp in zip(self.feature_columns, importances):
                    contributions[col] = round(float(imp) / total_imp, 4)
            return contributions

    # --------------------------------------------------------------- metadata

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "bundle_generation": self.bundle_generation,
            "trained_at": self.trained_at,
            "target": self.target,
            "target_description": self.target_description,
            "horizons": self.horizons,
            "primary_horizon": self.primary_horizon,
            "calibrated_horizons": sorted(self.calibrators.keys()),
            "class_prior": self.class_prior,
            "classification_threshold": self.classification_threshold,
            "risk_thresholds": self.risk_thresholds,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "quarantined_features": (
                sorted(LEAKAGE_QUARANTINED_FEATURES) if self.bundle_generation == "v1-legacy" else []
            ),
            "leakage_handling": (
                "Excluded from the design matrix at training time (no serve-time zeroing)."
                if self.bundle_generation == "v2-panel-trained"
                else "Zeroed at inference (legacy; causes train/serve skew)."
            ),
            "is_loaded": self.is_loaded,
            "is_degraded": self.is_degraded,
            "degraded_reason": self.degraded_reason,
        }


def load_training_metrics() -> Optional[Dict[str, Any]]:
    """Read the measured metrics artifact emitted by scripts/train_model.py.

    Returns None when the pipeline has not been run. Callers must surface that
    absence rather than substituting remembered or illustrative numbers.
    """
    path = settings.MODEL_METRICS_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.exception("Could not read model metrics artifact at %s", path)
        return None


def load_ordering_sensitivity() -> Optional[Dict[str, Any]]:
    """Read the reconstructed-time-axis robustness artifact, or None if absent."""
    path = settings.ORDERING_SENSITIVITY_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.exception("Could not read ordering sensitivity artifact at %s", path)
        return None


def load_label_validity() -> Optional[Dict[str, Any]]:
    """Read the label-validity artifact (is the label physical?), or None."""
    path = settings.LABEL_VALIDITY_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.exception("Could not read label validity artifact at %s", path)
        return None


def load_lead_time_report() -> Optional[Dict[str, Any]]:
    """Read the measured lead-time backtest artifact, or None if absent."""
    path = settings.LEAD_TIME_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.exception("Could not read lead-time artifact at %s", path)
        return None
