"""Model loading and inference service for MoSPI PAIMANA XGBoost predictor.

Leak-Free Preprocessing (Spec Section 1.2, step 2): features indicating
post-facto bureaucratic adjustments -- notably `has_revised_doc` (34.22%
gain-based target proxy) and `current_delay_months_robust` (5.22%) -- are
strictly quarantined from the predictive early warning feature vector.
The legacy bundle still carries those columns, so they are zeroed at
inference time rather than removed, keeping the booster's expected
feature layout intact.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.paimana_contracts import ProjectInput
from app.paimana_shap import explain_prediction_shap

logger = logging.getLogger("paimana.model_service")

# Statutory leakage quarantine list (Spec Section 4.1)
LEAKAGE_QUARANTINED_FEATURES: frozenset[str] = frozenset(
    {
        "has_revised_doc",
        "current_delay_months_robust",
    }
)


class ModelService:
    """Manages the lifecycle, leak-free input normalization, and inference of the trained XGBoost model."""

    _instance: Optional["ModelService"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.imputer = None
        self.feature_columns: List[str] = []
        self.classification_threshold: float = 0.45
        self.risk_thresholds: Dict[str, float] = {"high": 75.0, "medium": 50.0}
        self.model_name: str = "PAIMANA Project Schedule Risk XGBoost"
        self.target: str = "future_schedule_risk"
        self.target_description: str = "Whether the project's next reported revised date of completion is pushed later"
        self.is_loaded: bool = False

        self.load_model()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def load_model(self) -> None:
        """Loads the trained model bundle and patches cross-version scikit-learn attributes."""
        if not self.model_path.exists():
            logger.error(f"Model file not found at: {self.model_path}")
            self.is_loaded = False
            return

        try:
            logger.info(f"Loading model bundle from: {self.model_path}")
            bundle = joblib.load(self.model_path)

            self.model = bundle.get("model")
            self.imputer = bundle.get("imputer")
            self.feature_columns = bundle.get("feature_columns", [])
            self.classification_threshold = float(bundle.get("classification_threshold", 0.45))
            self.risk_thresholds = bundle.get("risk_thresholds", {"high": 75.0, "medium": 50.0})
            self.model_name = bundle.get("model_name", "PAIMANA Schedule Risk Predictor")
            self.target = bundle.get("target", "future_schedule_risk")
            self.target_description = bundle.get(
                "target_description",
                "Probability of project schedule overrun / delayed revised date of completion",
            )

            # Compatibility patch: scikit-learn 1.6.1 -> 1.9.1 SimpleImputer attribute change
            if self.imputer is not None:
                if not hasattr(self.imputer, "_fill_dtype") and hasattr(self.imputer, "_fit_dtype"):
                    self.imputer._fill_dtype = self.imputer._fit_dtype
                elif not hasattr(self.imputer, "_fill_dtype"):
                    self.imputer._fill_dtype = np.float64

            self.is_loaded = True
            logger.info(
                f"Successfully loaded '{self.model_name}' with {len(self.feature_columns)} features "
                f"(quarantined leakage proxies: {sorted(LEAKAGE_QUARANTINED_FEATURES & set(self.feature_columns))})."
            )
        except Exception as exc:
            logger.exception(f"Failed to load model bundle: {exc}")
            self.is_loaded = False

    def build_leak_free_frame(self, project: ProjectInput) -> pd.DataFrame:
        """
        Converts a validated ProjectInput into the model's exact feature frame
        with quarantined outcome proxies hard-zeroed (leak-free preprocessing).
        """
        feature_dict = project.to_feature_dict(self.feature_columns)
        for col in self.feature_columns:
            if col in LEAKAGE_QUARANTINED_FEATURES:
                feature_dict[col] = 0.0
        return pd.DataFrame([feature_dict])[self.feature_columns]

    def predict(self, project: ProjectInput) -> Tuple[float, int, Dict[str, float]]:
        """
        Executes leak-free model inference for a single project.

        Returns:
            probability (float 0.0 - 1.0): P_model, probability of schedule risk.
            decision (int): 1 if probability >= classification_threshold else 0.
            feature_contributions (dict): Per-project TreeSHAP attribution,
                normalized to absolute-contribution shares (sums to 1.0).
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Cannot execute prediction.")

        df = self.build_leak_free_frame(project)

        # Apply imputer if present
        if self.imputer is not None:
            input_array = self.imputer.transform(df)
        else:
            input_array = df.values

        # Predict probability
        proba = self.model.predict_proba(input_array)
        risk_probability = float(proba[0][1])
        decision = 1 if risk_probability >= self.classification_threshold else 0

        # Feature contribution mapping
        contributions: Dict[str, float] = self._shap_contributions(input_array)

        return risk_probability, decision, contributions

    def explain(
        self, project: ProjectInput, top_k: int = 5
    ) -> Tuple[float, float, list]:
        """
        Returns (p_model, base_rate_probability, top_k SHAPDriver list) for a
        project using native C++ TreeSHAP (Spec Section 3.2.3).

        Raises RuntimeError when the model is not loaded or attribution fails
        irrecoverably; callers own the fallback policy.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Cannot explain prediction.")

        df = self.build_leak_free_frame(project)
        if self.imputer is not None:
            input_array = self.imputer.transform(df)
        else:
            input_array = df.values

        p_model = float(self.model.predict_proba(input_array)[0][1])
        base_rate, drivers = explain_prediction_shap(
            booster=self.model,
            feature_row=input_array,
            feature_names=self.feature_columns,
            top_k=top_k,
        )
        return p_model, base_rate, drivers

    def _shap_contributions(self, input_array: np.ndarray) -> Dict[str, float]:
        """Per-project TreeSHAP contributions, normalized to abs-weight shares.

        Falls back to global `feature_importances_` only if the booster's
        `pred_contribs` path fails (it answers 'why THIS project', not
        'why the model in general').
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

    def get_metadata(self) -> Dict[str, Any]:
        """Returns model diagnostic metadata."""
        return {
            "model_name": self.model_name,
            "target": self.target,
            "target_description": self.target_description,
            "classification_threshold": self.classification_threshold,
            "risk_thresholds": self.risk_thresholds,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "quarantined_features": sorted(LEAKAGE_QUARANTINED_FEATURES),
            "is_loaded": self.is_loaded,
        }
