"""Governance evaluation service: leak-free ML inference + GovScore lattice.

Single evaluation pipeline shared by the single-project, batch, CSV, and
portfolio endpoints (Spec Section 1.2 data flow):

    CUF snapshot -> leak-free features -> P_model + TreeSHAP
                 -> RuleFloor (F1..F5) -> GovScore = max(100*P_model, RuleFloor)
                 -> Capital-at-Risk -> ProjectGovernanceAssessment
"""

from __future__ import annotations

from typing import Optional

from app.paimana_contracts import ProjectGovernanceAssessment, ProjectInput
from app.paimana_engine import assess_project
from app.services.model_service import ModelService


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested but the model bundle is unavailable."""


def evaluate_project(
    project: ProjectInput,
    include_drivers: bool = True,
) -> ProjectGovernanceAssessment:
    """
    Full single-project governance assessment.

    Raises:
        ModelNotLoadedError: if the XGBoost bundle is not available.
    """
    model_service = ModelService.get_instance()
    if not model_service.is_loaded:
        raise ModelNotLoadedError("Predictive model bundle is not loaded.")

    shap_drivers = None
    base_rate_probability: Optional[float] = None

    if include_drivers:
        # explain() runs inference and native C++ TreeSHAP in one pass.
        p_model, base_rate_probability, shap_drivers = model_service.explain(project, top_k=5)
    else:
        # Fast path for whole-portfolio scoring: skip TreeSHAP, but go through
        # predict() so the isotonic calibrator is still applied. Reaching into
        # the raw booster here would silently serve uncalibrated probabilities
        # on the portfolio leaderboard while the single-project endpoint served
        # calibrated ones -- and Capital-at-Risk multiplies that number by
        # rupees, so the two views would disagree in crores.
        p_model, _decision, _contribs = model_service.predict(project)
        # The class prior is the honest reference point for "how does this
        # project compare to the portfolio", not a hardcoded 0.5.
        base_rate_probability = (
            float(model_service.class_prior)
            if model_service.class_prior is not None
            else 0.5
        )

    return assess_project(
        project=project,
        p_model=p_model,
        base_rate_probability=base_rate_probability if base_rate_probability is not None else 0.5,
        shap_drivers=shap_drivers,
    )
