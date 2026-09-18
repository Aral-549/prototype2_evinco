"""MoSPI PAIMANA Predictive Infrastructure Monitoring Platform.

Backend configuration and constants.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    # v2 (leak-free, panel-trained, calibrated) is preferred; the v1 bundle is
    # kept only as a fallback so the API still starts on a checkout that has not
    # run scripts/train_model.py yet.
    roots = [
        Path(__file__).resolve().parent.parent.parent / "model",
        Path(__file__).resolve().parent.parent / "model",
        Path(__file__).resolve().parent.parent,
    ]
    candidates = [r / "paimana_schedule_risk_v2.pkl" for r in roots]
    candidates += [r / "paimana_schedule_risk_xgboost.pkl" for r in roots]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _resolve_frontend_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "frontend",
        Path(__file__).resolve().parent.parent / "frontend",
        Path(__file__).resolve().parent / "templates",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _resolve_real_data_path() -> Path:
    env_path = os.getenv("REAL_DATA_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "data" / "paimana_real_projects.json",
        Path(__file__).resolve().parent.parent / "data" / "paimana_real_projects.json",
        Path(__file__).resolve().parent / "data" / "paimana_real_projects.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _resolve_artifact(filename: str) -> Path:
    """Locate a training artifact emitted by scripts/train_model.py.

    These files are the single source of truth for every performance number the
    API reports; the serving layer must never carry hardcoded metrics.
    """
    roots = [
        Path(__file__).resolve().parent.parent.parent / "model",
        Path(__file__).resolve().parent.parent / "model",
    ]
    for root in roots:
        candidate = root / filename
        if candidate.exists():
            return candidate
    return roots[0] / filename


def _resolve_model_metrics_path() -> Path:
    return _resolve_artifact("paimana_model_metrics.json")


def _resolve_lead_time_path() -> Path:
    return _resolve_artifact("paimana_lead_time.json")


def _resolve_ordering_path() -> Path:
    return _resolve_artifact("paimana_ordering_sensitivity.json")


def _resolve_label_validity_path() -> Path:
    return _resolve_artifact("paimana_label_validity.json")


class Settings(BaseModel):
    PROJECT_NAME: str = "MoSPI PAIMANA Predictive Intelligence API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Base directory paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    MODEL_PATH: Path = Field(default_factory=_resolve_model_path)
    FRONTEND_DIR: Path = Field(default_factory=_resolve_frontend_dir)
    REAL_DATA_PATH: Path = Field(default_factory=_resolve_real_data_path)
    MODEL_METRICS_PATH: Path = Field(default_factory=_resolve_model_metrics_path)
    LEAD_TIME_PATH: Path = Field(default_factory=_resolve_lead_time_path)
    ORDERING_SENSITIVITY_PATH: Path = Field(default_factory=_resolve_ordering_path)
    LABEL_VALIDITY_PATH: Path = Field(default_factory=_resolve_label_validity_path)
    
    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
    
    # CORS Origins (allow common frontend ports)
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "*",
    ]


settings = Settings()
