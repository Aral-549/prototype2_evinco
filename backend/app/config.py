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
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "model" / "paimana_schedule_risk_xgboost.pkl",
        Path(__file__).resolve().parent.parent / "model" / "paimana_schedule_risk_xgboost.pkl",
        Path(__file__).resolve().parent.parent / "paimana_schedule_risk_xgboost.pkl",
    ]
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


class Settings(BaseModel):
    PROJECT_NAME: str = "MoSPI PAIMANA Predictive Intelligence API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Base directory paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    MODEL_PATH: Path = Field(default_factory=_resolve_model_path)
    FRONTEND_DIR: Path = Field(default_factory=_resolve_frontend_dir)
    REAL_DATA_PATH: Path = Field(default_factory=_resolve_real_data_path)
    
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
