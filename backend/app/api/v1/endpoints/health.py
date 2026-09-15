"""Health and diagnostics endpoints."""

from fastapi import APIRouter
from app.config import settings
from app.schemas.analytics import HealthResponse
from app.services.model_service import ModelService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="System Health & Model Status")
def get_health() -> HealthResponse:
    """Returns platform operational health and loaded model diagnostics."""
    service = ModelService.get_instance()
    meta = service.get_metadata()
    
    return HealthResponse(
        status="HEALTHY" if service.is_loaded else "DEGRADED",
        service_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        model_loaded=service.is_loaded,
        model_name=meta.get("model_name", "Unknown"),
        model_target=meta.get("target", "Unknown"),
        classification_threshold=meta.get("classification_threshold", 0.45),
        feature_count=meta.get("feature_count", 0),
        features=meta.get("feature_columns", []),
        quarantined_features=meta.get("quarantined_features", []),
    )
