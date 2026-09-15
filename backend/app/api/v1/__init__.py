"""API v1 router composition."""

from fastapi import APIRouter
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.predict import router as predict_router
from app.api.v1.endpoints.portfolio import router as portfolio_router
from app.api.v1.endpoints.analytics import router as analytics_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router, tags=["Health & Diagnostics"])
api_v1_router.include_router(predict_router, tags=["Predictions & Inference"])
api_v1_router.include_router(portfolio_router, tags=["Portfolio Monitoring & CaR"])
api_v1_router.include_router(analytics_router, tags=["Governance Analytics & CUF Gap"])
