"""FastAPI entrypoint for MoSPI PAIMANA Predictive Intelligence Platform.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.api import api_v1_router
from app.services.model_service import ModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paimana.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle events: Preload models and verify assets on startup."""
    logger.info("Initializing MoSPI PAIMANA Predictive Intelligence Platform...")
    service = ModelService.get_instance()
    if service.is_loaded:
        logger.info(f"Model service ready: {service.model_name} with {len(service.feature_columns)} features.")
    else:
        logger.warning(f"Model could not be loaded from {settings.MODEL_PATH}. Check asset existence.")
    yield
    logger.info("Shutting down PAIMANA Predictive Intelligence Platform.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "An AI-powered Predictive Analytics & Early Warning Decision Support System for "
        "Central Sector Infrastructure Projects (MoSPI PAIMANA / OCMS). "
        "Forecasts schedule slippage, computes Capital-at-Risk (CaR), addresses Common Upload Form (CUF) "
        "information bounds, and provides actionable governance recommendations."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for frontend access (React, Next.js, Vite, Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

def _get_dashboard_html() -> str | None:
    candidates = [
        settings.FRONTEND_DIR / "index.html",
        settings.FRONTEND_DIR / "dashboard.html",
        settings.BASE_DIR / "app" / "templates" / "dashboard.html",
    ]
    for c in candidates:
        if c.exists():
            return c.read_text(encoding="utf-8")
    return None


# Serve the frontend directory so the dashboard can load vendored assets.
# Tailwind is vendored locally (frontend/vendor/) rather than pulled from
# cdn.tailwindcss.com at runtime: the page takes ALL of its layout from
# Tailwind classes, so on a machine without internet -- a real possibility at
# a demo venue -- the CDN version renders as an unstyled wall of text.
if settings.FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(settings.FRONTEND_DIR)),
        name="static",
    )


@app.get("/dashboard", summary="Executive Command Center", response_class=HTMLResponse, tags=["Frontend"])
def get_dashboard() -> HTMLResponse:
    """Serves the Executive UI/UX Dashboard."""
    content = _get_dashboard_html()
    if content:
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>Dashboard template not found</h1>", status_code=404)


@app.get("/", summary="API Root", tags=["Root"])
def root() -> JSONResponse:
    """Returns an executive operational overview and interactive documentation links."""
    service = ModelService.get_instance()
    return JSONResponse(
        content={
            "platform": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "sponsor": "Ministry of Statistics and Programme Implementation (MoSPI)",
            "problem_statement": "SIH26103 - Use case on web-based integrated project-monitoring platform",
            "dashboard_url": "/dashboard",
            "documentation_url": "/docs",
            "health_check_url": f"{settings.API_V1_STR}/health",
            "portfolio_summary_url": f"{settings.API_V1_STR}/portfolio/summary",
            "cuf_gap_analysis_url": f"{settings.API_V1_STR}/analytics/cuf-gap",
            "status": "ONLINE" if service.is_loaded else "DEGRADED (Model Not Found)",
        }
    )
