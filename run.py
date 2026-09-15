#!/usr/bin/env python
"""MoSPI PAIMANA Predictive Intelligence Platform.

Root entrypoint to launch the unified platform server:
    python run.py
"""

import sys
from pathlib import Path

# Add backend directory to Python sys.path so app modules resolve seamlessly
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("=" * 70)
    print("  MoSPI PAIMANA Predictive Infrastructure Monitoring Platform")
    print(f"  Interactive Dashboard: http://localhost:{settings.PORT}/dashboard")
    print(f"  Swagger API Docs:      http://localhost:{settings.PORT}/docs")
    print(f"  Health Check:          http://localhost:{settings.PORT}/api/v1/health")
    print("=" * 70)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
