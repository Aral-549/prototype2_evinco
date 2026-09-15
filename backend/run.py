#!/usr/bin/env python
"""Dev server entrypoint: `python run.py` or `uvicorn app.main:app`."""
import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
