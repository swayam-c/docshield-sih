"""DOCSHIELD AI — FastAPI application entrypoint."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import analyze, cases, health
from app.config import get_settings
from app.db import init_db

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app() -> FastAPI:
    settings = get_settings()
    init_db()

    app = FastAPI(
        title="DOCSHIELD AI",
        description=(
            "AI-based identity & document screening platform (SIH 26188). "
            "Phases 1–30 complete — screening assist only."
        ),
        version=__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    # health also at /api/health via router; expose root alias
    app.include_router(analyze.router)
    app.include_router(cases.router)

    if FRONTEND_DIR.exists():
        assets = FRONTEND_DIR / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/")
        def index():
            return FileResponse(FRONTEND_DIR / "index.html")

        @app.get("/dashboard")
        def dashboard():
            return FileResponse(FRONTEND_DIR / "dashboard.html")

    @app.get("/api")
    def api_root():
        return {
            "service": "DOCSHIELD AI",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
