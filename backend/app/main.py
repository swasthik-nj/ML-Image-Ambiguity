"""FastAPI entrypoint for the Image Ambiguity Prediction API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from image_ambiguity.config import get_settings
from image_ambiguity.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("backend")
settings = get_settings()
settings.ensure_directories()

app = FastAPI(
    title="Image Ambiguity Prediction API",
    description=(
        "Explainable image ambiguity prediction using human and "
        "AI-generated caption diversity with computer vision features."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for local development and deployment checks."""
    return {"status": "ok", "environment": settings.app_env}


@app.get("/")
def root() -> dict[str, str]:
    """API root metadata."""
    logger.info("Root endpoint requested")
    return {
        "project": "Explainable Image Ambiguity Prediction",
        "docs": "/docs",
        "health": "/health",
    }
