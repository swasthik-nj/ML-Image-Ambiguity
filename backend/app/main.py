"""FastAPI entrypoint for the Image Ambiguity Prediction API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router as ambiguity_router
from backend.app.schemas import HealthResponse, RootResponse
from image_ambiguity.config import get_settings
from image_ambiguity.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("backend")
settings = get_settings()
settings.ensure_directories()

app = FastAPI(
    title="Image Ambiguity Prediction API",
    description=(
        "REST API for explainable image ambiguity prediction.\n\n"
        "Upload an image, extract caption-diversity + OpenCV features, "
        "predict Low/Medium/High ambiguity, and inspect SHAP explanations.\n\n"
        "**Typical flow:** `POST /upload` → `POST /predict` or `POST /explain` "
        "with the returned `upload_id`. You may also upload the image "
        "directly on `/features`, `/predict`, and `/explain`."
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

app.include_router(ambiguity_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness probe for local development and deployment checks."""
    return HealthResponse(status="ok", environment=settings.app_env)


@app.get("/", response_model=RootResponse, tags=["system"])
def root() -> RootResponse:
    """API root metadata and endpoint index."""
    logger.info("Root endpoint requested")
    return RootResponse(
        project="Explainable Image Ambiguity Prediction",
        docs="/docs",
        health="/health",
        endpoints=[
            "/upload",
            "/features",
            "/predict",
            "/explain",
            "/compare",
            "/health",
            "/docs",
        ],
    )
