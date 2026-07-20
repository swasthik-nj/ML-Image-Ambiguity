"""Backend service layer (use-cases over domain packages)."""

from backend.app.services.inference import (
    AmbiguityInferenceService,
    get_inference_service,
)

__all__ = ["AmbiguityInferenceService", "get_inference_service"]
