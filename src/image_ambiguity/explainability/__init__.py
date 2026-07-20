"""Explainability utilities (feature attribution, caption contribution)."""

from image_ambiguity.explainability.shap_explainer import (
    PredictionExplanation,
    SHAPExplainer,
)

__all__ = ["PredictionExplanation", "SHAPExplainer"]
