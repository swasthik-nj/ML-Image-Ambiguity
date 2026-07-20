"""Pydantic response / request models for the Ambiguity Prediction API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    environment: str


class RootResponse(BaseModel):
    project: str
    docs: str
    health: str
    endpoints: list[str]


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    content_type: str
    size_bytes: int
    width: int
    height: int
    mode: str
    coco_image_id: int | None = None
    coco_captions: list[str] = Field(default_factory=list)
    message: str = "Image uploaded successfully"


class OpenCVFeatures(BaseModel):
    edge_density: float
    entropy: float
    brightness: float
    contrast: float
    color_variance: float
    texture: float


class CaptionDiversityFeatures(BaseModel):
    average_similarity: float
    minimum_similarity: float
    maximum_similarity: float
    std_similarity: float
    caption_diversity: float
    n_captions: int
    n_pairs: int


class FeaturesResponse(BaseModel):
    upload_id: str | None = None
    caption_source: str = Field(
        default="user",
        description="Caption origin: user, coco_human, or blip",
    )
    opencv_features: OpenCVFeatures
    caption_diversity: CaptionDiversityFeatures
    captions: list[str]
    feature_vector: dict[str, float]


class PredictionResponse(BaseModel):
    upload_id: str | None = None
    predicted_ambiguity: str = Field(
        description="Ambiguity class: Low, Medium, or High"
    )
    confidence: float
    probabilities: dict[str, float]
    caption_source: str = Field(
        default="user",
        description="Caption origin: user, coco_human, or blip",
    )
    caption_diversity: CaptionDiversityFeatures
    opencv_features: OpenCVFeatures
    captions: list[str]
    feature_vector: dict[str, float]


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class ShapExplanation(BaseModel):
    row_index: int
    predicted_class: str
    predicted_probability: float
    base_value: float
    contributions: list[dict[str, Any]]


class ExplainResponse(PredictionResponse):
    shap_explanation: ShapExplanation
    summary: str


class ErrorResponse(BaseModel):
    detail: str


class HistogramBin(BaseModel):
    bin: str
    count: int


class CompareResponse(BaseModel):
    human_diversity: float | None
    ai_diversity: float | None
    n_human: int
    n_ai: int
    n_compared: int
    human_available: bool
    ai_available: bool
    message: str
    human_histogram: list[HistogramBin] = Field(default_factory=list)
    ai_histogram: list[HistogramBin] = Field(default_factory=list)
