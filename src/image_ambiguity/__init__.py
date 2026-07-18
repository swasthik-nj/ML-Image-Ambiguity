"""Explainable Image Ambiguity Prediction research package."""

from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.caption_diversity import (
    CaptionDiversityAnalyzer,
    DiversityMetrics,
)
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.pipeline.dataset_builder import MLDatasetBuilder
from image_ambiguity.pipeline.label_generator import AmbiguityLabelGenerator

__all__ = [
    "AmbiguityLabelGenerator",
    "CaptionDiversityAnalyzer",
    "CocoDatasetLoader",
    "DiversityMetrics",
    "MLDatasetBuilder",
    "OpenCVFeatureExtractor",
    "SentenceEmbeddingGenerator",
]
__version__ = "0.1.0"

