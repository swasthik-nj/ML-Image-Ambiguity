"""Feature engineering for caption diversity and computer-vision signals."""

from image_ambiguity.features.blip_captions import (
    BlipCaptionGenerator,
    BlipCaptionResult,
    CaptionComparison,
)
from image_ambiguity.features.caption_diversity import (
    CaptionDiversityAnalyzer,
    DiversityMetrics,
)
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator

__all__ = [
    "BlipCaptionGenerator",
    "BlipCaptionResult",
    "CaptionComparison",
    "CaptionDiversityAnalyzer",
    "DiversityMetrics",
    "OpenCVFeatureExtractor",
    "SentenceEmbeddingGenerator",
]
