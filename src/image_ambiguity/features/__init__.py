"""Feature engineering for caption diversity and computer-vision signals."""

from image_ambiguity.features.blip_captions import (
    BlipCaptionGenerator,
    BlipCaptionResult,
    CaptionComparison,
)
from image_ambiguity.features.caption_clustering import (
    CaptionClusterResult,
    cluster_captions_by_similarity,
)
from image_ambiguity.features.caption_diversity import (
    CaptionDiversityAnalyzer,
    DiversityMetrics,
)
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.features.visual_simplicity import (
    VisualSimplicityAssessment,
    VisualSimplicityConfig,
    assess_visual_simplicity,
)

__all__ = [
    "BlipCaptionGenerator",
    "BlipCaptionResult",
    "CaptionClusterResult",
    "CaptionComparison",
    "CaptionDiversityAnalyzer",
    "DiversityMetrics",
    "OpenCVFeatureExtractor",
    "SentenceEmbeddingGenerator",
    "VisualSimplicityAssessment",
    "VisualSimplicityConfig",
    "assess_visual_simplicity",
    "cluster_captions_by_similarity",
]
