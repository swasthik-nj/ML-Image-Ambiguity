"""Explainable Image Ambiguity Prediction research package."""

from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator

__all__ = ["CocoDatasetLoader", "SentenceEmbeddingGenerator"]
__version__ = "0.1.0"
