"""Unit tests for visual simplicity gating and caption clustering."""

from __future__ import annotations

import numpy as np
import pytest

from image_ambiguity.features.caption_clustering import cluster_captions_by_similarity
from image_ambiguity.features.visual_simplicity import (
    VisualSimplicityConfig,
    assess_visual_simplicity,
)


class TestVisualSimplicity:
    def test_simple_apple_like_features(self) -> None:
        opencv = {
            "edge_density": 0.01,
            "entropy": 5.5,
            "texture": 120.0,
            "contrast": 25.0,
            "color_variance": 800.0,
            "brightness": 200.0,
        }
        result = assess_visual_simplicity(opencv)
        assert result.is_simple is True
        assert result.simple_votes >= 3
        assert 0.0 < result.simplicity_score <= 1.0

    def test_complex_busy_features(self) -> None:
        opencv = {
            "edge_density": 0.12,
            "entropy": 7.7,
            "texture": 3500.0,
            "contrast": 75.0,
            "color_variance": 5200.0,
            "brightness": 110.0,
        }
        result = assess_visual_simplicity(opencv)
        assert result.is_simple is False

    def test_disabled_gate_never_simple(self) -> None:
        opencv = {
            "edge_density": 0.01,
            "entropy": 5.0,
            "texture": 50.0,
            "contrast": 10.0,
            "color_variance": 100.0,
            "brightness": 220.0,
        }
        cfg = VisualSimplicityConfig(enabled=False)
        result = assess_visual_simplicity(opencv, cfg)
        assert result.is_simple is False
        assert result.enabled is False

    def test_thresholds_are_configurable(self) -> None:
        opencv = {
            "edge_density": 0.05,
            "entropy": 7.4,
            "texture": 1000.0,
            "contrast": 60.0,
            "color_variance": 4000.0,
            "brightness": 100.0,
        }
        loose = VisualSimplicityConfig(
            edge_density_max=0.10,
            entropy_max=7.8,
            texture_max=2000.0,
            contrast_max=80.0,
            color_variance_max=5000.0,
            brightness_min=None,
            min_simple_votes=3,
            min_vote_fraction=0.5,
        )
        assert assess_visual_simplicity(opencv, loose).is_simple is True


class TestCaptionClustering:
    def test_merges_near_duplicate_meanings(self) -> None:
        captions = [
            "a red apple on a white background",
            "an apple sitting on white",
            "red apple against white backdrop",
            "a human face carved in rock",
        ]
        # Dim-2 unit-ish vectors: first three similar, fourth orthogonal.
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.98, 0.1],
                [0.97, 0.05],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result = cluster_captions_by_similarity(
            captions,
            embeddings,
            similarity_threshold=0.85,
        )
        assert result.applied is True
        assert result.n_clusters == 2
        assert len(result.representatives) == 2

    def test_single_cluster_keeps_two_members_for_diversity(self) -> None:
        captions = ["apple one", "apple two", "apple three"]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.98, 0.02],
            ],
            dtype=np.float64,
        )
        result = cluster_captions_by_similarity(
            captions,
            embeddings,
            similarity_threshold=0.9,
        )
        assert result.n_clusters == 1
        assert len(result.representatives) == 2

    def test_apply_false_preserves_inputs(self) -> None:
        captions = ["a", "b", "c"]
        embeddings = np.eye(3)
        result = cluster_captions_by_similarity(
            captions,
            embeddings,
            apply=False,
        )
        assert result.applied is False
        assert result.representatives == captions
