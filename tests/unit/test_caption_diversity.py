"""Unit tests for :class:`CaptionDiversityAnalyzer`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from image_ambiguity.features.caption_diversity import CaptionDiversityAnalyzer


@pytest.fixture
def analyzer() -> CaptionDiversityAnalyzer:
    return CaptionDiversityAnalyzer()


@pytest.fixture
def identical_embeddings() -> np.ndarray:
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.stack([vector, vector, vector], axis=0)


@pytest.fixture
def orthogonal_embeddings() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


class TestCaptionDiversityAnalyzer:
    def test_identical_embeddings_zero_diversity(
        self,
        analyzer: CaptionDiversityAnalyzer,
        identical_embeddings: np.ndarray,
    ) -> None:
        metrics = analyzer.compute(identical_embeddings)

        assert metrics.n_captions == 3
        assert metrics.n_pairs == 3
        assert metrics.average_similarity == pytest.approx(1.0)
        assert metrics.min_similarity == pytest.approx(1.0)
        assert metrics.max_similarity == pytest.approx(1.0)
        assert metrics.std_similarity == pytest.approx(0.0)
        assert metrics.diversity_score == pytest.approx(0.0)

    def test_orthogonal_embeddings_full_diversity(
        self,
        analyzer: CaptionDiversityAnalyzer,
        orthogonal_embeddings: np.ndarray,
    ) -> None:
        metrics = analyzer.compute(orthogonal_embeddings)

        assert metrics.average_similarity == pytest.approx(0.0, abs=1e-6)
        assert metrics.diversity_score == pytest.approx(1.0, abs=1e-6)

    def test_diversity_formula(
        self, analyzer: CaptionDiversityAnalyzer
    ) -> None:
        a = np.array([1.0, 0.0], dtype=np.float64)
        b = np.array([0.8, 0.6], dtype=np.float64)
        embeddings = np.stack([a, b], axis=0)

        metrics = analyzer.compute(embeddings)
        assert metrics.average_similarity == pytest.approx(0.8, abs=1e-6)
        assert metrics.diversity_score == pytest.approx(0.2, abs=1e-6)
        assert metrics.diversity_score == pytest.approx(
            1.0 - metrics.average_similarity
        )

    def test_requires_at_least_two_embeddings(
        self, analyzer: CaptionDiversityAnalyzer
    ) -> None:
        with pytest.raises(ValueError, match="At least two"):
            analyzer.compute(np.ones((1, 4), dtype=np.float32))

    def test_rejects_non_2d(
        self, analyzer: CaptionDiversityAnalyzer
    ) -> None:
        with pytest.raises(ValueError, match="2-D"):
            analyzer.compute(np.ones((4,), dtype=np.float32))

    def test_to_dict_and_json_roundtrip(
        self,
        analyzer: CaptionDiversityAnalyzer,
        identical_embeddings: np.ndarray,
        tmp_path: Path,
    ) -> None:
        metrics = analyzer.compute(identical_embeddings)
        payload = analyzer.to_dict(
            metrics, image_id=42, captions=["a", "b", "c"]
        )

        assert payload["image_id"] == 42
        assert payload["diversity_score"] == 0.0
        assert payload["formula"].startswith("diversity_score")

        path = analyzer.save_json(
            metrics,
            tmp_path / "out.json",
            image_id=42,
            captions=["a", "b", "c"],
        )
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["n_captions"] == 3
        assert loaded["captions"] == ["a", "b", "c"]

    def test_save_csv(
        self,
        analyzer: CaptionDiversityAnalyzer,
        identical_embeddings: np.ndarray,
        tmp_path: Path,
    ) -> None:
        metrics = analyzer.compute(identical_embeddings)
        path = analyzer.save_csv(metrics, tmp_path / "out.csv", image_id=7)
        text = path.read_text(encoding="utf-8")
        assert "diversity_score" in text
        assert "7" in text

    def test_visualize_saves_png(
        self,
        analyzer: CaptionDiversityAnalyzer,
        identical_embeddings: np.ndarray,
        tmp_path: Path,
    ) -> None:
        metrics = analyzer.compute(identical_embeddings)

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        fake_fig.colorbar = MagicMock()
        fake_fig.tight_layout = MagicMock()
        fake_fig.savefig = MagicMock()
        fake_ax.imshow.return_value = MagicMock()

        fake_pyplot = ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = MagicMock(return_value=(fake_fig, fake_ax))
        fake_pyplot.close = MagicMock()
        fake_pyplot.show = MagicMock()
        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.pyplot = fake_pyplot
        fake_matplotlib.get_backend = MagicMock(return_value="agg")
        fake_matplotlib.use = MagicMock()

        with patch.dict(
            sys.modules,
            {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot},
        ):
            saved = analyzer.visualize(
                metrics,
                tmp_path / "heat.png",
                captions=["a", "b", "c"],
                show=False,
            )

        assert saved is not None
        assert saved.name == "heat.png"
        fake_fig.savefig.assert_called_once()
