"""Unit tests for :class:`OpenCVFeatureExtractor`."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from image_ambiguity.features.cv_features import OpenCVFeatureExtractor


@pytest.fixture
def extractor() -> OpenCVFeatureExtractor:
    return OpenCVFeatureExtractor()


@pytest.fixture
def gray_image() -> np.ndarray:
    # Smooth gradient + a bright square (edges + texture).
    image = np.zeros((64, 64), dtype=np.uint8)
    image[:, :] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
    image[20:40, 20:40] = 255
    return image


@pytest.fixture
def color_image(gray_image: np.ndarray) -> np.ndarray:
    # Synthetic BGR image with different channel statistics.
    b = gray_image
    g = np.roll(gray_image, 5, axis=0)
    r = np.roll(gray_image, 10, axis=1)
    return np.stack([b, g, r], axis=-1)


class TestOpenCVFeatureExtractor:
    def test_extract_grayscale_keys_and_ranges(
        self, extractor: OpenCVFeatureExtractor, gray_image: np.ndarray
    ) -> None:
        features = extractor.extract(gray_image)

        assert set(extractor.FEATURE_KEYS).issubset(features.keys())
        assert features["is_grayscale"] is True
        assert features["channels"] == 1
        assert features["color_variance"] == 0.0
        assert 0.0 <= features["edge_density"] <= 1.0
        assert 0.0 <= features["entropy"] <= 8.0
        assert 0.0 <= features["brightness"] <= 255.0
        assert features["width"] == 64
        assert features["height"] == 64
        assert features["resolution"] == 64 * 64

    def test_extract_color_has_nonzero_color_variance(
        self, extractor: OpenCVFeatureExtractor, color_image: np.ndarray
    ) -> None:
        features = extractor.extract(color_image)

        assert features["is_grayscale"] is False
        assert features["channels"] == 3
        assert features["color_variance"] > 0.0

    def test_uniform_image_low_edge_and_texture(
        self, extractor: OpenCVFeatureExtractor
    ) -> None:
        flat = np.full((32, 32), 128, dtype=np.uint8)
        features = extractor.extract(flat)

        assert features["edge_density"] == pytest.approx(0.0)
        assert features["texture"] == pytest.approx(0.0, abs=1e-6)
        assert features["brightness"] == pytest.approx(128.0)
        assert features["contrast"] == pytest.approx(0.0, abs=1e-6)

    def test_to_csv_row_and_save_csv(
        self,
        extractor: OpenCVFeatureExtractor,
        gray_image: np.ndarray,
        tmp_path: Path,
    ) -> None:
        features = extractor.extract(gray_image)
        row = extractor.to_csv_row(
            features, image_id=1, file_name="x.jpg"
        )
        assert row["image_id"] == 1
        assert "edge_density" in row

        path = extractor.save_csv(
            features, tmp_path / "feats.csv", image_id=1, file_name="x.jpg"
        )
        text = path.read_text(encoding="utf-8")
        assert "edge_density" in text
        assert "entropy" in text

    def test_load_image_missing_raises(
        self, extractor: OpenCVFeatureExtractor, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            extractor.load_image(tmp_path / "missing.jpg")

    def test_invalid_canny_thresholds(self) -> None:
        with pytest.raises(ValueError, match="canny_threshold2"):
            OpenCVFeatureExtractor(canny_threshold1=200, canny_threshold2=100)

    def test_visualize_debug_saves_png(
        self,
        extractor: OpenCVFeatureExtractor,
        color_image: np.ndarray,
        tmp_path: Path,
    ) -> None:
        fake_fig = MagicMock()
        fake_axes = [MagicMock() for _ in range(4)]
        axes_arr = MagicMock()
        axes_arr.ravel.return_value = fake_axes
        fake_fig.suptitle = MagicMock()
        fake_fig.tight_layout = MagicMock()
        fake_fig.savefig = MagicMock()

        fake_pyplot = ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = MagicMock(return_value=(fake_fig, axes_arr))
        fake_pyplot.close = MagicMock()
        fake_pyplot.show = MagicMock()
        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.pyplot = fake_pyplot
        fake_matplotlib.get_backend = MagicMock(return_value="Agg")
        fake_matplotlib.use = MagicMock()

        with patch.dict(
            sys.modules,
            {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot},
        ):
            saved = extractor.visualize_debug(
                color_image, tmp_path / "debug.png", show=False
            )

        assert saved is not None
        assert saved.name == "debug.png"
        fake_fig.savefig.assert_called_once()
