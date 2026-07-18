"""Unit tests for :class:`MLDatasetBuilder`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from image_ambiguity.features.caption_diversity import CaptionDiversityAnalyzer
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.pipeline.dataset_builder import (
    DATASET_COLUMNS,
    MLDatasetBuilder,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ANNOTATION_FILE = FIXTURES / "captions_tiny.json"
IMAGE_DIR = FIXTURES / "images"


def _fake_embedding_generator(dim: int = 8) -> MagicMock:
    """Return a mock embedding generator with deterministic encode()."""
    generator = MagicMock()
    generator.model = None

    def _load_model() -> MagicMock:
        generator.model = MagicMock()
        return generator.model

    def _generate_embeddings(captions: list[str]) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(tuple(captions))) % (2**32))
        return rng.normal(size=(len(captions), dim)).astype(np.float32)

    generator.load_model.side_effect = _load_model
    generator.generate_embeddings.side_effect = _generate_embeddings
    return generator


@pytest.fixture
def builder() -> MLDatasetBuilder:
    return MLDatasetBuilder(
        ANNOTATION_FILE,
        IMAGE_DIR,
        embedding_generator=_fake_embedding_generator(),
        diversity_analyzer=CaptionDiversityAnalyzer(),
        cv_extractor=OpenCVFeatureExtractor(),
    )


class TestMLDatasetBuilder:
    def test_select_image_ids_returns_all_when_sample_size_none(
        self, builder: MLDatasetBuilder
    ) -> None:
        ids = builder.select_image_ids(sample_size=None)
        assert set(ids) == {1, 2}

    def test_select_image_ids_samples_subset(
        self, builder: MLDatasetBuilder
    ) -> None:
        ids = builder.select_image_ids(sample_size=1, seed=0)
        assert len(ids) == 1
        assert ids[0] in {1, 2}

    def test_build_returns_expected_columns_and_row_count(
        self, builder: MLDatasetBuilder
    ) -> None:
        df = builder.build(image_ids=[1, 2])

        assert list(df.columns) == DATASET_COLUMNS
        assert len(df) == 2
        assert set(df["image_id"]) == {1, 2}
        assert df["image_id"].dtype == np.int64

    def test_build_has_no_missing_values_after_impute(
        self, builder: MLDatasetBuilder
    ) -> None:
        df = builder.build(image_ids=[1, 2])
        assert df.isna().sum().sum() == 0

    def test_build_diversity_values_are_valid(
        self, builder: MLDatasetBuilder
    ) -> None:
        df = builder.build(image_ids=[1, 2])
        row1 = df.loc[df["image_id"] == 1].iloc[0]
        # image 1 has 2 captions -> exactly one pair -> valid similarity in [-1, 1]
        assert -1.0 <= row1["average_similarity"] <= 1.0
        assert row1["caption_diversity"] == pytest.approx(
            1.0 - row1["average_similarity"]
        )

    def test_build_handles_single_caption_image_with_nan_then_impute(
        self, builder: MLDatasetBuilder
    ) -> None:
        # image 2 has only 1 caption in the fixture -> diversity columns
        # should be imputed (no NaN survives in the final frame).
        df = builder.build(image_ids=[1, 2])
        row2 = df.loc[df["image_id"] == 2].iloc[0]
        assert not pd.isna(row2["caption_diversity"])

    def test_save_csv_writes_file(
        self, builder: MLDatasetBuilder, tmp_path: Path
    ) -> None:
        df = builder.build(image_ids=[1, 2])
        path = builder.save_csv(df, tmp_path / "human_dataset.csv")

        assert path.is_file()
        reloaded = pd.read_csv(path)
        assert len(reloaded) == 2
        assert list(reloaded.columns) == DATASET_COLUMNS

    def test_handle_missing_values_imputes_with_median(self) -> None:
        df = pd.DataFrame(
            {
                "image_id": [1, 2, 3],
                "average_similarity": [0.2, np.nan, 0.8],
                "minimum_similarity": [0.1, 0.5, np.nan],
                "maximum_similarity": [0.3, 0.6, 0.9],
                "std_similarity": [0.01, 0.02, 0.03],
                "caption_diversity": [0.8, np.nan, 0.2],
                "edge_density": [0.1, 0.2, 0.3],
                "entropy": [5.0, 6.0, 7.0],
                "brightness": [100.0, 110.0, 120.0],
                "contrast": [10.0, 20.0, 30.0],
                "color_variance": [1.0, 2.0, 3.0],
                "texture": [50.0, 60.0, 70.0],
            }
        )

        result = MLDatasetBuilder.handle_missing_values(df)

        assert result.isna().sum().sum() == 0
        assert result.loc[1, "average_similarity"] == pytest.approx(0.5)
        assert result["image_id"].dtype == np.int64

    def test_handle_missing_values_empty_df(self) -> None:
        df = pd.DataFrame(columns=DATASET_COLUMNS)
        result = MLDatasetBuilder.handle_missing_values(df)
        assert result.empty

    def test_build_skips_unknown_image_id(
        self, builder: MLDatasetBuilder
    ) -> None:
        df = builder.build(image_ids=[1, 999])
        assert len(df) == 1
        assert df.iloc[0]["image_id"] == 1

    def test_repr(self, builder: MLDatasetBuilder) -> None:
        text = repr(builder)
        assert "MLDatasetBuilder" in text
