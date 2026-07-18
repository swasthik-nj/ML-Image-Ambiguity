"""Unit tests for :class:`AmbiguityLabelGenerator`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from image_ambiguity.pipeline.label_generator import AmbiguityLabelGenerator


@pytest.fixture
def generator() -> AmbiguityLabelGenerator:
    return AmbiguityLabelGenerator(low_max=0.35, medium_max=0.65)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [1, 2, 3, 4, 5],
            "caption_diversity": [0.10, 0.34, 0.50, 0.65, 0.90],
            "edge_density": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )


class TestLabelValue:
    def test_low_bucket(self, generator: AmbiguityLabelGenerator) -> None:
        assert generator.label_value(0.0) == "Low"
        assert generator.label_value(0.34) == "Low"

    def test_medium_bucket(self, generator: AmbiguityLabelGenerator) -> None:
        assert generator.label_value(0.35) == "Medium"
        assert generator.label_value(0.64) == "Medium"

    def test_high_bucket(self, generator: AmbiguityLabelGenerator) -> None:
        assert generator.label_value(0.65) == "High"
        assert generator.label_value(1.0) == "High"

    def test_nan_raises(self, generator: AmbiguityLabelGenerator) -> None:
        with pytest.raises(ValueError):
            generator.label_value(float("nan"))

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError):
            AmbiguityLabelGenerator(low_max=0.7, medium_max=0.5)


class TestLabelDataframe:
    def test_appends_expected_labels(
        self, generator: AmbiguityLabelGenerator, sample_df: pd.DataFrame
    ) -> None:
        labeled = generator.label_dataframe(sample_df)

        assert "ambiguity_label" in labeled.columns
        assert list(labeled["ambiguity_label"]) == [
            "Low",
            "Low",
            "Medium",
            "High",
            "High",
        ]
        # Original dataframe is untouched.
        assert "ambiguity_label" not in sample_df.columns

    def test_missing_column_raises(
        self, generator: AmbiguityLabelGenerator
    ) -> None:
        df = pd.DataFrame({"other_col": [0.1, 0.2]})
        with pytest.raises(KeyError):
            generator.label_dataframe(df)

    def test_nan_values_become_unknown(
        self, generator: AmbiguityLabelGenerator
    ) -> None:
        df = pd.DataFrame({"caption_diversity": [0.2, np.nan, 0.8]})
        labeled = generator.label_dataframe(df)
        assert list(labeled["ambiguity_label"]) == ["Low", "Unknown", "High"]


class TestComputeStatistics:
    def test_counts_and_percentages(
        self, generator: AmbiguityLabelGenerator, sample_df: pd.DataFrame
    ) -> None:
        labeled = generator.label_dataframe(sample_df)
        stats = generator.compute_statistics(labeled)

        assert stats["n_images"] == 5
        assert stats["label_counts"]["Low"] == 2
        assert stats["label_counts"]["Medium"] == 1
        assert stats["label_counts"]["High"] == 2
        assert pytest.approx(stats["label_percentages"]["Low"], abs=0.01) == 40.0
        assert "per_label" in stats
        assert stats["per_label"]["High"]["count"] == 2

    def test_missing_label_column_raises(
        self, generator: AmbiguityLabelGenerator, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(KeyError):
            generator.compute_statistics(sample_df)


class TestSaveOutputs:
    def test_save_csv_writes_file(
        self,
        generator: AmbiguityLabelGenerator,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        labeled = generator.label_dataframe(sample_df)
        destination = tmp_path / "labeled.csv"

        saved_path = generator.save_csv(labeled, destination)

        assert saved_path.exists()
        reloaded = pd.read_csv(saved_path)
        assert "ambiguity_label" in reloaded.columns
        assert len(reloaded) == len(sample_df)

    def test_save_statistics_json_writes_file(
        self,
        generator: AmbiguityLabelGenerator,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        labeled = generator.label_dataframe(sample_df)
        stats = generator.compute_statistics(labeled)
        destination = tmp_path / "stats.json"

        saved_path = generator.save_statistics_json(stats, destination)

        assert saved_path.exists()
        with saved_path.open(encoding="utf-8") as handle:
            reloaded = json.load(handle)
        assert reloaded["n_images"] == 5


class TestPlotDistribution:
    def test_plot_saved_without_display(
        self,
        generator: AmbiguityLabelGenerator,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        labeled = generator.label_dataframe(sample_df)
        destination = tmp_path / "plot.png"

        fake_pyplot = MagicMock()
        fake_fig = MagicMock()
        fake_axes = [MagicMock(), MagicMock()]
        fake_pyplot.subplots.return_value = (fake_fig, fake_axes)

        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.get_backend = MagicMock(return_value="agg")
        fake_matplotlib.use = MagicMock()

        with patch.dict(
            sys.modules,
            {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot},
        ):
            saved_path = generator.plot_distribution(labeled, destination)

        assert saved_path == destination.resolve()
        fake_fig.savefig.assert_called_once()
        fake_pyplot.close.assert_called_once_with(fake_fig)

    def test_plot_without_path_returns_none(
        self, generator: AmbiguityLabelGenerator, sample_df: pd.DataFrame
    ) -> None:
        labeled = generator.label_dataframe(sample_df)

        fake_pyplot = MagicMock()
        fake_fig = MagicMock()
        fake_axes = [MagicMock(), MagicMock()]
        fake_pyplot.subplots.return_value = (fake_fig, fake_axes)

        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.get_backend = MagicMock(return_value="agg")
        fake_matplotlib.use = MagicMock()

        with patch.dict(
            sys.modules,
            {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot},
        ):
            saved_path = generator.plot_distribution(labeled, path=None)

        assert saved_path is None
