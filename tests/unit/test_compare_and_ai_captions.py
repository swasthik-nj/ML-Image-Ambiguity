"""Unit tests for AI caption helpers and human/AI diversity compare."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from image_ambiguity.pipeline.ai_captions import flatten_generated_captions
from image_ambiguity.pipeline.compare_datasets import compare_mean_diversity


class TestFlattenGeneratedCaptions:
    def test_preserves_order_and_deduplicates(self) -> None:
        generated = {
            "beam_search": ["a red bike", "a blue car"],
            "top_k": ["a red bike", "people walking"],
            "nucleus": ["a busy street"],
        }
        captions = flatten_generated_captions(generated)
        assert captions == [
            "a red bike",
            "a blue car",
            "people walking",
            "a busy street",
        ]


class TestCompareMeanDiversity:
    def test_shared_image_ids(self, tmp_path: Path) -> None:
        human = pd.DataFrame(
            {
                "image_id": [1, 2, 3],
                "caption_diversity": [0.10, 0.20, 0.30],
            }
        )
        ai = pd.DataFrame(
            {
                "image_id": [2, 3, 4],
                "caption_diversity": [0.40, 0.50, 0.90],
            }
        )
        human_path = tmp_path / "human_dataset.csv"
        ai_path = tmp_path / "ai_dataset.csv"
        human.to_csv(human_path, index=False)
        ai.to_csv(ai_path, index=False)

        human_mean, ai_mean, n_images = compare_mean_diversity(
            human_path, ai_path
        )
        assert n_images == 2
        assert human_mean == pytest.approx(0.25)
        assert ai_mean == pytest.approx(0.45)
