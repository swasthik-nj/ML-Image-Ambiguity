"""Compare caption-diversity statistics across human and AI datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_diversity_frame(path: str | Path) -> pd.DataFrame:
    """Load a dataset CSV and validate diversity columns."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    if "caption_diversity" not in df.columns:
        raise ValueError(f"{dataset_path} is missing caption_diversity column")
    if df.empty:
        raise ValueError(f"{dataset_path} contains no rows")
    return df


def compare_mean_diversity(
    human_path: str | Path,
    ai_path: str | Path,
) -> tuple[float, float, int]:
    """Compare mean caption diversity on shared ``image_id`` rows when possible.

    Returns:
        ``(human_mean, ai_mean, n_images)``
    """
    human_df = load_diversity_frame(human_path)
    ai_df = load_diversity_frame(ai_path)

    if "image_id" in human_df.columns and "image_id" in ai_df.columns:
        merged = human_df.merge(
            ai_df,
            on="image_id",
            how="inner",
            suffixes=("_human", "_ai"),
        )
        if not merged.empty:
            return (
                float(merged["caption_diversity_human"].mean()),
                float(merged["caption_diversity_ai"].mean()),
                int(len(merged)),
            )

    return (
        float(human_df["caption_diversity"].mean()),
        float(ai_df["caption_diversity"].mean()),
        int(min(len(human_df), len(ai_df))),
    )
