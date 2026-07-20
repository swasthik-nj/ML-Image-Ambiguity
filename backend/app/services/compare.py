"""Human vs AI dataset diversity comparison for the frontend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from image_ambiguity.config import get_settings
from image_ambiguity.logging_config import get_logger

logger = get_logger("backend.services.compare")


def _histogram(series: pd.Series, bins: int = 8) -> list[dict[str, Any]]:
    values = series.dropna().astype(float).to_numpy()
    if values.size == 0:
        return []
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    rows: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        left = edges[index]
        right = edges[index + 1]
        rows.append(
            {
                "bin": f"{left:.2f}-{right:.2f}",
                "count": int(count),
            }
        )
    return rows


def compare_datasets(
    *,
    human_path: Path | None = None,
    ai_path: Path | None = None,
) -> dict[str, Any]:
    """Return mean caption diversity and histograms for human/AI CSVs."""
    settings = get_settings()
    human_csv = human_path or (settings.dataset_dir / "human_dataset.csv")
    ai_csv = ai_path or (settings.dataset_dir / "ai_dataset.csv")

    human_df = (
        pd.read_csv(human_csv)
        if human_csv.is_file()
        else pd.DataFrame(columns=["caption_diversity"])
    )
    ai_df = (
        pd.read_csv(ai_csv)
        if ai_csv.is_file()
        else pd.DataFrame(columns=["caption_diversity"])
    )

    human_available = (
        not human_df.empty and "caption_diversity" in human_df.columns
    )
    ai_available = not ai_df.empty and "caption_diversity" in ai_df.columns

    human_diversity = (
        float(human_df["caption_diversity"].mean()) if human_available else None
    )
    ai_diversity = (
        float(ai_df["caption_diversity"].mean()) if ai_available else None
    )

    n_compared = 0
    if (
        human_available
        and ai_available
        and "image_id" in human_df.columns
        and "image_id" in ai_df.columns
    ):
        merged = human_df.merge(ai_df, on="image_id", how="inner", suffixes=("_h", "_a"))
        n_compared = int(len(merged))
        if n_compared:
            human_diversity = float(merged["caption_diversity_h"].mean())
            ai_diversity = float(merged["caption_diversity_a"].mean())

    if human_available and ai_available:
        message = (
            f"Compared {n_compared or min(len(human_df), len(ai_df))} images "
            "using caption_diversity means from human_dataset.csv and "
            "ai_dataset.csv."
        )
    elif human_available:
        message = (
            "Loaded human_dataset.csv. ai_dataset.csv is missing — "
            "build it with the AI caption diversity pipeline to enable "
            "full comparison."
        )
    elif ai_available:
        message = "Loaded ai_dataset.csv only; human_dataset.csv is missing."
    else:
        message = (
            "No dataset CSVs found under dataset/. "
            "Run create_dataset / create_ai_dataset first."
        )

    logger.info(
        "Compare datasets human=%s ai=%s",
        human_available,
        ai_available,
    )
    return {
        "human_diversity": human_diversity,
        "ai_diversity": ai_diversity,
        "n_human": int(len(human_df)) if human_available else 0,
        "n_ai": int(len(ai_df)) if ai_available else 0,
        "n_compared": n_compared,
        "human_available": human_available,
        "ai_available": ai_available,
        "message": message,
        "human_histogram": _histogram(human_df["caption_diversity"])
        if human_available
        else [],
        "ai_histogram": _histogram(ai_df["caption_diversity"])
        if ai_available
        else [],
    }
