"""Rule-based ambiguity label generation from caption diversity scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir

logger = get_logger("pipeline.label_generator")

LABELS = ("Low", "Medium", "High")


class AmbiguityLabelGenerator:
    """Assign categorical ambiguity labels from a caption diversity score.

    Rules (default thresholds)::

        0.00 <= diversity < 0.35  -> "Low"
        0.35 <= diversity < 0.65  -> "Medium"
        0.65 <= diversity <= 1.00 -> "High"

    Args:
        low_max: Upper (exclusive) bound of the "Low" bucket.
        medium_max: Upper (exclusive) bound of the "Medium" bucket;
            values at or above this are "High".
        source_column: Name of the diversity column to read.
        label_column: Name of the label column to write.
    """

    def __init__(
        self,
        low_max: float = 0.35,
        medium_max: float = 0.65,
        *,
        source_column: str = "caption_diversity",
        label_column: str = "ambiguity_label",
    ) -> None:
        if not (0.0 < low_max < medium_max < 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 < low_max < medium_max < 1, "
                f"got low_max={low_max}, medium_max={medium_max}"
            )
        self.low_max = float(low_max)
        self.medium_max = float(medium_max)
        self.source_column = source_column
        self.label_column = label_column

    def label_value(self, diversity: float) -> str:
        """Classify a single caption diversity score.

        Args:
            diversity: Caption diversity score, expected in ``[0, 1]``.

        Returns:
            One of ``"Low"``, ``"Medium"``, or ``"High"``.

        Raises:
            ValueError: If ``diversity`` is NaN.
        """
        if pd.isna(diversity):
            raise ValueError("diversity must not be NaN")
        value = float(diversity)
        if value < self.low_max:
            return "Low"
        if value < self.medium_max:
            return "Medium"
        return "High"

    def label_dataframe(
        self, df: pd.DataFrame, *, column: str | None = None
    ) -> pd.DataFrame:
        """Append an ambiguity label column derived from a diversity column.

        Args:
            df: Input dataset containing the diversity column.
            column: Optional override for the source column name.

        Returns:
            A copy of ``df`` with an added, ordered categorical
            ``label_column``.

        Raises:
            KeyError: If the source column is missing.
        """
        source = column or self.source_column
        if source not in df.columns:
            raise KeyError(f"Column '{source}' not found in dataset")

        result = df.copy()
        values = result[source].astype(float)
        if values.isna().any():
            logger.warning(
                "%s row(s) have NaN %s; labeling as 'Unknown'",
                int(values.isna().sum()),
                source,
            )

        labels = values.apply(
            lambda v: self.label_value(v) if pd.notna(v) else "Unknown"
        )
        categories = [*LABELS, "Unknown"] if labels.eq("Unknown").any() else list(LABELS)
        result[self.label_column] = pd.Categorical(
            labels, categories=categories, ordered=True
        )
        logger.info(
            "Labeled %s rows: %s",
            len(result),
            result[self.label_column].value_counts().to_dict(),
        )
        return result

    def compute_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute label distribution and per-label diversity statistics.

        Args:
            df: Dataset already labeled via :meth:`label_dataframe`.

        Returns:
            Dictionary with overall counts/percentages and per-label
            diversity mean/std/min/max.

        Raises:
            KeyError: If the label column is missing.
        """
        if self.label_column not in df.columns:
            raise KeyError(
                f"Column '{self.label_column}' not found; "
                "call label_dataframe() first"
            )

        n_total = int(len(df))
        counts = df[self.label_column].value_counts()
        counts = counts.reindex(
            [c for c in df[self.label_column].cat.categories], fill_value=0
        )
        percentages = (
            (counts / n_total * 100.0).round(2) if n_total else counts.astype(float)
        )

        per_label: dict[str, dict[str, float | int]] = {}
        for label in counts.index:
            subset = df.loc[df[self.label_column] == label, self.source_column]
            subset = subset.dropna()
            per_label[str(label)] = {
                "count": int(len(subset)),
                "mean_diversity": float(subset.mean()) if len(subset) else float("nan"),
                "std_diversity": float(subset.std()) if len(subset) else float("nan"),
                "min_diversity": float(subset.min()) if len(subset) else float("nan"),
                "max_diversity": float(subset.max()) if len(subset) else float("nan"),
            }

        diversity_all = df[self.source_column].dropna()
        stats: dict[str, Any] = {
            "n_images": n_total,
            "thresholds": {"low_max": self.low_max, "medium_max": self.medium_max},
            "label_counts": {str(k): int(v) for k, v in counts.items()},
            "label_percentages": {str(k): float(v) for k, v in percentages.items()},
            "diversity_overall": {
                "mean": float(diversity_all.mean()) if len(diversity_all) else float("nan"),
                "std": float(diversity_all.std()) if len(diversity_all) else float("nan"),
                "min": float(diversity_all.min()) if len(diversity_all) else float("nan"),
                "max": float(diversity_all.max()) if len(diversity_all) else float("nan"),
            },
            "per_label": per_label,
        }
        return stats

    def save_statistics_json(
        self, statistics: dict[str, Any], path: str | Path
    ) -> Path:
        """Write the statistics dictionary to a JSON file.

        Args:
            statistics: Output of :meth:`compute_statistics`.
            path: Destination ``.json`` path.

        Returns:
            Resolved path written to disk.
        """
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".json")
        ensure_dir(destination.parent)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(statistics, handle, indent=2, ensure_ascii=False)
        logger.info("Saved label statistics to %s", destination)
        return destination.resolve()

    def save_csv(self, df: pd.DataFrame, path: str | Path) -> Path:
        """Save the labeled dataset to CSV.

        Args:
            df: Labeled dataset.
            path: Destination ``.csv`` path.

        Returns:
            Resolved path written to disk.
        """
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".csv")
        ensure_dir(destination.parent)
        df.to_csv(destination, index=False)
        logger.info("Saved labeled dataset to %s (%s rows)", destination, len(df))
        return destination.resolve()

    def plot_distribution(
        self,
        df: pd.DataFrame,
        path: str | Path | None = None,
        *,
        show: bool = False,
    ) -> Path | None:
        """Plot the diversity histogram (with threshold lines) and label counts.

        Args:
            df: Labeled dataset.
            path: Optional destination ``.png`` path.
            show: If ``True``, display an interactive window.

        Returns:
            Resolved save path when ``path`` is provided, else ``None``.
        """
        import matplotlib

        if not show and matplotlib.get_backend().lower() != "agg":
            try:
                matplotlib.use("Agg", force=False)
            except Exception:  # noqa: BLE001 - backend choice is best-effort
                pass
        import matplotlib.pyplot as plt

        colors = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728", "Unknown": "#7f7f7f"}
        categories = list(df[self.label_column].cat.categories)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: diversity histogram with threshold boundaries.
        axes[0].hist(
            df[self.source_column].dropna(),
            bins=20,
            range=(0.0, 1.0),
            color="#4c72b0",
            edgecolor="white",
        )
        axes[0].axvline(self.low_max, color="black", linestyle="--", linewidth=1)
        axes[0].axvline(self.medium_max, color="black", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Caption diversity")
        axes[0].set_ylabel("Number of images")
        axes[0].set_title(
            f"Diversity distribution (thresholds={self.low_max}, {self.medium_max})"
        )

        # Panel 2: label counts.
        counts = df[self.label_column].value_counts().reindex(categories, fill_value=0)
        bar_colors = [colors.get(str(c), "#4c72b0") for c in categories]
        axes[1].bar([str(c) for c in categories], counts.values, color=bar_colors)
        for index, value in enumerate(counts.values):
            axes[1].text(index, value, str(int(value)), ha="center", va="bottom")
        axes[1].set_xlabel("Ambiguity label")
        axes[1].set_ylabel("Number of images")
        axes[1].set_title("Label distribution")

        fig.tight_layout()

        saved: Path | None = None
        if path is not None:
            destination = Path(path)
            if destination.suffix == "":
                destination = destination.with_suffix(".png")
            ensure_dir(destination.parent)
            fig.savefig(destination, dpi=150, bbox_inches="tight")
            saved = destination.resolve()
            logger.info("Saved label distribution plot to %s", saved)

        if show:
            plt.show()
        else:
            plt.close(fig)
        return saved

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"low_max={self.low_max}, medium_max={self.medium_max})"
        )
