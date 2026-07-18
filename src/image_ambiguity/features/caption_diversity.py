"""Caption diversity metrics derived from Sentence-BERT embeddings."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir

logger = get_logger("features.caption_diversity")


@dataclass(frozen=True)
class DiversityMetrics:
    """Caption diversity summary for one image's caption set.

    Attributes:
        pairwise_similarity: Upper-triangle pairwise cosine similarities
            as a 1-D array of length ``n_pairs``.
        similarity_matrix: Full ``(n, n)`` cosine similarity matrix.
        average_similarity: Mean of pairwise cosine similarities.
        min_similarity: Minimum pairwise cosine similarity.
        max_similarity: Maximum pairwise cosine similarity.
        std_similarity: Standard deviation of pairwise similarities.
        diversity_score: ``1 - average_similarity``.
        n_captions: Number of caption embeddings.
        n_pairs: Number of unique caption pairs.
    """

    pairwise_similarity: np.ndarray
    similarity_matrix: np.ndarray
    average_similarity: float
    min_similarity: float
    max_similarity: float
    std_similarity: float
    diversity_score: float
    n_captions: int
    n_pairs: int


class CaptionDiversityAnalyzer:
    """Compute caption diversity scores from Sentence-BERT embeddings.

    Diversity is defined as::

        Diversity = 1 - Average Similarity

    where Average Similarity is the mean of all unique pairwise cosine
    similarities among the caption embeddings for one image.

    Args:
        eps: Numerical floor used when L2-normalizing embeddings.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        if eps <= 0:
            raise ValueError("eps must be > 0")
        self.eps = eps

    def pairwise_cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Return the full pairwise cosine similarity matrix.

        Args:
            embeddings: Array of shape ``(n_captions, dim)``.

        Returns:
            Similarity matrix of shape ``(n_captions, n_captions)``.
        """
        array = self._validate_embeddings(embeddings)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        normalized = array / np.clip(norms, self.eps, None)
        return normalized @ normalized.T

    def compute(self, embeddings: np.ndarray) -> DiversityMetrics:
        """Compute diversity metrics for a set of caption embeddings.

        Args:
            embeddings: Array of shape ``(n_captions, dim)`` with
                ``n_captions >= 2``.

        Returns:
            :class:`DiversityMetrics` with pairwise and aggregate scores.

        Raises:
            ValueError: If fewer than two embeddings are provided.
        """
        array = self._validate_embeddings(embeddings)
        if array.shape[0] < 2:
            raise ValueError(
                "At least two caption embeddings are required to "
                "compute pairwise diversity."
            )

        similarity_matrix = self.pairwise_cosine_similarity(array)
        n = similarity_matrix.shape[0]
        upper = similarity_matrix[np.triu_indices(n, k=1)]
        average = float(np.mean(upper))
        diversity = float(1.0 - average)

        metrics = DiversityMetrics(
            pairwise_similarity=upper.astype(np.float64, copy=False),
            similarity_matrix=similarity_matrix.astype(np.float64, copy=False),
            average_similarity=average,
            min_similarity=float(np.min(upper)),
            max_similarity=float(np.max(upper)),
            std_similarity=float(np.std(upper)),
            diversity_score=diversity,
            n_captions=n,
            n_pairs=int(upper.size),
        )
        logger.info(
            "Diversity computed: n=%s pairs=%s avg_sim=%.4f diversity=%.4f",
            metrics.n_captions,
            metrics.n_pairs,
            metrics.average_similarity,
            metrics.diversity_score,
        )
        return metrics

    def to_dict(
        self,
        metrics: DiversityMetrics,
        *,
        image_id: int | None = None,
        captions: Sequence[str] | None = None,
        round_digits: int = 6,
    ) -> dict[str, Any]:
        """Serialize metrics to a JSON-friendly dictionary.

        Args:
            metrics: Computed diversity metrics.
            image_id: Optional COCO image id for provenance.
            captions: Optional caption texts aligned with rows.
            round_digits: Decimal places for floating-point fields.

        Returns:
            Nested dictionary suitable for ``json.dump``.
        """
        payload: dict[str, Any] = {
            "n_captions": metrics.n_captions,
            "n_pairs": metrics.n_pairs,
            "average_similarity": round(metrics.average_similarity, round_digits),
            "min_similarity": round(metrics.min_similarity, round_digits),
            "max_similarity": round(metrics.max_similarity, round_digits),
            "std_similarity": round(metrics.std_similarity, round_digits),
            "diversity_score": round(metrics.diversity_score, round_digits),
            "formula": "diversity_score = 1 - average_similarity",
            "pairwise_similarity": [
                round(float(value), round_digits)
                for value in metrics.pairwise_similarity.tolist()
            ],
            "similarity_matrix": [
                [round(float(value), round_digits) for value in row]
                for row in metrics.similarity_matrix.tolist()
            ],
        }
        if image_id is not None:
            payload["image_id"] = image_id
        if captions is not None:
            payload["captions"] = list(captions)
        return payload

    def save_json(
        self,
        metrics: DiversityMetrics,
        path: str | Path,
        *,
        image_id: int | None = None,
        captions: Sequence[str] | None = None,
    ) -> Path:
        """Write metrics to a JSON file.

        Args:
            metrics: Computed diversity metrics.
            path: Destination ``.json`` path.
            image_id: Optional image id stored in the payload.
            captions: Optional caption texts stored in the payload.

        Returns:
            Resolved path written to disk.
        """
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".json")
        ensure_dir(destination.parent)
        payload = self.to_dict(
            metrics, image_id=image_id, captions=captions
        )
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        logger.info("Saved diversity JSON to %s", destination)
        return destination.resolve()

    def save_csv(
        self,
        metrics: DiversityMetrics,
        path: str | Path,
        *,
        image_id: int | None = None,
    ) -> Path:
        """Write a flat summary row to CSV.

        Args:
            metrics: Computed diversity metrics.
            path: Destination ``.csv`` path.
            image_id: Optional image id stored as the first column.

        Returns:
            Resolved path written to disk.
        """
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".csv")
        ensure_dir(destination.parent)

        fieldnames = [
            "image_id",
            "n_captions",
            "n_pairs",
            "average_similarity",
            "min_similarity",
            "max_similarity",
            "std_similarity",
            "diversity_score",
        ]
        row = {
            "image_id": "" if image_id is None else image_id,
            "n_captions": metrics.n_captions,
            "n_pairs": metrics.n_pairs,
            "average_similarity": f"{metrics.average_similarity:.6f}",
            "min_similarity": f"{metrics.min_similarity:.6f}",
            "max_similarity": f"{metrics.max_similarity:.6f}",
            "std_similarity": f"{metrics.std_similarity:.6f}",
            "diversity_score": f"{metrics.diversity_score:.6f}",
        }

        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

        logger.info("Saved diversity CSV to %s", destination)
        return destination.resolve()

    def visualize(
        self,
        metrics: DiversityMetrics,
        path: str | Path | None = None,
        *,
        captions: Sequence[str] | None = None,
        title: str | None = None,
        show: bool = False,
    ) -> Path | None:
        """Plot a pairwise cosine-similarity heatmap.

        Args:
            metrics: Computed diversity metrics.
            path: Optional destination image path (``.png`` recommended).
            captions: Optional short labels for axes.
            title: Optional plot title.
            show: If ``True``, call ``plt.show()``.

        Returns:
            Resolved save path when ``path`` is provided, else ``None``.
        """
        import matplotlib

        # Prefer a non-interactive backend so CLI demos work headlessly.
        if not show and matplotlib.get_backend().lower() != "agg":
            try:
                matplotlib.use("Agg", force=False)
            except Exception:  # noqa: BLE001 - backend choice is best-effort
                pass
        import matplotlib.pyplot as plt

        matrix = metrics.similarity_matrix
        n = metrics.n_captions
        labels = self._axis_labels(n, captions)

        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Caption")
        ax.set_ylabel("Caption")

        plot_title = title or (
            f"Caption similarity  |  avg={metrics.average_similarity:.2f}  "
            f"diversity={metrics.diversity_score:.2f}"
        )
        ax.set_title(plot_title)

        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] < 0.55 else "black",
                    fontsize=8,
                )

        fig.tight_layout()

        saved: Path | None = None
        if path is not None:
            destination = Path(path)
            if destination.suffix == "":
                destination = destination.with_suffix(".png")
            ensure_dir(destination.parent)
            fig.savefig(destination, dpi=150, bbox_inches="tight")
            saved = destination.resolve()
            logger.info("Saved diversity heatmap to %s", saved)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
        array = np.asarray(embeddings, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (n_captions, dim), got shape {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError("embeddings must contain at least one row")
        if not np.isfinite(array).all():
            raise ValueError("embeddings contain NaN or Inf values")
        return array

    @staticmethod
    def _axis_labels(n: int, captions: Sequence[str] | None) -> list[str]:
        if captions is None:
            return [f"C{i + 1}" for i in range(n)]
        if len(captions) != n:
            raise ValueError(
                f"len(captions)={len(captions)} does not match n_captions={n}"
            )
        labels: list[str] = []
        for index, caption in enumerate(captions, start=1):
            short = " ".join(str(caption).split())
            if len(short) > 28:
                short = short[:25] + "..."
            labels.append(f"C{index}: {short}")
        return labels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps})"
