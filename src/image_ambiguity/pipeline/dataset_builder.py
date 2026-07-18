"""Merge caption-diversity and OpenCV features into an ML-ready dataset."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.caption_diversity import CaptionDiversityAnalyzer
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir, timed

logger = get_logger("pipeline.dataset_builder")

DATASET_COLUMNS = [
    "image_id",
    "average_similarity",
    "minimum_similarity",
    "maximum_similarity",
    "std_similarity",
    "caption_diversity",
    "edge_density",
    "entropy",
    "brightness",
    "contrast",
    "color_variance",
    "texture",
]

DEFAULT_SAMPLE_SIZE = 300


class MLDatasetBuilder:
    """Build a merged, tabular ML dataset from two feature families.

    For each COCO image, this class combines:

    - Caption-diversity statistics (from
      :class:`SentenceEmbeddingGenerator` + :class:`CaptionDiversityAnalyzer`)
    - Classical OpenCV image features (from :class:`OpenCVFeatureExtractor`)

    into a single row, then assembles all rows into a
    :class:`pandas.DataFrame` with missing values imputed.

    Args:
        annotation_file: Path to a COCO captions_*.json file.
        image_dir: Directory containing the matching image files.
        embedding_generator: Optional pre-configured embedding generator.
        diversity_analyzer: Optional pre-configured diversity analyzer.
        cv_extractor: Optional pre-configured OpenCV feature extractor.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_dir: str | Path,
        *,
        embedding_generator: SentenceEmbeddingGenerator | None = None,
        diversity_analyzer: CaptionDiversityAnalyzer | None = None,
        cv_extractor: OpenCVFeatureExtractor | None = None,
    ) -> None:
        self.loader = CocoDatasetLoader(annotation_file, image_dir)
        self.image_dir = Path(image_dir)
        self.embedding_generator = embedding_generator or SentenceEmbeddingGenerator()
        self.diversity_analyzer = diversity_analyzer or CaptionDiversityAnalyzer()
        self.cv_extractor = cv_extractor or OpenCVFeatureExtractor()
        self._model_loaded = False

    def _ensure_model(self) -> None:
        if not self._model_loaded:
            self.embedding_generator.load_model()
            self._model_loaded = True

    def select_image_ids(
        self, sample_size: int | None, seed: int | None = None
    ) -> list[int]:
        """Pick image ids to include in the dataset.

        Args:
            sample_size: Number of images to sample. ``None`` uses all
                images in the annotation file.
            seed: Random seed for reproducible sampling.

        Returns:
            List of COCO image ids.
        """
        if self.loader.coco is None:
            self.loader.load_annotations()
        assert self.loader.coco is not None

        image_ids = list(self.loader.coco.getImgIds())
        if sample_size is not None and sample_size < len(image_ids):
            rng = random.Random(seed)
            image_ids = rng.sample(image_ids, sample_size)
        return image_ids

    def build(
        self,
        image_ids: Sequence[int] | None = None,
        *,
        sample_size: int | None = DEFAULT_SAMPLE_SIZE,
        seed: int | None = 42,
    ) -> pd.DataFrame:
        """Build the merged dataset.

        Args:
            image_ids: Explicit image ids to process. If ``None``, ids are
                sampled via :meth:`select_image_ids`.
            sample_size: Number of images to sample when ``image_ids`` is
                not provided.
            seed: Random seed for reproducible sampling.

        Returns:
            DataFrame with one row per image and columns from
            :data:`DATASET_COLUMNS`.
        """
        if self.loader.coco is None:
            self.loader.load_annotations()
        assert self.loader.coco is not None

        ids = (
            list(image_ids)
            if image_ids is not None
            else self.select_image_ids(sample_size, seed)
        )

        captions_by_image = self._collect_captions(ids)
        embeddings_by_image = self._encode_all_captions(captions_by_image)

        rows: list[dict[str, Any]] = []
        with timed(f"build_dataset:n={len(ids)}"):
            for image_id in ids:
                if image_id not in self.loader.coco.imgs:
                    logger.warning("Skipping unknown image_id=%s", image_id)
                    continue
                rows.append(
                    self._build_row(
                        image_id,
                        captions_by_image.get(image_id, []),
                        embeddings_by_image.get(image_id),
                    )
                )

        df = pd.DataFrame(rows, columns=DATASET_COLUMNS)
        df = self.handle_missing_values(df)
        logger.info("Built dataset with %s rows", len(df))
        return df

    def _collect_captions(self, ids: Sequence[int]) -> dict[int, list[str]]:
        captions_by_image: dict[int, list[str]] = {}
        assert self.loader.coco is not None
        for image_id in ids:
            if image_id not in self.loader.coco.imgs:
                continue
            captions_by_image[image_id] = self.loader.get_captions(image_id)
        return captions_by_image

    def _encode_all_captions(
        self, captions_by_image: dict[int, list[str]]
    ) -> dict[int, np.ndarray]:
        """Batch-encode every eligible image's captions in one model call."""
        flat_captions: list[str] = []
        spans: dict[int, tuple[int, int]] = {}
        for image_id, captions in captions_by_image.items():
            if len(captions) < 2:
                continue
            start = len(flat_captions)
            flat_captions.extend(captions)
            spans[image_id] = (start, len(flat_captions))

        if not flat_captions:
            return {}

        self._ensure_model()
        with timed(f"encode_all_captions:n={len(flat_captions)}"):
            all_embeddings = self.embedding_generator.generate_embeddings(
                flat_captions
            )

        return {
            image_id: all_embeddings[start:end]
            for image_id, (start, end) in spans.items()
        }

    def _build_row(
        self,
        image_id: int,
        captions: list[str],
        embeddings: np.ndarray | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"image_id": image_id}
        row.update(self._diversity_columns(image_id, captions, embeddings))
        row.update(self._cv_columns(image_id))
        return row

    def _diversity_columns(
        self,
        image_id: int,
        captions: list[str],
        embeddings: np.ndarray | None,
    ) -> dict[str, Any]:
        if embeddings is None or len(captions) < 2:
            logger.warning(
                "image_id=%s has %s caption(s); diversity set to NaN",
                image_id,
                len(captions),
            )
            return {
                "average_similarity": np.nan,
                "minimum_similarity": np.nan,
                "maximum_similarity": np.nan,
                "std_similarity": np.nan,
                "caption_diversity": np.nan,
            }
        try:
            metrics = self.diversity_analyzer.compute(embeddings)
            return {
                "average_similarity": metrics.average_similarity,
                "minimum_similarity": metrics.min_similarity,
                "maximum_similarity": metrics.max_similarity,
                "std_similarity": metrics.std_similarity,
                "caption_diversity": metrics.diversity_score,
            }
        except Exception:  # noqa: BLE001 - keep pipeline resilient per-row
            logger.exception(
                "Caption diversity computation failed for image_id=%s", image_id
            )
            return {
                "average_similarity": np.nan,
                "minimum_similarity": np.nan,
                "maximum_similarity": np.nan,
                "std_similarity": np.nan,
                "caption_diversity": np.nan,
            }

    def _cv_columns(self, image_id: int) -> dict[str, Any]:
        assert self.loader.coco is not None
        try:
            file_name = self.loader.coco.loadImgs(image_id)[0]["file_name"]
            image_path = self.image_dir / file_name
            features = self.cv_extractor.extract(image_path)
            return {
                "edge_density": features["edge_density"],
                "entropy": features["entropy"],
                "brightness": features["brightness"],
                "contrast": features["contrast"],
                "color_variance": features["color_variance"],
                "texture": features["texture"],
            }
        except Exception:  # noqa: BLE001 - keep pipeline resilient per-row
            logger.exception(
                "CV feature extraction failed for image_id=%s", image_id
            )
            return {
                "edge_density": np.nan,
                "entropy": np.nan,
                "brightness": np.nan,
                "contrast": np.nan,
                "color_variance": np.nan,
                "texture": np.nan,
            }

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing numeric values without dropping any rows.

        Each NaN cell (e.g. from images with fewer than two captions, or
        an unreadable image file) is filled with that column's median
        across the dataset, falling back to ``0.0`` if the entire column
        is missing.

        Args:
            df: Dataset possibly containing NaN values.

        Returns:
            Dataset with all NaN values imputed and ``image_id`` coerced
            to ``int64``.
        """
        if df.empty:
            return df

        numeric_cols = [c for c in df.columns if c != "image_id"]
        n_missing = int(df[numeric_cols].isna().sum().sum())
        if n_missing:
            logger.warning(
                "Imputing %s missing value(s) with column medians", n_missing
            )
        for col in numeric_cols:
            if df[col].isna().any():
                median = df[col].median()
                fill_value = 0.0 if pd.isna(median) else median
                df[col] = df[col].fillna(fill_value)

        df["image_id"] = df["image_id"].astype("int64")
        return df

    def save_csv(self, df: pd.DataFrame, path: str | Path) -> Path:
        """Save the dataset to CSV.

        Args:
            df: Dataset produced by :meth:`build`.
            path: Destination ``.csv`` path.

        Returns:
            Resolved path written to disk.
        """
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".csv")
        ensure_dir(destination.parent)
        df.to_csv(destination, index=False)
        logger.info("Saved ML dataset to %s (%s rows)", destination, len(df))
        return destination.resolve()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"annotation_file={self.loader.annotation_file!s}, "
            f"image_dir={self.image_dir!s})"
        )
