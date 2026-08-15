"""Build a vision-only dataset for ambiguity prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir

logger = get_logger("pipeline.vision_dataset_builder")

VISION_FEATURE_COLUMNS = (
    "edge_density",
    "entropy",
    "brightness",
    "contrast",
    "color_variance",
    "texture",
)

class VisionOnlyDatasetBuilder:
    """Build a tabular ML dataset from existing OpenCV features.

    Reads the merged dataset (e.g. human_dataset.csv), extracts only the
    OpenCV feature columns and the ambiguity label, imputes any missing
    values with the median, and saves the result. This dataset is strictly
    vision-only, meaning all caption/BLIP-derived columns are dropped.

    Args:
        input_csv: Path to the existing labeled dataset (e.g. human_dataset.csv).
    """

    def __init__(self, input_csv: str | Path) -> None:
        self.input_csv = Path(input_csv)

    def build(self) -> pd.DataFrame:
        """Build the vision-only dataset.

        Returns:
            DataFrame with image_id, OpenCV features, and ambiguity_label.
        """
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input dataset not found at {self.input_csv}")

        df = pd.read_csv(self.input_csv)
        
        # We need image_id, the 6 OpenCV features, and ambiguity_label.
        required_cols = ["image_id"] + list(VISION_FEATURE_COLUMNS)
        if "ambiguity_label" in df.columns:
            required_cols.append("ambiguity_label")
            
        missing_cols = [c for c in required_cols if c not in df.columns and c != "ambiguity_label"]
        if missing_cols:
            raise KeyError(f"Input dataset is missing required vision columns: {missing_cols}")

        vision_df = df[[c for c in required_cols if c in df.columns]].copy()
        
        # Median imputation for missing numeric values (never drop rows)
        numeric_cols = [c for c in vision_df.columns if c not in ("image_id", "ambiguity_label")]
        for col in numeric_cols:
            if vision_df[col].isna().any():
                median = vision_df[col].median()
                fill_value = 0.0 if pd.isna(median) else median
                vision_df[col] = vision_df[col].fillna(fill_value)
                
        logger.info("Built vision-only dataset with %s rows", len(vision_df))
        return vision_df

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
        logger.info("Saved vision-only dataset to %s (%s rows)", destination, len(df))
        return destination.resolve()
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_csv={self.input_csv!s})"
