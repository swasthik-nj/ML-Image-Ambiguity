"""Build a combined Vision + CLIP dataset for ambiguity prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.features.clip_embeddings import CLIPFeatureExtractor

logger = get_logger("pipeline.vision_clip_dataset_builder")

VISION_FEATURE_COLUMNS = (
    "edge_density",
    "entropy",
    "brightness",
    "contrast",
    "color_variance",
    "texture",
)

CLIP_COLUMNS = tuple(f"clip_{i}" for i in range(512))


class VisionClipDatasetBuilder:
    """Build a tabular ML dataset from OpenCV and CLIP features.

    Reads the merged dataset (e.g. human_dataset.csv) to get the image IDs
    and ambiguity labels. For each image, it extracts the 6 OpenCV features
    (or uses the existing ones from the CSV if they are valid) and extracts
    the 512-dimensional CLIP embedding.
    """

    def __init__(self, input_csv: str | Path, image_dir: str | Path) -> None:
        self.input_csv = Path(input_csv)
        self.image_dir = Path(image_dir)
        self.cv_extractor = OpenCVFeatureExtractor()
        self.clip_extractor = CLIPFeatureExtractor()

    def _resolve_image_path(self, image_id: Any) -> Path:
        """Resolve an image_id to a physical path."""
        # Try COCO format
        try:
            filename = f"{int(image_id):012d}.jpg"
            path = self.image_dir / filename
            if path.exists():
                return path
        except ValueError:
            pass
            
        # Fallback for custom images if they exist directly
        path = self.image_dir / str(image_id)
        if path.exists():
            return path
            
        raise FileNotFoundError(f"Could not locate image for id {image_id} in {self.image_dir}")

    def build(self) -> pd.DataFrame:
        """Build the combined dataset."""
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input dataset not found at {self.input_csv}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found at {self.image_dir}")

        df = pd.read_csv(self.input_csv)
        
        if "image_id" not in df.columns or "ambiguity_label" not in df.columns:
            raise KeyError("Input dataset must contain 'image_id' and 'ambiguity_label'")

        rows = []
        logger.info(f"Extracting Vision+CLIP features for {len(df)} images...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            image_id = row["image_id"]
            label = row["ambiguity_label"]
            
            try:
                img_path = self._resolve_image_path(image_id)
                
                # We can reuse the CV features if they already exist in the CSV
                # to save time, or we can re-extract them. Let's reuse them if present.
                has_cv = all(c in row for c in VISION_FEATURE_COLUMNS) and not any(pd.isna(row.get(c)) for c in VISION_FEATURE_COLUMNS)
                
                if has_cv:
                    cv_feats = {c: row[c] for c in VISION_FEATURE_COLUMNS}
                else:
                    cv_feats = self.cv_extractor.extract(img_path)
                
                # Extract CLIP features
                clip_feats = self.clip_extractor.extract(img_path)
                
                # Combine
                out_row = {"image_id": image_id}
                out_row.update(cv_feats)
                for i, val in enumerate(clip_feats):
                    out_row[f"clip_{i}"] = val
                out_row["ambiguity_label"] = label
                
                rows.append(out_row)
            except Exception as e:
                logger.error(f"Failed to process image {image_id}: {e}")

        combined_df = pd.DataFrame(rows)
        logger.info(f"Successfully built dataset with {len(combined_df)} rows and {len(combined_df.columns)} columns.")
        return combined_df

    def save_csv(self, df: pd.DataFrame, path: str | Path) -> Path:
        """Save the dataset to CSV."""
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".csv")
        ensure_dir(destination.parent)
        df.to_csv(destination, index=False)
        logger.info("Saved Vision+CLIP dataset to %s (%s rows)", destination, len(df))
        return destination.resolve()
