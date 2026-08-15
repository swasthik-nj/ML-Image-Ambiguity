"""Utility script to extract OpenCV features from custom images and append them to the dataset."""

import argparse
from pathlib import Path
import pandas as pd
import uuid

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.logging_config import setup_logging, get_logger

logger = get_logger("add_custom_images")

def parse_args():
    parser = argparse.ArgumentParser(description="Add custom images to human_dataset.csv")
    parser.add_argument(
        "--images-dir", 
        type=Path, 
        required=True, 
        help="Directory containing the new images to add"
    )
    parser.add_argument(
        "--label", 
        type=str, 
        default="High",
        choices=["Low", "Medium", "High"],
        help="The ambiguity label to assign to all these images"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "human_dataset.csv",
        help="Path to the dataset CSV to append to"
    )
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()

    if not args.images_dir.is_dir():
        logger.error(f"Images directory not found: {args.images_dir}")
        return

    if not args.dataset.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        return

    extractor = OpenCVFeatureExtractor()
    new_rows = []
    
    image_paths = list(args.images_dir.glob("*.*"))
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    logger.info(f"Found {len(image_paths)} files in {args.images_dir}. Extracting features...")
    
    for path in image_paths:
        if path.suffix.lower() not in valid_exts:
            continue
            
        try:
            features = extractor.extract(path)
            
            # Generate a pseudo-numeric image_id to match COCO style
            image_id = str(uuid.uuid4().int)[:10]
            
            row = {
                "image_id": image_id,
                "average_similarity": pd.NA,
                "minimum_similarity": pd.NA,
                "maximum_similarity": pd.NA,
                "std_similarity": pd.NA,
                "caption_diversity": pd.NA,
                "edge_density": features["edge_density"],
                "entropy": features["entropy"],
                "brightness": features["brightness"],
                "contrast": features["contrast"],
                "color_variance": features["color_variance"],
                "texture": features["texture"],
                "ambiguity_label": args.label
            }
            new_rows.append(row)
            logger.info(f"Processed: {path.name}")
        except Exception as e:
            logger.error(f"Failed to process {path.name}: {e}")

    if not new_rows:
        logger.warning("No valid images were processed.")
        return

    # Append to dataset
    new_df = pd.DataFrame(new_rows)
    existing_df = pd.read_csv(args.dataset)
    
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(args.dataset, index=False)
    
    logger.info(f"Successfully added {len(new_df)} new '{args.label}' images to {args.dataset}")
    logger.info("You can now run `python src/train_vision_only.py` to train with the new data!")

if __name__ == "__main__":
    main()
