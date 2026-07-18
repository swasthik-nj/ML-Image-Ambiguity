"""Runnable demo for OpenCV image feature extraction.

Usage (from project root)::

    python src/feature_extractor.py
    python src/feature_extractor.py --image-id 397133
    python src/feature_extractor.py --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_ambiguity.config import get_settings
from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.logging_config import setup_logging

DEFAULT_IMAGE_ID = 397133
DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT / "dataset" / "annotations" / "captions_val2017.json"
)
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "val2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OpenCV image features for a COCO / local image."
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=DEFAULT_IMAGE_ID,
        help=f"COCO image id (default: {DEFAULT_IMAGE_ID})",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional direct path to an image file (overrides --image-id)",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help="Path to captions/instances JSON used to resolve file names",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing image files",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the visual debugging panel",
    )
    return parser.parse_args()


def resolve_image_path(args: argparse.Namespace) -> tuple[Path, int | None, str]:
    """Return (image_path, image_id, file_name)."""
    if args.image is not None:
        path = Path(args.image)
        return path, None, path.name

    image_dir = Path(args.image_dir)
    annotation_file = Path(args.annotation_file)

    # Prefer COCO metadata when available; fall back to padded file name.
    file_name = f"{args.image_id:012d}.jpg"
    if annotation_file.is_file() and image_dir.is_dir():
        try:
            loader = CocoDatasetLoader(annotation_file, image_dir)
            loader.load_annotations()
            assert loader.coco is not None
            imgs = loader.coco.loadImgs(args.image_id)
            if imgs:
                file_name = imgs[0]["file_name"]
        except Exception:  # noqa: BLE001 - fall back to conventional name
            pass

    path = image_dir / file_name
    return path, args.image_id, file_name


def main() -> None:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    settings.ensure_directories()

    features_dir = settings.results_dir / "cv_features"
    features_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = settings.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    image_path, image_id, file_name = resolve_image_path(args)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}. "
            "Pass --image path/to/file.jpg or use a val2017 image id."
        )

    extractor = OpenCVFeatureExtractor()
    features = extractor.extract(image_path)
    row = extractor.to_csv_row(
        features, image_id=image_id, file_name=file_name
    )

    stem = str(image_id) if image_id is not None else Path(file_name).stem
    csv_path = extractor.save_csv(
        features,
        features_dir / f"cv_features_{stem}.csv",
        image_id=image_id,
        file_name=file_name,
    )
    json_path = features_dir / f"cv_features_{stem}.json"
    json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    debug_path = extractor.visualize_debug(
        image_path,
        figures_dir / f"cv_debug_{stem}.png",
        title=f"CV features — {file_name}",
        show=args.show,
    )

    print(f"Image: {image_path}")
    if image_id is not None:
        print(f"Image ID: {image_id}")
    print(f"Size: {features['width']} x {features['height']}")
    print(f"Channels: {features['channels']}")
    print()
    print("Edge Density")
    print(f"{features['edge_density']:.2f}")
    print()
    print("Entropy")
    print(f"{features['entropy']:.2f}")
    print()
    print("Brightness")
    print(f"{features['brightness']:.0f}")
    print()
    print("Contrast")
    print(f"{features['contrast']:.2f}")
    print()
    print("Color Variance")
    print(f"{features['color_variance']:.2f}")
    print()
    print("Texture")
    print(f"{features['texture']:.2f}")
    print()
    print("Image Resolution")
    print(features["resolution"])
    print()
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    if debug_path is not None:
        print(f"Plot: {debug_path}")


if __name__ == "__main__":
    main()
