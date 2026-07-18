"""Runnable demo for COCO caption loading.

Usage (from project root)::

    python src/coco_loader.py
    python src/coco_loader.py --image-id 397133
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_ambiguity.data.coco_loader import CocoDatasetLoader


DEFAULT_IMAGE_ID = 391895
DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT / "dataset" / "annotations" / "captions_train2017.json"
)
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "train2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print COCO captions for a given image id."
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=DEFAULT_IMAGE_ID,
        help=f"COCO image id (default: {DEFAULT_IMAGE_ID})",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help="Path to captions_*.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing image files (optional for caption-only demo)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Image dir may be missing for a captions-only check; create a dummy path
    # that exists so the loader can index annotations.
    image_dir = args.image_dir
    if not image_dir.is_dir():
        # Fall back to val2017 if present, else project root (captions-only).
        val_dir = PROJECT_ROOT / "dataset" / "val2017"
        image_dir = val_dir if val_dir.is_dir() else PROJECT_ROOT

    loader = CocoDatasetLoader(args.annotation_file, image_dir)
    loader.load_annotations()

    image_id = args.image_id
    captions = loader.get_captions(image_id)

    print(f"Image ID: {image_id}")
    print()
    for index, caption in enumerate(captions, start=1):
        print(f"Caption {index}:")
        print(caption)
        print()


if __name__ == "__main__":
    main()
