"""Runnable demo that builds the merged human_dataset.csv.

Merges caption-diversity features (Sentence-BERT + cosine similarity) and
OpenCV computer-vision features into a single ML-ready dataset.

Usage (from project root)::

    python src/create_dataset.py
    python src/create_dataset.py --sample-size 300 --output dataset/human_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_ambiguity.config import get_settings
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.logging_config import setup_logging
from image_ambiguity.pipeline.dataset_builder import (
    DEFAULT_SAMPLE_SIZE,
    MLDatasetBuilder,
)

DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT / "dataset" / "annotations" / "captions_val2017.json"
)
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "val2017"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "human_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge caption-diversity and OpenCV features into "
            "human_dataset.csv."
        )
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
        help="Directory containing image files",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of images to sample (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV path (default: dataset/human_dataset.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    settings.ensure_directories()

    builder = MLDatasetBuilder(
        annotation_file=args.annotation_file,
        image_dir=args.image_dir,
        embedding_generator=SentenceEmbeddingGenerator(
            model_name=settings.sentence_model_name,
            device=settings.device,
            batch_size=settings.embedding_batch_size,
        ),
    )

    df = builder.build(sample_size=args.sample_size, seed=settings.random_seed)
    saved = builder.save_csv(df, args.output)

    print("Dataset created")
    print(f"{len(df)} rows")
    print()
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Saved to: {saved}")
    print()
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
