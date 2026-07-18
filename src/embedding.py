"""Runnable demo for Sentence-BERT caption embeddings.

Usage (from project root)::

    python src/embedding.py
    python src/embedding.py --image-id 397133
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
from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.logging_config import setup_logging

DEFAULT_IMAGE_ID = 391895
DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT / "dataset" / "annotations" / "captions_train2017.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Sentence-BERT embeddings for COCO captions."
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
        "--output",
        type=Path,
        default=None,
        help="Output .npz path (default: results/embeddings/<image_id>.npz)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    settings.ensure_directories()

    image_dir = PROJECT_ROOT / "dataset" / "val2017"
    if not image_dir.is_dir():
        image_dir = PROJECT_ROOT

    loader = CocoDatasetLoader(args.annotation_file, image_dir)
    loader.load_annotations()
    captions = loader.get_captions(args.image_id)

    print(f"Image ID: {args.image_id}")
    print(f"Captions: {len(captions)}")
    for index, caption in enumerate(captions, start=1):
        print(f"  {index}. {caption}")
    print()

    generator = SentenceEmbeddingGenerator(
        model_name=settings.sentence_model_name,
        device=settings.device,
        batch_size=settings.embedding_batch_size,
    )
    generator.load_model()
    embeddings = generator.generate_embeddings(captions)

    output = args.output or (
        settings.embeddings_dir / f"embeddings_{args.image_id}.npz"
    )
    saved = generator.save_embeddings(embeddings, output, captions=captions)
    loaded = generator.load_embeddings(saved)

    print(f"Model: {generator.model_name}")
    print(f"Device: {generator.device}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Saved to: {saved}")
    print(f"Reload check shape: {loaded.shape}")
    print()
    print("First caption embedding (first 8 dims):")
    print(embeddings[0, :8])


if __name__ == "__main__":
    main()
