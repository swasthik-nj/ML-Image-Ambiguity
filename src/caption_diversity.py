"""Runnable demo for caption diversity from Sentence-BERT embeddings.

Usage (from project root)::

    python src/caption_diversity.py
    python src/caption_diversity.py --image-id 397133
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_ambiguity.config import get_settings
from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.caption_diversity import CaptionDiversityAnalyzer
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.logging_config import setup_logging

DEFAULT_IMAGE_ID = 391895
DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT / "dataset" / "annotations" / "captions_train2017.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute caption diversity from Sentence-BERT embeddings."
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
        "--embeddings",
        type=Path,
        default=None,
        help="Optional precomputed .npz embeddings (skips model encode)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the similarity heatmap window",
    )
    return parser.parse_args()


def load_or_generate_embeddings(
    args: argparse.Namespace,
    settings: Any,
) -> tuple[Any, list[str]]:
    """Load saved embeddings or generate them from COCO captions."""
    embeddings_path = args.embeddings or (
        settings.embeddings_dir / f"embeddings_{args.image_id}.npz"
    )

    generator = SentenceEmbeddingGenerator(
        model_name=settings.sentence_model_name,
        device=settings.device,
        batch_size=settings.embedding_batch_size,
    )

    image_dir = PROJECT_ROOT / "dataset" / "val2017"
    if not image_dir.is_dir():
        image_dir = PROJECT_ROOT

    loader = CocoDatasetLoader(args.annotation_file, image_dir)
    loader.load_annotations()
    captions = loader.get_captions(args.image_id)

    if Path(embeddings_path).is_file():
        embeddings = generator.load_embeddings(embeddings_path)
        return embeddings, captions

    generator.load_model()
    embeddings = generator.generate_embeddings(captions)
    generator.save_embeddings(embeddings, embeddings_path, captions=captions)
    return embeddings, captions


def main() -> None:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    settings.ensure_directories()

    diversity_dir = settings.results_dir / "diversity"
    diversity_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = settings.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    embeddings, captions = load_or_generate_embeddings(args, settings)

    analyzer = CaptionDiversityAnalyzer()
    metrics = analyzer.compute(embeddings)

    json_path = analyzer.save_json(
        metrics,
        diversity_dir / f"diversity_{args.image_id}.json",
        image_id=args.image_id,
        captions=captions,
    )
    csv_path = analyzer.save_csv(
        metrics,
        diversity_dir / f"diversity_{args.image_id}.csv",
        image_id=args.image_id,
    )
    figure_path = analyzer.visualize(
        metrics,
        figures_dir / f"similarity_{args.image_id}.png",
        captions=captions,
        title=f"Image {args.image_id} caption similarity",
        show=args.show,
    )

    print(f"Image ID: {args.image_id}")
    print(f"Captions: {metrics.n_captions}")
    print(f"Pairs: {metrics.n_pairs}")
    print()
    print("Average Similarity")
    print(f"{metrics.average_similarity:.2f}")
    print()
    print("Minimum Similarity")
    print(f"{metrics.min_similarity:.2f}")
    print()
    print("Maximum Similarity")
    print(f"{metrics.max_similarity:.2f}")
    print()
    print("Standard Deviation")
    print(f"{metrics.std_similarity:.2f}")
    print()
    print("Caption Diversity")
    print(f"{metrics.diversity_score:.2f}")
    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    if figure_path is not None:
        print(f"Plot: {figure_path}")


if __name__ == "__main__":
    main()
