"""Build ai_dataset.csv from BLIP captions + the caption-diversity pipeline.

Reuses the same image ids (and by default the same OpenCV columns) as
``human_dataset.csv``, replacing human COCO captions with BLIP generations.

Usage (from project root)::

    python src/create_ai_dataset.py
    python src/create_ai_dataset.py --human-dataset dataset/human_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_ambiguity.config import get_settings
from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.blip_captions import (
    ALL_STRATEGIES,
    DEFAULT_MODEL_NAME,
    BlipCaptionGenerator,
)
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.logging_config import setup_logging
from image_ambiguity.pipeline.ai_captions import collect_blip_captions
from image_ambiguity.pipeline.dataset_builder import (
    DATASET_COLUMNS,
    MLDatasetBuilder,
)

DEFAULT_HUMAN_DATASET = PROJECT_ROOT / "dataset" / "human_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "ai_dataset.csv"
DEFAULT_CACHE = PROJECT_ROOT / "results" / "captions" / "blip_caption_cache.json"
CV_COLUMNS = [
    "edge_density",
    "entropy",
    "brightness",
    "contrast",
    "color_variance",
    "texture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the caption-diversity + CV feature pipeline on BLIP "
            "captions and write dataset/ai_dataset.csv."
        )
    )
    parser.add_argument(
        "--human-dataset",
        type=Path,
        default=DEFAULT_HUMAN_DATASET,
        help="Human dataset CSV used to select matching image ids",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="COCO captions JSON (default: settings.annotation_file)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Image directory (default: settings.image_dir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV (default: dataset/ai_dataset.csv)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="BLIP caption cache JSON path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"BLIP model id (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cpu/cuda/auto)",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=3,
        help="Captions per decoding strategy (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images (debug)",
    )
    parser.add_argument(
        "--force-blip",
        action="store_true",
        help="Regenerate BLIP captions even if cached",
    )
    parser.add_argument(
        "--recompute-cv",
        action="store_true",
        help="Re-extract OpenCV features instead of copying from human CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    settings.ensure_directories()

    if not args.human_dataset.is_file():
        raise FileNotFoundError(
            f"Human dataset not found: {args.human_dataset}. "
            "Run python src/create_dataset.py first."
        )

    human_df = pd.read_csv(args.human_dataset)
    if "image_id" not in human_df.columns:
        raise ValueError("human_dataset.csv must contain an image_id column")

    image_ids = [int(value) for value in human_df["image_id"].tolist()]
    if args.limit is not None:
        image_ids = image_ids[: max(0, args.limit)]
        human_df = human_df[human_df["image_id"].isin(image_ids)].copy()

    annotation_file = args.annotation_file or settings.annotation_file
    image_dir = args.image_dir or settings.image_dir
    device = args.device or settings.device

    loader = CocoDatasetLoader(annotation_file, image_dir)
    loader.load_annotations()

    blip = BlipCaptionGenerator(
        model_name=args.model,
        device=device,
        num_return_sequences=args.num_return_sequences,
    )
    captions_by_image = collect_blip_captions(
        loader,
        image_ids,
        generator=blip,
        cache_path=args.cache,
        strategies=ALL_STRATEGIES,
        force=args.force_blip,
    )

    builder = MLDatasetBuilder(
        annotation_file=annotation_file,
        image_dir=image_dir,
        embedding_generator=SentenceEmbeddingGenerator(
            model_name=settings.sentence_model_name,
            device=settings.device,
            batch_size=settings.embedding_batch_size,
        ),
    )

    ai_df = builder.build(
        image_ids=image_ids,
        captions_by_image=captions_by_image,
    )

    # Keep CV features identical to the human dataset for a fair comparison
    # unless the caller asks to recompute them.
    if not args.recompute_cv and all(col in human_df.columns for col in CV_COLUMNS):
        human_cv = human_df.set_index("image_id")[CV_COLUMNS]
        ai_df = ai_df.set_index("image_id")
        ai_df.loc[:, CV_COLUMNS] = human_cv.loc[ai_df.index, CV_COLUMNS]
        ai_df = ai_df.reset_index()

    ai_df = ai_df[DATASET_COLUMNS]
    saved = builder.save_csv(ai_df, args.output)

    print("AI dataset created")
    print(f"{len(ai_df)} rows")
    print()
    print(f"Columns: {', '.join(ai_df.columns)}")
    print(f"Saved to: {saved}")
    print(f"BLIP cache: {args.cache}")
    print()
    print(f"Mean caption_diversity: {ai_df['caption_diversity'].mean():.4f}")
    print()
    print(ai_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
    raise SystemExit(0)
