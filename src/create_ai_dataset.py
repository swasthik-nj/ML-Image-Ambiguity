"""Build ai_dataset.csv from BLIP captions + the caption-diversity pipeline.

Usage (from project root)::

    python src/create_ai_dataset.py
    python src/create_ai_dataset.py --limit 20
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
from image_ambiguity.pipeline.dataset_builder import DATASET_COLUMNS, MLDatasetBuilder

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
        description="Build dataset/ai_dataset.csv from BLIP captions."
    )
    parser.add_argument(
        "--human-dataset",
        type=Path,
        default=DEFAULT_HUMAN_DATASET,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-return-sequences", type=int, default=2)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on images (e.g. 20 for a quick check)",
    )
    parser.add_argument("--force-blip", action="store_true")
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
    image_ids = [int(value) for value in human_df["image_id"].tolist()]
    if args.limit is not None:
        image_ids = image_ids[: max(0, args.limit)]
        human_df = human_df[human_df["image_id"].isin(image_ids)].copy()

    loader = CocoDatasetLoader(settings.annotation_file, settings.image_dir)
    loader.load_annotations()

    blip = BlipCaptionGenerator(
        model_name=args.model,
        device=args.device or settings.device,
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
        annotation_file=settings.annotation_file,
        image_dir=settings.image_dir,
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

    if all(col in human_df.columns for col in CV_COLUMNS):
        human_cv = human_df.set_index("image_id")[CV_COLUMNS]
        ai_df = ai_df.set_index("image_id")
        ai_df.loc[:, CV_COLUMNS] = human_cv.loc[ai_df.index, CV_COLUMNS]
        ai_df = ai_df.reset_index()

    ai_df = ai_df[DATASET_COLUMNS]
    saved = builder.save_csv(ai_df, args.output)

    print("AI dataset created")
    print(f"{len(ai_df)} rows")
    print(f"Saved to: {saved}")
    print(f"Mean caption_diversity: {ai_df['caption_diversity'].mean():.4f}")


if __name__ == "__main__":
    main()
    raise SystemExit(0)
