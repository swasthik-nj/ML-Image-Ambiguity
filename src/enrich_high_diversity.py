"""Mine additional High-diversity COCO images and append them to the dataset.

Keeps label thresholds unchanged (High >= 0.65). Finds unused val2017 images
whose human-caption diversity is at least ``--high-min``, extracts OpenCV
features, and appends them to ``human_dataset.csv``.

Usage (from project root)::

    python src/enrich_high_diversity.py
    python src/enrich_high_diversity.py --max-high 40 --high-min 0.65
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_ambiguity.config import get_settings
from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.caption_diversity import CaptionDiversityAnalyzer
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.logging_config import setup_logging
from image_ambiguity.pipeline.dataset_builder import DATASET_COLUMNS, MLDatasetBuilder
from image_ambiguity.pipeline.label_generator import AmbiguityLabelGenerator

DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "human_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "human_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append High-diversity COCO images to reduce class imbalance."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--high-min",
        type=float,
        default=0.65,
        help="Minimum caption_diversity to treat as High (default: 0.65)",
    )
    parser.add_argument(
        "--max-high",
        type=int,
        default=40,
        help="Maximum number of new High images to append (default: 40)",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=2000,
        help="Max unused images to scan for diversity (default: 2000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--low-max",
        type=float,
        default=0.35,
        help="Low threshold used when re-labeling (default: 0.35)",
    )
    parser.add_argument(
        "--medium-max",
        type=float,
        default=0.65,
        help="Medium/High threshold used when re-labeling (default: 0.65)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    settings = get_settings()

    if not args.input.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.input}")

    existing = pd.read_csv(args.input)
    existing_ids = set(existing["image_id"].astype(int).tolist())

    loader = CocoDatasetLoader(settings.annotation_file, settings.image_dir)
    loader.load_annotations()
    assert loader.coco is not None

    all_ids = list(loader.coco.getImgIds())
    unused = [image_id for image_id in all_ids if image_id not in existing_ids]
    rng = np.random.default_rng(args.seed)
    if args.scan_limit and len(unused) > args.scan_limit:
        unused = list(rng.choice(unused, size=args.scan_limit, replace=False))

    print(f"Existing rows: {len(existing)}")
    print(f"Scanning unused images: {len(unused)}")

    captions_by_image: dict[int, list[str]] = {}
    for image_id in unused:
        captions = loader.get_captions(int(image_id))
        if len(captions) >= 2:
            captions_by_image[int(image_id)] = captions

    embedder = SentenceEmbeddingGenerator(
        model_name=settings.sentence_model_name,
        device=settings.device,
        batch_size=settings.embedding_batch_size,
    )
    embedder.load_model()
    analyzer = CaptionDiversityAnalyzer()

    flat: list[str] = []
    spans: dict[int, tuple[int, int]] = {}
    for image_id, captions in captions_by_image.items():
        start = len(flat)
        flat.extend(captions)
        spans[image_id] = (start, len(flat))

    embeddings = embedder.generate_embeddings(flat)
    scored: list[tuple[int, float]] = []
    for image_id, (start, end) in spans.items():
        metrics = analyzer.compute(embeddings[start:end])
        scored.append((image_id, float(metrics.diversity_score)))

    high_candidates = [
        (image_id, score)
        for image_id, score in scored
        if score >= args.high_min
    ]
    high_candidates.sort(key=lambda item: item[1], reverse=True)
    selected = high_candidates[: max(0, args.max_high)]

    print(f"High candidates found: {len(high_candidates)}")
    print(f"Selected to append: {len(selected)}")

    if not selected:
        print("No new High-diversity images found. Dataset unchanged.")
        return

    builder = MLDatasetBuilder(
        annotation_file=settings.annotation_file,
        image_dir=settings.image_dir,
        embedding_generator=embedder,
        diversity_analyzer=analyzer,
        cv_extractor=OpenCVFeatureExtractor(),
    )
    selected_ids = [image_id for image_id, _ in selected]
    selected_captions = {
        image_id: captions_by_image[image_id] for image_id in selected_ids
    }
    new_df = builder.build(
        image_ids=selected_ids,
        captions_by_image=selected_captions,
    )

    # Keep only DATASET_COLUMNS from existing if extra columns (labels) exist.
    base_existing = existing[
        [col for col in DATASET_COLUMNS if col in existing.columns]
    ].copy()
    merged = pd.concat([base_existing, new_df[DATASET_COLUMNS]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_id"], keep="first")

    labeler = AmbiguityLabelGenerator(
        low_max=args.low_max,
        medium_max=args.medium_max,
    )
    labeled = labeler.label_dataframe(merged)
    labeled.to_csv(args.output, index=False)

    counts = labeled["ambiguity_label"].value_counts().to_dict()
    print()
    print(f"Saved enriched dataset: {args.output} ({len(labeled)} rows)")
    print("Label counts:")
    for label in ("Low", "Medium", "High"):
        print(f"  {label:<8} {counts.get(label, 0)}")
    print()
    print("Next: python src/train.py --balance-classes")


if __name__ == "__main__":
    main()
