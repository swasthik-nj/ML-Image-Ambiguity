"""Runnable demo for BLIP caption generation.

Usage (from project root)::

    python src/blip_caption.py
    python src/blip_caption.py --image-id 397133
    python src/blip_caption.py --image dataset/val2017/000000397133.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

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
from image_ambiguity.logging_config import setup_logging

DEFAULT_IMAGE_ID = 397133


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BLIP captions with beam search, top-k, and nucleus "
            "sampling; compare against COCO captions; save JSON."
        )
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
        help="Optional direct image path (skips COCO image lookup)",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="Path to captions_*.json (default: settings.annotation_file)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory with COCO images (default: settings.image_dir)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face BLIP model id (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, cuda, or auto (default: settings.device)",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=3,
        help="Captions per decoding strategy (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: results/captions/blip_<id>.json)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip COCO caption comparison",
    )
    return parser.parse_args()


def _print_result(payload: dict) -> None:
    print(f"Image ID: {payload.get('image_id')}")
    print(f"File: {payload.get('file_name')}")
    print(f"Model: {payload.get('model')}")
    print(f"Device: {payload.get('device')}")
    print()

    print("Generated captions")
    for strategy in ALL_STRATEGIES:
        captions = payload.get("generated_captions", {}).get(strategy, [])
        print(f"  [{strategy}]")
        if not captions:
            print("    (none)")
            continue
        for index, caption in enumerate(captions, start=1):
            print(f"    {index}. {caption}")
    print()

    coco = payload.get("coco_captions") or []
    if coco:
        print("COCO captions")
        for index, caption in enumerate(coco, start=1):
            print(f"  {index}. {caption}")
        print()

    comparison = payload.get("comparison")
    if comparison:
        print("Comparison")
        print(f"  COCO count: {comparison['n_coco']}")
        print(f"  Generated count: {comparison['n_generated']}")
        exact = comparison.get("exact_matches") or []
        print(f"  Exact matches: {len(exact)}")
        for caption in exact:
            print(f"    - {caption}")
        print("  Best word overlap (Jaccard) per strategy:")
        for strategy, rows in (comparison.get("best_overlaps") or {}).items():
            print(f"    [{strategy}]")
            for row in rows:
                print(
                    f"      gen={row['generated']!r} | "
                    f"coco={row['best_coco']!r} | "
                    f"jaccard={row['jaccard']:.4f}"
                )
        print()


def main() -> None:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    settings.ensure_directories()

    captions_dir = settings.results_dir / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = args.annotation_file or settings.annotation_file
    image_dir = args.image_dir or settings.image_dir
    device = args.device or settings.device

    loader = CocoDatasetLoader(annotation_file, image_dir)
    loader.load_annotations()

    image_id = args.image_id
    coco_captions: list[str] = []
    file_name: str | None = None

    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        file_name = image_path.name
        if not args.no_compare and image_id in loader.coco.imgs:  # type: ignore[union-attr]
            coco_captions = loader.get_captions(image_id)
            file_name = loader.coco.loadImgs(image_id)[0]["file_name"]  # type: ignore[union-attr]
    else:
        image = loader.get_image(image_id)
        file_name = loader.coco.loadImgs(image_id)[0]["file_name"]  # type: ignore[union-attr]
        if not args.no_compare:
            coco_captions = loader.get_captions(image_id)

    generator = BlipCaptionGenerator(
        model_name=args.model,
        device=device,
        num_return_sequences=args.num_return_sequences,
    )
    generator.load_model()

    result = generator.caption_image(
        image,
        image_id=image_id,
        file_name=file_name,
        coco_captions=None if args.no_compare else coco_captions,
    )

    output = args.output or (captions_dir / f"blip_{image_id}.json")
    saved = generator.save_result(result, output)
    payload = result.to_dict()

    _print_result(payload)
    print(f"JSON: {saved}")
    print()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
    # Torch/transformers can keep worker threads alive on Windows; exit explicitly.
    raise SystemExit(0)
