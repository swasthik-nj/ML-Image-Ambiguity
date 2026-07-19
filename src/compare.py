"""Compare mean caption diversity of human vs AI datasets.

Usage (from project root)::

    python src/compare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_ambiguity.logging_config import setup_logging
from image_ambiguity.pipeline.compare_datasets import compare_mean_diversity

DEFAULT_HUMAN = PROJECT_ROOT / "dataset" / "human_dataset.csv"
DEFAULT_AI = PROJECT_ROOT / "dataset" / "ai_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare human vs AI caption diversity scores."
    )
    parser.add_argument(
        "--human-dataset",
        type=Path,
        default=DEFAULT_HUMAN,
        help="Path to human_dataset.csv",
    )
    parser.add_argument(
        "--ai-dataset",
        type=Path,
        default=DEFAULT_AI,
        help="Path to ai_dataset.csv",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Digits after the decimal point (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    human_diversity, ai_diversity, _n_images = compare_mean_diversity(
        args.human_dataset,
        args.ai_dataset,
    )

    fmt = f".{max(0, args.decimals)}f"
    print("Human Diversity")
    print()
    print(format(human_diversity, fmt))
    print()
    print("AI Diversity")
    print()
    print(format(ai_diversity, fmt))


if __name__ == "__main__":
    main()
