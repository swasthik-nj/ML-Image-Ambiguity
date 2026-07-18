"""Runnable demo that appends rule-based ambiguity labels to human_dataset.csv.

Reads the ``caption_diversity`` column produced by ``create_dataset.py``,
buckets each image into Low / Medium / High ambiguity, appends the label
column, writes statistics + plots, and saves the labeled CSV.

Usage (from project root)::

    python src/generate_labels.py
    python src/generate_labels.py --input dataset/human_dataset.csv --output dataset/human_dataset.csv
    python src/generate_labels.py --low-max 0.35 --medium-max 0.65
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

from image_ambiguity.logging_config import setup_logging
from image_ambiguity.pipeline.label_generator import AmbiguityLabelGenerator

DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "human_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "human_dataset.csv"
DEFAULT_STATS_PATH = PROJECT_ROOT / "results" / "labels" / "label_statistics.json"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "results" / "figures" / "ambiguity_label_distribution.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rule-based ambiguity labels from caption diversity."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the merged dataset CSV (default: dataset/human_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV path (default: overwrite dataset/human_dataset.csv)",
    )
    parser.add_argument(
        "--low-max",
        type=float,
        default=0.35,
        help="Upper bound (exclusive) of the Low bucket (default: 0.35)",
    )
    parser.add_argument(
        "--medium-max",
        type=float,
        default=0.65,
        help="Upper bound (exclusive) of the Medium bucket (default: 0.65)",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_PATH,
        help="Destination JSON path for statistics",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Destination PNG path for the distribution plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.input}. Run 'python src/create_dataset.py' first."
        )

    df = pd.read_csv(args.input)

    generator = AmbiguityLabelGenerator(low_max=args.low_max, medium_max=args.medium_max)
    labeled_df = generator.label_dataframe(df)
    stats = generator.compute_statistics(labeled_df)

    saved_csv = generator.save_csv(labeled_df, args.output)
    saved_stats = generator.save_statistics_json(stats, args.stats_output)
    saved_plot = generator.plot_distribution(
        labeled_df, args.plot_output, show=args.show
    )

    print("Ambiguity labels generated")
    print(f"{len(labeled_df)} rows")
    print()
    print("Label counts:")
    for label, count in stats["label_counts"].items():
        pct = stats["label_percentages"].get(label, 0.0)
        print(f"  {label:<8} {count:>4}  ({pct:.2f}%)")
    print()
    print(
        "Overall diversity: "
        f"mean={stats['diversity_overall']['mean']:.4f}  "
        f"std={stats['diversity_overall']['std']:.4f}  "
        f"min={stats['diversity_overall']['min']:.4f}  "
        f"max={stats['diversity_overall']['max']:.4f}"
    )
    print()
    print(f"Dataset saved to: {saved_csv}")
    print(f"Statistics saved to: {saved_stats}")
    if saved_plot is not None:
        print(f"Plot saved to: {saved_plot}")
    print()
    print(labeled_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
