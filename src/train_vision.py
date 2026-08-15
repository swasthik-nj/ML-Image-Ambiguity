"""Runnable demo that trains ambiguity classifiers using ONLY OpenCV features.

Trains a Random Forest and an XGBoost classifier to predict the
rule-based ambiguity_label (Low / Medium / High) from only the computer
vision feature columns. Performs an 80/20 split, baseline cross-validation,
grid-search hyperparameter tuning, reports metrics, and saves the best
model as best_vision_model.joblib.

Usage (from project root)::

    python src/train_vision.py
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
from image_ambiguity.models.trainer import DISPLAY_NAMES, ModelResult, ModelTrainer
from image_ambiguity.utils.common import save_json
import pandas as pd

DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "human_dataset.csv"
DEFAULT_VISION_OUTPUT = PROJECT_ROOT / "dataset" / "vision_dataset.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "vision_training_metrics.json"

VISION_FEATURE_COLUMNS = (
    "edge_density",
    "entropy",
    "brightness",
    "contrast",
    "color_variance",
    "texture",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classifiers on vision-only features to predict ambiguity_label."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the labeled dataset CSV (default: dataset/human_dataset.csv)",
    )
    parser.add_argument(
        "--vision-output",
        type=Path,
        default=DEFAULT_VISION_OUTPUT,
        help="Destination path for vision dataset (default: dataset/vision_dataset.csv)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory to save trained models (default: models/)",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Destination JSON path for metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Requested number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--select-by",
        choices=["accuracy", "precision", "recall", "f1", "roc_auc"],
        default="f1",
        help="Metric used to choose the best model to save (default: f1)",
    )
    parser.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Oversample minority classes in the training split (default: on)",
    )
    return parser.parse_args()


def print_result(result: ModelResult) -> None:
    display_name = DISPLAY_NAMES.get(result.name, result.name)
    print(f"--- {display_name} (Vision-Only) ---")
    if result.baseline_cv_accuracy is not None:
        print(
            f"Baseline {result.cv_folds_used}-fold CV Accuracy: "
            f"{result.baseline_cv_accuracy * 100:.2f}%"
        )
    else:
        print("Baseline CV: skipped (too few samples per class)")
    if result.tuned_cv_score is not None:
        print(f"Tuned CV Score: {result.tuned_cv_score * 100:.2f}%")
        print(f"Best Params: {result.best_params}")
    else:
        print("Hyperparameter tuning: skipped (too few samples per class)")

    metrics = result.metrics
    print(f"{display_name} Accuracy: {metrics['accuracy'] * 100:.0f}%")
    print(f"{display_name} Precision: {metrics['precision'] * 100:.2f}%")
    print(f"{display_name} Recall: {metrics['recall'] * 100:.2f}%")
    print(f"{display_name} F1: {metrics['f1'] * 100:.2f}%")
    roc = metrics.get("roc_auc")
    print(f"{display_name} ROC-AUC: {roc * 100:.2f}%" if roc is not None else f"{display_name} ROC-AUC: n/a")
    print()


def main() -> None:
    args = parse_args()
    setup_logging()

    trainer = ModelTrainer(
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        balance_classes=args.balance_classes,
    )
    
    # Override features to ONLY use vision features
    trainer.FEATURE_COLUMNS = VISION_FEATURE_COLUMNS

    df = trainer.load_dataset(args.input)
    
    # Save a copy of the dataset with only vision columns (and label/image_id)
    cols_to_keep = ["image_id"] + list(VISION_FEATURE_COLUMNS)
    if "ambiguity_label" in df.columns:
        cols_to_keep.append("ambiguity_label")
    
    vision_df = df[[c for c in cols_to_keep if c in df.columns]]
    vision_df.to_csv(args.vision_output, index=False)
    print(f"Saved vision-only dataset to {args.vision_output} ({len(vision_df)} rows)")
    
    # Train
    results = trainer.train(df)

    print("Vision-Only Training complete")
    print(f"Dataset: {args.input} ({len(df)} rows)")
    if "ambiguity_label" in df.columns:
        print("Label counts:")
        print(df["ambiguity_label"].value_counts().to_string())
    print(f"Class balancing: {'on' if args.balance_classes else 'off'}")
    print()
    for result in results.values():
        print_result(result)

    best = ModelTrainer.select_best(results, metric=args.select_by)
    best_path = trainer.save_model(best.model, args.models_dir / "best_vision_model.joblib")
    
    metrics_payload = {
        "best_model": best.name,
        "selected_by": args.select_by,
        "models": {
            name: {
                "display_name": DISPLAY_NAMES.get(name, name),
                "best_params": result.best_params,
                "cv_folds_used": result.cv_folds_used,
                "baseline_cv_accuracy": result.baseline_cv_accuracy,
                "tuned_cv_score": result.tuned_cv_score,
                "metrics": result.metrics,
            }
            for name, result in results.items()
        },
    }
    metrics_path = save_json(metrics_payload, args.metrics_output)

    print(f"Best Vision Model: {best.display_name} ({args.select_by}={best.metrics[args.select_by]:.4f})")
    print(f"Best Vision Model saved to: {best_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
