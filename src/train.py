"""Runnable demo that trains, tunes, evaluates, and saves ambiguity classifiers.

Trains a Random Forest and an XGBoost classifier to predict the
rule-based ambiguity_label (Low / Medium / High) from the caption
diversity + OpenCV feature columns in human_dataset.csv. Performs an
80/20 split, baseline cross-validation, grid-search hyperparameter
tuning, reports Accuracy / Precision / Recall / F1 / ROC-AUC per model,
and saves the best-performing model with joblib.

Usage (from project root)::

    python src/train.py
    python src/train.py --input dataset/human_dataset.csv --cv-folds 5
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

DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "human_dataset.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "training_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Random Forest + XGBoost classifiers on human_dataset.csv "
            "to predict ambiguity_label."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the labeled dataset CSV (default: dataset/human_dataset.csv)",
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
        help="Fraction of data held out for testing (default: 0.2, i.e. 80/20)",
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
        default="accuracy",
        help="Metric used to choose the best model to save (default: accuracy)",
    )
    return parser.parse_args()


def print_result(result: ModelResult) -> None:
    display_name = DISPLAY_NAMES.get(result.name, result.name)
    print(f"--- {display_name} ---")
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
    )
    df = trainer.load_dataset(args.input)
    results = trainer.train(df)

    print("Training complete")
    print(f"Dataset: {args.input} ({len(df)} rows)")
    print()
    for result in results.values():
        print_result(result)

    best = ModelTrainer.select_best(results, metric=args.select_by)
    best_path = trainer.save_model(best.model, args.models_dir / "best_model.joblib")
    for name, result in results.items():
        trainer.save_model(result.model, args.models_dir / f"{name}.joblib")

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

    print(f"Best model: {best.display_name} ({args.select_by}={best.metrics[args.select_by]:.4f})")
    print(f"Best model saved to: {best_path}")
    print(f"All models saved to: {args.models_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
