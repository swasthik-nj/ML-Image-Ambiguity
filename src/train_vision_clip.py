"""Runnable demo that trains ambiguity classifiers using OpenCV + CLIP features.

Builds the vision_clip_dataset.csv from human_dataset.csv, trains a Random Forest
and an XGBoost classifier to predict the ambiguity_label using the 6 OpenCV
features plus 512 CLIP features. Performs an 80/20 split, cross-validation, 
hyperparameter tuning, and saves the best model to vision_clip_model.joblib.

Usage (from project root)::

    python src/train_vision_clip.py
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
from image_ambiguity.pipeline.vision_clip_dataset_builder import VisionClipDatasetBuilder, VISION_FEATURE_COLUMNS, CLIP_COLUMNS
from image_ambiguity.utils.common import save_json

DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "human_dataset.csv"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "val2017"
DEFAULT_VISION_CLIP_OUTPUT = PROJECT_ROOT / "dataset" / "vision_clip_dataset.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "vision_clip_training_metrics.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classifiers on Vision+CLIP features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the human dataset CSV",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Path to the COCO val2017 image directory",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_VISION_CLIP_OUTPUT,
        help="Destination path for vision+clip dataset",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory to save trained models",
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
        help="Fraction of data held out for testing",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Requested number of cross-validation folds",
    )
    parser.add_argument(
        "--select-by",
        choices=["accuracy", "precision", "recall", "f1", "roc_auc"],
        default="f1",
        help="Metric used to choose the best model to save (default: f1 for macro-F1)",
    )
    parser.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Oversample minority classes in the training split",
    )
    return parser.parse_args()


def print_result(result: ModelResult) -> None:
    display_name = DISPLAY_NAMES.get(result.name, result.name)
    print(f"--- {display_name} (Vision+CLIP) ---")
    if result.baseline_cv_accuracy is not None:
        print(
            f"Baseline {result.cv_folds_used}-fold CV Accuracy: "
            f"{result.baseline_cv_accuracy * 100:.2f}%"
        )
    else:
        print("Baseline CV: skipped (too few samples per class)")
    if result.tuned_cv_score is not None:
        print(f"Tuned CV Score (f1_macro): {result.tuned_cv_score * 100:.2f}%")
        print(f"Best Params: {result.best_params}")
    else:
        print("Hyperparameter tuning: skipped (too few samples per class)")

    metrics = result.metrics
    print(f"{display_name} Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"{display_name} Macro Precision: {metrics['precision'] * 100:.2f}%")
    print(f"{display_name} Macro Recall: {metrics['recall'] * 100:.2f}%")
    print(f"{display_name} Macro F1: {metrics['f1'] * 100:.2f}%")
    roc = metrics.get("roc_auc")
    print(f"{display_name} ROC-AUC: {roc * 100:.2f}%" if roc is not None else f"{display_name} ROC-AUC: n/a")
    print()
    
    # Print Confusion Matrix
    if "confusion_matrix" in metrics:
        print(f"{display_name} Confusion Matrix (Low, Medium, High):")
        for row in metrics["confusion_matrix"]:
            print(f"  {row}")
        print()


def main() -> None:
    args = parse_args()
    setup_logging()

    if not args.output_csv.exists():
        print("Building vision+clip dataset. This may take a while to extract CLIP embeddings...")
        builder = VisionClipDatasetBuilder(input_csv=args.input, image_dir=args.image_dir)
        vision_clip_df = builder.build()
        builder.save_csv(vision_clip_df, args.output_csv)
    else:
        import pandas as pd
        print(f"Loading existing vision+clip dataset from {args.output_csv}...")
        vision_clip_df = pd.read_csv(args.output_csv)

    feature_cols = list(VISION_FEATURE_COLUMNS) + list(CLIP_COLUMNS)

    # Print a worked example row to confirm columns line up
    if not vision_clip_df.empty:
        example_row = vision_clip_df.iloc[0]
        print(f"\nWorked Example Row:")
        print(f"image_id {example_row['image_id']} -> ", end="")
        features_str = ", ".join(f"{col} {example_row[col]:.4f}" for col in VISION_FEATURE_COLUMNS if col in example_row)
        print(f"{features_str}, clip_0 {example_row['clip_0']:.4f}... -> label {example_row.get('ambiguity_label', 'Unknown')}\n")

    trainer = ModelTrainer(
        feature_columns=feature_cols,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        scoring="f1_macro",
        balance_classes=args.balance_classes,
    )
    
    results = trainer.train(vision_clip_df)

    print("Vision+CLIP Training complete")
    print(f"Dataset: {args.output_csv} ({len(vision_clip_df)} rows, {len(feature_cols)} features)")
    if "ambiguity_label" in vision_clip_df.columns:
        print("Label counts:")
        print(vision_clip_df["ambiguity_label"].value_counts().to_string())
    print(f"Class balancing: {'on' if args.balance_classes else 'off'}")
    print()
    for result in results.values():
        print_result(result)

    best = ModelTrainer.select_best(results, metric=args.select_by)
    best_path = trainer.save_model(best.model, args.models_dir / "vision_clip_model.joblib")
    
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

    print(f"Best Vision+CLIP Model: {best.display_name} ({args.select_by}={best.metrics[args.select_by]:.4f})")
    print(f"Best Vision+CLIP Model saved to: {best_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
