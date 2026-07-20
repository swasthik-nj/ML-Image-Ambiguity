"""Runnable demo that explains the Random Forest ambiguity classifier with SHAP.

Loads the trained Random Forest model (models/random_forest.joblib) and
the labeled dataset (dataset/human_dataset.csv), then produces:

* A Top Feature Importance bar chart (mean |SHAP value| across classes).
* A Summary (beeswarm) plot for one ambiguity class.
* A Waterfall plot explaining one specific prediction.
* A Force plot (static, matplotlib-rendered) for that same prediction.
* A structured, human-readable explanation of that one prediction.

Usage (from project root)::

    python src/shap_analysis.py
    python src/shap_analysis.py --class-name High --image-id 261982
    python src/shap_analysis.py --model models/random_forest.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from image_ambiguity.explainability.shap_explainer import SHAPExplainer
from image_ambiguity.logging_config import setup_logging
from image_ambiguity.models.trainer import FEATURE_COLUMNS, LABEL_ORDER, ModelTrainer
from image_ambiguity.utils.common import save_json

DEFAULT_MODEL = PROJECT_ROOT / "models" / "random_forest.joblib"
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "human_dataset.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_EXPLANATIONS_DIR = PROJECT_ROOT / "results" / "explanations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explain the Random Forest ambiguity classifier with SHAP: "
            "top feature importance, a summary plot, and a waterfall + "
            "force plot for one prediction."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the trained model (default: models/random_forest.joblib)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the labeled dataset CSV (default: dataset/human_dataset.csv)",
    )
    parser.add_argument(
        "--class-name",
        choices=list(LABEL_ORDER),
        default="High",
        help="Ambiguity class to use for the summary plot (default: High)",
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=None,
        help="Specific image_id to explain (overrides --index)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Row position (0-based) to explain, if --image-id is not given",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of features to show in the importance chart (default: 10)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory to save figures (default: results/figures)",
    )
    parser.add_argument(
        "--explanations-dir",
        type=Path,
        default=DEFAULT_EXPLANATIONS_DIR,
        help="Directory to save the JSON explanation (default: results/explanations)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model}. Run 'python src/train.py' first."
        )

    trainer = ModelTrainer()
    model = trainer.load_model(args.model)
    df = trainer.load_dataset(args.input)

    X, y = trainer.prepare_data(df)
    image_ids = df.loc[X.index, "image_id"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    explainer = SHAPExplainer(
        model, feature_names=FEATURE_COLUMNS, class_names=list(LABEL_ORDER)
    )
    shap_values = explainer.compute_shap_values(X)

    # --- Top Feature Importance (aggregated across all classes) ---
    importance_df = explainer.top_feature_importance(shap_values, top_n=args.top_n)
    importance_path = explainer.plot_top_feature_importance(
        importance_df, args.figures_dir / "shap_top_feature_importance.png"
    )
    importance_csv = args.explanations_dir / "shap_top_feature_importance.csv"
    importance_csv.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_csv, index=False)

    # --- Summary (beeswarm) plot for the requested class ---
    class_index = LABEL_ORDER.index(args.class_name)
    summary_path = explainer.summary_plot(
        shap_values,
        X,
        class_index=class_index,
        path=args.figures_dir / f"shap_summary_{args.class_name.lower()}.png",
    )

    # --- Pick one prediction to explain ---
    predictions = model.predict(X)
    if args.image_id is not None:
        matches = image_ids[image_ids == args.image_id]
        if matches.empty:
            raise ValueError(f"image_id {args.image_id} not found in dataset")
        row_index = int(matches.index[0])
    elif args.index is not None:
        row_index = args.index
    else:
        matching = np.where(predictions == class_index)[0]
        row_index = int(matching[0]) if len(matching) else 0

    predicted_class_index = int(predictions[row_index])
    image_id = int(image_ids.iloc[row_index])

    waterfall_path = explainer.waterfall_plot(
        shap_values,
        row_index,
        class_index=predicted_class_index,
        path=args.figures_dir / f"shap_waterfall_image_{image_id}.png",
    )
    force_path = explainer.force_plot(
        shap_values,
        row_index,
        class_index=predicted_class_index,
        path=args.figures_dir / f"shap_force_image_{image_id}.png",
    )

    explanation = explainer.explain_prediction(
        shap_values, X, row_index, class_index=predicted_class_index
    )
    explanation_path = save_json(
        {"image_id": image_id, **explanation.to_dict()},
        args.explanations_dir / f"shap_explanation_image_{image_id}.json",
    )

    print("SHAP explainability analysis complete")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.input} ({len(X)} rows explained)")
    print()
    print(f"Top {args.top_n} Feature Importance:")
    print(importance_df.to_string(index=False))
    print(f"  -> plot: {importance_path}")
    print(f"  -> csv:  {importance_csv.resolve()}")
    print()
    print(f"Summary plot ({args.class_name} class): {summary_path}")
    print()
    print(f"Explaining one prediction: image_id={image_id} (row {row_index})")
    print(explanation.summary_text())
    print(f"  -> waterfall plot: {waterfall_path}")
    print(f"  -> force plot:     {force_path}")
    print(f"  -> explanation:    {explanation_path}")


if __name__ == "__main__":
    main()
