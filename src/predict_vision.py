"""Predict ambiguity for a single image using the vision-only model.

Usage::
    python src/predict_vision.py path/to/your/image.jpg
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.logging_config import setup_logging

VISION_FEATURE_COLUMNS = [
    "edge_density",
    "entropy",
    "brightness",
    "contrast",
    "color_variance",
    "texture",
]

def main():
    parser = argparse.ArgumentParser(description="Predict image ambiguity using only OpenCV features.")
    parser.add_argument("image_path", type=Path, help="Path to the image to analyze")
    parser.add_argument(
        "--model", 
        type=Path, 
        default=PROJECT_ROOT / "models" / "best_vision_model.joblib",
        help="Path to the trained vision-only model"
    )
    args = parser.parse_args()
    
    setup_logging()

    if not args.image_path.is_file():
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
        
    if not args.model.is_file():
        print(f"Error: Model not found at {args.model}. Run 'python src/train_vision.py' first.")
        sys.exit(1)

    # 1. Extract CV features
    print(f"Extracting features from: {args.image_path.name}...")
    extractor = OpenCVFeatureExtractor()
    features = extractor.extract(args.image_path)
    
    # 2. Format features as a DataFrame (to match training)
    # The columns must match the order in VISION_FEATURE_COLUMNS
    row = {col: features[col] for col in VISION_FEATURE_COLUMNS}
    df = pd.DataFrame([row])
    
    # 3. Load Model and Predict
    print(f"Loading model: {args.model.name}...")
    model = joblib.load(args.model)
    
    prediction_idx = model.predict(df)[0]
    
    # Map index back to label. The trainer uses: 0=Low, 1=Medium, 2=High
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    label_str = label_map.get(prediction_idx, f"Unknown ({prediction_idx})")
    
    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        print(f"\n--- Prediction Results ---")
        print(f"Predicted Ambiguity: **{label_str}**")
        print(f"Probabilities: Low={probs[0]:.1%}, Medium={probs[1]:.1%}, High={probs[2]:.1%}")
    else:
        print(f"\n--- Prediction Results ---")
        print(f"Predicted Ambiguity: **{label_str}**")
        
    print("\nExtracted Features:")
    for k, v in row.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
