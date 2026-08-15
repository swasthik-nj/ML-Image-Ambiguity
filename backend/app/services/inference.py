"""Inference service: features, prediction, and SHAP for uploaded images."""

from __future__ import annotations

import json
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image

from image_ambiguity.config import Settings, get_settings
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.features.clip_embeddings import CLIPFeatureExtractor
from image_ambiguity.logging_config import get_logger
from image_ambiguity.models.trainer import LABEL_ORDER, ModelTrainer
from image_ambiguity.pipeline.vision_clip_dataset_builder import VISION_FEATURE_COLUMNS, CLIP_COLUMNS
from image_ambiguity.utils.common import ensure_dir

logger = get_logger("backend.services.inference")

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CV_FEATURE_KEYS = VISION_FEATURE_COLUMNS
VISION_CLIP_FEATURE_KEYS = tuple(list(VISION_FEATURE_COLUMNS) + list(CLIP_COLUMNS))

class AmbiguityInferenceService:
    """End-to-end feature extraction, prediction, and explanation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.upload_dir = ensure_dir(self.settings.results_dir / "uploads")
        self.cv_extractor = OpenCVFeatureExtractor()
        self.clip_extractor = CLIPFeatureExtractor()
        self.trainer = ModelTrainer()
        self._model: Any | None = None
        self._loaded_model_path: Path | None = None
        self._shap_explainer: Any | None = None

    def _require_model(self) -> Any:
        if self._model is None:
            path = self.settings.models_dir / "vision_clip_model.joblib"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Vision+CLIP model not found at {path}. "
                    "Please run `python src/train_vision_clip.py` first."
                )
            logger.info("Loading Vision+CLIP classifier from %s", path)
            self._model = self.trainer.load_model(path)
            self._loaded_model_path = path
        return self._model

    def read_upload_meta(self, upload_id: str) -> dict[str, Any] | None:
        meta_path = self.upload_dir / f"{upload_id}.json"
        if not meta_path.is_file():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _ensure_shap(self) -> Any:
        if self._shap_explainer is None:
            from image_ambiguity.explainability.shap_explainer import SHAPExplainer

            model = self._require_model()
            self._shap_explainer = SHAPExplainer(
                model,
                feature_names=VISION_CLIP_FEATURE_KEYS,
                class_names=list(LABEL_ORDER),
            )
        return self._shap_explainer

    def save_upload(
        self,
        *,
        filename: str,
        data: bytes,
    ) -> dict[str, Any]:
        """Persist an uploaded image and return upload metadata."""
        if not data:
            raise ValueError("Uploaded file is empty")

        suffix = Path(filename or "upload.jpg").suffix.lower() or ".jpg"
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(
                f"Unsupported image type '{suffix}'. "
                f"Allowed: {sorted(ALLOWED_IMAGE_SUFFIXES)}"
            )

        upload_id = uuid.uuid4().hex
        path = self.upload_dir / f"{upload_id}{suffix}"
        path.write_bytes(data)

        # Validate it opens as an image.
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode

        original_name = Path(filename).name if filename else path.name
        meta = {
            "upload_id": upload_id,
            "filename": original_name,
            "content_type": f"image/{suffix.lstrip('.')}",
            "path": str(path),
            "size_bytes": len(data),
            "width": width,
            "height": height,
            "mode": mode,
        }
        (self.upload_dir / f"{upload_id}.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved upload %s (%s bytes)", upload_id, len(data))
        return meta

    def resolve_image_path(
        self,
        *,
        upload_id: str | None = None,
        image_path: str | Path | None = None,
    ) -> Path:
        """Resolve an on-disk image from upload id or explicit path."""
        if image_path is not None:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            return path

        if not upload_id:
            raise ValueError("Provide either an uploaded file or upload_id")

        matches = list(self.upload_dir.glob(f"{upload_id}.*"))
        image_matches = [
            path
            for path in matches
            if path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        ]
        if not image_matches:
            raise FileNotFoundError(f"Unknown upload_id: {upload_id}")
        return image_matches[0]

    def extract_features(self, image_path: Path) -> dict[str, Any]:
        """Extract Vision+CLIP features for the image."""
        opencv_dict = self.cv_extractor.extract(image_path)
        opencv = {key: float(opencv_dict[key]) for key in CV_FEATURE_KEYS}
        
        clip_list = self.clip_extractor.extract(image_path)
        
        feature_row = {}
        feature_row.update(opencv)
        for i, val in enumerate(clip_list):
            feature_row[f"clip_{i}"] = float(val)
            
        return {
            "feature_mode": "vision_clip",
            "image_path": str(image_path),
            "opencv_features": opencv,
            "clip_features": clip_list,
            "feature_vector": feature_row,
        }

    def predict_from_features(self, feature_row: pd.DataFrame, model: Any) -> dict[str, Any]:
        """Run the trained classifier on a feature row."""
        proba = model.predict_proba(feature_row)[0]
        class_index = int(proba.argmax())
        label = LABEL_ORDER[class_index]
        probabilities = {
            LABEL_ORDER[index]: float(proba[index]) for index in range(len(LABEL_ORDER))
        }
        return {
            "predicted_ambiguity": label,
            "predicted_class_index": class_index,
            "confidence": float(proba[class_index]),
            "probabilities": probabilities,
        }

    def predict(
        self,
        image_path: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Full prediction payload for an image."""
        features = self.extract_features(image_path)
        model = self._require_model()
        
        feature_df = pd.DataFrame([features["feature_vector"]], columns=list(VISION_CLIP_FEATURE_KEYS)).astype(float)
        prediction = self.predict_from_features(feature_df, model)
        return {
            **prediction,
            "feature_mode": features["feature_mode"],
            "opencv_features": features["opencv_features"],
            "feature_vector": features["feature_vector"],
        }

    def explain(
        self,
        image_path: Path,
        top_n: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prediction plus SHAP feature contributions."""
        prediction = self.predict(image_path)
        explainer = self._ensure_shap()

        feature_df = pd.DataFrame([prediction["feature_vector"]], columns=list(VISION_CLIP_FEATURE_KEYS)).astype(float)
        shap_values = explainer.compute_shap_values(feature_df)
        explanation = explainer.explain_prediction(
            shap_values,
            feature_df,
            row_index=0,
            top_n=top_n,
        )
        return {
            **prediction,
            "shap_explanation": explanation.to_dict(),
            "summary": explanation.summary_text(top_n=top_n),
        }


@lru_cache(maxsize=1)
def get_inference_service() -> AmbiguityInferenceService:
    """Return a process-wide inference service singleton."""
    return AmbiguityInferenceService()


def reset_inference_service() -> None:
    """Clear the cached inference service (tests / config reloads)."""
    get_inference_service.cache_clear()

