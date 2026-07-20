"""Inference service: features, prediction, and SHAP for uploaded images."""

from __future__ import annotations

import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image

from image_ambiguity.config import Settings, get_settings
from image_ambiguity.features.blip_captions import (
    ALL_STRATEGIES,
    BlipCaptionGenerator,
)
from image_ambiguity.features.caption_diversity import CaptionDiversityAnalyzer
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.logging_config import get_logger
from image_ambiguity.models.trainer import FEATURE_COLUMNS, LABEL_ORDER, ModelTrainer
from image_ambiguity.utils.common import ensure_dir

logger = get_logger("backend.services.inference")

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CV_FEATURE_KEYS = (
    "edge_density",
    "entropy",
    "brightness",
    "contrast",
    "color_variance",
    "texture",
)


def flatten_blip_captions(generated: dict[str, list[str]]) -> list[str]:
    """Flatten BLIP strategy outputs into a unique ordered caption list."""
    seen: set[str] = set()
    captions: list[str] = []
    for strategy in ALL_STRATEGIES:
        for caption in generated.get(strategy, []):
            text = " ".join(str(caption).strip().split())
            if text and text not in seen:
                seen.add(text)
                captions.append(text)
    return captions


class AmbiguityInferenceService:
    """End-to-end feature extraction, prediction, and explanation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.upload_dir = ensure_dir(self.settings.results_dir / "uploads")
        self.cv_extractor = OpenCVFeatureExtractor()
        self.diversity_analyzer = CaptionDiversityAnalyzer()
        self.embedding_generator = SentenceEmbeddingGenerator(
            model_name=self.settings.sentence_model_name,
            device=self.settings.device,
            batch_size=self.settings.embedding_batch_size,
        )
        self.blip_generator = BlipCaptionGenerator(
            device=self.settings.device,
            num_return_sequences=2,
        )
        self.trainer = ModelTrainer()
        self._model: Any | None = None
        self._loaded_model_path: Path | None = None
        self._embeddings_loaded = False
        self._blip_loaded = False
        self._shap_explainer: Any | None = None

    @property
    def model_path(self) -> Path:
        return self.settings.models_dir / "best_model.joblib"

    def _candidate_model_paths(self) -> list[Path]:
        """Prefer best_model, then sklearn Random Forest if XGBoost is missing."""
        models_dir = self.settings.models_dir
        return [
            models_dir / "best_model.joblib",
            models_dir / "random_forest.joblib",
            models_dir / "xgboost.joblib",
        ]

    def _require_model(self) -> Any:
        if self._model is None:
            errors: list[str] = []
            for path in self._candidate_model_paths():
                if not path.is_file():
                    continue
                try:
                    logger.info("Loading classifier from %s", path)
                    self._model = self.trainer.load_model(path)
                    self._loaded_model_path = path
                    return self._model
                except Exception as exc:  # noqa: BLE001 - try next artifact
                    message = f"{path.name}: {exc}"
                    logger.warning("Failed to load %s (%s)", path, exc)
                    errors.append(message)
                    # XGBoost-backed joblib needs the xgboost package installed.
                    if "xgboost" in str(exc).lower():
                        continue

            detail = "; ".join(errors) if errors else "no model files found"
            raise FileNotFoundError(
                "Could not load a trained classifier. "
                f"{detail}. Install xgboost (`pip install xgboost`) "
                "or run python src/train.py."
            )
        return self._model

    def _ensure_embeddings(self) -> None:
        if not self._embeddings_loaded:
            self.embedding_generator.load_model()
            self._embeddings_loaded = True

    def _ensure_blip(self) -> None:
        if not self._blip_loaded:
            self.blip_generator.load_model()
            self._blip_loaded = True

    def _ensure_shap(self) -> Any:
        if self._shap_explainer is None:
            from image_ambiguity.explainability.shap_explainer import SHAPExplainer

            model = self._require_model()
            self._shap_explainer = SHAPExplainer(
                model,
                feature_names=FEATURE_COLUMNS,
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

        meta = {
            "upload_id": upload_id,
            "filename": Path(filename).name if filename else path.name,
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

    def generate_captions(self, image_path: Path) -> list[str]:
        """Generate BLIP captions for an image."""
        self._ensure_blip()
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            generated = self.blip_generator.generate_all(rgb)
        captions = flatten_blip_captions(generated)
        if len(captions) < 2:
            raise RuntimeError(
                "BLIP produced fewer than 2 captions; cannot compute diversity"
            )
        return captions

    def compute_opencv_features(self, image_path: Path) -> dict[str, float]:
        """Extract the OpenCV features used by the classifier."""
        features = self.cv_extractor.extract(image_path)
        return {key: float(features[key]) for key in CV_FEATURE_KEYS}

    def compute_diversity_features(
        self,
        captions: Sequence[str],
    ) -> dict[str, Any]:
        """Embed captions and compute diversity metrics."""
        if len(captions) < 2:
            raise ValueError("At least 2 captions are required for diversity")
        self._ensure_embeddings()
        embeddings = self.embedding_generator.generate_embeddings(list(captions))
        metrics = self.diversity_analyzer.compute(embeddings)
        return {
            "average_similarity": float(metrics.average_similarity),
            "minimum_similarity": float(metrics.min_similarity),
            "maximum_similarity": float(metrics.max_similarity),
            "std_similarity": float(metrics.std_similarity),
            "caption_diversity": float(metrics.diversity_score),
            "n_captions": int(metrics.n_captions),
            "n_pairs": int(metrics.n_pairs),
            "captions": list(captions),
        }

    def build_feature_row(
        self,
        *,
        diversity: dict[str, Any],
        opencv: dict[str, float],
    ) -> pd.DataFrame:
        """Assemble a one-row DataFrame matching ``FEATURE_COLUMNS``."""
        row = {
            "average_similarity": diversity["average_similarity"],
            "minimum_similarity": diversity["minimum_similarity"],
            "maximum_similarity": diversity["maximum_similarity"],
            "std_similarity": diversity["std_similarity"],
            "caption_diversity": diversity["caption_diversity"],
            **opencv,
        }
        return pd.DataFrame([row], columns=list(FEATURE_COLUMNS)).astype(float)

    def extract_features(
        self,
        image_path: Path,
        *,
        captions: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return OpenCV + caption-diversity features for one image."""
        resolved_captions = (
            list(captions) if captions is not None else self.generate_captions(image_path)
        )
        opencv = self.compute_opencv_features(image_path)
        diversity = self.compute_diversity_features(resolved_captions)
        feature_row = self.build_feature_row(diversity=diversity, opencv=opencv)
        return {
            "image_path": str(image_path),
            "opencv_features": opencv,
            "caption_diversity": {
                key: diversity[key]
                for key in (
                    "average_similarity",
                    "minimum_similarity",
                    "maximum_similarity",
                    "std_similarity",
                    "caption_diversity",
                    "n_captions",
                    "n_pairs",
                )
            },
            "captions": diversity["captions"],
            "feature_vector": feature_row.iloc[0].to_dict(),
        }

    def predict_from_features(self, feature_row: pd.DataFrame) -> dict[str, Any]:
        """Run the trained classifier on a feature row."""
        model = self._require_model()
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
        *,
        captions: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Full prediction payload for an image."""
        features = self.extract_features(image_path, captions=captions)
        feature_row = pd.DataFrame(
            [features["feature_vector"]], columns=list(FEATURE_COLUMNS)
        )
        prediction = self.predict_from_features(feature_row)
        return {
            **prediction,
            "caption_diversity": features["caption_diversity"],
            "opencv_features": features["opencv_features"],
            "captions": features["captions"],
            "feature_vector": features["feature_vector"],
        }

    def explain(
        self,
        image_path: Path,
        *,
        captions: Sequence[str] | None = None,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Prediction plus SHAP feature contributions."""
        prediction = self.predict(image_path, captions=captions)
        feature_row = pd.DataFrame(
            [prediction["feature_vector"]], columns=list(FEATURE_COLUMNS)
        )
        explainer = self._ensure_shap()
        shap_values = explainer.compute_shap_values(feature_row)
        explanation = explainer.explain_prediction(
            shap_values,
            feature_row,
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
