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
from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.blip_captions import (
    BLIP_MODE_DIVERSE,
    BLIP_MODE_STABLE,
    BlipCaptionGenerator,
    flatten_strategy_captions,
)
from image_ambiguity.features.caption_clustering import cluster_captions_by_similarity
from image_ambiguity.features.caption_diversity import CaptionDiversityAnalyzer
from image_ambiguity.features.cv_features import OpenCVFeatureExtractor
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator
from image_ambiguity.features.visual_simplicity import (
    VisualSimplicityConfig,
    assess_visual_simplicity,
)
from image_ambiguity.logging_config import get_logger
from image_ambiguity.models.trainer import FEATURE_COLUMNS, LABEL_ORDER, ModelTrainer
from image_ambiguity.utils.common import ensure_dir

logger = get_logger("backend.services.inference")

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_FILENAME_RE = re.compile(
    r"(?:^|[/\\])0*(\d{1,12})\.(?:jpe?g|png|bmp|webp)$",
    re.IGNORECASE,
)
CV_FEATURE_KEYS = (
    "edge_density",
    "entropy",
    "brightness",
    "contrast",
    "color_variance",
    "texture",
)


def parse_coco_image_id(filename: str | None) -> int | None:
    """Extract a COCO image id from names like ``000000538236.jpg``."""
    if not filename:
        return None
    match = COCO_FILENAME_RE.search(str(filename).strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def flatten_blip_captions(generated: dict[str, list[str]]) -> list[str]:
    """Flatten BLIP strategy outputs into a unique ordered caption list."""
    return flatten_strategy_captions(generated)


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
            num_return_sequences=self.settings.blip_diverse_num_return_sequences,
            temperature=self.settings.blip_diverse_temperature,
            top_p=self.settings.blip_diverse_top_p,
            top_k=self.settings.blip_diverse_top_k,
        )
        self.trainer = ModelTrainer()
        self._model: Any | None = None
        self._loaded_model_path: Path | None = None
        self._embeddings_loaded = False
        self._blip_loaded = False
        self._shap_explainer: Any | None = None
        self._coco_loader: CocoDatasetLoader | None = None

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

    def _ensure_coco(self) -> CocoDatasetLoader | None:
        """Lazy-load COCO captions; return None if annotations are unavailable."""
        if self._coco_loader is not None:
            return self._coco_loader
        annotation = self.settings.annotation_file
        image_dir = self.settings.image_dir
        if not Path(annotation).is_file() or not Path(image_dir).is_dir():
            logger.warning("COCO annotations/images unavailable; skipping lookup")
            return None
        loader = CocoDatasetLoader(annotation, image_dir)
        loader.load_annotations()
        self._coco_loader = loader
        return loader

    def lookup_coco_captions(
        self,
        *,
        filename: str | None = None,
        image_id: int | None = None,
    ) -> list[str]:
        """Return human COCO captions for a filename or image id."""
        resolved_id = image_id if image_id is not None else parse_coco_image_id(filename)
        if resolved_id is None:
            return []
        loader = self._ensure_coco()
        if loader is None:
            return []
        try:
            captions = loader.get_captions(int(resolved_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("COCO caption lookup failed for id=%s (%s)", resolved_id, exc)
            return []
        cleaned = [" ".join(str(c).strip().split()) for c in captions if str(c).strip()]
        # Preserve order, drop exact duplicates.
        seen: set[str] = set()
        unique: list[str] = []
        for caption in cleaned:
            key = caption.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(caption)
        return unique

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

        original_name = Path(filename).name if filename else path.name
        coco_image_id = parse_coco_image_id(original_name)
        coco_captions = (
            self.lookup_coco_captions(image_id=coco_image_id)
            if coco_image_id is not None
            else []
        )
        meta = {
            "upload_id": upload_id,
            "filename": original_name,
            "content_type": f"image/{suffix.lstrip('.')}",
            "path": str(path),
            "size_bytes": len(data),
            "width": width,
            "height": height,
            "mode": mode,
            "coco_image_id": coco_image_id,
            "coco_captions": coco_captions,
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

    def visual_simplicity_config(self) -> VisualSimplicityConfig:
        """Build visual-gate config from settings."""
        return VisualSimplicityConfig(
            enabled=self.settings.visual_gate_enabled,
            edge_density_max=self.settings.visual_gate_edge_density_max,
            entropy_max=self.settings.visual_gate_entropy_max,
            texture_max=self.settings.visual_gate_texture_max,
            contrast_max=self.settings.visual_gate_contrast_max,
            color_variance_max=self.settings.visual_gate_color_variance_max,
            brightness_min=self.settings.visual_gate_brightness_min,
            min_simple_votes=self.settings.visual_gate_min_simple_votes,
            min_vote_fraction=self.settings.visual_gate_min_vote_fraction,
        )

    def assess_image_simplicity(
        self,
        opencv: dict[str, float],
    ) -> dict[str, Any]:
        """Assess visual simplicity from OpenCV features (gating signal only)."""
        assessment = assess_visual_simplicity(
            opencv,
            self.visual_simplicity_config(),
        )
        return assessment.to_dict()

    def generate_captions(
        self,
        image_path: Path,
        *,
        mode: str = BLIP_MODE_DIVERSE,
    ) -> tuple[list[str], dict[str, Any]]:
        """Generate BLIP captions for an image in stable or diverse mode."""
        self._ensure_blip()
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            generated = self.blip_generator.generate_for_mode(
                rgb,
                mode,
                stable_num_beams=self.settings.blip_stable_num_beams,
                stable_num_return_sequences=(
                    self.settings.blip_stable_num_return_sequences
                ),
                stable_max_length=self.settings.blip_stable_max_length,
                stable_use_mild_prompts=self.settings.blip_stable_use_mild_prompts,
            )
        captions = flatten_blip_captions(generated)
        if len(captions) < 2:
            raise RuntimeError(
                "BLIP produced fewer than 2 captions; cannot compute diversity"
            )
        meta = {
            "blip_mode": mode,
            "n_raw_captions": len(captions),
            "strategies": sorted(generated.keys()),
        }
        return captions, meta

    def maybe_cluster_captions(
        self,
        captions: Sequence[str],
        *,
        apply: bool,
    ) -> tuple[list[str], dict[str, Any]]:
        """Optionally cluster captions by SBERT similarity before diversity."""
        cleaned = [" ".join(str(c).strip().split()) for c in captions if str(c).strip()]
        if not apply:
            reps = list(cleaned)
            if len(reps) == 1:
                reps = [reps[0], reps[0]]
            return reps, {
                "representatives": reps,
                "cluster_ids": list(range(len(cleaned))),
                "n_input": len(cleaned),
                "n_clusters": len(cleaned),
                "similarity_threshold": float(
                    self.settings.blip_cluster_similarity_threshold
                ),
                "applied": False,
            }

        self._ensure_embeddings()
        embeddings = self.embedding_generator.generate_embeddings(cleaned)
        clustered = cluster_captions_by_similarity(
            cleaned,
            embeddings,
            similarity_threshold=self.settings.blip_cluster_similarity_threshold,
            apply=True,
        )
        return clustered.representatives, clustered.to_dict()

    def resolve_captions(
        self,
        image_path: Path,
        *,
        captions: Sequence[str] | None = None,
        upload_id: str | None = None,
        force_blip: bool = False,
        visual_is_simple: bool = False,
    ) -> tuple[list[str], str, dict[str, Any]]:
        """Choose user, COCO-human, or BLIP captions.

        Returns:
            ``(captions, source, meta)`` where source is ``user``,
            ``coco_human``, or ``blip``.
        """
        meta: dict[str, Any] = {
            "blip_mode": None,
            "clustering": None,
        }
        if captions is not None and len(captions) >= 2:
            return list(captions), "user", meta

        if not force_blip:
            upload_meta = self.read_upload_meta(upload_id) if upload_id else None
            filename = (upload_meta or {}).get("filename") or image_path.name
            coco_captions = list((upload_meta or {}).get("coco_captions") or [])
            if len(coco_captions) < 2:
                coco_captions = self.lookup_coco_captions(filename=str(filename))
            if len(coco_captions) >= 2:
                logger.info(
                    "Using %s COCO human captions for %s",
                    len(coco_captions),
                    filename,
                )
                return coco_captions, "coco_human", meta

        mode = BLIP_MODE_STABLE if visual_is_simple else BLIP_MODE_DIVERSE
        blip_captions, blip_meta = self.generate_captions(image_path, mode=mode)
        meta.update(blip_meta)

        apply_cluster = (
            self.settings.blip_cluster_when_simple
            if visual_is_simple
            else self.settings.blip_cluster_when_complex
        )
        clustered, cluster_meta = self.maybe_cluster_captions(
            blip_captions,
            apply=apply_cluster,
        )
        meta["clustering"] = cluster_meta
        meta["n_captions_before_cluster"] = len(blip_captions)
        meta["n_captions_after_cluster"] = len(clustered)
        logger.info(
            "BLIP mode=%s simple=%s cluster=%s raw=%s clustered=%s",
            mode,
            visual_is_simple,
            apply_cluster,
            len(blip_captions),
            len(clustered),
        )
        return clustered, "blip", meta

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
        upload_id: str | None = None,
        force_blip: bool = False,
    ) -> dict[str, Any]:
        """Return OpenCV + caption-diversity features for one image.

        Visual simplicity only selects stable vs diverse BLIP and clustering.
        It does not hardcode Low/Medium/High or change label thresholds.
        """
        opencv = self.compute_opencv_features(image_path)
        visual = self.assess_image_simplicity(opencv)
        resolved_captions, caption_source, caption_meta = self.resolve_captions(
            image_path,
            captions=captions,
            upload_id=upload_id,
            force_blip=force_blip,
            visual_is_simple=bool(visual.get("is_simple")),
        )
        diversity = self.compute_diversity_features(resolved_captions)
        feature_row = self.build_feature_row(diversity=diversity, opencv=opencv)
        return {
            "image_path": str(image_path),
            "caption_source": caption_source,
            "opencv_features": opencv,
            "visual_simplicity": visual,
            "caption_pipeline": caption_meta,
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
        upload_id: str | None = None,
        force_blip: bool = False,
    ) -> dict[str, Any]:
        """Full prediction payload for an image."""
        features = self.extract_features(
            image_path,
            captions=captions,
            upload_id=upload_id,
            force_blip=force_blip,
        )
        feature_row = pd.DataFrame(
            [features["feature_vector"]], columns=list(FEATURE_COLUMNS)
        )
        prediction = self.predict_from_features(feature_row)
        return {
            **prediction,
            "caption_source": features["caption_source"],
            "caption_diversity": features["caption_diversity"],
            "opencv_features": features["opencv_features"],
            "visual_simplicity": features.get("visual_simplicity"),
            "caption_pipeline": features.get("caption_pipeline"),
            "captions": features["captions"],
            "feature_vector": features["feature_vector"],
        }

    def explain(
        self,
        image_path: Path,
        *,
        captions: Sequence[str] | None = None,
        upload_id: str | None = None,
        force_blip: bool = False,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Prediction plus SHAP feature contributions."""
        prediction = self.predict(
            image_path,
            captions=captions,
            upload_id=upload_id,
            force_blip=force_blip,
        )
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


def reset_inference_service() -> None:
    """Clear the cached inference service (tests / config reloads)."""
    get_inference_service.cache_clear()
