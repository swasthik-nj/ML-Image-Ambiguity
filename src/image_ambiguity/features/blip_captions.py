"""BLIP image caption generation with multiple decoding strategies."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from image_ambiguity.features.sentence_embeddings import resolve_device
from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir, save_json, timed

logger = get_logger("features.blip_captions")

DEFAULT_MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEFAULT_MAX_LENGTH = 50
DEFAULT_NUM_BEAMS = 5
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_NUM_RETURN_SEQUENCES = 3
DEFAULT_TEMPERATURE = 1.0

STRATEGY_BEAM = "beam_search"
STRATEGY_TOP_K = "top_k"
STRATEGY_NUCLEUS = "nucleus"
ALL_STRATEGIES = (STRATEGY_BEAM, STRATEGY_TOP_K, STRATEGY_NUCLEUS)


@dataclass(slots=True)
class CaptionComparison:
    """Side-by-side comparison of BLIP captions against COCO references."""

    n_coco: int
    n_generated: int
    exact_matches: list[str] = field(default_factory=list)
    best_overlaps: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BlipCaptionResult:
    """Structured BLIP caption output for one image."""

    image_id: int | None
    file_name: str | None
    model: str
    device: str
    generated_captions: dict[str, list[str]]
    coco_captions: list[str] = field(default_factory=list)
    comparison: CaptionComparison | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "image_id": self.image_id,
            "file_name": self.file_name,
            "model": self.model,
            "device": self.device,
            "generated_captions": self.generated_captions,
            "coco_captions": self.coco_captions,
        }
        if self.comparison is not None:
            payload["comparison"] = self.comparison.to_dict()
        return payload


def _normalize_caption(text: str) -> str:
    """Lowercase and strip punctuation for loose caption matching."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokenize(text: str) -> set[str]:
    """Return word tokens for Jaccard overlap."""
    return {token for token in _normalize_caption(text).split() if token}


def word_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between word sets of two captions."""
    left = _tokenize(a)
    right = _tokenize(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def compare_with_coco(
    generated: dict[str, list[str]],
    coco_captions: list[str],
) -> CaptionComparison:
    """Compare generated captions against COCO reference captions.

    Reports exact normalized matches and, per strategy, the best word-overlap
    COCO caption for each generated string.
    """
    coco_norm = {_normalize_caption(c): c for c in coco_captions}
    flat_generated = [c for captions in generated.values() for c in captions]
    exact = sorted(
        {
            caption
            for caption in flat_generated
            if _normalize_caption(caption) in coco_norm
        }
    )

    best_overlaps: dict[str, list[dict[str, Any]]] = {}
    for strategy, captions in generated.items():
        rows: list[dict[str, Any]] = []
        for caption in captions:
            if not coco_captions:
                rows.append(
                    {
                        "generated": caption,
                        "best_coco": None,
                        "jaccard": 0.0,
                    }
                )
                continue
            scored = [(ref, word_jaccard(caption, ref)) for ref in coco_captions]
            best_ref, score = max(scored, key=lambda item: item[1])
            rows.append(
                {
                    "generated": caption,
                    "best_coco": best_ref,
                    "jaccard": round(score, 4),
                }
            )
        best_overlaps[strategy] = rows

    return CaptionComparison(
        n_coco=len(coco_captions),
        n_generated=len(flat_generated),
        exact_matches=exact,
        best_overlaps=best_overlaps,
    )


class BlipCaptionGenerator:
    """Generate image captions with BLIP using multiple decoding strategies.

    Strategies:
        * ``beam_search`` — deterministic beam search
        * ``top_k`` — top-k sampling
        * ``nucleus`` — nucleus (top-p) sampling

    Args:
        model_name: Hugging Face BLIP model id.
        device: Target device (``cpu``, ``cuda``, or ``auto``).
        max_length: Maximum generated caption length.
        num_beams: Beam width for beam search.
        top_k: ``k`` for top-k sampling.
        top_p: Nucleus probability mass.
        num_return_sequences: Captions returned per sampling strategy.
        temperature: Softmax temperature for sampling strategies.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = "auto",
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        num_return_sequences: int = DEFAULT_NUM_RETURN_SEQUENCES,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        if num_beams < 1:
            raise ValueError("num_beams must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_length = max_length
        self.num_beams = max(num_beams, num_return_sequences)
        self.top_k = top_k
        self.top_p = top_p
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature
        self.processor: Any | None = None
        self.model: Any | None = None
        self._torch: Any | None = None

    def load_model(self) -> tuple[Any, Any]:
        """Load BLIP processor and conditional generation model.

        Returns:
            ``(processor, model)`` pair.

        Raises:
            ImportError: If ``transformers`` / ``torch`` are unavailable.
            RuntimeError: If model loading fails.
        """
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            ) from exc

        logger.info(
            "Loading BLIP model '%s' on device=%s",
            self.model_name,
            self.device,
        )
        with timed(f"load_model:{self.model_name}"):
            try:
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name
                )
                self.model.to(self.device)
                self.model.eval()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to load BLIP model '{self.model_name}' "
                    f"on device '{self.device}'."
                ) from exc

        self._torch = torch
        logger.info("BLIP model loaded successfully: %s", self.model_name)
        return self.processor, self.model

    def _require_model(self) -> tuple[Any, Any, Any]:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        if self._torch is None:
            import torch

            self._torch = torch
        return self.processor, self.model, self._torch

    def _prepare_inputs(self, image: Image.Image) -> dict[str, Any]:
        processor, _, _ = self._require_model()
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _decode(self, sequences: Any) -> list[str]:
        processor, _, _ = self._require_model()
        texts = processor.batch_decode(sequences, skip_special_tokens=True)
        seen: set[str] = set()
        captions: list[str] = []
        for text in texts:
            caption = " ".join(text.strip().split())
            if not caption or caption in seen:
                continue
            seen.add(caption)
            captions.append(caption)
        return captions

    def generate_beam_search(self, image: Image.Image) -> list[str]:
        """Generate captions with beam search decoding."""
        _, model, torch = self._require_model()
        inputs = self._prepare_inputs(image)
        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                num_return_sequences=min(
                    self.num_return_sequences, self.num_beams
                ),
                early_stopping=True,
            )
        return self._decode(sequences)

    def generate_top_k(self, image: Image.Image) -> list[str]:
        """Generate captions with top-k sampling."""
        _, model, torch = self._require_model()
        inputs = self._prepare_inputs(image)
        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=True,
                top_k=self.top_k,
                temperature=self.temperature,
                num_return_sequences=self.num_return_sequences,
            )
        return self._decode(sequences)

    def generate_nucleus(self, image: Image.Image) -> list[str]:
        """Generate captions with nucleus (top-p) sampling."""
        _, model, torch = self._require_model()
        inputs = self._prepare_inputs(image)
        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
                num_return_sequences=self.num_return_sequences,
            )
        return self._decode(sequences)

    def generate_all(
        self,
        image: Image.Image,
        *,
        strategies: tuple[str, ...] = ALL_STRATEGIES,
    ) -> dict[str, list[str]]:
        """Run all requested decoding strategies on one image.

        Args:
            image: RGB PIL image.
            strategies: Subset of ``beam_search``, ``top_k``, ``nucleus``.

        Returns:
            Mapping of strategy name → list of caption strings.
        """
        generators = {
            STRATEGY_BEAM: self.generate_beam_search,
            STRATEGY_TOP_K: self.generate_top_k,
            STRATEGY_NUCLEUS: self.generate_nucleus,
        }
        unknown = set(strategies) - set(generators)
        if unknown:
            raise ValueError(f"Unknown strategies: {sorted(unknown)}")

        results: dict[str, list[str]] = {}
        for strategy in strategies:
            with timed(f"blip_generate:{strategy}"):
                results[strategy] = generators[strategy](image)
            logger.info(
                "Strategy %s produced %s caption(s)",
                strategy,
                len(results[strategy]),
            )
        return results

    def caption_image(
        self,
        image: Image.Image,
        *,
        image_id: int | None = None,
        file_name: str | None = None,
        coco_captions: list[str] | None = None,
        strategies: tuple[str, ...] = ALL_STRATEGIES,
    ) -> BlipCaptionResult:
        """Generate captions and optionally compare with COCO references.

        Args:
            image: Input RGB image.
            image_id: Optional COCO image id for the JSON record.
            file_name: Optional source file name.
            coco_captions: Optional human COCO captions for comparison.
            strategies: Decoding strategies to run.

        Returns:
            :class:`BlipCaptionResult` ready for JSON serialization.
        """
        generated = self.generate_all(image, strategies=strategies)
        coco = list(coco_captions or [])
        comparison = compare_with_coco(generated, coco) if coco else None
        return BlipCaptionResult(
            image_id=image_id,
            file_name=file_name,
            model=self.model_name,
            device=self.device,
            generated_captions=generated,
            coco_captions=coco,
            comparison=comparison,
        )

    def save_result(
        self,
        result: BlipCaptionResult,
        path: str | Path,
    ) -> Path:
        """Save a caption result as pretty-printed JSON."""
        destination = Path(path)
        ensure_dir(destination.parent)
        saved = save_json(result.to_dict(), destination)
        logger.info("Saved BLIP captions to %s", saved)
        return saved
