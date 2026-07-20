"""Batch BLIP caption generation with on-disk caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from image_ambiguity.features.blip_captions import (
    ALL_STRATEGIES,
    BlipCaptionGenerator,
)
from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir, load_json, save_json, timed

logger = get_logger("pipeline.ai_captions")

DEFAULT_CACHE_PATH = Path("results/captions/blip_caption_cache.json")


def flatten_generated_captions(generated: dict[str, list[str]]) -> list[str]:
    """Flatten strategy → captions mapping into a unique ordered list."""
    seen: set[str] = set()
    captions: list[str] = []
    for strategy in ALL_STRATEGIES:
        for caption in generated.get(strategy, []):
            text = " ".join(str(caption).strip().split())
            if text and text not in seen:
                seen.add(text)
                captions.append(text)
    return captions


def load_caption_cache(path: str | Path) -> dict[int, list[str]]:
    cache_path = Path(path)
    if not cache_path.is_file():
        return {}
    raw = load_json(cache_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Caption cache must be a JSON object: {cache_path}")
    return {int(key): list(value) for key, value in raw.items()}


def save_caption_cache(
    captions_by_image: dict[int, list[str]],
    path: str | Path,
) -> Path:
    payload = {
        str(image_id): captions
        for image_id, captions in sorted(captions_by_image.items())
    }
    return save_json(payload, path)


def collect_blip_captions(
    loader: Any,
    image_ids: Sequence[int],
    *,
    generator: BlipCaptionGenerator | None = None,
    cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    strategies: tuple[str, ...] = ALL_STRATEGIES,
    force: bool = False,
    seed: int = 42,
) -> dict[int, list[str]]:
    """Generate BLIP captions for image ids, reusing cache when possible."""
    cache: dict[int, list[str]] = {}
    if cache_path is not None and not force:
        cache = load_caption_cache(cache_path)

    missing = [
        int(image_id)
        for image_id in image_ids
        if force or int(image_id) not in cache or len(cache[int(image_id)]) < 2
    ]

    if missing:
        blip = generator or BlipCaptionGenerator()
        if blip.model is None:
            blip.load_model()

        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass

        logger.info(
            "Generating BLIP captions for %s image(s) (%s cached)",
            len(missing),
            len(image_ids) - len(missing),
        )
        with timed(f"blip_batch:n={len(missing)}"):
            for index, image_id in enumerate(missing, start=1):
                try:
                    image = loader.get_image(image_id)
                    generated = blip.generate_all(image, strategies=strategies)
                    captions = flatten_generated_captions(generated)
                    cache[image_id] = captions
                    logger.info(
                        "[%s/%s] image_id=%s -> %s caption(s)",
                        index,
                        len(missing),
                        image_id,
                        len(captions),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "BLIP captioning failed for image_id=%s", image_id
                    )
                    cache.setdefault(image_id, [])

                if cache_path is not None and index % 5 == 0:
                    ensure_dir(Path(cache_path).parent)
                    save_caption_cache(cache, cache_path)

        if cache_path is not None:
            ensure_dir(Path(cache_path).parent)
            save_caption_cache(cache, cache_path)
    else:
        logger.info(
            "Using cached BLIP captions for all %s image(s)", len(image_ids)
        )

    return {
        int(image_id): list(cache.get(int(image_id), []))
        for image_id in image_ids
    }
