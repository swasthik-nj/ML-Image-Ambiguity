"""Sentence-BERT embedding generation for caption diversity features."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir, timed

logger = get_logger("features.sentence_embeddings")

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


def resolve_device(device: str | None = None) -> str:
    """Pick a compute device, preferring CUDA when available.

    Args:
        device: Explicit device string (``cpu``, ``cuda``, ``cuda:0``, …).
            Use ``None`` or ``"auto"`` to select automatically.

    Returns:
        Device string suitable for SentenceTransformer.
    """
    if device is not None and device.lower() not in {"", "auto"}:
        return device

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed; falling back to CPU.")
        return "cpu"

    if torch.cuda.is_available():
        selected = "cuda"
        logger.info("CUDA is available; using device=%s", selected)
        return selected

    logger.info("CUDA not available; using device=cpu")
    return "cpu"


class SentenceEmbeddingGenerator:
    """Generate, persist, and reload Sentence-BERT caption embeddings.

    Uses ``all-MiniLM-L6-v2`` by default with batched inference and
    automatic GPU selection when CUDA is present.

    Args:
        model_name: Hugging Face / Sentence-Transformers model id.
        device: Target device (``cpu``, ``cuda``, or ``auto``).
        batch_size: Batch size for ``encode`` calls.
        normalize_embeddings: Whether to L2-normalize output vectors.
        show_progress_bar: Forwarded to SentenceTransformer.encode.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.model: Any | None = None

    def load_model(self) -> Any:
        """Load the Sentence-BERT model onto the selected device.

        Returns:
            The loaded :class:`~sentence_transformers.SentenceTransformer`
            instance.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
            RuntimeError: If model loading fails.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info(
            "Loading Sentence-BERT model '%s' on device=%s",
            self.model_name,
            self.device,
        )
        with timed(f"load_model:{self.model_name}"):
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:  # noqa: BLE001 - surface load failures clearly
                raise RuntimeError(
                    f"Failed to load Sentence-BERT model '{self.model_name}' "
                    f"on device '{self.device}'."
                ) from exc

        logger.info("Model loaded successfully: %s", self.model_name)
        return self.model

    def _require_model(self) -> Any:
        """Return the loaded model or raise if ``load_model`` was not called."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        return self.model

    def generate_embeddings(self, captions: Sequence[str]) -> np.ndarray:
        """Encode captions into a 2-D NumPy embedding matrix.

        Args:
            captions: Sequence of caption strings.

        Returns:
            Array of shape ``(n_captions, embedding_dim)`` with dtype
            ``float32``.

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError: If ``captions`` is empty or contains non-strings.
        """
        model = self._require_model()

        if not captions:
            raise ValueError("captions must be a non-empty sequence of strings.")

        cleaned: list[str] = []
        for index, caption in enumerate(captions):
            if not isinstance(caption, str):
                raise ValueError(
                    f"captions[{index}] must be str, got {type(caption).__name__}"
                )
            text = caption.strip()
            if not text:
                raise ValueError(f"captions[{index}] is empty after stripping.")
            cleaned.append(text)

        logger.info(
            "Generating embeddings for %s captions (batch_size=%s, device=%s)",
            len(cleaned),
            self.batch_size,
            self.device,
        )

        with timed(f"generate_embeddings:n={len(cleaned)}"):
            vectors = model.encode(
                cleaned,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=self.show_progress_bar,
            )

        embeddings = np.asarray(vectors, dtype=np.float32)
        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D embeddings, got shape {embeddings.shape}"
            )

        logger.info(
            "Generated embeddings with shape=%s dtype=%s",
            embeddings.shape,
            embeddings.dtype,
        )
        return embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: str | Path,
        *,
        captions: Sequence[str] | None = None,
    ) -> Path:
        """Persist embeddings (and optional captions) to disk.

        Saves a ``.npz`` archive with keys ``embeddings`` and, when provided,
        ``captions``. Parent directories are created automatically.

        Args:
            embeddings: Embedding matrix from :meth:`generate_embeddings`.
            path: Destination path (``.npz`` recommended).
            captions: Optional caption strings aligned with rows of
                ``embeddings``.

        Returns:
            Resolved path written to disk.

        Raises:
            ValueError: If shapes are invalid or caption count mismatches.
        """
        array = np.asarray(embeddings)
        if array.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D, got shape {array.shape}"
            )
        if captions is not None and len(captions) != array.shape[0]:
            raise ValueError(
                f"len(captions)={len(captions)} does not match "
                f"embeddings.shape[0]={array.shape[0]}"
            )

        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".npz")

        ensure_dir(destination.parent)
        payload: dict[str, Any] = {
            "embeddings": np.asarray(array, dtype=np.float32),
            "model_name": np.array(self.model_name),
            "device": np.array(self.device),
        }
        if captions is not None:
            payload["captions"] = np.asarray(list(captions), dtype=object)

        np.savez_compressed(destination, **payload)
        logger.info(
            "Saved embeddings to %s (shape=%s)",
            destination,
            array.shape,
        )
        return destination.resolve()

    def load_embeddings(self, path: str | Path) -> np.ndarray:
        """Load embeddings previously written by :meth:`save_embeddings`.

        Args:
            path: Path to a ``.npz`` (or legacy ``.npy``) file.

        Returns:
            Embedding matrix as ``float32`` NumPy array.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file format is unsupported or invalid.
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Embeddings file not found: {source}")

        logger.info("Loading embeddings from %s", source)

        if source.suffix == ".npy":
            embeddings = np.load(source)
        elif source.suffix == ".npz":
            with np.load(source, allow_pickle=True) as data:
                if "embeddings" not in data:
                    raise ValueError(
                        f"Archive {source} does not contain key 'embeddings'"
                    )
                embeddings = data["embeddings"]
                if "model_name" in data:
                    logger.info(
                        "Archive model_name=%s",
                        data["model_name"].item()
                        if data["model_name"].shape == ()
                        else data["model_name"],
                    )
        else:
            raise ValueError(
                f"Unsupported embeddings file type: {source.suffix} "
                "(expected .npz or .npy)"
            )

        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Loaded embeddings must be 2-D, got {array.shape}")

        logger.info("Loaded embeddings with shape=%s", array.shape)
        return array

    def __repr__(self) -> str:
        status = "loaded" if self.model is not None else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"device={self.device!r}, "
            f"batch_size={self.batch_size}, "
            f"status={status})"
        )
