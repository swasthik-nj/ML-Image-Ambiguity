"""Unit tests for :class:`SentenceEmbeddingGenerator`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from image_ambiguity.features.sentence_embeddings import (
    SentenceEmbeddingGenerator,
    resolve_device,
)


class TestResolveDevice:
    def test_explicit_device_is_respected(self) -> None:
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda:0") == "cuda:0"

    def test_auto_uses_cuda_when_available(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert resolve_device("auto") == "cuda"

    def test_auto_falls_back_to_cpu(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert resolve_device(None) == "cpu"

    def test_missing_torch_falls_back_to_cpu(self) -> None:
        # ``sys.modules['torch'] = None`` makes ``import torch`` raise ImportError.
        with patch.dict("sys.modules", {"torch": None}):
            assert resolve_device("auto") == "cpu"


class TestSentenceEmbeddingGenerator:
    def test_load_model_sets_model(self) -> None:
        generator = SentenceEmbeddingGenerator(device="cpu", batch_size=8)
        fake_model = MagicMock(name="SentenceTransformer")

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_model,
        ) as ctor:
            loaded = generator.load_model()

        ctor.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        assert loaded is fake_model
        assert generator.model is fake_model

    def test_generate_embeddings_returns_float32_numpy(self) -> None:
        generator = SentenceEmbeddingGenerator(device="cpu", batch_size=2)
        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones((2, 4), dtype=np.float64)
        generator.model = fake_model

        captions = ["a red bike", "a blue car"]
        embeddings = generator.generate_embeddings(captions)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32
        assert embeddings.shape == (2, 4)
        fake_model.encode.assert_called_once()
        kwargs = fake_model.encode.call_args.kwargs
        assert kwargs["batch_size"] == 2
        assert kwargs["convert_to_numpy"] is True
        assert kwargs["normalize_embeddings"] is True

    def test_generate_embeddings_requires_loaded_model(self) -> None:
        generator = SentenceEmbeddingGenerator(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            generator.generate_embeddings(["caption"])

    def test_generate_embeddings_rejects_empty_input(self) -> None:
        generator = SentenceEmbeddingGenerator(device="cpu")
        generator.model = MagicMock()
        with pytest.raises(ValueError, match="non-empty"):
            generator.generate_embeddings([])

    def test_generate_embeddings_rejects_blank_caption(self) -> None:
        generator = SentenceEmbeddingGenerator(device="cpu")
        generator.model = MagicMock()
        with pytest.raises(ValueError, match="empty"):
            generator.generate_embeddings(["  "])

    def test_save_and_load_embeddings_roundtrip(self, tmp_path: Path) -> None:
        generator = SentenceEmbeddingGenerator(device="cpu")
        embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
        captions = ["a", "b", "c"]
        path = tmp_path / "caption_embeddings.npz"

        saved = generator.save_embeddings(
            embeddings, path, captions=captions
        )
        loaded = generator.load_embeddings(saved)

        assert saved.is_file()
        np.testing.assert_allclose(loaded, embeddings)
        assert loaded.dtype == np.float32

    def test_load_embeddings_supports_npy(self, tmp_path: Path) -> None:
        generator = SentenceEmbeddingGenerator(device="cpu")
        embeddings = np.random.randn(5, 8).astype(np.float32)
        path = tmp_path / "vectors.npy"
        np.save(path, embeddings)

        loaded = generator.load_embeddings(path)
        np.testing.assert_allclose(loaded, embeddings)

    def test_save_embeddings_caption_mismatch_raises(self, tmp_path: Path) -> None:
        generator = SentenceEmbeddingGenerator(device="cpu")
        embeddings = np.zeros((2, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="does not match"):
            generator.save_embeddings(
                embeddings,
                tmp_path / "bad.npz",
                captions=["only-one"],
            )

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            SentenceEmbeddingGenerator(batch_size=0)

    def test_repr_includes_status(self) -> None:
        generator = SentenceEmbeddingGenerator(device="cpu")
        assert "not loaded" in repr(generator)
        generator.model = MagicMock()
        assert "loaded" in repr(generator)
