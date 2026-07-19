"""Unit tests for BLIP caption generation helpers and generator API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_ambiguity.features.blip_captions import (
    BlipCaptionGenerator,
    compare_with_coco,
    word_jaccard,
)


class TestComparisonHelpers:
    def test_word_jaccard_identical(self) -> None:
        assert word_jaccard("A red bike", "a red bike!") == 1.0

    def test_word_jaccard_disjoint(self) -> None:
        assert word_jaccard("cat", "dog") == 0.0

    def test_compare_with_coco_finds_exact_and_overlap(self) -> None:
        generated = {
            "beam_search": ["a man cooking in a kitchen"],
            "top_k": ["someone prepares food"],
            "nucleus": ["A man cooking in a kitchen."],
        }
        coco = [
            "A man cooking in a kitchen.",
            "A chef prepares pizza.",
        ]
        comparison = compare_with_coco(generated, coco)

        assert comparison.n_coco == 2
        assert comparison.n_generated == 3
        assert "a man cooking in a kitchen" in {
            c.lower().rstrip(".") for c in comparison.exact_matches
        } or any(
            "man cooking" in match.lower() for match in comparison.exact_matches
        )
        assert "beam_search" in comparison.best_overlaps
        best = comparison.best_overlaps["beam_search"][0]
        assert best["jaccard"] > 0.5


class TestBlipCaptionGenerator:
    def test_generate_requires_loaded_model(self) -> None:
        generator = BlipCaptionGenerator(device="cpu")
        image = Image.new("RGB", (64, 64), color=(255, 0, 0))
        with pytest.raises(RuntimeError, match="load_model"):
            generator.generate_beam_search(image)

    def test_caption_image_with_mocked_model(self, tmp_path: Path) -> None:
        generator = BlipCaptionGenerator(
            device="cpu",
            num_return_sequences=2,
            num_beams=2,
        )

        fake_processor = MagicMock()
        fake_processor.return_value = {"pixel_values": MagicMock()}
        # Support both processor(images=...) call style and attribute access.
        fake_inputs = MagicMock()
        fake_inputs.items.return_value = [("pixel_values", MagicMock(to=MagicMock()))]
        fake_processor.side_effect = None
        fake_processor.return_value = fake_inputs
        fake_processor.batch_decode.side_effect = [
            ["a kitchen with a stove", "a person cooking food"],
            ["people preparing a meal", "a busy kitchen scene"],
            ["a cook stands near an oven", "food is being prepared"],
        ]

        fake_model = MagicMock()
        fake_model.generate.return_value = MagicMock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = None

        fake_torch = MagicMock()
        fake_torch.inference_mode.return_value.__enter__ = MagicMock(
            return_value=None
        )
        fake_torch.inference_mode.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with (
            patch(
                "transformers.BlipProcessor.from_pretrained",
                return_value=fake_processor,
            ),
            patch(
                "transformers.BlipForConditionalGeneration.from_pretrained",
                return_value=fake_model,
            ),
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch(
                "image_ambiguity.features.blip_captions.resolve_device",
                return_value="cpu",
            ),
        ):
            # Bypass real load_model import path by injecting fakes.
            generator.processor = fake_processor
            generator.model = fake_model
            generator._torch = fake_torch
            generator.device = "cpu"

            # Make processor(...) return tensors-like mapping.
            def _proc(**_kwargs):
                tensor = MagicMock()
                tensor.to.return_value = tensor
                return {"pixel_values": tensor}

            fake_processor.side_effect = _proc

            image = Image.new("RGB", (64, 64), color=(0, 128, 255))
            result = generator.caption_image(
                image,
                image_id=397133,
                file_name="000000397133.jpg",
                coco_captions=["A person cooking food in a kitchen."],
            )

        assert result.image_id == 397133
        assert set(result.generated_captions) == {
            "beam_search",
            "top_k",
            "nucleus",
        }
        assert result.comparison is not None
        assert result.comparison.n_coco == 1

        out = generator.save_result(result, tmp_path / "blip_397133.json")
        assert out.is_file()
        text = out.read_text(encoding="utf-8")
        assert "generated_captions" in text
        assert "coco_captions" in text
