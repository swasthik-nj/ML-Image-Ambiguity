"""Unit tests for :class:`image_ambiguity.data.coco_loader.CocoDatasetLoader`."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pycocotools.coco import COCO

from image_ambiguity import CocoDatasetLoader

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ANNOTATION_FILE = FIXTURES / "captions_tiny.json"
IMAGE_DIR = FIXTURES / "images"


@pytest.fixture
def loader() -> CocoDatasetLoader:
    """Return a loader pointed at the tiny fixture dataset."""
    return CocoDatasetLoader(ANNOTATION_FILE, IMAGE_DIR)


@pytest.fixture
def loaded_loader(loader: CocoDatasetLoader) -> CocoDatasetLoader:
    """Return a loader with annotations already indexed."""
    loader.load_annotations()
    return loader


class TestLoadAnnotations:
    def test_load_annotations_returns_coco(self, loader: CocoDatasetLoader) -> None:
        coco = loader.load_annotations()

        assert isinstance(coco, COCO)
        assert loader.coco is coco
        assert len(loader) == 2

    def test_load_annotations_missing_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"
        loader = CocoDatasetLoader(missing, IMAGE_DIR)

        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            loader.load_annotations()

    def test_load_annotations_missing_image_dir_raises(self, tmp_path: Path) -> None:
        loader = CocoDatasetLoader(ANNOTATION_FILE, tmp_path / "no_images")

        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            loader.load_annotations()


class TestGetImage:
    def test_get_image_returns_rgb_pil_image(
        self, loaded_loader: CocoDatasetLoader
    ) -> None:
        image = loaded_loader.get_image(1)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (8, 8)

    def test_get_image_unknown_id_raises(
        self, loaded_loader: CocoDatasetLoader
    ) -> None:
        with pytest.raises(KeyError, match="Unknown image_id"):
            loaded_loader.get_image(999)

    def test_get_image_before_load_raises(self, loader: CocoDatasetLoader) -> None:
        with pytest.raises(RuntimeError, match="load_annotations"):
            loader.get_image(1)

    def test_get_image_missing_file_raises(
        self, loaded_loader: CocoDatasetLoader, tmp_path: Path
    ) -> None:
        loaded_loader.image_dir = tmp_path

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            loaded_loader.get_image(1)


class TestGetCaptions:
    def test_get_captions_returns_expected_texts(
        self, loaded_loader: CocoDatasetLoader
    ) -> None:
        captions = loaded_loader.get_captions(1)

        assert captions == ["a red square", "solid red image"]

    def test_get_captions_single_caption(
        self, loaded_loader: CocoDatasetLoader
    ) -> None:
        captions = loaded_loader.get_captions(2)

        assert captions == ["a green square"]

    def test_get_captions_unknown_id_raises(
        self, loaded_loader: CocoDatasetLoader
    ) -> None:
        with pytest.raises(KeyError, match="Unknown image_id"):
            loaded_loader.get_captions(999)

    def test_get_captions_before_load_raises(self, loader: CocoDatasetLoader) -> None:
        with pytest.raises(RuntimeError, match="load_annotations"):
            loader.get_captions(1)


class TestShowImageWithCaptions:
    def test_show_image_with_captions_calls_pyplot(
        self, loaded_loader: CocoDatasetLoader, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_pyplot = ModuleType("matplotlib.pyplot")
        fake_pyplot.imshow = MagicMock()
        fake_pyplot.axis = MagicMock()
        fake_pyplot.title = MagicMock()
        fake_pyplot.show = MagicMock()
        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.pyplot = fake_pyplot

        with patch.dict(
            sys.modules,
            {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot},
        ):
            loaded_loader.show_image_with_captions(1)

        fake_pyplot.imshow.assert_called_once()
        fake_pyplot.axis.assert_called_once_with("off")
        fake_pyplot.title.assert_called_once_with("image_id=1")
        fake_pyplot.show.assert_called_once()

        captured = capsys.readouterr().out
        assert "a red square" in captured
        assert "solid red image" in captured


class TestGetRandomSample:
    def test_get_random_sample_structure(
        self, loaded_loader: CocoDatasetLoader
    ) -> None:
        sample = loaded_loader.get_random_sample(seed=0)

        assert set(sample.keys()) == {"image_id", "image", "captions"}
        assert sample["image_id"] in {1, 2}
        assert isinstance(sample["image"], Image.Image)
        assert isinstance(sample["captions"], list)
        assert all(isinstance(c, str) for c in sample["captions"])
        assert sample["captions"] == loaded_loader.get_captions(sample["image_id"])

    def test_get_random_sample_is_reproducible(
        self, loaded_loader: CocoDatasetLoader
    ) -> None:
        first = loaded_loader.get_random_sample(seed=7)
        second = loaded_loader.get_random_sample(seed=7)

        assert first["image_id"] == second["image_id"]
        assert first["captions"] == second["captions"]

    def test_get_random_sample_before_load_raises(
        self, loader: CocoDatasetLoader
    ) -> None:
        with pytest.raises(RuntimeError, match="load_annotations"):
            loader.get_random_sample()

    def test_get_random_sample_empty_dataset_raises(
        self, loaded_loader: CocoDatasetLoader
    ) -> None:
        assert loaded_loader.coco is not None
        with patch.object(loaded_loader.coco, "getImgIds", return_value=[]):
            with pytest.raises(RuntimeError, match="no images"):
                loaded_loader.get_random_sample()


class TestRepr:
    def test_repr_includes_paths_and_status(self, loader: CocoDatasetLoader) -> None:
        text = repr(loader)

        assert "CocoDatasetLoader" in text
        assert "not loaded" in text
        assert str(ANNOTATION_FILE) in text or "captions_tiny.json" in text

        loader.load_annotations()
        assert "loaded" in repr(loader)


def test_real_val2017_paths_optional() -> None:
    """Smoke-test against the real dataset when it is present locally."""
    project_root = Path(__file__).resolve().parents[2]
    annotation_file = (
        project_root / "dataset" / "annotations" / "captions_val2017.json"
    )
    image_dir = project_root / "dataset" / "val2017"

    if not annotation_file.is_file() or not image_dir.is_dir():
        pytest.skip("Full COCO val2017 dataset is not available")

    loader = CocoDatasetLoader(annotation_file, image_dir)
    coco: Any = loader.load_annotations()

    assert len(loader) == 5000
    image_id = coco.getImgIds()[0]
    captions = loader.get_captions(image_id)
    image = loader.get_image(image_id)

    assert len(captions) >= 1
    assert isinstance(image, Image.Image)
