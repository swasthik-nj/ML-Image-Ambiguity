"""COCO caption dataset loader built on pycocotools."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image
from pycocotools.coco import COCO

from image_ambiguity.logging_config import get_logger

logger = get_logger("data.coco_loader")


class CocoDatasetLoader:
    """Load and inspect COCO caption annotations with matching image files.

    Uses ``pycocotools.coco.COCO`` for annotation indexing and lookup.
    Image pixels are read from ``image_dir`` via Pillow.

    Args:
        annotation_file: Path to a COCO captions JSON file
            (e.g. ``dataset/annotations/captions_val2017.json``).
        image_dir: Directory containing the corresponding image files
            (e.g. ``dataset/val2017``).
    """

    def __init__(self, annotation_file: str | Path, image_dir: str | Path) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.coco: COCO | None = None

    def load_annotations(self) -> COCO:
        """Load caption annotations and build the pycocotools index.

        Returns:
            The initialized :class:`~pycocotools.coco.COCO` instance.

        Raises:
            FileNotFoundError: If the annotation file or image directory
                does not exist.
        """
        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}"
            )
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        logger.info("Loading COCO annotations from %s", self.annotation_file)
        self.coco = COCO(str(self.annotation_file))
        logger.info("Indexed %s images", len(self))
        return self.coco

    def _require_coco(self) -> COCO:
        """Return the loaded COCO API, or raise if annotations are unloaded."""
        if self.coco is None:
            raise RuntimeError(
                "Annotations are not loaded. Call load_annotations() first."
            )
        return self.coco

    def get_image(self, image_id: int) -> Image.Image:
        """Open the image file for a COCO image id.

        Args:
            image_id: COCO image identifier.

        Returns:
            RGB :class:`~PIL.Image.Image` for the requested id.

        Raises:
            RuntimeError: If annotations have not been loaded.
            KeyError: If ``image_id`` is not present in the annotations.
            FileNotFoundError: If the image file is missing on disk.
        """
        coco = self._require_coco()
        if image_id not in coco.imgs:
            raise KeyError(f"Unknown image_id: {image_id}")

        imgs = coco.loadImgs(image_id)
        file_name = imgs[0]["file_name"]
        image_path = self.image_dir / file_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    def get_captions(self, image_id: int) -> list[str]:
        """Return all caption strings for a COCO image id.

        Args:
            image_id: COCO image identifier.

        Returns:
            List of caption texts associated with the image.

        Raises:
            RuntimeError: If annotations have not been loaded.
            KeyError: If ``image_id`` is not present in the annotations.
        """
        coco = self._require_coco()
        if image_id not in coco.imgs:
            raise KeyError(f"Unknown image_id: {image_id}")

        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        return [ann["caption"] for ann in anns]

    def show_image_with_captions(self, image_id: int) -> None:
        """Display an image and print its captions.

        Args:
            image_id: COCO image identifier.

        Raises:
            RuntimeError: If annotations have not been loaded.
            KeyError: If ``image_id`` is not present in the annotations.
            FileNotFoundError: If the image file is missing on disk.
        """
        import matplotlib.pyplot as plt

        image = self.get_image(image_id)
        captions = self.get_captions(image_id)

        plt.imshow(image)
        plt.axis("off")
        plt.title(f"image_id={image_id}")
        plt.show()

        print(f"Captions for image_id={image_id}:")
        for caption in captions:
            print(f"- {caption}")

    def get_random_sample(self, seed: int | None = None) -> dict[str, Any]:
        """Pick a random image and return its id, image, and captions.

        Args:
            seed: Optional random seed for reproducible sampling.

        Returns:
            Dictionary with keys ``image_id``, ``image``, and ``captions``.

        Raises:
            RuntimeError: If annotations have not been loaded or the
                dataset contains no images.
        """
        coco = self._require_coco()
        image_ids = coco.getImgIds()
        if not image_ids:
            raise RuntimeError("Dataset contains no images.")

        rng = random.Random(seed)
        image_id = rng.choice(image_ids)
        return {
            "image_id": image_id,
            "image": self.get_image(image_id),
            "captions": self.get_captions(image_id),
        }

    def __len__(self) -> int:
        """Return the number of images in the loaded annotation set."""
        coco = self._require_coco()
        return len(coco.getImgIds())

    def __repr__(self) -> str:
        status = "loaded" if self.coco is not None else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"annotation_file={self.annotation_file!s}, "
            f"image_dir={self.image_dir!s}, "
            f"status={status})"
        )
