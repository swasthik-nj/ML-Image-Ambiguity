"""OpenCV-based low-level image feature extraction."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir

logger = get_logger("features.cv_features")


class OpenCVFeatureExtractor:
    """Extract classical computer-vision features with OpenCV.

    Features
    --------
    - Edge Density: fraction of Canny edge pixels
    - Entropy: Shannon entropy of the grayscale intensity histogram
    - Brightness: mean grayscale intensity in ``[0, 255]``
    - Contrast: standard deviation of grayscale intensity
    - Color Variance: mean of per-channel variances (0 for grayscale)
    - Texture: variance of the Laplacian (focus / texture energy)
    - Image Resolution: width, height, and total pixels

    Handles both single-channel (grayscale) and 3-channel (BGR/RGB)
    images. Paths are loaded with ``cv2.imread`` (BGR).

    Args:
        canny_threshold1: Lower hysteresis threshold for Canny.
        canny_threshold2: Upper hysteresis threshold for Canny.
        blur_ksize: Odd kernel size for optional Gaussian pre-blur
            before edge detection (``0`` / ``1`` disables blur).
    """

    FEATURE_KEYS = (
        "edge_density",
        "entropy",
        "brightness",
        "contrast",
        "color_variance",
        "texture",
        "width",
        "height",
        "resolution",
        "channels",
        "is_grayscale",
    )

    def __init__(
        self,
        canny_threshold1: float = 100.0,
        canny_threshold2: float = 200.0,
        blur_ksize: int = 3,
    ) -> None:
        if canny_threshold1 < 0 or canny_threshold2 < 0:
            raise ValueError("Canny thresholds must be non-negative")
        if canny_threshold2 < canny_threshold1:
            raise ValueError("canny_threshold2 must be >= canny_threshold1")
        if blur_ksize < 0:
            raise ValueError("blur_ksize must be >= 0")
        if blur_ksize > 1 and blur_ksize % 2 == 0:
            raise ValueError("blur_ksize must be odd when > 1")

        self.canny_threshold1 = float(canny_threshold1)
        self.canny_threshold2 = float(canny_threshold2)
        self.blur_ksize = int(blur_ksize)

    def load_image(self, path: str | Path) -> np.ndarray:
        """Load an image from disk with OpenCV (BGR or grayscale).

        Args:
            path: Path to an image file.

        Returns:
            Image array (``HxW`` or ``HxWxC``), ``uint8``.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If OpenCV cannot decode the file.
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Image file not found: {source}")

        # IMREAD_UNCHANGED preserves grayscale vs color as stored on disk.
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to decode image: {source}")
        logger.info("Loaded image %s with shape=%s", source, image.shape)
        return image

    def extract(self, image: np.ndarray | str | Path) -> dict[str, Any]:
        """Extract the full feature dictionary from an image.

        Args:
            image: NumPy image array, or a filesystem path.

        Returns:
            Dictionary of feature names to numeric values.
        """
        array = self.load_image(image) if isinstance(image, (str, Path)) else image
        array = self._validate_image(array)
        gray, is_grayscale = self._to_grayscale(array)

        edge_density = self._edge_density(gray)
        entropy = self._entropy(gray)
        brightness = self._brightness(gray)
        contrast = self._contrast(gray)
        color_variance = self._color_variance(array, is_grayscale=is_grayscale)
        texture = self._texture(gray)
        height, width = gray.shape[:2]

        features: dict[str, Any] = {
            "edge_density": float(edge_density),
            "entropy": float(entropy),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "color_variance": float(color_variance),
            "texture": float(texture),
            "width": int(width),
            "height": int(height),
            "resolution": int(width * height),
            "channels": 1 if is_grayscale else int(array.shape[2]),
            "is_grayscale": bool(is_grayscale),
        }
        logger.info(
            "Extracted CV features: edge=%.4f entropy=%.4f brightness=%.2f",
            features["edge_density"],
            features["entropy"],
            features["brightness"],
        )
        return features

    def to_csv_row(
        self,
        features: dict[str, Any],
        *,
        image_id: int | str | None = None,
        file_name: str | None = None,
    ) -> dict[str, Any]:
        """Convert a feature dictionary into a flat CSV-ready row.

        Args:
            features: Output of :meth:`extract`.
            image_id: Optional identifier for the image.
            file_name: Optional source file name.

        Returns:
            Ordered row dictionary suitable for ``csv.DictWriter``.
        """
        row: dict[str, Any] = {}
        if image_id is not None:
            row["image_id"] = image_id
        if file_name is not None:
            row["file_name"] = file_name
        for key in self.FEATURE_KEYS:
            if key in features:
                row[key] = features[key]
        return row

    def save_csv(
        self,
        features: dict[str, Any],
        path: str | Path,
        *,
        image_id: int | str | None = None,
        file_name: str | None = None,
    ) -> Path:
        """Write one feature row to a CSV file.

        Args:
            features: Output of :meth:`extract`.
            path: Destination ``.csv`` path.
            image_id: Optional image identifier.
            file_name: Optional source file name.

        Returns:
            Resolved path written to disk.
        """
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".csv")
        ensure_dir(destination.parent)

        row = self.to_csv_row(
            features, image_id=image_id, file_name=file_name
        )
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        logger.info("Saved CV feature CSV to %s", destination)
        return destination.resolve()

    def visualize_debug(
        self,
        image: np.ndarray | str | Path,
        path: str | Path | None = None,
        *,
        title: str | None = None,
        show: bool = False,
    ) -> Path | None:
        """Create a visual debugging panel (original / gray / edges / Laplacian).

        Args:
            image: Source image array or path.
            path: Optional destination ``.png`` path.
            title: Optional figure title.
            show: If ``True``, display an interactive window.

        Returns:
            Resolved save path when ``path`` is provided, else ``None``.
        """
        import matplotlib

        if not show and matplotlib.get_backend().lower() != "agg":
            try:
                matplotlib.use("Agg", force=False)
            except Exception:  # noqa: BLE001 - backend choice is best-effort
                pass
        import matplotlib.pyplot as plt

        array = self.load_image(image) if isinstance(image, (str, Path)) else image
        array = self._validate_image(array)
        gray, is_grayscale = self._to_grayscale(array)
        edges = self._canny(gray)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)
        if laplacian_abs.max() > 0:
            laplacian_vis = (
                255.0 * (laplacian_abs / laplacian_abs.max())
            ).astype(np.uint8)
        else:
            laplacian_vis = np.zeros_like(gray, dtype=np.uint8)

        # Display color in RGB for matplotlib; grayscale stays single-channel.
        if is_grayscale:
            display = gray
            cmap_original = "gray"
        else:
            display = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            cmap_original = None

        features = self.extract(array)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()

        axes[0].imshow(display, cmap=cmap_original)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(gray, cmap="gray")
        axes[1].set_title("Grayscale")
        axes[1].axis("off")

        axes[2].imshow(edges, cmap="gray")
        axes[2].set_title(f"Canny edges (density={features['edge_density']:.2f})")
        axes[2].axis("off")

        axes[3].imshow(laplacian_vis, cmap="magma")
        axes[3].set_title(f"Laplacian texture (var={features['texture']:.1f})")
        axes[3].axis("off")

        fig.suptitle(
            title
            or (
                f"CV features  |  brightness={features['brightness']:.0f}  "
                f"entropy={features['entropy']:.2f}  "
                f"contrast={features['contrast']:.1f}"
            ),
            fontsize=12,
        )
        fig.tight_layout()

        saved: Path | None = None
        if path is not None:
            destination = Path(path)
            if destination.suffix == "":
                destination = destination.with_suffix(".png")
            ensure_dir(destination.parent)
            fig.savefig(destination, dpi=150, bbox_inches="tight")
            saved = destination.resolve()
            logger.info("Saved CV debug visualization to %s", saved)

        if show:
            plt.show()
        else:
            plt.close(fig)
        return saved

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _edge_density(self, gray: np.ndarray) -> float:
        edges = self._canny(gray)
        return float(np.count_nonzero(edges) / edges.size)

    def _canny(self, gray: np.ndarray) -> np.ndarray:
        working = gray
        if self.blur_ksize > 1:
            working = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        return cv2.Canny(
            working,
            self.canny_threshold1,
            self.canny_threshold2,
        )

    def _entropy(self, gray: np.ndarray) -> float:
        # Shannon entropy of the 8-bit intensity histogram (bits).
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = float(hist.sum())
        if total <= 0:
            return 0.0
        probs = hist / total
        entropy = 0.0
        for probability in probs:
            if probability > 0:
                entropy -= float(probability) * math.log2(float(probability))
        return entropy

    @staticmethod
    def _brightness(gray: np.ndarray) -> float:
        return float(np.mean(gray))

    @staticmethod
    def _contrast(gray: np.ndarray) -> float:
        # Population std (ddof=0) — common OpenCV / image-processing convention.
        return float(np.std(gray))

    @staticmethod
    def _color_variance(image: np.ndarray, *, is_grayscale: bool) -> float:
        if is_grayscale:
            return 0.0
        # Mean of per-channel variances in BGR/RGB space.
        channel_vars = [float(np.var(image[:, :, c])) for c in range(image.shape[2])]
        return float(np.mean(channel_vars))

    @staticmethod
    def _texture(gray: np.ndarray) -> float:
        # Laplacian variance: higher => more high-frequency texture / sharpness.
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> tuple[np.ndarray, bool]:
        if image.ndim == 2:
            return image, True
        if image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0], True
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), False
        if image.ndim == 3 and image.shape[2] == 4:
            # Drop alpha; treat as BGR.
            bgr = image[:, :, :3]
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), False
        raise ValueError(f"Unsupported image shape: {image.shape}")

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.size == 0:
            raise ValueError("Image array is empty")
        if array.ndim not in (2, 3):
            raise ValueError(
                f"Image must be 2-D or 3-D, got shape {array.shape}"
            )
        if array.dtype != np.uint8:
            # Best-effort convert float images in [0, 1] or [0, 255].
            if np.issubdtype(array.dtype, np.floating):
                max_val = float(np.nanmax(array)) if array.size else 0.0
                scaled = array * 255.0 if max_val <= 1.0 + 1e-6 else array
                array = np.clip(scaled, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"canny=({self.canny_threshold1}, {self.canny_threshold2}), "
            f"blur_ksize={self.blur_ksize})"
        )
