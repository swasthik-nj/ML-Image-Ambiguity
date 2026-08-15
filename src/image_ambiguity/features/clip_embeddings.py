"""Extract semantic image embeddings using OpenAI CLIP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from image_ambiguity.logging_config import get_logger

logger = get_logger("features.clip_embeddings")


class CLIPFeatureExtractor:
    """Extract 512-dimensional semantic embeddings using CLIP.

    This uses the huggingface ``transformers`` library to load
    ``openai/clip-vit-base-patch32``. The model is loaded lazily on the
    first call to ``extract``.
    
    Args:
        model_name: The HuggingFace model hub name for the CLIP model.
        device: Device to run the model on ('cpu', 'cuda', etc.).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        logger.info("Loading CLIP model: %s on %s", self.model_name, self.device)
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def extract(self, image: str | Path | Image.Image) -> list[float]:
        """Extract the embedding for a single image.

        Args:
            image: Path to the image file or a loaded PIL Image.

        Returns:
            A list of floats representing the 512-dimensional embedding.
        """
        self._ensure_loaded()

        if isinstance(image, (str, Path)):
            img_obj = Image.open(image).convert("RGB")
        else:
            img_obj = image.convert("RGB")

        inputs = self._processor(images=img_obj, return_tensors="pt").to(self.device)
        
        # We only need the image features, not text.
        # We use torch.no_grad to save memory and avoid tracking history.
        import torch
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        # Normalize the embeddings to match typical CLIP usage (optional but standard)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Convert to a flat list of floats
        embedding = image_features.squeeze(0).cpu().tolist()
        return embedding
