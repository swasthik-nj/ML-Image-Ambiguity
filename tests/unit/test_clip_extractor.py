"""Unit tests for the CLIP feature extractor."""

from PIL import Image

from image_ambiguity.features.clip_embeddings import CLIPFeatureExtractor


def test_clip_feature_extractor() -> None:
    """Test that the CLIP extractor correctly produces a 512-dim embedding."""
    extractor = CLIPFeatureExtractor()
    
    # Create a dummy image
    image = Image.new("RGB", (224, 224), color="red")
    
    embedding = extractor.extract(image)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 512
    # Ensure all elements are floats
    assert all(isinstance(x, float) for x in embedding)
