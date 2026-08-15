"""Unit tests for the AmbiguityInferenceService using Vision+CLIP features."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from PIL import Image

from image_ambiguity.config import Settings
from backend.app.services.inference import AmbiguityInferenceService, CV_FEATURE_KEYS


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    return Settings(
        results_dir=tmp_path,
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def mock_service(mock_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> AmbiguityInferenceService:
    service = AmbiguityInferenceService(settings=mock_settings)

    # Mock extractors
    service.cv_extractor = MagicMock()
    service.cv_extractor.extract.return_value = {key: 0.5 for key in CV_FEATURE_KEYS}
    
    service.clip_extractor = MagicMock()
    service.clip_extractor.extract.return_value = [0.1] * 512

    # Mock model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.1, 0.8, 0.1]]
    mock_model.classes_ = [0, 1, 2]
    service._model = mock_model

    return service


def test_extract_features(mock_service: AmbiguityInferenceService) -> None:
    """Test extracting combined Vision and CLIP features."""
    dummy_path = Path("test.jpg")
    features = mock_service.extract_features(dummy_path)

    assert features["feature_mode"] == "vision_clip"
    assert "opencv_features" in features
    assert "clip_features" in features
    
    # Check that OpenCV and CLIP were extracted
    assert len(features["opencv_features"]) == len(CV_FEATURE_KEYS)
    assert len(features["clip_features"]) == 512
    
    # Check feature vector assembly
    vector = features["feature_vector"]
    assert len(vector) == 6 + 512
    assert "clip_0" in vector
    assert "edge_density" in vector


def test_predict(mock_service: AmbiguityInferenceService) -> None:
    """Test full prediction payload."""
    dummy_path = Path("test.jpg")
    
    prediction = mock_service.predict(dummy_path)
    
    assert prediction["predicted_ambiguity"] == "Medium"
    assert prediction["confidence"] == 0.8
    assert prediction["feature_mode"] == "vision_clip"
    assert "feature_vector" in prediction
