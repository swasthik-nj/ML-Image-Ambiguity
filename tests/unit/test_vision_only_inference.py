import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
import numpy as np

from backend.app.services.inference import AmbiguityInferenceService, CV_FEATURE_KEYS

@pytest.fixture
def inference_service():
    service = AmbiguityInferenceService()
    # Mock models and extractors so we don't load huge models in unit tests
    service.cv_extractor = MagicMock()
    service.blip_generator = MagicMock()
    service.embedding_generator = MagicMock()
    service._vision_model = MagicMock()
    service._vision_model.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]]) # Predicts Medium
    service._model = MagicMock()
    service._model.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
    return service

def test_inference_non_coco_bypasses_blip(inference_service):
    # Setup dummy CV extraction
    dummy_cv = {k: 0.5 for k in CV_FEATURE_KEYS}
    inference_service.cv_extractor.extract.return_value = dummy_cv
    
    # Path that doesn't look like a COCO image
    non_coco_path = Path("custom_upload.jpg")
    
    result = inference_service.extract_features(non_coco_path)
    
    assert result["feature_mode"] == "vision_only"
    assert result["caption_source"] == "none (vision_only)"
    assert result["caption_pipeline"] is None
    assert result["captions"] == []
    
    # Verify BLIP and embeddings were NEVER called
    inference_service.blip_generator.generate_for_mode.assert_not_called()
    inference_service.embedding_generator.generate_embeddings.assert_not_called()

def test_inference_coco_uses_blip_or_human_captions(inference_service):
    dummy_cv = {k: 0.5 for k in CV_FEATURE_KEYS}
    inference_service.cv_extractor.extract.return_value = dummy_cv
    
    # Path that looks like a COCO image
    coco_path = Path("000000123456.jpg")
    
    # Mock COCO lookup to return captions
    with patch.object(inference_service, 'lookup_coco_captions', return_value=["A dog", "A cute dog"]):
        # Mock diversity features
        with patch.object(inference_service, 'compute_diversity_features', return_value={
            "average_similarity": 0.5,
            "minimum_similarity": 0.4,
            "maximum_similarity": 0.6,
            "std_similarity": 0.1,
            "caption_diversity": 0.5,
            "n_captions": 2,
            "n_pairs": 1,
            "captions": ["A dog", "A cute dog"]
        }):
            result = inference_service.extract_features(coco_path)
            
            assert result["feature_mode"] == "caption+vision"
            assert result["caption_source"] == "coco_human"
            assert result["caption_diversity"]["caption_diversity"] == 0.5

def test_predict_non_coco_uses_vision_model(inference_service):
    dummy_cv = {k: 0.5 for k in CV_FEATURE_KEYS}
    inference_service.cv_extractor.extract.return_value = dummy_cv
    
    non_coco_path = Path("custom_upload.jpg")
    
    result = inference_service.predict(non_coco_path)
    
    assert result["feature_mode"] == "vision_only"
    assert result["predicted_ambiguity"] == "Medium"
    
    # Vision model should have been called, 11-D model should NOT have been called
    inference_service._vision_model.predict_proba.assert_called_once()
    inference_service._model.predict_proba.assert_not_called()
