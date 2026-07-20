"""Integration tests for upload / features / predict / explain endpoints."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.inference import AmbiguityInferenceService
from image_ambiguity.models.trainer import FEATURE_COLUMNS, LABEL_ORDER


@pytest.fixture
def tiny_jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 48), color=(120, 80, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AmbiguityInferenceService:
    settings = MagicMock()
    settings.results_dir = tmp_path / "results"
    settings.models_dir = tmp_path / "models"
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.sentence_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    settings.device = "cpu"
    settings.embedding_batch_size = 8
    settings.ensure_directories = MagicMock()

    svc = AmbiguityInferenceService(settings=settings)

    # Lightweight OpenCV path still runs for real on the tiny JPEG.
    svc.generate_captions = MagicMock(  # type: ignore[method-assign]
        return_value=["a red bike on the street", "a bicycle parked outdoors"]
    )
    svc.compute_diversity_features = MagicMock(  # type: ignore[method-assign]
        return_value={
            "average_similarity": 0.80,
            "minimum_similarity": 0.70,
            "maximum_similarity": 0.90,
            "std_similarity": 0.05,
            "caption_diversity": 0.20,
            "n_captions": 2,
            "n_pairs": 1,
            "captions": ["a red bike on the street", "a bicycle parked outdoors"],
        }
    )

    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
    svc._model = fake_model

    fake_explanation = MagicMock()
    fake_explanation.to_dict.return_value = {
        "row_index": 0,
        "predicted_class": "Medium",
        "predicted_probability": 0.7,
        "base_value": 0.33,
        "contributions": [
            {
                "feature": "caption_diversity",
                "value": 0.20,
                "shap_value": 0.12,
            }
        ],
    }
    fake_explanation.summary_text.return_value = (
        "Predicted class: Medium (probability=70.00%)"
    )
    fake_shap = MagicMock()
    fake_shap.compute_shap_values.return_value = object()
    fake_shap.explain_prediction.return_value = fake_explanation
    svc._shap_explainer = fake_shap

    monkeypatch.setattr(
        "backend.app.api.routes.get_inference_service",
        lambda: svc,
    )
    monkeypatch.setattr(
        "backend.app.services.inference.get_inference_service",
        lambda: svc,
    )
    return svc


@pytest.fixture
def client(service: AmbiguityInferenceService) -> TestClient:
    return TestClient(app)


def test_root_lists_endpoints(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "/predict" in body["endpoints"]
    assert "/explain" in body["endpoints"]


def test_upload_returns_upload_id(
    client: TestClient, tiny_jpeg_bytes: bytes
) -> None:
    response = client.post(
        "/upload",
        files={"file": ("sample.jpg", tiny_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["upload_id"]
    assert body["filename"] == "sample.jpg"
    assert body["width"] == 64
    assert body["height"] == 48


def test_features_with_provided_captions(
    client: TestClient, tiny_jpeg_bytes: bytes
) -> None:
    response = client.post(
        "/features",
        files={"file": ("sample.jpg", tiny_jpeg_bytes, "image/jpeg")},
        data={
            "captions": '["caption one about a scene", "caption two about objects"]'
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "opencv_features" in body
    assert body["caption_diversity"]["caption_diversity"] == pytest.approx(0.20)
    assert len(body["captions"]) == 2
    assert set(body["feature_vector"]) == set(FEATURE_COLUMNS)


def test_predict_returns_ambiguity(
    client: TestClient, tiny_jpeg_bytes: bytes
) -> None:
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", tiny_jpeg_bytes, "image/jpeg")},
        data={
            "captions": '["a kitchen with a stove", "a person cooking food"]'
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_ambiguity"] in LABEL_ORDER
    assert body["predicted_ambiguity"] == "Medium"
    assert "caption_diversity" in body
    assert "opencv_features" in body
    assert set(body["probabilities"]) == set(LABEL_ORDER)


def test_explain_includes_shap(
    client: TestClient, tiny_jpeg_bytes: bytes
) -> None:
    response = client.post(
        "/explain",
        files={"file": ("sample.jpg", tiny_jpeg_bytes, "image/jpeg")},
        data={
            "captions": '["a cat on a sofa", "a kitten resting indoors"]',
            "top_n": "3",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_ambiguity"] == "Medium"
    assert "shap_explanation" in body
    assert body["shap_explanation"]["predicted_class"] == "Medium"
    assert "summary" in body


def test_predict_with_upload_id_flow(
    client: TestClient, tiny_jpeg_bytes: bytes
) -> None:
    upload = client.post(
        "/upload",
        files={"file": ("sample.jpg", tiny_jpeg_bytes, "image/jpeg")},
    )
    upload_id = upload.json()["upload_id"]
    response = client.post(
        "/predict",
        data={
            "upload_id": upload_id,
            "captions": '["one caption", "another caption"]',
        },
    )
    assert response.status_code == 200
    assert response.json()["upload_id"] == upload_id


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/upload",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
