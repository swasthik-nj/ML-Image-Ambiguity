"""REST routes for upload, features, prediction, and SHAP explanation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.schemas import (
    CompareResponse,
    ErrorResponse,
    ExplainResponse,
    FeaturesResponse,
    PredictionResponse,
    UploadResponse,
)
from backend.app.services.compare import compare_datasets
from backend.app.services.inference import (
    AmbiguityInferenceService,
    get_inference_service,
)

router = APIRouter(tags=["ambiguity"])


def get_service() -> AmbiguityInferenceService:
    return get_inference_service()


async def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    filename = file.filename or "upload.jpg"
    return filename, data


def _unprocessable(detail: str) -> HTTPException:
    """Return HTTP 422 using the current Starlette status constant."""
    code = getattr(
        status,
        "HTTP_422_UNPROCESSABLE_CONTENT",
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    )
    return HTTPException(status_code=code, detail=detail)


def _parse_captions(captions_json: str | None) -> list[str] | None:
    """Parse optional captions from Swagger/form input.

    Accepts:
    - empty / omitted → ``None`` (BLIP will generate captions)
    - JSON array: ``["caption one", "caption two"]``
    - plain lines / ``;``-separated text (at least 2 captions)
    """
    if captions_json is None:
        return None
    text = captions_json.strip()
    if not text or text in {"string", "null", "None"}:
        # Swagger often pre-fills the literal word "string"
        return None

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _unprocessable(
                'captions must be valid JSON, e.g. ["caption one", "caption two"]'
            ) from exc
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            raise _unprocessable("captions JSON must be an array of strings")
        captions = [item.strip() for item in parsed if item.strip()]
    else:
        parts = [part.strip() for part in text.replace(";", "\n").splitlines()]
        captions = [part for part in parts if part]

    if len(captions) == 1:
        raise _unprocessable(
            "Provide at least 2 captions, or leave captions empty to use BLIP"
        )
    if not captions:
        return None
    return captions


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=str(exc),
    )


def _clean_optional_text(value: str | None) -> str | None:
    """Normalize Swagger placeholder defaults like ``string`` / ``null``."""
    if value is None:
        return None
    text = value.strip()
    if not text or text in {"string", "null", "None"}:
        return None
    return text


async def _resolve_image(
    service: AmbiguityInferenceService,
    *,
    file: UploadFile | None,
    upload_id: str | None,
) -> tuple[Path, str | None]:
    """Return ``(image_path, upload_id)`` from multipart file and/or id."""
    # Swagger may send an empty file part; ignore it when no real filename.
    has_file = (
        file is not None
        and bool(file.filename)
        and file.filename not in {"string", "null"}
    )
    if has_file:
        filename, data = await _read_upload(file)
        try:
            meta = service.save_upload(filename=filename, data=data)
        except ValueError as exc:
            raise _http_error(exc) from exc
        return Path(meta["path"]), meta["upload_id"]

    cleaned_upload_id = _clean_optional_text(upload_id)
    if cleaned_upload_id:
        try:
            path = service.resolve_image_path(upload_id=cleaned_upload_id)
        except (FileNotFoundError, ValueError) as exc:
            raise _http_error(exc) from exc
        return path, cleaned_upload_id

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Provide an image file upload or an existing upload_id",
    )


@router.get(
    "/compare",
    response_model=CompareResponse,
    summary="Compare human vs AI caption diversity",
)
def compare_human_ai() -> CompareResponse:
    """Return mean caption diversity and histograms from dataset CSVs."""
    payload = compare_datasets()
    return CompareResponse(**payload)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Upload an image",
)
async def upload_image(
    file: Annotated[UploadFile, File(description="Image file (jpg/png/webp/bmp)")],
    service: Annotated[AmbiguityInferenceService, Depends(get_service)],
) -> UploadResponse:
    """Store an uploaded image and return an ``upload_id`` for later calls."""
    filename, data = await _read_upload(file)
    try:
        meta = service.save_upload(filename=filename, data=data)
    except ValueError as exc:
        raise _http_error(exc) from exc

    return UploadResponse(
        upload_id=meta["upload_id"],
        filename=meta["filename"],
        content_type=meta["content_type"],
        size_bytes=meta["size_bytes"],
        width=meta["width"],
        height=meta["height"],
        mode=meta["mode"],
        coco_image_id=meta.get("coco_image_id"),
        coco_captions=list(meta.get("coco_captions") or []),
    )


@router.post(
    "/features",
    response_model=FeaturesResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Extract OpenCV and caption-diversity features",
)
async def extract_features(
    service: Annotated[AmbiguityInferenceService, Depends(get_service)],
    file: Annotated[
        UploadFile | None,
        File(description="Image file (optional if upload_id is provided)"),
    ] = None,
    upload_id: Annotated[
        str | None,
        Form(description="Previously returned upload id"),
    ] = None,
    captions: Annotated[
        str | None,
        Form(
            description=(
                'Optional JSON array of captions, e.g. ["cap1","cap2"]. '
                "If omitted, uses COCO human captions when available, else BLIP."
            ),
        ),
    ] = None,
    force_blip: Annotated[
        bool,
        Form(description="Force BLIP even when COCO human captions exist"),
    ] = False,
) -> FeaturesResponse:
    """Return OpenCV features and caption-diversity metrics for an image."""
    image_path, resolved_upload_id = await _resolve_image(
        service, file=file, upload_id=upload_id
    )
    parsed_captions = _parse_captions(captions)
    try:
        payload = service.extract_features(
            image_path,
            captions=parsed_captions,
            upload_id=resolved_upload_id,
            force_blip=force_blip,
        )
    except Exception as exc:  # noqa: BLE001 - map domain errors to HTTP
        raise _http_error(exc) from exc

    return FeaturesResponse(
        upload_id=resolved_upload_id,
        caption_source=payload["caption_source"],
        opencv_features=payload["opencv_features"],
        caption_diversity=payload["caption_diversity"],
        captions=payload["captions"],
        feature_vector=payload["feature_vector"],
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict image ambiguity",
)
async def predict_ambiguity(
    service: Annotated[AmbiguityInferenceService, Depends(get_service)],
    file: Annotated[
        UploadFile | None,
        File(description="Image file (optional if upload_id is provided)"),
    ] = None,
    upload_id: Annotated[
        str | None,
        Form(description="Previously returned upload id"),
    ] = None,
    captions: Annotated[
        str | None,
        Form(
            description=(
                'Optional JSON array of captions, e.g. ["cap1","cap2"]. '
                "If omitted, uses COCO human captions when available, else BLIP."
            ),
        ),
    ] = None,
    force_blip: Annotated[
        bool,
        Form(description="Force BLIP even when COCO human captions exist"),
    ] = False,
) -> PredictionResponse:
    """Predict Low/Medium/High ambiguity and return supporting features."""
    image_path, resolved_upload_id = await _resolve_image(
        service, file=file, upload_id=upload_id
    )
    parsed_captions = _parse_captions(captions)
    try:
        payload = service.predict(
            image_path,
            captions=parsed_captions,
            upload_id=resolved_upload_id,
            force_blip=force_blip,
        )
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc) from exc

    return PredictionResponse(
        upload_id=resolved_upload_id,
        predicted_ambiguity=payload["predicted_ambiguity"],
        confidence=payload["confidence"],
        probabilities=payload["probabilities"],
        caption_source=payload["caption_source"],
        caption_diversity=payload["caption_diversity"],
        opencv_features=payload["opencv_features"],
        captions=payload["captions"],
        feature_vector=payload["feature_vector"],
    )


@router.post(
    "/explain",
    response_model=ExplainResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict and explain with SHAP",
)
async def explain_prediction(
    service: Annotated[AmbiguityInferenceService, Depends(get_service)],
    file: Annotated[
        UploadFile | None,
        File(description="Image file (optional if upload_id is provided)"),
    ] = None,
    upload_id: Annotated[
        str | None,
        Form(description="Previously returned upload id"),
    ] = None,
    captions: Annotated[
        str | None,
        Form(
            description=(
                'Optional JSON array of captions, e.g. ["cap1","cap2"]. '
                "If omitted, uses COCO human captions when available, else BLIP."
            ),
        ),
    ] = None,
    force_blip: Annotated[
        bool,
        Form(description="Force BLIP even when COCO human captions exist"),
    ] = False,
    top_n: Annotated[
        int,
        Form(description="Number of top SHAP contributions to highlight", ge=1, le=20),
    ] = 5,
) -> ExplainResponse:
    """Return predicted ambiguity plus a SHAP feature attribution explanation."""
    image_path, resolved_upload_id = await _resolve_image(
        service, file=file, upload_id=upload_id
    )
    parsed_captions = _parse_captions(captions)
    try:
        payload = service.explain(
            image_path,
            captions=parsed_captions,
            upload_id=resolved_upload_id,
            force_blip=force_blip,
            top_n=top_n,
        )
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc) from exc

    return ExplainResponse(
        upload_id=resolved_upload_id,
        predicted_ambiguity=payload["predicted_ambiguity"],
        confidence=payload["confidence"],
        probabilities=payload["probabilities"],
        caption_source=payload["caption_source"],
        caption_diversity=payload["caption_diversity"],
        opencv_features=payload["opencv_features"],
        captions=payload["captions"],
        feature_vector=payload["feature_vector"],
        shap_explanation=payload["shap_explanation"],
        summary=payload["summary"],
    )
