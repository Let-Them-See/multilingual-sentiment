"""Prediction routes — single and batch inference."""

from __future__ import annotations

import hashlib
import logging
import random

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.models.inference import predict_batch, predict_single
from backend.models.schemas import (
    BatchPredictResponse,
    PredictBatchRequest,
    PredictRequest,
    PredictResponse,
    SentimentLabel,
)

def _mock_sentiment(text: str) -> dict:
    """Deterministic mock prediction for demo mode (no model loaded)."""
    rng = random.Random(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32))
    labels = ["positive", "neutral", "negative"]
    raw = [rng.random() for _ in range(3)]
    total = sum(raw)
    probs = [r / total for r in raw]
    idx = probs.index(max(probs))
    return {
        "label": labels[idx],
        "label_id": idx,
        "confidence": round(max(probs), 4),
        "probabilities": {
            "positive": round(probs[0], 4),
            "neutral":  round(probs[1], 4),
            "negative": round(probs[2], 4),
        },
        "language_detected": None,
        "inference_ms": round(rng.uniform(12, 40), 1),
    }

logger = logging.getLogger("routes.predict")
router = APIRouter(prefix="/predict", tags=["Inference"])


def _get_redis(request: Request):
    return getattr(request.app.state, "redis", None)


def _get_model(request: Request):
    return getattr(request.app.state, "model", None)


def _get_tokenizer(request: Request):
    return getattr(request.app.state, "tokenizer", None)


def _get_device(request: Request) -> str:
    return getattr(request.app.state, "device", "cpu")


@router.post("/", response_model=PredictResponse, summary="Single-text sentiment prediction")
async def predict(
    body: PredictRequest,
    request: Request,
):
    """Predict sentiment for a single text.

    - Checks Redis cache (5-min TTL) before running inference
    - Returns label, confidence, per-class probabilities, and timing
    """
    model = _get_model(request)
    tokenizer = _get_tokenizer(request)
    redis = _get_redis(request)
    device = _get_device(request)

    if model is None or tokenizer is None:
        prediction = _mock_sentiment(body.text)
    else:
        prediction = predict_single(
            model=model,
            tokenizer=tokenizer,
            text=body.text,
            language=body.language,
            redis_client=redis,
            device=device,
        )

    return PredictResponse(
        text=body.text,
        prediction=SentimentLabel(**prediction),
    )


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch sentiment prediction (max 50 texts)",
)
async def predict_batch_endpoint(
    body: PredictBatchRequest,
    request: Request,
):
    """Predict sentiment for up to 50 texts in a single call.

    Returns individual predictions plus total batch inference time.
    """
    if len(body.texts) > 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch size cannot exceed 50 texts.",
        )

    model = _get_model(request)
    tokenizer = _get_tokenizer(request)
    redis = _get_redis(request)
    device = _get_device(request)

    if model is None or tokenizer is None:
        predictions = [_mock_sentiment(t) for t in body.texts]
        batch_ms = sum(p["latency_ms"] for p in predictions)
    else:
        predictions, batch_ms = predict_batch(
            model=model,
            tokenizer=tokenizer,
            texts=body.texts,
            language=body.language,
            redis_client=redis,
            device=device,
        )

    results = [
        PredictResponse(text=text, prediction=SentimentLabel(**pred))
        for text, pred in zip(body.texts, predictions)
    ]

    return BatchPredictResponse(
        results=results,
        total=len(results),
        batch_inference_ms=batch_ms,
    )
