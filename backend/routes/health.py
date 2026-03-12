"""Health check route — GET /health."""

from __future__ import annotations

import logging

import torch
from fastapi import APIRouter, Request

from backend.models.schemas import HealthResponse

logger = logging.getLogger("routes.health")
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="API health check")
async def health(request: Request):
    """Return service health status including model readiness and GPU info."""
    model_loaded = getattr(request.app.state, "model", None) is not None
    tokenizer_loaded = getattr(request.app.state, "tokenizer", None) is not None

    cache_alive = False
    redis = getattr(request.app.state, "redis", None)
    if redis is not None:
        try:
            redis.ping()
            cache_alive = True
        except Exception:
            pass

    return HealthResponse(
        status="ok" if (model_loaded and tokenizer_loaded) else "degraded",
        model_loaded=model_loaded and tokenizer_loaded,
        cache_alive=cache_alive,
        gpu_available=torch.cuda.is_available(),
    )
