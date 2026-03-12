"""Bias audit route — POST /bias/check."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from backend.models.schemas import BiasCheckRequest, BiasCheckResponse

logger = logging.getLogger("routes.bias")
router = APIRouter(prefix="/bias", tags=["Bias Audit"])


@router.post(
    "/check",
    response_model=BiasCheckResponse,
    summary="Run multi-dimensional bias audit on a labeled batch",
)
async def check_bias(
    body: BiasCheckRequest,
    request: Request,
):
    """Audit model predictions on *body.texts* for 5 bias dimensions.

    Requires at least 5 labeled samples. For statistically meaningful
    results, send ≥ 200 samples covering all languages and brands.

    Returns:
        BiasCheckResponse with per-dimension scores and flag messages.
    """
    model = getattr(request.app.state, "model", None)
    tokenizer = getattr(request.app.state, "tokenizer", None)

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    if len(body.texts) != len(body.labels):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"texts ({len(body.texts)}) and labels ({len(body.labels)}) must have the same length.",
        )

    try:
        from backend.bias.checker import BiasChecker

        device = getattr(request.app.state, "device", "cpu")
        checker = BiasChecker(model=model, tokenizer=tokenizer, device=device)

        report = checker.run_full_audit(
            texts=body.texts,
            labels=body.labels,
            brands=body.brands,
        )

        return BiasCheckResponse(
            overall_bias_score=report.overall_bias_score,
            gender_bias_score=report.gender_bias_score,
            regional_gap=report.regional_gap,
            script_gap=report.script_gap,
            brand_gap=report.brand_gap,
            class_precision=report.class_precision,
            class_recall=report.class_recall,
            bias_flags=report.bias_flags,
            sample_count=report.sample_count,
        )

    except Exception as exc:
        logger.exception("Bias check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bias audit failed: {exc}",
        ) from exc
