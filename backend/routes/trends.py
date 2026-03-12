"""Sentiment trends route — aggregated time-series data."""

from __future__ import annotations

import logging
import random
from datetime import date, timedelta

from fastapi import APIRouter, Query

from backend.models.schemas import TrendPoint, TrendsResponse

logger = logging.getLogger("routes.trends")
router = APIRouter(prefix="/trends", tags=["Analytics"])

BRANDS = [
    "Jio", "Airtel", "Zomato", "Swiggy", "Flipkart",
    "Amazon", "Meesho", "Nykaa", "LensKart", "BigBasket",
]


def _generate_mock_trend(brand: str, days: int) -> list[TrendPoint]:
    """Generate deterministic mock trend data (replace with DB query in prod).

    Args:
        brand: Brand name.
        days: Number of historical days.

    Returns:
        List of TrendPoint objects.
    """
    rng = random.Random(hash(brand))
    points = []
    today = date.today()

    pos_base = rng.uniform(0.35, 0.55)
    neg_base = rng.uniform(0.10, 0.25)
    volume_base = rng.randint(200, 800)

    for i in range(days, 0, -1):
        d = today - timedelta(days=i)
        noise = rng.uniform(-0.05, 0.05)
        pos = min(0.90, max(0.05, pos_base + noise))
        neg = min(0.90, max(0.05, neg_base - noise * 0.5))
        neu = max(0.0, 1.0 - pos - neg)
        volume = max(10, volume_base + rng.randint(-100, 100))
        points.append(
            TrendPoint(
                date=d.isoformat(),
                brand=brand,
                positive_pct=round(pos * 100, 1),
                neutral_pct=round(neu * 100, 1),
                negative_pct=round(neg * 100, 1),
                volume=volume,
            )
        )
    return points


@router.get(
    "/",
    response_model=TrendsResponse,
    summary="Get sentiment trend for a brand over N days",
)
async def get_trends(
    brand: str = Query(..., description="Brand name (case-sensitive)"),
    days: int = Query(default=30, ge=1, le=365, description="Number of historical days"),
):
    """Return time-series sentiment distribution for *brand* over the last *days* days.

    In production, replace ``_generate_mock_trend`` with a database query
    that aggregates stored predictions by (brand, date) buckets.
    """
    if brand not in BRANDS:
        # Still return data for any brand — mock generator handles unknown strings
        logger.info("Unknown brand requested: %s — generating mock data.", brand)

    data = _generate_mock_trend(brand, days)
    total = sum(p.volume for p in data)

    return TrendsResponse(
        brand=brand,
        days=days,
        data=data,
        total_samples=total,
    )


@router.get(
    "/brands",
    response_model=list[str],
    summary="List all supported brands",
)
async def list_brands():
    """Return the canonical list of supported brand names."""
    return BRANDS
