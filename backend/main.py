"""
FastAPI application entry point.

Startup sequence:
  1. Load tokenizer + model (LoRA + optional 4-bit quant)
  2. Connect to Redis (graceful fallback if unavailable)
  3. Register all routers
  4. Expose OpenAPI docs at /docs

Run locally:
    uvicorn backend.main:app --reload --port 8000

Environment variables (see .env.example):
    BASE_MODEL_ID   — HF model hub ID or local path
    HF_MODEL_REPO   — Optional LoRA adapter repo
    REDIS_URL       — Redis connection string
"""

from __future__ import annotations

import logging
import os

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models.loader import load_model_and_tokenizer
from backend.routes.bias import router as bias_router
from backend.routes.health import router as health_router
from backend.routes.predict import router as predict_router
from backend.routes.trends import router as trends_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multilingual Indian Sentiment API",
    description=(
        "Production inference API for LoRA-fine-tuned sentiment analysis "
        "over Hindi, Tamil, Bengali, Telugu, Marathi, and code-mix text.\n\n"
        "**GitHub**: https://github.com/vedant/multilingual-sentiment  \n"
        "**Paper**: arXiv:2024.XXXXX"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
_origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(predict_router, prefix="/api/v1")
app.include_router(trends_router, prefix="/api/v1")
app.include_router(bias_router, prefix="/api/v1")


# ─── Lifecycle ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    """Load model, tokenizer, and connect to Redis on startup."""
    logger.info("═══ Starting up Multilingual Sentiment API ═══")

    # 1. Load model
    try:
        model, tokenizer = load_model_and_tokenizer()
        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Model loaded on device: %s", app.state.device)
    except Exception as exc:
        logger.error("Model load FAILED: %s — running without model.", exc)
        app.state.model = None
        app.state.tokenizer = None
        app.state.device = "cpu"

    # 2. Connect to Redis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis as redis_lib

        client = redis_lib.from_url(redis_url, decode_responses=True)
        client.ping()
        app.state.redis = client
        logger.info("Redis connected: %s", redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — caching disabled.", exc)
        app.state.redis = None

    logger.info("═══ API startup complete ═══")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Gracefully close Redis connection."""
    redis = getattr(app.state, "redis", None)
    if redis is not None:
        try:
            redis.close()
        except Exception:
            pass
    logger.info("API shutdown complete.")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — redirects to docs."""
    return {
        "message": "Multilingual Indian Sentiment API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }
