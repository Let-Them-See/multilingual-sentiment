"""
Async inference engine with Redis caching and timing.

Provides:
  - predict_single()  : Single-text inference with soft-max probabilities
  - predict_batch()   : Batched inference with timing
  - get_cache_key()   : Deterministic cache key from text + language
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("inference")

LABEL_NAMES = {0: "positive", 1: "neutral", 2: "negative"}


def get_cache_key(text: str, language: Optional[str] = None) -> str:
    """Build a deterministic Redis cache key.

    Args:
        text: Input text.
        language: Optional language hint.

    Returns:
        SHA-256 hex string prefixed with ``"senti:"``
    """
    payload = f"{text}|{language or ''}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"senti:{digest}"


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        logits: 1-D logit array.

    Returns:
        Probability array summing to 1.
    """
    shifted = logits - logits.max()
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum()


def _run_inference(
    model,
    tokenizer,
    texts: list[str],
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run batched model forward pass.

    Args:
        model: HuggingFace classification model.
        tokenizer: Matching tokenizer.
        texts: Input text list.
        device: Torch device string.

    Returns:
        Tuple of (predictions array, softmax_probs array, inference_ms float).
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = np.array([_softmax(row) for row in logits])
    preds = probs.argmax(axis=1)
    return preds, probs, elapsed_ms


def predict_single(
    model,
    tokenizer,
    text: str,
    language: Optional[str] = None,
    redis_client=None,
    device: str = "cpu",
) -> dict:
    """Predict sentiment for a single text with optional Redis caching.

    Args:
        model: Loaded classification model.
        tokenizer: Matching tokenizer.
        text: Input string.
        language: Optional language hint.
        redis_client: Redis client or None (disables caching).
        device: Torch device.

    Returns:
        Dict with keys: label, label_id, confidence, probabilities,
        language_detected, inference_ms.
    """
    cache_key = get_cache_key(text, language)

    # Check cache
    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.debug("Cache HIT for key: %s", cache_key)
                return json.loads(cached)
        except Exception as exc:
            logger.warning("Redis GET failed: %s", exc)

    preds, probs, inf_ms = _run_inference(model, tokenizer, [text], device)

    label_id = int(preds[0])
    prob_row = probs[0]

    result = {
        "label": LABEL_NAMES[label_id],
        "label_id": label_id,
        "confidence": float(prob_row[label_id]),
        "probabilities": {
            LABEL_NAMES[i]: float(prob_row[i]) for i in range(3)
        },
        "language_detected": language,
        "inference_ms": round(inf_ms, 2),
    }

    # Write to cache (TTL = 5 minutes)
    if redis_client is not None:
        try:
            redis_client.setex(cache_key, 300, json.dumps(result))
        except Exception as exc:
            logger.warning("Redis SET failed: %s", exc)

    return result


def predict_batch(
    model,
    tokenizer,
    texts: list[str],
    language: Optional[str] = None,
    redis_client=None,
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[list[dict], float]:
    """Predict sentiment for a batch of texts.

    Checks cache per-sample; runs inference only on cache misses.

    Args:
        model: Loaded classification model.
        tokenizer: Matching tokenizer.
        texts: Input text list.
        language: Optional language override for all texts.
        redis_client: Redis client or None.
        device: Torch device.
        batch_size: Maximum inference batch size.

    Returns:
        Tuple of (list of prediction dicts, total batch inference ms).
    """
    results: list[dict | None] = [None] * len(texts)
    uncached_indices: list[int] = []

    # Check cache for all texts
    for i, text in enumerate(texts):
        cache_key = get_cache_key(text, language)
        if redis_client is not None:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    results[i] = json.loads(cached)
                    continue
            except Exception:
                pass
        uncached_indices.append(i)

    total_inf_ms = 0.0

    # Batch inference on uncached samples
    for start in range(0, len(uncached_indices), batch_size):
        batch_indices = uncached_indices[start : start + batch_size]
        batch_texts = [texts[i] for i in batch_indices]

        preds, probs, inf_ms = _run_inference(
            model, tokenizer, batch_texts, device
        )
        total_inf_ms += inf_ms

        for j, orig_idx in enumerate(batch_indices):
            label_id = int(preds[j])
            prob_row = probs[j]
            prediction = {
                "label": LABEL_NAMES[label_id],
                "label_id": label_id,
                "confidence": float(prob_row[label_id]),
                "probabilities": {
                    LABEL_NAMES[k]: float(prob_row[k]) for k in range(3)
                },
                "language_detected": language,
                "inference_ms": round(inf_ms / len(batch_indices), 2),
            }
            results[orig_idx] = prediction

            # Write to cache
            if redis_client is not None:
                cache_key = get_cache_key(texts[orig_idx], language)
                try:
                    redis_client.setex(cache_key, 300, json.dumps(prediction))
                except Exception:
                    pass

    # All results should be filled
    return [r for r in results if r is not None], round(total_inf_ms, 2)
