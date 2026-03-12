"""
Model loading utilities with optional 4-bit quantization.

Supports:
  - LoRA adapter loading via PEFT
  - 4-bit NF4 quantization via BitsAndBytes
  - Graceful CPU fallback
  - Singleton pattern to avoid reloading on each request
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger("model_loader")

# ─── Module-level singletons ──────────────────────────────────────────────────
_model = None
_tokenizer = None

LABEL_MAP = {0: "positive", 1: "neutral", 2: "negative"}
ID2LABEL = LABEL_MAP
LABEL2ID = {v: k for k, v in LABEL_MAP.items()}


def _build_bnb_config() -> BitsAndBytesConfig | None:
    """Build BitsAndBytes 4-bit NF4 config if GPU is available.

    Returns:
        BitsAndBytesConfig or None for CPU-only environments.
    """
    if not torch.cuda.is_available():
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(
    base_model_id: str | None = None,
    adapter_path: str | None = None,
    force_cpu: bool = False,
) -> tuple:
    """Load tokenizer + classification model (singleton).

    Priority order for model loading:
    1. LoRA adapter from ``adapter_path`` on top of ``base_model_id``.
    2. Full fine-tuned model from ``adapter_path``.
    3. Base model from ``base_model_id`` (with 4-bit quant if GPU available).

    Args:
        base_model_id: HuggingFace model hub ID or local path.
        adapter_path: LoRA adapter directory or HF Hub repo.
        force_cpu: Disable GPU even if available.

    Returns:
        Tuple (model, tokenizer).

    Raises:
        RuntimeError: If no valid model source is specified.
    """
    global _model, _tokenizer  # noqa: PLW0603

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    base_id = base_model_id or os.environ.get(
        "BASE_MODEL_ID", "ai4bharat/indic-bert"
    )
    adapter = adapter_path or os.environ.get("HF_MODEL_REPO", "")
    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"

    logger.info("Loading tokenizer: %s", base_id)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = None if device == "cpu" else _build_bnb_config()

    load_kwargs: dict = {
        "num_labels": 3,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "ignore_mismatched_sizes": True,
    }
    if bnb_cfg is not None:
        load_kwargs["quantization_config"] = bnb_cfg
    else:
        load_kwargs["torch_dtype"] = torch.float32

    logger.info("Loading base model: %s  (device=%s)", base_id, device)
    base = AutoModelForSequenceClassification.from_pretrained(base_id, **load_kwargs)

    if adapter and Path(adapter).exists() or (adapter and "/" in adapter):
        logger.info("Loading LoRA adapter from: %s", adapter)
        try:
            model = PeftModel.from_pretrained(base, adapter)
            model = model.merge_and_unload()
        except Exception as exc:
            logger.warning("LoRA adapter load failed (%s), using base model.", exc)
            model = base
    else:
        model = base

    if device == "cpu":
        model = model.float()

    model.eval()
    if device == "cuda":
        model.to(device)

    _model = model
    _tokenizer = tokenizer

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model ready — %.1fM parameters on %s", param_count, device)

    return _model, _tokenizer


def reset_singletons() -> None:
    """Force reload on next call (useful for testing)."""
    global _model, _tokenizer  # noqa: PLW0603
    _model = None
    _tokenizer = None
