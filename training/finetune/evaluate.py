"""
Evaluation utilities for multilingual sentiment model.

Computes overall metrics, per-language F1, per-brand F1,
normalized confusion matrix, Cohen's Kappa, and AUC-ROC.
All results are logged to W&B.

Usage:
    from training.finetune.evaluate import compute_all_metrics
    metrics = compute_all_metrics(preds, labels, languages, brands)
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger("evaluate")

SENTIMENT_CLASSES = ["positive", "negative", "neutral"]
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def compute_per_group_f1(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    group_name: str = "language",
) -> dict[str, float]:
    """Compute macro-averaged F1 separately for each unique group value.

    Args:
        preds: Predicted label indices (N,).
        labels: Ground-truth label indices (N,).
        groups: Per-sample group string (e.g., language or brand name).
        group_name: Human-readable name used as key prefix.

    Returns:
        Dict mapping "{group_name}_{value}" → macro F1 float.
    """
    metrics: dict[str, float] = {}
    unique_groups = sorted(set(groups))

    for group_val in unique_groups:
        mask = np.array([g == group_val for g in groups])
        if mask.sum() < 2:
            continue
        f1 = f1_score(
            labels[mask],
            preds[mask],
            average="macro",
            zero_division=0,
        )
        key = f"f1_{group_name}_{group_val}".replace(" ", "_").replace("'", "")
        metrics[key] = float(f1)

    return metrics


def compute_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute confusion matrix, optionally row-normalized.

    Args:
        preds: Predicted label indices.
        labels: True label indices.
        normalize: If True, normalize each row to sum to 1.

    Returns:
        Confusion matrix as 2D numpy array.
    """
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
    return cm


def compute_auc_roc(
    probs: Optional[np.ndarray],
    labels: np.ndarray,
    n_classes: int = 3,
) -> float:
    """Compute one-vs-rest macro AUC-ROC.

    Args:
        probs: Softmax probabilities (N, n_classes) or None.
        labels: True label indices (N,).
        n_classes: Number of classes.

    Returns:
        Macro-averaged AUC-ROC score, or 0.0 if unavailable.
    """
    if probs is None:
        return 0.0
    try:
        return float(
            roc_auc_score(
                labels,
                probs,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError as exc:
        logger.warning("AUC-ROC computation failed: %s", exc)
        return 0.0


def compute_all_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    languages: Optional[list[str]] = None,
    brands: Optional[list[str]] = None,
    probs: Optional[np.ndarray] = None,
    log_to_wandb: bool = True,
    step: Optional[int] = None,
) -> dict[str, float]:
    """Compute the full evaluation metric suite.

    Args:
        preds: Predicted class indices (N,).
        labels: Ground-truth class indices (N,).
        languages: Per-sample language strings for per-language F1.
        brands: Per-sample brand strings for per-brand F1.
        probs: Softmax probabilities (N, 3) for AUC-ROC.
        log_to_wandb: Whether to push metrics to W&B.
        step: Global step for W&B logging.

    Returns:
        Dict of all computed metric names and float values.
    """
    metrics: dict[str, float] = {}

    # ── Overall metrics ────────────────────────────────────────────────────
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["f1_macro"] = float(
        f1_score(labels, preds, average="macro", zero_division=0)
    )

    per_class_f1 = f1_score(
        labels, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    for cls_id, cls_name in enumerate(SENTIMENT_CLASSES):
        metrics[f"f1_{cls_name}"] = float(per_class_f1[cls_id])

    metrics["cohen_kappa"] = float(cohen_kappa_score(labels, preds))
    metrics["auc_roc"] = compute_auc_roc(probs, labels)

    # ── Per-language metrics ────────────────────────────────────────────────
    if languages is not None:
        lang_f1 = compute_per_group_f1(preds, labels, languages, "language")
        metrics.update(lang_f1)

    # ── Per-brand metrics ───────────────────────────────────────────────────
    if brands is not None:
        flat_brands = [b[0] if isinstance(b, list) and b else "unknown" for b in brands]
        brand_f1 = compute_per_group_f1(preds, labels, flat_brands, "brand")
        metrics.update(brand_f1)

    # ── Log to W&B ─────────────────────────────────────────────────────────
    if log_to_wandb and WANDB_AVAILABLE and wandb.run:
        log_dict = {f"eval/{k}": v for k, v in metrics.items()}
        wandb.log(log_dict, step=step)

        # Confusion matrix as W&B heatmap
        cm = compute_confusion_matrix(preds, labels)
        wandb.log(
            {
                "eval/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=labels.tolist(),
                    preds=preds.tolist(),
                    class_names=SENTIMENT_CLASSES,
                )
            },
            step=step,
        )

    logger.info(
        "Accuracy=%.4f  F1-macro=%.4f  Kappa=%.4f  AUC=%.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["cohen_kappa"],
        metrics["auc_roc"],
    )

    return metrics


def make_compute_metrics(
    id2label: dict[int, str],
    eval_dataset,
) -> callable:
    """Factory: return a compute_metrics function for HF Trainer.

    Injects language/brand metadata from eval_dataset into the
    metrics computation.

    Args:
        id2label: Map from label index to string.
        eval_dataset: HuggingFace Dataset with 'language' and 'brands' columns.

    Returns:
        Callable compatible with Trainer's compute_metrics API.
    """
    languages = eval_dataset["language"] if "language" in eval_dataset.column_names else None
    brands = eval_dataset["brands"] if "brands" in eval_dataset.column_names else None

    def compute_metrics(eval_pred) -> dict[str, float]:
        """Compute metrics from HF Trainer EvalPrediction.

        Args:
            eval_pred: EvalPrediction namedtuple (predictions, label_ids).

        Returns:
            Dict of metric names to float values.
        """
        logits, labels = eval_pred
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        preds = np.argmax(logits, axis=-1)

        return compute_all_metrics(
            preds=preds,
            labels=labels,
            languages=list(languages) if languages is not None else None,
            brands=list(brands) if brands is not None else None,
            probs=probs,
            log_to_wandb=True,
        )

    return compute_metrics
