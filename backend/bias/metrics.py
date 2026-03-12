"""
Quantitative bias metrics for token fertility, calibration, and fairness.

Provides:
  - tokenization_fertility()     : char count / token count per language
  - calibration_curve_data()     : confidence bins for reliability diagrams
  - counterfactual_consistency() : % prediction flips between paired inputs
  - inter_annotator_agreement()  : Cohen's Kappa on multi-annotator labels
  - demographic_parity_gap()     : ΔP(ŷ=positive | group) across groups

Usage:
    from backend.bias.metrics import compute_bias_metrics
    metrics = compute_bias_metrics(model, tokenizer, dataset)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy
from sklearn.calibration import calibration_curve
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger("bias_metrics")


def tokenization_fertility(
    tokenizer,
    texts: list[str],
    languages: list[str],
) -> dict[str, float]:
    """Compute chars-per-token (fertility) by language.

    Lower fertility = language is well supported by the tokenizer vocabulary.
    Indic scripts often show higher fertility on Latin-trained tokenizers.

    Args:
        tokenizer: HuggingFace tokenizer.
        texts: List of sample strings.
        languages: Parallel list of language labels.

    Returns:
        Dict mapping language to mean fertility (chars / token).
    """
    lang_fertility: dict[str, list[float]] = {}

    for text, lang in zip(texts, languages):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = max(len(token_ids), 1)
        n_chars = len(text)
        fertility = n_chars / n_tokens

        lang_fertility.setdefault(lang, []).append(fertility)

    return {
        lang: float(np.mean(fertilities))
        for lang, fertilities in lang_fertility.items()
    }


def calibration_curve_data(
    softmax_probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Compute calibration curve for reliability diagrams.

    Compares predicted confidence (max softmax) vs. actual accuracy.
    Well-calibrated models follow the diagonal y = x.

    Args:
        softmax_probs: Array of shape (N, C) with per-class probabilities.
        labels: Integer array of true labels, shape (N,).
        n_bins: Number of confidence bins.

    Returns:
        Dict with keys "mean_predicted_value" and "fraction_of_positives".
    """
    max_probs = softmax_probs.max(axis=1)
    pred_labels = softmax_probs.argmax(axis=1)
    correct = (pred_labels == labels).astype(float)

    fraction_pos, mean_pred = calibration_curve(
        correct,
        max_probs,
        n_bins=n_bins,
        strategy="quantile",
    )
    ece = float(
        np.sum(np.abs(fraction_pos - mean_pred)) / n_bins
    )
    logger.info("Expected Calibration Error (ECE) = %.4f", ece)

    return {
        "mean_predicted_value": mean_pred.tolist(),
        "fraction_of_positives": fraction_pos.tolist(),
        "ece": ece,
    }


def counterfactual_consistency(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    device: str = "cpu",
) -> float:
    """Measure what % of counterfactual pairs produce label flips.

    Args:
        model: Sentiment model.
        tokenizer: Matching tokenizer.
        pairs: List of (original_text, perturbed_text) pairs.
        device: Torch device.

    Returns:
        Flip rate float in [0, 1].
    """
    model.eval()
    model.to(device)

    originals = [p[0] for p in pairs]
    perturbed = [p[1] for p in pairs]

    def batch_predict(texts: list[str]) -> np.ndarray:
        all_preds = []
        for i in range(0, len(texts), 32):
            enc = tokenizer(
                texts[i : i + 32],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        return np.array(all_preds)

    orig_preds = batch_predict(originals)
    pert_preds = batch_predict(perturbed)
    flip_rate = float(np.mean(orig_preds != pert_preds))

    logger.info(
        "Counterfactual consistency: %.1f%% of %d pairs flipped",
        flip_rate * 100,
        len(pairs),
    )
    return flip_rate


def inter_annotator_agreement(
    annotations_a: list[int],
    annotations_b: list[int],
) -> float:
    """Cohen's Kappa between two annotators.

    Args:
        annotations_a: Label list from annotator A.
        annotations_b: Label list from annotator B.

    Returns:
        Cohen's Kappa coefficient (−1 to 1, >0.6 = substantial).
    """
    kappa = cohen_kappa_score(annotations_a, annotations_b)
    logger.info("Inter-annotator Cohen's Kappa = %.4f", kappa)
    return float(kappa)


def demographic_parity_gap(
    predictions: np.ndarray,
    groups: list[str],
    positive_label: int = 0,
) -> dict[str, float]:
    """Compute P(ŷ = positive | group) for each group.

    Demographic parity requires equal positive rates across groups.
    Gap = max(rates) − min(rates).

    Args:
        predictions: Predicted label index array.
        groups: Per-sample group identifier.
        positive_label: Label index treated as "positive" (default 0).

    Returns:
        Dict with group → positive_rate, plus "gap" key.
    """
    unique_groups = sorted(set(groups))
    rates: dict[str, float] = {}

    for group in unique_groups:
        group_preds = np.array(
            [predictions[i] for i, g in enumerate(groups) if g == group]
        )
        if len(group_preds) == 0:
            continue
        rates[group] = float(np.mean(group_preds == positive_label))

    if len(rates) >= 2:
        vals = list(rates.values())
        rates["gap"] = float(max(vals) - min(vals))
        logger.info(
            "Demographic parity gap = %.3f across groups: %s",
            rates["gap"],
            list(rates.keys())[:-1],
        )

    return rates


def compute_bias_metrics(
    model,
    tokenizer,
    texts: list[str],
    labels: list[int],
    languages: list[str],
    groups: Optional[list[str]] = None,
    device: str = "cpu",
) -> dict:
    """Aggregate all bias metrics in a single pass.

    Args:
        model: Sentiment classification model.
        tokenizer: Matching tokenizer.
        texts: Evaluation texts.
        labels: True sentiment labels.
        languages: Per-sample language identifier.
        groups: Optional grouping variable for demographic parity.
        device: Torch device.

    Returns:
        Dict with all bias metric values.
    """
    model.eval()
    model.to(device)

    # Forward pass for calibration
    all_probs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    softmax_probs = np.vstack(all_probs)
    labels_arr = np.array(labels)

    fertility = tokenization_fertility(tokenizer, texts, languages)
    calibration = calibration_curve_data(softmax_probs, labels_arr)

    dp_gap = {}
    if groups:
        dp_gap = demographic_parity_gap(
            softmax_probs.argmax(axis=1), groups
        )

    return {
        "tokenization_fertility": fertility,
        "calibration": calibration,
        "demographic_parity": dp_gap,
        "ece": calibration.get("ece", 0.0),
    }
