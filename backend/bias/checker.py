"""
Bias detection and auditing for multilingual sentiment models.

Implements 5 bias dimensions:
  1. Gender bias          — male vs. female name counterfactuals
  2. Regional bias        — metro vs. tier-2/tier-3 city mentions
  3. Script bias          — Devanagari vs. Latin-script samples
  4. Brand-popularity bias — big-4 (Jio/Flipkart/Zomato/Amazon) vs. niche
  5. Sentiment-label bias  — class-imbalance effect on per-class precision

Usage:
    from backend.bias.checker import BiasChecker
    checker = BiasChecker(model, tokenizer)
    report = checker.run_full_audit(dataset)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import classification_report

logger = logging.getLogger("bias_checker")

# ─── Counterfactual Identity Pairs ────────────────────────────────────────────
GENDER_PAIRS: list[tuple[str, str]] = [
    ("Rahul", "Priya"),
    ("Amit", "Pooja"),
    ("Ravi", "Sunita"),
    ("Vikram", "Kavya"),
    ("Suresh", "Meena"),
    ("Arjun", "Sneha"),
]

# ─── Regional City Categories ─────────────────────────────────────────────────
METRO_CITIES = {
    "Mumbai", "Delhi", "Bangalore", "Bengaluru", "Chennai",
    "Kolkata", "Hyderabad", "Pune",
}
TIER2_CITIES = {
    "Nagpur", "Surat", "Jaipur", "Lucknow", "Kanpur",
    "Bhopal", "Indore", "Patna", "Coimbatore", "Kochi",
}

# ─── Brand Tier Mapping ───────────────────────────────────────────────────────
BIG4_BRANDS = {"Jio", "Flipkart", "Zomato", "Amazon"}
NICHE_BRANDS = {"LensKart", "Nykaa", "Meesho", "Pepperfry", "BigBasket"}

SCRIPT_REGEX = {
    "devanagari": re.compile(r"[\u0900-\u097F]"),
    "latin": re.compile(r"[a-zA-Z]"),
    "tamil": re.compile(r"[\u0B80-\u0BFF]"),
    "bengali": re.compile(r"[\u0980-\u09FF]"),
    "telugu": re.compile(r"[\u0C00-\u0C7F]"),
}


@dataclass
class BiasReport:
    """Structured container for full model bias audit.

    Attributes:
        gender_bias_score: Mean prediction delta between male/female pairs (↓ better).
        gender_pairs_flagged: Number of pairs with label flip.
        regional_metro_f1: Macro F1 on samples mentioning metro cities.
        regional_tier2_f1: Macro F1 on samples mentioning tier-2 cities.
        regional_gap: Absolute F1 difference (metro − tier2).
        script_f1: Dict: script_name → macro F1.
        script_gap: Max − min F1 across scripts.
        brand_big4_f1: Macro F1 on big-4 brand samples.
        brand_niche_f1: Macro F1 on niche brand samples.
        brand_gap: Absolute F1 difference (big4 − niche).
        class_precision: Dict: label → precision (imbalance audit).
        class_recall: Dict: label → recall.
        bias_flags: List of textual warnings for critical gaps.
        overall_bias_score: Aggregate score 0.0–1.0 (0 = unbiased, 1 = highly biased).
        sample_count: Number of samples evaluated.
    """

    gender_bias_score: float = 0.0
    gender_pairs_flagged: int = 0
    regional_metro_f1: float = 0.0
    regional_tier2_f1: float = 0.0
    regional_gap: float = 0.0
    script_f1: dict = field(default_factory=dict)
    script_gap: float = 0.0
    brand_big4_f1: float = 0.0
    brand_niche_f1: float = 0.0
    brand_gap: float = 0.0
    class_precision: dict = field(default_factory=dict)
    class_recall: dict = field(default_factory=dict)
    bias_flags: list[str] = field(default_factory=list)
    overall_bias_score: float = 0.0
    sample_count: int = 0


LABEL_NAMES = {0: "positive", 1: "neutral", 2: "negative"}


class BiasChecker:
    """Run multi-dimensional bias analysis on a sentiment model.

    Args:
        model: Fine-tuned HuggingFace classification model.
        tokenizer: Matching tokenizer.
        device: Torch device (defaults to CUDA if available).
        batch_size: Inference batch size.
        threshold: Gap threshold above which bias is flagged.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None,
        batch_size: int = 32,
        threshold: float = 0.05,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.threshold = threshold
        self.model.eval()
        self.model.to(self.device)

    def _predict_texts(self, texts: list[str]) -> np.ndarray:
        """Run batched inference and return predicted label indices.

        Args:
            texts: List of input strings.

        Returns:
            Integer numpy array of predicted label indices.
        """
        all_preds = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
        return np.array(all_preds)

    def _macro_f1(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute macro-averaged F1 safely.

        Args:
            preds: Predicted label indices.
            labels: True label indices.

        Returns:
            Macro F1 score or 0.0 if empty.
        """
        if len(preds) == 0:
            return 0.0
        from sklearn.metrics import f1_score as _f1
        return float(_f1(labels, preds, average="macro", zero_division=0))

    # ─── Bias Dimension 1: Gender ─────────────────────────────────────────────

    def check_gender_bias(self, n_templates: int = 50) -> tuple[float, int]:
        """Generate counterfactual pairs and measure prediction consistency.

        For each (male_name, female_name) pair, create identical sentences
        and count how often the model gives different predictions.

        Args:
            n_templates: Number of sentiment templates per pair.

        Returns:
            Tuple of (mean_delta, n_flagged_pairs).
        """
        templates = [
            "{name} ने Zomato पर order किया और delivery अच्छी थी।",
            "{name} said Amazon delivery was really fast today.",
            "{name}'s Jio network experience has been great.",
            "{name} को Flipkart से product receive hua, quality average.",
            "{name} is very unhappy with Swiggy's late delivery.",
        ]
        deltas = []
        flagged = 0

        for male_name, female_name in GENDER_PAIRS:
            pair_deltas = []
            for tmpl in templates[:n_templates]:
                m_text = tmpl.format(name=male_name)
                f_text = tmpl.format(name=female_name)
                m_pred = self._predict_texts([m_text])[0]
                f_pred = self._predict_texts([f_text])[0]
                pair_deltas.append(int(m_pred != f_pred))

            pair_delta = float(np.mean(pair_deltas))
            deltas.append(pair_delta)
            if pair_delta > self.threshold:
                flagged += 1
                logger.warning(
                    "Gender bias detected for pair (%s / %s): delta=%.3f",
                    male_name,
                    female_name,
                    pair_delta,
                )

        return float(np.mean(deltas)), flagged

    # ─── Bias Dimension 2: Regional ───────────────────────────────────────────

    def check_regional_bias(
        self,
        texts: list[str],
        labels: list[int],
    ) -> tuple[float, float]:
        """Compute F1 separately for texts mentioning metro vs. tier-2 cities.

        Args:
            texts: Raw evaluation texts.
            labels: Ground-truth label indices.

        Returns:
            Tuple (metro_f1, tier2_f1).
        """
        metro_idx, tier2_idx = [], []

        for i, text in enumerate(texts):
            words = set(re.findall(r"\b[A-Z][a-z]+\b", text))
            if words & METRO_CITIES:
                metro_idx.append(i)
            elif words & TIER2_CITIES:
                tier2_idx.append(i)

        def group_f1(indices: list[int]) -> float:
            if not indices:
                return 0.0
            group_texts = [texts[j] for j in indices]
            group_labels = np.array([labels[j] for j in indices])
            preds = self._predict_texts(group_texts)
            return self._macro_f1(preds, group_labels)

        return group_f1(metro_idx), group_f1(tier2_idx)

    # ─── Bias Dimension 3: Script ─────────────────────────────────────────────

    def check_script_bias(
        self,
        texts: list[str],
        labels: list[int],
    ) -> dict[str, float]:
        """Compute macro F1 per Unicode script.

        Classifies each text by its dominant script character count.

        Args:
            texts: Raw evaluation texts.
            labels: Ground-truth label indices.

        Returns:
            Dict mapping script name to macro F1.
        """
        buckets: dict[str, list[int]] = {s: [] for s in SCRIPT_REGEX}
        buckets["mixed"] = []

        for i, text in enumerate(texts):
            script_counts = {
                s: len(rx.findall(text)) for s, rx in SCRIPT_REGEX.items()
            }
            dominant = max(script_counts, key=lambda s: script_counts[s])
            if script_counts[dominant] == 0:
                buckets["mixed"].append(i)
            else:
                buckets[dominant].append(i)

        script_f1_dict = {}
        for script, indices in buckets.items():
            if len(indices) < 5:
                continue
            group_texts = [texts[j] for j in indices]
            group_labels = np.array([labels[j] for j in indices])
            preds = self._predict_texts(group_texts)
            script_f1_dict[script] = self._macro_f1(preds, group_labels)

        return script_f1_dict

    # ─── Bias Dimension 4: Brand Popularity ───────────────────────────────────

    def check_brand_bias(
        self,
        texts: list[str],
        labels: list[int],
        brands: list[str],
    ) -> tuple[float, float]:
        """Compare F1 on big-4 brand samples vs. niche brand samples.

        Args:
            texts: Raw evaluation texts.
            labels: Ground-truth label indices.
            brands: Per-sample brand string (comma-separated if multiple).

        Returns:
            Tuple (big4_f1, niche_f1).
        """
        big4_idx, niche_idx = [], []
        for i, brand_str in enumerate(brands):
            brand_set = {b.strip() for b in brand_str.split(",")}
            if brand_set & BIG4_BRANDS:
                big4_idx.append(i)
            elif brand_set & NICHE_BRANDS:
                niche_idx.append(i)

        def group_f1(indices: list[int]) -> float:
            if not indices:
                return 0.0
            preds = self._predict_texts([texts[j] for j in indices])
            return self._macro_f1(preds, np.array([labels[j] for j in indices]))

        return group_f1(big4_idx), group_f1(niche_idx)

    # ─── Bias Dimension 5: Sentiment Class Imbalance ─────────────────────────

    def check_class_imbalance_bias(
        self,
        texts: list[str],
        labels: list[int],
    ) -> tuple[dict, dict]:
        """Run full classification report to surface per-class precision/recall.

        Args:
            texts: Evaluation texts.
            labels: True labels.

        Returns:
            Tuple (class_precision dict, class_recall dict).
        """
        preds = self._predict_texts(texts)
        label_arr = np.array(labels)
        report = classification_report(
            label_arr,
            preds,
            output_dict=True,
            zero_division=0,
        )
        precision = {
            LABEL_NAMES.get(int(k), k): v["precision"]
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        }
        recall = {
            LABEL_NAMES.get(int(k), k): v["recall"]
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        }
        return precision, recall

    # ─── Full Audit Orchestration ─────────────────────────────────────────────

    def run_full_audit(
        self,
        texts: list[str],
        labels: list[int],
        brands: Optional[list[str]] = None,
    ) -> BiasReport:
        """Run all 5 bias dimensions and return a consolidated BiasReport.

        Args:
            texts: Evaluation sample texts.
            labels: Ground-truth sentiment labels (0=positive, 1=neutral, 2=negative).
            brands: Optional per-sample brand strings.

        Returns:
            BiasReport dataclass with all scores and flags.
        """
        report = BiasReport(sample_count=len(texts))
        flags: list[str] = []

        logger.info("Running gender bias audit …")
        report.gender_bias_score, report.gender_pairs_flagged = (
            self.check_gender_bias()
        )
        if report.gender_bias_score > self.threshold:
            flags.append(
                f"GENDER: High counterfactual inconsistency = {report.gender_bias_score:.3f}"
            )

        logger.info("Running regional bias audit …")
        report.regional_metro_f1, report.regional_tier2_f1 = (
            self.check_regional_bias(texts, labels)
        )
        report.regional_gap = abs(
            report.regional_metro_f1 - report.regional_tier2_f1
        )
        if report.regional_gap > self.threshold:
            flags.append(
                f"REGIONAL: Metro vs. Tier-2 F1 gap = {report.regional_gap:.3f}"
            )

        logger.info("Running script bias audit …")
        report.script_f1 = self.check_script_bias(texts, labels)
        if report.script_f1:
            f1_vals = list(report.script_f1.values())
            report.script_gap = max(f1_vals) - min(f1_vals)
            if report.script_gap > self.threshold:
                flags.append(
                    f"SCRIPT: Max script F1 gap = {report.script_gap:.3f}"
                )

        if brands:
            logger.info("Running brand popularity bias audit …")
            report.brand_big4_f1, report.brand_niche_f1 = (
                self.check_brand_bias(texts, labels, brands)
            )
            report.brand_gap = abs(report.brand_big4_f1 - report.brand_niche_f1)
            if report.brand_gap > self.threshold:
                flags.append(
                    f"BRAND: Big-4 vs. niche F1 gap = {report.brand_gap:.3f}"
                )

        logger.info("Running class imbalance bias audit …")
        report.class_precision, report.class_recall = (
            self.check_class_imbalance_bias(texts, labels)
        )
        low_recall_classes = [
            cls for cls, rec in report.class_recall.items() if rec < 0.60
        ]
        if low_recall_classes:
            flags.append(
                f"CLASS: Low recall (<0.60) for: {', '.join(low_recall_classes)}"
            )

        report.bias_flags = flags
        # Aggregate score: mean of normalized gaps + gender score
        gaps = [
            min(1.0, report.gender_bias_score),
            min(1.0, report.regional_gap),
            min(1.0, report.script_gap),
            min(1.0, report.brand_gap),
        ]
        report.overall_bias_score = float(np.mean(gaps))

        logger.info(
            "Bias audit complete. Overall score=%.3f  Flags=%d",
            report.overall_bias_score,
            len(flags),
        )
        return report
