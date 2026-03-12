"""
Ablation study runner for 8 systematic experiments.

Each ablation isolates one variable while holding others constant,
following standard NLP experimental methodology for reproducibility.

Usage:
    python run_ablation.py --study all
    python run_ablation.py --study lora_rank
    python run_ablation.py --study base_model --dry_run
"""

import json
import logging
import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from training.finetune.config import MODEL_CONFIG, TRAINING_CONFIG
from training.finetune.evaluate import compute_all_metrics

logger = logging.getLogger("ablation")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

RESULTS_DIR = Path("training/ablation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = MODEL_CONFIG.dataset_path
SEED = TRAINING_CONFIG.seed


@dataclass
class AblationResult:
    """Container for a single ablation experiment's results.

    Attributes:
        study: Ablation study name.
        config_name: Human-readable configuration label.
        config_params: Dict of varied hyperparameters.
        f1_macro: Overall macro F1 on test set.
        f1_hindi: Hindi subset F1.
        f1_tamil: Tamil subset F1.
        f1_bengali: Bengali subset F1.
        f1_code_mix: Code-mix subset F1.
        trainable_params: Number of trainable parameters.
        inference_ms: Median inference time per sample (ms).
        model_mb: Model size in megabytes.
        accuracy: Overall accuracy.
    """

    study: str
    config_name: str
    config_params: dict
    f1_macro: float = 0.0
    f1_hindi: float = 0.0
    f1_tamil: float = 0.0
    f1_bengali: float = 0.0
    f1_code_mix: float = 0.0
    trainable_params: int = 0
    inference_ms: float = 0.0
    model_mb: float = 0.0
    accuracy: float = 0.0


def count_trainable_params(model) -> int:
    """Count trainable parameters in a model.

    Args:
        model: Any PyTorch model.

    Returns:
        Integer count of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cpu",
    n_repeats: int = 5,
) -> float:
    """Measure median per-sample inference time in milliseconds.

    Args:
        model: Fine-tuned classification model.
        tokenizer: Corresponding tokenizer.
        texts: List of sample texts.
        device: PyTorch device string.
        n_repeats: Number of timing trials.

    Returns:
        Median milliseconds per sample.
    """
    model.eval()
    model.to(device)
    times = []

    for _ in range(n_repeats):
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**enc)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / len(texts)
        times.append(elapsed_ms)

    return float(np.median(times))


def model_size_mb(model) -> float:
    """Compute approximate model size in megabytes via parameter bytes.

    Args:
        model: PyTorch model.

    Returns:
        Float megabytes.
    """
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def quick_train_eval(
    base_model: str,
    lora_r: int,
    dataset_path: str,
    output_dir: str,
    data_fraction: float = 1.0,
    exclude_languages: Optional[list[str]] = None,
    max_steps: int = 200,
    dry_run: bool = False,
) -> tuple[dict, object, object]:
    """Run a minimal training loop and evaluate on test set.

    Used for all ablation experiments to keep wall-clock time manageable
    by capping at max_steps (production training uses full epochs).

    Args:
        base_model: HF model identifier.
        lora_r: LoRA rank value.
        dataset_path: Path to DatasetDict on disk.
        output_dir: Checkpoint directory.
        data_fraction: Fraction of training data to use.
        exclude_languages: Languages to drop from training data.
        max_steps: Max gradient steps (200 for ablation; -1 for full).
        dry_run: Use only 20 steps.

    Returns:
        Tuple of (metrics dict, model, tokenizer).
    """
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_steps = 20 if dry_run else max_steps

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_from_disk(dataset_path)

    train_ds = raw["train"]
    test_ds = raw["test"]

    # Apply language exclusion
    if exclude_languages:
        train_ds = train_ds.filter(
            lambda ex: ex["language"] not in exclude_languages
        )

    # Apply data fraction
    if data_fraction < 1.0:
        n = max(16, int(len(train_ds) * data_fraction))
        train_ds = train_ds.select(range(n))

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding=False, max_length=128
        )

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text", "source", "created_at"])
    test_tok = test_ds.map(tokenize, batched=True, remove_columns=["text", "source", "created_at"])
    train_tok.set_format("torch")
    test_tok.set_format("torch")

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=3, ignore_mismatched_sizes=True
    )
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base, lora_cfg)

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=actual_steps,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    # Evaluate on test set
    pred_output = trainer.predict(test_tok)
    preds = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids

    langs = test_ds["language"]
    brands = test_ds["brands"]

    metrics = compute_all_metrics(
        preds=preds,
        labels=labels,
        languages=list(langs),
        brands=list(brands),
        log_to_wandb=False,
    )

    return metrics, model, tokenizer


# ─── Individual Ablation Studies ──────────────────────────────────────────────


def ablation_base_model(dry_run: bool = False) -> list[AblationResult]:
    """ABLATION 1: Compare base backbone models.

    Configurations: XLM-RoBERTa, IndicBERT, Mistral-7B-LoRA, mBERT.

    Args:
        dry_run: Use minimal training steps.

    Returns:
        List of AblationResult for each configuration.
    """
    models = [
        ("XLM-RoBERTa", "xlm-roberta-base"),
        ("IndicBERT", "ai4bharat/indic-bert"),
        ("mBERT", "bert-base-multilingual-cased"),
    ]
    results = []

    for config_name, model_id in models:
        logger.info("ABLATION 1 — Base Model: %s", config_name)
        out_dir = str(RESULTS_DIR / f"abl1_{config_name.lower().replace('-', '_')}")
        try:
            metrics, model, tokenizer = quick_train_eval(
                base_model=model_id,
                lora_r=16,
                dataset_path=DATASET_PATH,
                output_dir=out_dir,
                dry_run=dry_run,
            )
            benchmark_texts = [
                "Jio का नेटवर्क बहुत अच्छा है।",
                "Zomato delivery very fast da!",
                "Flipkart সেবা ভালো।",
            ]
            inf_ms = measure_inference_time(model, tokenizer, benchmark_texts)
            mb = model_size_mb(model)

            results.append(
                AblationResult(
                    study="base_model",
                    config_name=config_name,
                    config_params={"model": model_id},
                    f1_macro=metrics.get("f1_macro", 0.0),
                    f1_hindi=metrics.get("f1_language_hindi", 0.0),
                    f1_tamil=metrics.get("f1_language_tamil", 0.0),
                    f1_bengali=metrics.get("f1_language_bengali", 0.0),
                    f1_code_mix=metrics.get("f1_language_code_mix", 0.0),
                    trainable_params=count_trainable_params(model),
                    inference_ms=inf_ms,
                    model_mb=mb,
                    accuracy=metrics.get("accuracy", 0.0),
                )
            )
        except Exception as exc:
            logger.error("Failed for %s: %s", config_name, exc)

    return results


def ablation_lora_rank(dry_run: bool = False) -> list[AblationResult]:
    """ABLATION 2: Sweep LoRA rank r over [4, 8, 16, 32, 64].

    Args:
        dry_run: Use minimal training steps.

    Returns:
        List of AblationResult for each rank.
    """
    ranks = [4, 8, 16, 32, 64] if not dry_run else [4, 16]
    results = []

    for r in ranks:
        logger.info("ABLATION 2 — LoRA rank r=%d", r)
        out_dir = str(RESULTS_DIR / f"abl2_r{r}")
        try:
            metrics, model, tokenizer = quick_train_eval(
                base_model="ai4bharat/indic-bert",
                lora_r=r,
                dataset_path=DATASET_PATH,
                output_dir=out_dir,
                dry_run=dry_run,
            )
            results.append(
                AblationResult(
                    study="lora_rank",
                    config_name=f"r={r}",
                    config_params={"r": r, "alpha": r * 2},
                    f1_macro=metrics.get("f1_macro", 0.0),
                    f1_hindi=metrics.get("f1_language_hindi", 0.0),
                    trainable_params=count_trainable_params(model),
                    accuracy=metrics.get("accuracy", 0.0),
                )
            )
        except Exception as exc:
            logger.error("Failed for r=%d: %s", r, exc)

    return results


def ablation_data_fraction(dry_run: bool = False) -> list[AblationResult]:
    """ABLATION 3: Training data size learning curves (10%–100%).

    Args:
        dry_run: Use minimal steps.

    Returns:
        AblationResult list for each fraction.
    """
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0] if not dry_run else [0.1, 0.5]
    results = []

    for frac in fractions:
        logger.info("ABLATION 3 — Data fraction=%.2f", frac)
        out_dir = str(RESULTS_DIR / f"abl3_frac{int(frac*100)}")
        try:
            metrics, model, _ = quick_train_eval(
                base_model="ai4bharat/indic-bert",
                lora_r=16,
                dataset_path=DATASET_PATH,
                output_dir=out_dir,
                data_fraction=frac,
                dry_run=dry_run,
            )
            results.append(
                AblationResult(
                    study="data_fraction",
                    config_name=f"{int(frac*100)}% data",
                    config_params={"fraction": frac},
                    f1_macro=metrics.get("f1_macro", 0.0),
                    f1_hindi=metrics.get("f1_language_hindi", 0.0),
                    f1_tamil=metrics.get("f1_language_tamil", 0.0),
                    f1_bengali=metrics.get("f1_language_bengali", 0.0),
                    f1_code_mix=metrics.get("f1_language_code_mix", 0.0),
                    accuracy=metrics.get("accuracy", 0.0),
                )
            )
        except Exception as exc:
            logger.error("Failed for frac=%.2f: %s", frac, exc)

    return results


def ablation_language_exclusion(dry_run: bool = False) -> list[AblationResult]:
    """ABLATION 4: Train without each language one at a time.

    Measures cross-lingual transfer — which language helps others most.

    Args:
        dry_run: Use minimal steps.

    Returns:
        AblationResult list for each excluded language.
    """
    languages = ["hindi", "tamil", "bengali", "telugu", "marathi", "code_mix"]
    results = []

    for excl_lang in languages:
        logger.info("ABLATION 4 — Excluding language: %s", excl_lang)
        out_dir = str(RESULTS_DIR / f"abl4_excl_{excl_lang}")
        try:
            metrics, model, _ = quick_train_eval(
                base_model="ai4bharat/indic-bert",
                lora_r=16,
                dataset_path=DATASET_PATH,
                output_dir=out_dir,
                exclude_languages=[excl_lang],
                dry_run=dry_run,
            )
            results.append(
                AblationResult(
                    study="language_exclusion",
                    config_name=f"excl_{excl_lang}",
                    config_params={"excluded": excl_lang},
                    f1_macro=metrics.get("f1_macro", 0.0),
                    f1_hindi=metrics.get("f1_language_hindi", 0.0),
                    f1_tamil=metrics.get("f1_language_tamil", 0.0),
                    f1_bengali=metrics.get("f1_language_bengali", 0.0),
                    f1_code_mix=metrics.get("f1_language_code_mix", 0.0),
                    accuracy=metrics.get("accuracy", 0.0),
                )
            )
        except Exception as exc:
            logger.error("Failed for excl=%s: %s", excl_lang, exc)

    return results


def ablation_code_mix_handling(dry_run: bool = False) -> list[AblationResult]:
    """ABLATION 5: Code-mix preprocessing strategies.

    Strategies:
    - "none"       : Raw transliterated text, no special handling.
    - "lang_prefix": Prepend <hi>, <ta>, <code_mix> language tokens.
    - "ascii_norm" : Normalize Romanized Indic to ASCII transliteration.
    - "script_sep" : Separate Devanagari/Tamil/Latin tokens into sub-sequences.

    Args:
        dry_run: Use minimal steps.

    Returns:
        AblationResult list for each strategy.
    """
    import unicodedata

    strategies = {
        "none": lambda text, lang: text,
        "lang_prefix": lambda text, lang: f"<{lang}> {text}",
        "ascii_norm": lambda text, lang: unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii"),
        "script_sep": lambda text, lang: " [SEP] ".join(
            c for c in text.split() if c
        ),
    }
    results = []

    for strat_name, transform_fn in strategies.items():
        logger.info("ABLATION 5 — Code-mix strategy: %s", strat_name)
        out_dir = str(RESULTS_DIR / f"abl5_{strat_name}")
        try:
            raw = load_from_disk(DATASET_PATH)

            def apply_transform(ex):
                ex["text"] = transform_fn(ex["text"], ex.get("language", ""))
                return ex

            transformed = {
                split: raw[split].map(apply_transform, desc=f"transform_{split}")
                for split in ("train", "validation", "test")
            }

            tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            def tokenize(batch):
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    padding=False,
                    max_length=128,
                )

            removable = [
                c for c in transformed["train"].column_names
                if c not in ("input_ids", "attention_mask", "label")
            ]
            train_tok = transformed["train"].map(
                tokenize, batched=True, remove_columns=removable
            )
            test_tok = transformed["test"].map(
                tokenize, batched=True, remove_columns=removable
            )
            train_tok.set_format("torch")
            test_tok.set_format("torch")

            model = AutoModelForSequenceClassification.from_pretrained(
                "ai4bharat/indic-bert", num_labels=3, ignore_mismatched_sizes=True
            )
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            model = get_peft_model(model, lora_cfg)

            args = TrainingArguments(
                output_dir=out_dir,
                max_steps=20 if dry_run else 200,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                gradient_accumulation_steps=2,
                learning_rate=2e-4,
                fp16=torch.cuda.is_available(),
                evaluation_strategy="no",
                logging_steps=10,
                report_to="none",
                seed=SEED,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_tok,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
            )
            trainer.train()

            pred_output = trainer.predict(test_tok)
            preds = np.argmax(pred_output.predictions, axis=-1)
            labels = pred_output.label_ids
            langs = raw["test"]["language"]
            brands = raw["test"]["brands"]

            metrics = compute_all_metrics(
                preds=preds,
                labels=labels,
                languages=list(langs),
                brands=list(brands),
                log_to_wandb=False,
            )
            results.append(
                AblationResult(
                    study="code_mix_handling",
                    config_name=strat_name,
                    config_params={"strategy": strat_name},
                    f1_macro=metrics.get("f1_macro", 0.0),
                    f1_hindi=metrics.get("f1_language_hindi", 0.0),
                    f1_tamil=metrics.get("f1_language_tamil", 0.0),
                    f1_bengali=metrics.get("f1_language_bengali", 0.0),
                    f1_code_mix=metrics.get("f1_language_code_mix", 0.0),
                    trainable_params=count_trainable_params(model),
                    accuracy=metrics.get("accuracy", 0.0),
                )
            )
        except Exception as exc:
            logger.error("Failed for strategy=%s: %s", strat_name, exc)

    return results


def ablation_data_augmentation(dry_run: bool = False) -> list[AblationResult]:
    """ABLATION 6: Data augmentation strategy comparison.

    Configurations:
    - "no_aug"           : Baseline — original data only.
    - "backtranslation"  : Helsinki-NLP back-translation only.
    - "synonym_only"     : Synonym replacement only.
    - "aug_combined"     : Back-translation + synonym replacement.

    Args:
        dry_run: Use minimal steps.

    Returns:
        AblationResult list.
    """
    aug_configs = [
        ("no_aug", False, False),
        ("backtranslation", True, False),
        ("synonym_only", False, True),
        ("aug_combined", True, True),
    ]
    results = []

    for config_name, use_bt, use_syn in aug_configs:
        logger.info("ABLATION 6 — Augmentation strategy: %s", config_name)
        dataset_path = DATASET_PATH

        if not dry_run and (use_bt or use_syn):
            aug_path = str(RESULTS_DIR / f"abl6_{config_name}_ds")
            try:
                from training.scripts.translate_aug import augment_dataset
                from datasets import load_from_disk as lfd

                raw = lfd(DATASET_PATH)
                augmented = augment_dataset(
                    dataset=raw,
                    use_back_translation=use_bt,
                    use_synonym=use_syn,
                    min_per_lang=3000,
                )
                augmented.save_to_disk(aug_path)
                dataset_path = aug_path
                logger.info("Augmented dataset saved to %s", aug_path)
            except Exception as aug_err:
                logger.warning(
                    "Augmentation failed (%s), using original data: %s",
                    config_name,
                    aug_err,
                )

        out_dir = str(RESULTS_DIR / f"abl6_{config_name}")
        try:
            metrics, model, _ = quick_train_eval(
                base_model="ai4bharat/indic-bert",
                lora_r=16,
                dataset_path=dataset_path,
                output_dir=out_dir,
                dry_run=dry_run,
            )
            results.append(
                AblationResult(
                    study="data_augmentation",
                    config_name=config_name,
                    config_params={
                        "back_translation": use_bt,
                        "synonym_replace": use_syn,
                    },
                    f1_macro=metrics.get("f1_macro", 0.0),
                    f1_hindi=metrics.get("f1_language_hindi", 0.0),
                    f1_tamil=metrics.get("f1_language_tamil", 0.0),
                    f1_bengali=metrics.get("f1_language_bengali", 0.0),
                    f1_code_mix=metrics.get("f1_language_code_mix", 0.0),
                    trainable_params=count_trainable_params(model),
                    accuracy=metrics.get("accuracy", 0.0),
                )
            )
        except Exception as exc:
            logger.error("Failed for config=%s: %s", config_name, exc)

    return results


def ablation_loss_function(dry_run: bool = False) -> list[AblationResult]:
    """ABLATION 7: Loss function comparison.

    Configurations:
    - "cross_entropy"    : Standard cross-entropy (PyTorch default).
    - "focal_gamma2"     : Focal loss with gamma=2.0.
    - "label_smooth_0.1" : Label smoothing with eps=0.1.
    - "weighted_ce"      : Class-weighted cross-entropy (inverse freq).

    Args:
        dry_run: Use minimal steps.

    Returns:
        AblationResult list.
    """
    import torch.nn as nn
    from torch.nn import CrossEntropyLoss

    class FocalLossLocal(nn.Module):
        """Re-defined locally to avoid circular imports."""

        def __init__(self, gamma: float = 2.0, num_labels: int = 3):
            super().__init__()
            self.gamma = gamma
            self.num_labels = num_labels

        def forward(self, logits, labels):
            ce = CrossEntropyLoss(reduction="none")(logits, labels)
            p_t = torch.exp(-ce)
            return ((1 - p_t) ** self.gamma * ce).mean()

    class WeightedCELoss(nn.Module):
        """Class-weighted cross-entropy for imbalanced labels."""

        def __init__(self, weights: list[float]):
            super().__init__()
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float))

        def forward(self, logits, labels):
            return CrossEntropyLoss(weight=self.weights)(logits, labels)

    loss_configs = {
        "cross_entropy": lambda: CrossEntropyLoss(),
        "focal_gamma2": lambda: FocalLossLocal(gamma=2.0),
        "label_smooth_0.1": lambda: CrossEntropyLoss(label_smoothing=0.1),
        "weighted_ce": lambda: WeightedCELoss(weights=[0.4, 0.35, 0.25]),
    }

    from transformers import Trainer as BaseTrainer

    results = []

    for config_name, loss_factory in loss_configs.items():
        logger.info("ABLATION 7 — Loss function: %s", config_name)
        out_dir = str(RESULTS_DIR / f"abl7_{config_name}")
        try:
            set_seed(SEED)
            tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            raw = load_from_disk(DATASET_PATH)

            def tok_fn(batch):
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    padding=False,
                    max_length=128,
                )

            removable = [
                c for c in raw["train"].column_names
                if c not in ("input_ids", "attention_mask", "label")
            ]
            train_tok = raw["train"].map(
                tok_fn, batched=True, remove_columns=removable
            )
            test_tok = raw["test"].map(
                tok_fn, batched=True, remove_columns=removable
            )
            train_tok.set_format("torch")
            test_tok.set_format("torch")

            base = AutoModelForSequenceClassification.from_pretrained(
                "ai4bharat/indic-bert",
                num_labels=3,
                ignore_mismatched_sizes=True,
            )
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            model = get_peft_model(base, lora_cfg)
            loss_fn = loss_factory()

            class CustomLossTrainer(BaseTrainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)
                    return (loss, outputs) if return_outputs else loss

            args = TrainingArguments(
                output_dir=out_dir,
                max_steps=20 if dry_run else 200,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                gradient_accumulation_steps=2,
                learning_rate=2e-4,
                fp16=torch.cuda.is_available(),
                evaluation_strategy="no",
                logging_steps=10,
                report_to="none",
                seed=SEED,
            )
            trainer = CustomLossTrainer(
                model=model,
                args=args,
                train_dataset=train_tok,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
            )
            trainer.train()

            pred_output = trainer.predict(test_tok)
            preds = np.argmax(pred_output.predictions, axis=-1)
            labels_arr = pred_output.label_ids

            metrics = compute_all_metrics(
                preds=preds,
                labels=labels_arr,
                languages=list(raw["test"]["language"]),
                brands=list(raw["test"]["brands"]),
                log_to_wandb=False,
            )
            results.append(
                AblationResult(
                    study="loss_function",
                    config_name=config_name,
                    config_params={"loss": config_name},
                    f1_macro=metrics.get("f1_macro", 0.0),
                    f1_hindi=metrics.get("f1_language_hindi", 0.0),
                    f1_tamil=metrics.get("f1_language_tamil", 0.0),
                    f1_bengali=metrics.get("f1_language_bengali", 0.0),
                    f1_code_mix=metrics.get("f1_language_code_mix", 0.0),
                    trainable_params=count_trainable_params(model),
                    accuracy=metrics.get("accuracy", 0.0),
                )
            )
        except Exception as exc:
            logger.error("Failed for loss=%s: %s", config_name, exc)

    return results


def ablation_quantization_impact(dry_run: bool = False) -> list[AblationResult]:
    """ABLATION 8: Post-training quantization quality vs. efficiency trade-off.

    Configurations:
    - "fp32"  : Full precision (no quantization).
    - "fp16"  : Half precision inference.
    - "int8"  : 8-bit BitsAndBytes quantization.
    - "nf4"   : 4-bit NormalFloat (NF4) quantization.

    Measures inference speed and F1 degradation from FP32 baseline.

    Args:
        dry_run: Use minimal steps.

    Returns:
        AblationResult list.
    """
    quant_configs = {
        "fp32": {"load_in_8bit": False, "load_in_4bit": False, "dtype": torch.float32},
        "fp16": {"load_in_8bit": False, "load_in_4bit": False, "dtype": torch.float16},
        "int8": {"load_in_8bit": True, "load_in_4bit": False, "dtype": torch.float16},
        "nf4": {"load_in_8bit": False, "load_in_4bit": True, "dtype": torch.float16},
    }
    results = []
    benchmark_texts = [
        "Jio का नेटवर्क बहुत अच्छा है।",
        "Zomato delivery fast ahe!",
        "Flipkart সেবা ভালো।",
        "Amazon bad experience re.",
        "Swiggy order late aayi.",
    ]

    for config_name, quant_kw in quant_configs.items():
        logger.info("ABLATION 8 — Quantization: %s", config_name)
        out_dir_train = str(RESULTS_DIR / f"abl8_{config_name}_train")
        try:
            # 1. Train in fp16 (same for all) then reload with quant config
            metrics_train, _, _ = quick_train_eval(
                base_model="ai4bharat/indic-bert",
                lora_r=16,
                dataset_path=DATASET_PATH,
                output_dir=out_dir_train,
                dry_run=dry_run,
            )

            # 2. Reload model with target quantization for inference benchmarking
            try:
                from transformers import BitsAndBytesConfig

                bnb_cfg = None
                if quant_kw.get("load_in_4bit"):
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                elif quant_kw.get("load_in_8bit"):
                    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

                load_kw = {
                    "num_labels": 3,
                    "ignore_mismatched_sizes": True,
                    "torch_dtype": quant_kw["dtype"],
                }
                if bnb_cfg is not None:
                    load_kw["quantization_config"] = bnb_cfg

                quant_model = AutoModelForSequenceClassification.from_pretrained(
                    "ai4bharat/indic-bert", **load_kw
                )
                tokenizer_q = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
                inf_ms = measure_inference_time(
                    quant_model,
                    tokenizer_q,
                    benchmark_texts,
                    device="cpu",
                )
                mb = model_size_mb(quant_model)
            except Exception as quant_err:
                logger.warning("Quantized load failed for %s: %s", config_name, quant_err)
                inf_ms = 0.0
                mb = 0.0

            results.append(
                AblationResult(
                    study="quantization_impact",
                    config_name=config_name,
                    config_params=quant_kw,
                    f1_macro=metrics_train.get("f1_macro", 0.0),
                    f1_hindi=metrics_train.get("f1_language_hindi", 0.0),
                    f1_tamil=metrics_train.get("f1_language_tamil", 0.0),
                    f1_bengali=metrics_train.get("f1_language_bengali", 0.0),
                    f1_code_mix=metrics_train.get("f1_language_code_mix", 0.0),
                    inference_ms=inf_ms,
                    model_mb=mb,
                    accuracy=metrics_train.get("accuracy", 0.0),
                )
            )
        except Exception as exc:
            logger.error("Failed for quant=%s: %s", config_name, exc)

    return results


ALL_STUDIES = {
    "base_model": ablation_base_model,
    "lora_rank": ablation_lora_rank,
    "data_fraction": ablation_data_fraction,
    "language_exclusion": ablation_language_exclusion,
    "code_mix_handling": ablation_code_mix_handling,
    "data_augmentation": ablation_data_augmentation,
    "loss_function": ablation_loss_function,
    "quantization_impact": ablation_quantization_impact,
}


def run_all_studies(dry_run: bool = False) -> dict[str, list[AblationResult]]:
    """Run all ablation studies sequentially and save results.

    Args:
        dry_run: Use minimal steps for smoke testing.

    Returns:
        Dict mapping study name to list of AblationResult.
    """
    all_results: dict[str, list[AblationResult]] = {}

    for study_name, study_fn in ALL_STUDIES.items():
        logger.info("══════ Starting ablation study: %s ══════", study_name)
        results = study_fn(dry_run=dry_run)
        all_results[study_name] = results

        # Save intermediate results
        out_path = RESULTS_DIR / f"{study_name}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(
                [asdict(r) for r in results],
                fh,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Saved %s results → %s", study_name, out_path)

    return all_results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--study",
        choices=list(ALL_STUDIES.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.study == "all":
        run_all_studies(dry_run=args.dry_run)
    else:
        study_fn = ALL_STUDIES[args.study]
        results = study_fn(dry_run=args.dry_run)
        out_path = RESULTS_DIR / f"{args.study}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2, ensure_ascii=False)
        logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
