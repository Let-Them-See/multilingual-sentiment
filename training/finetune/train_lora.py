"""
Main LoRA fine-tuning script for multilingual Indian sentiment analysis.

Implements:
  - LoRA (r=16) on IndicBERT / Mistral-7B base models
  - Focal loss with class-weighted alpha
  - W&B experiment tracking with full suite of metrics
  - Per-language and per-brand F1 evaluation
  - Model merging, 4-bit quantization, ONNX export
  - Push to HuggingFace Hub

Usage:
    python train_lora.py
    python train_lora.py --base_model xlm-roberta-base --wandb_run baseline_xlmr
    python train_lora.py --dry_run  # 10 steps only for smoke-test
"""

import os
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk, DatasetDict
from dotenv import load_dotenv
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from training.finetune.config import (
    FOCAL_LOSS_CONFIG,
    LANGUAGES,
    LORA_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
)
from training.finetune.callbacks import (
    GradientNormCallback,
    GPUMemoryCallback,
    LanguageF1Callback,
    SamplePredictionsCallback,
)
from training.finetune.evaluate import make_compute_metrics

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_lora")

# ─── Focal Loss ───────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal loss for handling long-tail class distributions.

    Addresses the ~40/35/25 positive/negative/neutral imbalance
    in the Indian brand sentiment dataset.

    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"

    Attributes:
        gamma: Focusing parameter. Higher values down-weight easy examples.
        alpha: Per-class weights tensor.
        reduction: Loss aggregation method.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[list[float]] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize FocalLoss.

        Args:
            gamma: Scaling exponent for hard-example focus.
            alpha: Class weights list (one per class).
            reduction: "mean", "sum", or "none".
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output (B, num_classes).
            targets: Ground-truth label indices (B,).

        Returns:
            Scalar loss tensor.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather log prob and prob at target class
        log_p_t = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        p_t = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        loss = -((1 - p_t) ** self.gamma) * log_p_t

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ─── Custom Trainer with Focal Loss ──────────────────────────────────────────


class FocalLossTrainer(Trainer):
    """HuggingFace Trainer subclass with Focal Loss override.

    Attributes:
        focal_loss: FocalLoss module instance.
    """

    def __init__(self, *args, focal_loss: FocalLoss, **kwargs) -> None:
        """Initialize with focal_loss injected.

        Args:
            focal_loss: Pre-instantiated FocalLoss module.
            *args: Passed to Trainer.
            **kwargs: Passed to Trainer.
        """
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict,
        return_outputs: bool = False,
    ):
        """Override default CE loss with Focal Loss.

        Args:
            model: The LoRA-wrapped model.
            inputs: Tokenized batch dict.
            return_outputs: Whether to return model outputs alongside loss.

        Returns:
            Scalar loss, or (loss, outputs) tuple if return_outputs.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ─── Tokenization ─────────────────────────────────────────────────────────────


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> DatasetDict:
    """Tokenize all splits in a DatasetDict.

    Args:
        dataset: HuggingFace DatasetDict with "text" column.
        tokenizer: Pre-loaded tokenizer.
        max_length: Token sequence cap.

    Returns:
        DatasetDict with input_ids, attention_mask, token_type_ids columns.
    """

    def tokenize_fn(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "source", "created_at"],
    )
    tokenized.set_format("torch")
    return tokenized


# ─── Main Training Function ───────────────────────────────────────────────────


def run_training(
    base_model: str = MODEL_CONFIG.base_model,
    dataset_path: str = MODEL_CONFIG.dataset_path,
    output_dir: str = TRAINING_CONFIG.output_dir,
    wandb_run_name: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Full LoRA fine-tuning pipeline.

    Loads data → tokenizes → wraps with LoRA → trains with Focal Loss
    → evaluates → merges weights → quantizes → pushes to HF Hub.

    Args:
        base_model: HuggingFace model identifier for the backbone.
        dataset_path: Path to HuggingFace DatasetDict on disk.
        output_dir: Local directory for checkpoints.
        wandb_run_name: Optional W&B run name for this experiment.
        dry_run: If True, run only 10 steps for smoke-testing.
    """
    set_seed(TRAINING_CONFIG.seed)
    start_time = time.time()

    # ── W&B init ──────────────────────────────────────────────────────────
    try:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "indic-sentiment-lora"),
            entity=os.getenv("WANDB_ENTITY"),
            name=wandb_run_name or f"lora_r{LORA_CONFIG.r}_{base_model.split('/')[-1]}",
            config={
                "base_model": base_model,
                "lora_r": LORA_CONFIG.r,
                "lora_alpha": LORA_CONFIG.lora_alpha,
                "learning_rate": TRAINING_CONFIG.learning_rate,
                "epochs": TRAINING_CONFIG.num_train_epochs,
                "batch_size": TRAINING_CONFIG.per_device_train_batch_size,
                "focal_gamma": FOCAL_LOSS_CONFIG.gamma,
            },
        )
    except Exception as exc:
        logger.warning("W&B init failed (continuing without): %s", exc)

    # ── Load tokenizer and model ───────────────────────────────────────────
    logger.info("Loading tokenizer and model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=TRAINING_CONFIG.num_labels,
        id2label=MODEL_CONFIG.id2label,
        label2id=MODEL_CONFIG.label2id,
        ignore_mismatched_sizes=True,
    )

    # ── Apply LoRA ─────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=LORA_CONFIG.r,
        lora_alpha=LORA_CONFIG.lora_alpha,
        target_modules=LORA_CONFIG.target_modules,
        lora_dropout=LORA_CONFIG.lora_dropout,
        bias=LORA_CONFIG.bias,
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / total,
    )

    # ── Load dataset ───────────────────────────────────────────────────────
    logger.info("Loading dataset from %s …", dataset_path)
    raw_dataset = load_from_disk(dataset_path)

    tokenized = tokenize_dataset(raw_dataset, tokenizer, TRAINING_CONFIG.max_seq_length)
    train_ds = tokenized["train"]
    eval_ds = tokenized["validation"]

    if dry_run:
        train_ds = train_ds.select(range(min(160, len(train_ds))))
        eval_ds = eval_ds.select(range(min(64, len(eval_ds))))
        logger.warning("DRY RUN: using only %d train samples", len(train_ds))

    # ── Build training args ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if dry_run else TRAINING_CONFIG.num_train_epochs,
        per_device_train_batch_size=TRAINING_CONFIG.per_device_train_batch_size,
        per_device_eval_batch_size=TRAINING_CONFIG.per_device_eval_batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG.gradient_accumulation_steps,
        learning_rate=TRAINING_CONFIG.learning_rate,
        lr_scheduler_type=TRAINING_CONFIG.lr_scheduler_type,
        warmup_ratio=TRAINING_CONFIG.warmup_ratio,
        weight_decay=TRAINING_CONFIG.weight_decay,
        fp16=torch.cuda.is_available() and TRAINING_CONFIG.fp16,
        evaluation_strategy=TRAINING_CONFIG.evaluation_strategy,
        save_strategy=TRAINING_CONFIG.save_strategy,
        load_best_model_at_end=TRAINING_CONFIG.load_best_model_at_end,
        metric_for_best_model=TRAINING_CONFIG.metric_for_best_model,
        logging_steps=10 if dry_run else TRAINING_CONFIG.logging_steps,
        report_to=TRAINING_CONFIG.report_to if not dry_run else "none",
        dataloader_num_workers=2,
        seed=TRAINING_CONFIG.seed,
        max_steps=10 if dry_run else -1,
    )

    # ── Sample predictions for W&B logging ────────────────────────────────
    sample_texts = [
        "Jio का नेटवर्क बहुत अच्छा है!",
        "Zomato delivery bahut late ho gayi yaar.",
        "Flipkart सेल आज शुरू हुई।",
        "Swiggy super fast delivery da!",
        "BYJU'S app crash ho gaya phir se 😡",
    ]
    sample_labels = ["positive", "negative", "neutral", "positive", "negative"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Build callbacks ────────────────────────────────────────────────────
    eval_original = raw_dataset["validation"]
    callbacks = [
        LanguageF1Callback(LANGUAGES, MODEL_CONFIG.id2label),
        GradientNormCallback(),
        GPUMemoryCallback(),
        SamplePredictionsCallback(
            sample_texts=sample_texts,
            true_labels=sample_labels,
            tokenizer=tokenizer,
            id2label=MODEL_CONFIG.id2label,
            device=device,
        ),
    ]

    # ── Focal loss ─────────────────────────────────────────────────────────
    focal_loss = FocalLoss(
        gamma=FOCAL_LOSS_CONFIG.gamma,
        alpha=FOCAL_LOSS_CONFIG.alpha,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_compute_metrics(MODEL_CONFIG.id2label, eval_original),
        callbacks=callbacks,
        focal_loss=focal_loss,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    trainer.train()
    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f minutes", elapsed / 60)

    if not dry_run:
        _post_training(model, tokenizer, output_dir, base_model)


def _post_training(
    model,
    tokenizer,
    output_dir: str,
    base_model: str,
) -> None:
    """Merge LoRA weights, quantize, export, and push to Hub.

    Args:
        model: Trained PEFT model.
        tokenizer: Fine-tuned tokenizer.
        output_dir: Local checkpoint directory.
        base_model: Original backbone model name.
    """
    adapter_dir = Path(output_dir) / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("LoRA adapter saved → %s", adapter_dir)

    # ── Merge LoRA into base weights ──────────────────────────────────────
    logger.info("Merging LoRA weights into base model …")
    merged_model = model.merge_and_unload()
    merged_dir = Path(output_dir) / "merged"
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    logger.info("Merged model saved → %s", merged_dir)

    # ── 4-bit quantization ────────────────────────────────────────────────
    logger.info("Attempting 4-bit quantization …")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        quantized = AutoModelForSequenceClassification.from_pretrained(
            str(merged_dir),
            quantization_config=bnb_config,
            device_map="auto",
        )
        quantized_dir = Path(output_dir) / "quantized_4bit"
        quantized.save_pretrained(str(quantized_dir))
        logger.info("4-bit quantized model saved → %s", quantized_dir)
    except Exception as exc:
        logger.warning("4-bit quantization failed (bitsandbytes required): %s", exc)

    # ── HuggingFace Hub push ──────────────────────────────────────────────
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        adapter_repo = os.getenv("HF_MODEL_REPO", MODEL_CONFIG.hub_adapter_repo)
        merged_repo = os.getenv("HF_MERGED_REPO", MODEL_CONFIG.hub_merged_repo)

        logger.info("Pushing adapter to HF Hub: %s", adapter_repo)
        model.push_to_hub(adapter_repo, token=hf_token)
        tokenizer.push_to_hub(adapter_repo, token=hf_token)

        logger.info("Pushing merged model to HF Hub: %s", merged_repo)
        merged_model.push_to_hub(merged_repo, token=hf_token)
        tokenizer.push_to_hub(merged_repo, token=hf_token)
    else:
        logger.info("HF_TOKEN not set — skipping Hub push")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Indian sentiment")
    parser.add_argument(
        "--base_model",
        type=str,
        default=MODEL_CONFIG.base_model,
        choices=[
            "ai4bharat/indic-bert",
            "mistralai/Mistral-7B-v0.1",
            "xlm-roberta-base",
            "google/muril-base-cased",
        ],
    )
    parser.add_argument("--dataset_path", type=str, default=MODEL_CONFIG.dataset_path)
    parser.add_argument("--output_dir", type=str, default=TRAINING_CONFIG.output_dir)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="10-step smoke test")
    args = parser.parse_args()

    run_training(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        wandb_run_name=args.wandb_run_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
