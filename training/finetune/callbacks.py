"""
Custom HuggingFace training callbacks for detailed logging.

Tracks per-language F1, learning rate schedule, gradient norms,
GPU memory usage, and logs sample predictions to W&B.
"""

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger("callbacks")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed — W&B logging disabled")


class LanguageF1Callback(TrainerCallback):
    """Log per-language F1 scores to W&B after each evaluation epoch.

    Requires the Trainer to expose `eval_dataset` with a `language`
    column and `predictions` attribute populated by the compute_metrics
    function.

    Attributes:
        language_names: List of language strings to track.
        id2label: Mapping from label int to string.
    """

    def __init__(
        self,
        language_names: list[str],
        id2label: dict[int, str],
    ) -> None:
        """Initialize callback with language and label metadata.

        Args:
            language_names: Language strings present in dataset.
            id2label: Label index to string name mapping.
        """
        self.language_names = language_names
        self.id2label = id2label
        self._lang_metrics: dict[str, dict[str, float]] = {}

    def update_lang_metrics(
        self,
        metrics: dict[str, float],
    ) -> None:
        """Store per-language metrics computed externally.

        Args:
            metrics: Dict keyed as "f1_{language}" with float values.
        """
        for lang in self.language_names:
            key = f"f1_{lang}"
            if key in metrics:
                self._lang_metrics[lang] = metrics[key]

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log per-language F1 metrics to W&B if available.

        Args:
            args: Training arguments.
            state: Trainer state (epoch, global_step, etc.).
            control: Trainer control flags.
            metrics: Eval metrics dict passed by Trainer.
        """
        if not WANDB_AVAILABLE or not wandb.run:
            return
        if metrics is None:
            return

        lang_log: dict[str, float] = {}
        for lang in self.language_names:
            key = f"eval_f1_{lang}"
            if key in metrics:
                lang_log[f"lang_f1/{lang}"] = metrics[key]

        if lang_log:
            wandb.log(lang_log, step=state.global_step)
            logger.info(
                "Epoch %s lang F1: %s",
                state.epoch,
                {k: f"{v:.4f}" for k, v in lang_log.items()},
            )


class GradientNormCallback(TrainerCallback):
    """Log total gradient norm to W&B at every logging step.

    Useful for diagnosing exploding/vanishing gradient issues
    during LoRA fine-tuning.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        """Compute and log gradient norm after each optimizer step.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Control flags.
            model: The model being trained.
        """
        if model is None:
            return
        if state.global_step % args.logging_steps != 0:
            return

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.detach().norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        if WANDB_AVAILABLE and wandb.run:
            wandb.log({"train/grad_norm": total_norm}, step=state.global_step)


class GPUMemoryCallback(TrainerCallback):
    """Log GPU memory usage (allocated and reserved) to W&B."""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Sample GPU memory at each logging step.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
        """
        if not torch.cuda.is_available():
            return
        if state.global_step % args.logging_steps != 0:
            return

        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9    # GB

        if WANDB_AVAILABLE and wandb.run:
            wandb.log(
                {
                    "gpu/memory_allocated_gb": allocated,
                    "gpu/memory_reserved_gb": reserved,
                },
                step=state.global_step,
            )


class SamplePredictionsCallback(TrainerCallback):
    """Log a table of sample predictions to W&B each epoch.

    Shows text, true label, predicted label, and confidence
    for a fixed set of examples to visually track model quality.

    Attributes:
        sample_texts: Fixed list of texts to predict each epoch.
        true_labels: Corresponding ground-truth label strings.
        tokenizer: Tokenizer for encoding inputs.
        model: Model for inference.
        id2label: Label ID to string mapping.
        device: Inference device.
    """

    def __init__(
        self,
        sample_texts: list[str],
        true_labels: list[str],
        tokenizer: Any,
        id2label: dict[int, str],
        device: str = "cpu",
    ) -> None:
        """Initialize with a fixed sample set.

        Args:
            sample_texts: Representative texts from each language.
            true_labels: Ground-truth label strings.
            tokenizer: Tokenizer instance.
            id2label: Label index to string mapping.
            device: PyTorch device string.
        """
        self.sample_texts = sample_texts
        self.true_labels = true_labels
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = device

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        """Run inference on samples and log predictions table to W&B.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Control flags.
            model: Active model.
        """
        if model is None or not WANDB_AVAILABLE or not wandb.run:
            return

        model.eval()
        rows = []

        with torch.no_grad():
            for text, true_label in zip(self.sample_texts, self.true_labels):
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                ).to(self.device)
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                pred_id = int(np.argmax(probs))
                pred_label = self.id2label[pred_id]
                confidence = float(probs[pred_id])

                rows.append(
                    [
                        text[:80],
                        true_label,
                        pred_label,
                        f"{confidence:.3f}",
                        "✅" if pred_label == true_label else "❌",
                    ]
                )

        table = wandb.Table(
            columns=["Text", "True", "Predicted", "Confidence", "Correct"],
            data=rows,
        )
        wandb.log({"predictions/samples": table}, step=state.global_step)
