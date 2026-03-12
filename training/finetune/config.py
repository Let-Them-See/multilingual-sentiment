"""
Training configuration for LoRA fine-tuning of multilingual sentiment model.

All hyperparameters, model paths, and experiment settings are centralized
here. Import this module instead of hardcoding values elsewhere.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation configuration.

    Attributes:
        r: LoRA rank — number of decomposition dimensions.
        lora_alpha: Scaling factor; effective LR = lr * alpha/r.
        target_modules: Attention projection layers to adapt.
        lora_dropout: Dropout on LoRA weights.
        bias: Bias training strategy ("none" | "all" | "lora_only").
        task_type: PEFT task type string.
    """

    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "SEQ_CLS"


@dataclass
class TrainingConfig:
    """HuggingFace TrainingArguments wrapper with all hyperparameters.

    Attributes:
        output_dir: Local directory for checkpoints.
        num_train_epochs: Total training epochs.
        per_device_train_batch_size: Batch size per GPU during training.
        per_device_eval_batch_size: Batch size during evaluation.
        gradient_accumulation_steps: Steps to accumulate before update.
        learning_rate: Peak learning rate.
        lr_scheduler_type: LR schedule type.
        warmup_ratio: Fraction of steps for linear warmup.
        weight_decay: L2 regularization coefficient.
        fp16: Enable mixed-precision FP16 training.
        evaluation_strategy: Eval frequency ("epoch" | "steps").
        save_strategy: Checkpoint save frequency.
        load_best_model_at_end: Restore best checkpoint after training.
        metric_for_best_model: Primary metric for best-model selection.
        logging_steps: Log every N steps.
        report_to: Experiment tracking backend ("wandb" | "tensorboard").
        max_seq_length: Max tokenized sequence length.
        num_labels: Number of classification labels.
        seed: Random seed.
    """

    output_dir: str = "./lora-sentiment"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    evaluation_strategy: str = "epoch"
    save_strategy: str = "best"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    logging_steps: int = 50
    report_to: str = "wandb"
    max_seq_length: int = 128
    num_labels: int = 3
    seed: int = 42


@dataclass
class ModelConfig:
    """Model selection and HuggingFace Hub configuration.

    Attributes:
        base_model: Primary model identifier.
        secondary_model: Secondary model for ablation.
        baseline_model: Baseline model for comparison.
        hub_adapter_repo: HF Hub repo for LoRA adapter.
        hub_merged_repo: HF Hub repo for merged model.
        dataset_path: Local path or HF Hub dataset identifier.
        id2label: Label index to string mapping.
        label2id: Label string to index mapping.
    """

    base_model: str = "ai4bharat/indic-bert"
    secondary_model: str = "mistralai/Mistral-7B-v0.1"
    baseline_model: str = "xlm-roberta-base"
    hub_adapter_repo: str = "yourusername/indic-sentiment-lora"
    hub_merged_repo: str = "yourusername/indic-sentiment-merged"
    dataset_path: str = "training/data/augmented"
    id2label: dict[int, str] = field(
        default_factory=lambda: {0: "positive", 1: "negative", 2: "neutral"}
    )
    label2id: dict[str, int] = field(
        default_factory=lambda: {"positive": 0, "negative": 1, "neutral": 2}
    )


@dataclass
class FocalLossConfig:
    """Focal loss configuration for imbalanced class handling.

    Attributes:
        gamma: Focal loss exponent. Higher = more focus on hard examples.
        alpha: Per-class weight for positive/negative/neutral.
        reduction: Loss reduction method ("mean" | "sum").
    """

    gamma: float = 2.0
    alpha: list[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    reduction: str = "mean"


# ─── Default instances ────────────────────────────────────────────────────────
LORA_CONFIG = LoRAConfig()
TRAINING_CONFIG = TrainingConfig()
MODEL_CONFIG = ModelConfig()
FOCAL_LOSS_CONFIG = FocalLossConfig()

# ─── languages and brands ─────────────────────────────────────────────────────
LANGUAGES: list[str] = ["hindi", "tamil", "bengali", "telugu", "marathi", "code_mix"]
BRANDS: list[str] = [
    "Jio", "Zomato", "Flipkart", "BYJU'S", "Paytm",
    "Ola", "Swiggy", "Tata", "HDFC", "Airtel",
]
SENTIMENT_CLASSES: list[str] = ["positive", "negative", "neutral"]
