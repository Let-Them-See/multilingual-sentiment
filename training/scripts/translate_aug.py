"""
Back-translation and synonym-replacement data augmentation.

Augments low-resource language splits to improve model generalization.
Uses Helsinki NLP OPUS-MT translation models from HuggingFace.

Usage:
    python translate_aug.py --input_dir training/data/processed \
                            --output_dir training/data/augmented \
                            --target_per_lang 5000
"""

import logging
import argparse
import random
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import MarianMTModel, MarianTokenizer

logger = logging.getLogger("translate_aug")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# Helsinki-NLP OPUS-MT back-translation routes
# hi/mr/bn → English → back to source language
TRANSLATION_MODELS: dict[str, dict[str, str]] = {
    "hi": {
        "to_en": "Helsinki-NLP/opus-mt-hi-en",
        "from_en": "Helsinki-NLP/opus-mt-en-hi",
    },
    "ta": {
        "to_en": "Helsinki-NLP/opus-mt-ta-en",
        "from_en": "Helsinki-NLP/opus-mt-en-ta",
    },
    "bn": {
        "to_en": "Helsinki-NLP/opus-mt-bn-en",
        "from_en": "Helsinki-NLP/opus-mt-en-bn",
    },
    "mr": {
        "to_en": "Helsinki-NLP/opus-mt-mr-en",
        "from_en": "Helsinki-NLP/opus-mt-en-mr",
    },
}

SYNONYM_VOCAB: dict[str, list[str]] = {
    # Common positive adjectives for paraphrase augmentation
    "good": ["great", "excellent", "superb", "nice", "wonderful"],
    "bad": ["poor", "terrible", "awful", "horrible", "dreadful"],
    "fast": ["quick", "speedy", "rapid", "swift"],
    "slow": ["sluggish", "delayed", "late", "laggy"],
    "cheap": ["affordable", "budget-friendly", "economical", "inexpensive"],
    "expensive": ["costly", "pricey", "overpriced", "steep"],
    "love": ["like", "enjoy", "appreciate", "adore"],
    "hate": ["dislike", "detest", "despise", "loathe"],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache loaded models to avoid reloading
_model_cache: dict[str, tuple[MarianMTModel, MarianTokenizer]] = {}


def load_translation_model(
    model_name: str,
) -> tuple[MarianMTModel, MarianTokenizer]:
    """Load or retrieve cached MarianMT model and tokenizer.

    Args:
        model_name: HuggingFace model identifier string.

    Returns:
        Tuple of (model, tokenizer).
    """
    if model_name not in _model_cache:
        logger.info("Loading translation model: %s", model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        model.eval()
        _model_cache[model_name] = (model, tokenizer)
    return _model_cache[model_name]


def translate_batch(
    texts: list[str],
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    batch_size: int = 32,
) -> list[str]:
    """Translate a list of texts using a MarianMT model.

    Args:
        texts: List of source language strings.
        model: MarianMT model.
        tokenizer: Corresponding tokenizer.
        batch_size: Number of samples per inference batch.

    Returns:
        List of translated strings (same length as input).
    """
    results: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=4, max_new_tokens=128)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results


def back_translate(
    texts: list[str],
    language: str,
) -> Optional[list[str]]:
    """Apply back-translation augmentation: lang → English → lang.

    Args:
        texts: Source language strings.
        language: Language code (hi/ta/bn/mr).

    Returns:
        Back-translated strings, or None if models unavailable.
    """
    routes = TRANSLATION_MODELS.get(language)
    if not routes:
        return None

    try:
        to_en_model, to_en_tok = load_translation_model(routes["to_en"])
        from_en_model, from_en_tok = load_translation_model(routes["from_en"])

        english = translate_batch(texts, to_en_model, to_en_tok)
        back = translate_batch(english, from_en_model, from_en_tok)
        return back
    except Exception as exc:
        logger.warning("Back-translation failed for lang=%s: %s", language, exc)
        return None


def synonym_replace(text: str, n_replacements: int = 1) -> str:
    """Replace n random words with synonyms for code-mix augmentation.

    Args:
        text: Input English or code-mix string.
        n_replacements: How many words to replace.

    Returns:
        Augmented text string.
    """
    words = text.split()
    replaceable = [
        (i, w, SYNONYM_VOCAB[w.lower()])
        for i, w in enumerate(words)
        if w.lower() in SYNONYM_VOCAB
    ]
    random.shuffle(replaceable)
    for idx, (i, orig, synonyms) in enumerate(replaceable):
        if idx >= n_replacements:
            break
        words[i] = random.choice(synonyms)
    return " ".join(words)


def augment_dataset(
    dataset: DatasetDict,
    target_per_lang: int = 5000,
) -> DatasetDict:
    """Augment the training split using back-translation + synonym replacement.

    Only augments languages where the training count < target_per_lang.

    Args:
        dataset: HuggingFace DatasetDict with train/validation/test splits.
        target_per_lang: Minimum samples per language in training set.

    Returns:
        DatasetDict with augmented training split.
    """
    train = dataset["train"]
    df = train.to_pandas()

    augmented_rows: list[dict] = []

    for lang, lang_df in df.groupby("language"):
        count = len(lang_df)
        needed = max(0, target_per_lang - count)
        if needed == 0:
            logger.info("lang=%s already has %d samples, skipping aug", lang, count)
            continue

        logger.info("lang=%s: %d samples, augmenting by %d", lang, count, needed)

        # Sample randomly (with replacement if needed)
        sample_df = lang_df.sample(
            n=min(needed, len(lang_df)), replace=needed > len(lang_df), random_state=42
        )
        texts = sample_df["text"].tolist()
        labels = sample_df["label"].tolist()
        brands = sample_df["brands"].tolist()

        # Apply back-translation for Indian script languages
        bt_texts = back_translate(texts, str(lang))
        if bt_texts:
            for text, label, brand_list in zip(bt_texts, labels, brands):
                augmented_rows.append(
                    {
                        "text": text,
                        "label": label,
                        "language": lang,
                        "brands": brand_list,
                        "source": "back_translation",
                        "created_at": "",
                    }
                )
        else:
            # Fallback: synonym replacement for code-mix
            for text, label, brand_list in zip(texts, labels, brands):
                aug_text = synonym_replace(text)
                augmented_rows.append(
                    {
                        "text": aug_text,
                        "label": label,
                        "language": lang,
                        "brands": brand_list,
                        "source": "synonym_replace",
                        "created_at": "",
                    }
                )

    if augmented_rows:
        aug_dataset = Dataset.from_list(augmented_rows)
        from datasets import concatenate_datasets
        new_train = concatenate_datasets([train, aug_dataset])
        logger.info(
            "Training augmented: %d → %d samples",
            len(train),
            len(new_train),
        )
        return DatasetDict(
            {
                "train": new_train,
                "validation": dataset["validation"],
                "test": dataset["test"],
            }
        )

    return dataset


def main() -> None:
    """Entry point — load processed dataset, augment, save to disk."""
    parser = argparse.ArgumentParser(description="Augment low-resource training splits")
    parser.add_argument("--input_dir", type=str, default="training/data/processed")
    parser.add_argument("--output_dir", type=str, default="training/data/augmented")
    parser.add_argument(
        "--target_per_lang",
        type=int,
        default=5000,
        help="Minimum training samples per language",
    )
    args = parser.parse_args()

    logger.info("Loading dataset from %s …", args.input_dir)
    dataset = load_from_disk(args.input_dir)

    logger.info("Augmenting …")
    augmented = augment_dataset(dataset, args.target_per_lang)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    augmented.save_to_disk(str(out_dir))
    logger.info("Augmented dataset saved to %s", out_dir)

    for split, ds in augmented.items():
        logger.info("  %s: %d samples", split, len(ds))


if __name__ == "__main__":
    main()
