"""
Synthetic data generator for Indian brand sentiment.

Generates 10,000 synthetic samples per language when scraped data
is below the 5,000 sample threshold. Includes code-mixed samples.

Usage:
    python create_dataset.py --min_per_lang 5000 \
                             --synth_per_lang 10000 \
                             --push_to_hub
"""

import os
import json
import random
import logging
import argparse
from pathlib import Path
from itertools import product
from typing import Generator

from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("create_dataset")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

BRANDS = ["Jio", "Zomato", "Flipkart", "BYJU'S", "Paytm",
          "Ola", "Swiggy", "Tata", "HDFC", "Airtel"]
LABELS = ["positive", "negative", "neutral"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "hi": {
        "positive": [
            "{brand} का नेटवर्क बहुत अच्छा है।",
            "{brand} की सर्विस शानदार है!",
            "{brand} से मैं बहुत खुश हूं।",
            "{brand} ने कमाल कर दिया!",
            "{brand} बेहतरीन है, पूरी तरह से संतुष्ट हूं।",
        ],
        "negative": [
            "{brand} की सर्विस बहुत खराब है।",
            "{brand} का नेटवर्क बेकार है।",
            "{brand} ने धोखा दिया।",
            "{brand} से बिल्कुल खुश नहीं हूं।",
            "{brand} की टीम ने कोई मदद नहीं की।",
        ],
        "neutral": [
            "{brand} का नया प्लान देखा।",
            "{brand} ने एक नई सर्विस शुरू की है।",
            "{brand} के बारे में जानकारी चाहिए।",
            "क्या {brand} का ऑफर सही है?",
        ],
    },
    "ta": {
        "positive": [
            "{brand} சேவை மிகவும் நல்லது!",
            "{brand} மிகவும் திறமையான சேவை.",
            "{brand} மூலம் மிகவும் மகிழ்ச்சியாக இருக்கிறேன்.",
            "{brand} அருமையான வேலை செய்கிறது!",
        ],
        "negative": [
            "{brand} சேவை மோசமாக இருக்கிறது.",
            "{brand} நம்பகமற்றது.",
            "{brand} மூலம் ஏமாற்றப்பட்டேன்.",
            "{brand} தரம் மிகவும் குறைவாக இருக்கிறது.",
        ],
        "neutral": [
            "{brand} புதிய சலுகை வந்தது.",
            "{brand} பற்றி மேலும் தெரிய வேண்டும்.",
            "{brand} புதிய திட்டம் அறிவித்துள்ளது.",
        ],
    },
    "bn": {
        "positive": [
            "{brand} এর সেবা অসাধারণ!",
            "{brand} থেকে খুব ভালো অভিজ্ঞতা পেলাম।",
            "{brand} সত্যিই চমৎকার।",
            "{brand} এর পরিষেবা দারুণ।",
        ],
        "negative": [
            "{brand} এর সেবা একদম খারাপ।",
            "{brand} থেকে প্রতারিত হলাম।",
            "{brand} আমাকে হতাশ করেছে।",
            "{brand} এর নেটওয়ার্ক বাজে।",
        ],
        "neutral": [
            "{brand} নতুন অফার দিচ্ছে।",
            "{brand} সম্পর্কে তথ্য চাই।",
            "{brand} এর নতুন প্ল্যান কী?",
        ],
    },
    "te": {
        "positive": [
            "{brand} సేవ చాలా మంచిది!",
            "{brand} తో చాలా సంతోషంగా ఉన్నాను.",
            "{brand} అద్భుతమైన అనుభవం ఇచ్చింది.",
            "{brand} నిజంగా అనుకూలంగా ఉంది.",
        ],
        "negative": [
            "{brand} సేవ చాలా చెడ్డగా ఉంది.",
            "{brand} వల్ల నిరాశ కలిగింది.",
            "{brand} నెట్‌వర్క్ సమస్యలు ఉన్నాయి.",
            "{brand} నుండి మోసపోయాను.",
        ],
        "neutral": [
            "{brand} కొత్త ప్లాన్ వచ్చింది.",
            "{brand} గురించి అడగాలి.",
            "{brand} నేడు కొత్త ఆఫర్ ప్రకటించింది.",
        ],
    },
    "mr": {
        "positive": [
            "{brand} ची सेवा खूप चांगली आहे!",
            "{brand} कडून खूप छान अनुभव मिळाला.",
            "{brand} खरोखरच झकास आहे.",
            "{brand} मुळे मी खूप खूश आहे.",
        ],
        "negative": [
            "{brand} ची सेवा अत्यंत वाईट आहे.",
            "{brand} ने फसवणूक केली.",
            "{brand} नेटवर्क बेकार आहे.",
            "{brand} कडून निराश झालो.",
        ],
        "neutral": [
            "{brand} ने नवीन ऑफर आणले आहे.",
            "{brand} विषयी माहिती हवी आहे.",
            "{brand} चा नवा प्लान कसा आहे?",
        ],
    },
    "code_mix": {
        "positive": [
            "{brand} ka service bahut mast hai yaar!",
            "Aaj {brand} ne kamaal kar diya!",
            "{brand} is really awesome da!",
            "Finally {brand} ne sahi kar diya!",
            "{brand} super fast delivery bro!",
        ],
        "negative": [
            "{brand} ka network bilkul bakwas hai yaar.",
            "{brand} ne phir se cheat kiya 😡",
            "{brand} delivery itni late kyun hoti hai?",
            "{brand} app crash ho gaya phir se.",
            "Fed up with {brand} customer service bhai.",
        ],
        "neutral": [
            "Anyone using {brand} new plan?",
            "{brand} ne new offer diya hai check karo.",
            "Kya {brand} ka service theek hai?",
            "{brand} launched something new today.",
        ],
    },
}


def generate_synthetic(
    lang: str,
    n_samples: int,
    seed: int = 42,
) -> Generator[dict, None, None]:
    """Generate synthetic sentiment samples by filling brand-slot templates.

    Args:
        lang: Language code (hi/ta/bn/te/mr/code_mix).
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Yields:
        Dict with text, label, language, brands, source fields.
    """
    random.seed(seed)
    templates = TEMPLATES.get(lang, TEMPLATES["code_mix"])
    all_templates = [
        (label, tmpl)
        for label, tmpls in templates.items()
        for tmpl in tmpls
    ]
    generated = 0

    while generated < n_samples:
        label, template = random.choice(all_templates)
        brand = random.choice(BRANDS)
        text = template.replace("{brand}", brand)

        yield {
            "text": text,
            "label": LABEL2ID[label],
            "language": lang,
            "brands": [brand],
            "source": "synthetic",
            "created_at": "",
        }
        generated += 1


def build_or_augment_dataset(
    processed_dir: Path,
    min_per_lang: int = 5000,
    synth_per_lang: int = 10000,
) -> DatasetDict:
    """Load processed dataset and fill gaps with synthetic data.

    Args:
        processed_dir: Path to processed HuggingFace dataset directory.
        min_per_lang: Minimum samples per language before augmenting.
        synth_per_lang: Number of synthetic samples to generate per language.

    Returns:
        DatasetDict with combined real + synthetic data.
    """
    if processed_dir.exists():
        logger.info("Loading processed dataset from %s", processed_dir)
        dataset = load_from_disk(str(processed_dir))
        train_df = dataset["train"].to_pandas()
        lang_counts = train_df["language"].value_counts().to_dict()
    else:
        logger.warning("No processed dataset found — building from synthetic data only")
        lang_counts = {}
        dataset = None

    all_langs = ["hi", "ta", "bn", "te", "mr", "code_mix"]
    synthetic_rows: list[dict] = []

    for lang in all_langs:
        count = lang_counts.get(lang, 0)
        if count < min_per_lang:
            needed = synth_per_lang
            logger.info(
                "lang=%s: real=%d, generating %d synthetic samples",
                lang, count, needed,
            )
            synthetic_rows.extend(generate_synthetic(lang, needed))
        else:
            logger.info("lang=%s: real=%d (sufficient, no augmentation)", lang, count)

    if not synthetic_rows:
        assert dataset is not None
        return dataset

    synth_ds = Dataset.from_list(synthetic_rows)

    if dataset is not None:
        from datasets import concatenate_datasets
        new_train = concatenate_datasets([dataset["train"], synth_ds])
        return DatasetDict(
            {
                "train": new_train,
                "validation": dataset["validation"],
                "test": dataset["test"],
            }
        )
    else:
        # Pure synthetic fallback — create 80/10/10 split
        total = len(synth_ds)
        n_val = int(total * 0.1)
        n_test = int(total * 0.1)
        shuffled = synth_ds.shuffle(seed=42)
        return DatasetDict(
            {
                "train": shuffled.select(range(total - n_val - n_test)),
                "validation": shuffled.select(range(total - n_val - n_test, total - n_test)),
                "test": shuffled.select(range(total - n_test, total)),
            }
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Create final HuggingFace dataset")
    parser.add_argument("--processed_dir", default="training/data/processed")
    parser.add_argument("--output_dir", default="training/data/augmented")
    parser.add_argument("--min_per_lang", type=int, default=5000)
    parser.add_argument("--synth_per_lang", type=int, default=10000)
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push final dataset to HuggingFace Hub",
    )
    args = parser.parse_args()

    dataset = build_or_augment_dataset(
        Path(args.processed_dir),
        args.min_per_lang,
        args.synth_per_lang,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))

    for split, ds in dataset.items():
        logger.info("  %s: %d samples", split, len(ds))

    if args.push_to_hub:
        hub_repo = os.getenv(
            "HF_DATASET_REPO", "yourusername/indian-brand-sentiment-multilingual"
        )
        token = os.getenv("HF_TOKEN")
        if not token:
            logger.error("HF_TOKEN not set — cannot push to Hub")
            return
        dataset.push_to_hub(hub_repo, token=token)
        logger.info("Dataset pushed to https://huggingface.co/datasets/%s", hub_repo)


if __name__ == "__main__":
    main()
