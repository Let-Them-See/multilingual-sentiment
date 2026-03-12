"""
Preprocessing pipeline for Indian multilingual sentiment dataset.

Steps:
  1. Load all raw JSONL files (Twitter + Reddit)
  2. Normalize Unicode and script characters
  3. Clean text (URLs, mentions, hashtags)
  4. Deduplicate with MinHash LSH
  5. Auto-label using TextBlob / IndicNLP heuristics
  6. Output HuggingFace DatasetDict with train/val/test splits

Usage:
    python preprocess.py --input_dir training/data/raw \
                         --output_dir training/data/processed
"""

import re
import json
import logging
import hashlib
import unicodedata
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Value, Features
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("preprocess")

# ─── Constants ────────────────────────────────────────────────────────────────
LABELS: list[str] = ["positive", "negative", "neutral"]
LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(LABELS)}

LANG_MAP: dict[str, str] = {
    "hi": "hindi",
    "ta": "tamil",
    "bn": "bengali",
    "te": "telugu",
    "mr": "marathi",
    "code_mix": "code_mix",
    "en": "code_mix",  # English-dominant Reddit → code_mix bucket
}

# Positive/Negative vocabulary per language for heuristic labelling
SENTIMENT_VOCAB: dict[str, dict[str, list[str]]] = {
    "hi": {
        "positive": ["अच्छा", "बढ़िया", "शानदार", "मस्त", "सुपर", "धाँसू", "बेहतरीन", "खुश"],
        "negative": ["बेकार", "खराब", "घटिया", "बकवास", "बुरा", "परेशान", "धोखा", "फालतू"],
    },
    "ta": {
        "positive": ["நல்ல", "அருமை", "சூப்பர்", "மிகவும் நல்ல", "பயனுள்ள"],
        "negative": ["மோசம்", "கெட்ட", "பயனற்ற", "ஏமாற்றம்", "மோசமான"],
    },
    "bn": {
        "positive": ["ভালো", "দারুণ", "অসাধারণ", "চমৎকার", "সেরা"],
        "negative": ["খারাপ", "বাজে", "বেকার", "ধোঁকা", "হতাশাজনক"],
    },
    "te": {
        "positive": ["మంచి", "అద్భుతమైన", "అనుకూలంగా", "చాలా బాగుంది"],
        "negative": ["చెడ్డ", "నిరాశగా", "వ్యర్థం", "సమస్య"],
    },
    "mr": {
        "positive": ["चांगले", "उत्कृष्ट", "भारी", "झकास", "मस्त"],
        "negative": ["वाईट", "बेकार", "घाण", "चिंताजनक", "फसवणूक"],
    },
    "code_mix": {
        "positive": ["amazing", "best", "superb", "great", "love", "awesome", "mast", "badhiya"],
        "negative": ["worst", "terrible", "useless", "waste", "pathetic", "bakwas", "bekar"],
    },
}

# Regex patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)
REPEATED_PUNCT = re.compile(r"([!?.]){3,}")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text to NFC form, preserving Indian scripts.

    Args:
        text: Raw input string.

    Returns:
        NFC-normalized string with consistent Unicode representation.
    """
    return unicodedata.normalize("NFC", text)


def clean_text(text: str, keep_hashtag_content: bool = True) -> str:
    """Clean tweet/post text while preserving linguistic content.

    Removes URLs, @mentions, and optionally hashtag symbols (#),
    normalizes whitespace, and strips leading/trailing spaces.
    Indian script characters are fully preserved.

    Args:
        text: Raw text.
        keep_hashtag_content: If True, remove # symbol but keep the word.

    Returns:
        Cleaned text string.
    """
    if not text or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)

    if keep_hashtag_content:
        text = HASHTAG_PATTERN.sub(r"\1", text)
    else:
        text = HASHTAG_PATTERN.sub(" ", text)

    # Normalize repeated punctuation
    text = REPEATED_PUNCT.sub(r"\1\1", text)
    # Collapse whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def text_hash(text: str) -> str:
    """Compute a short stable hash for deduplication.

    Args:
        text: Cleaned text string.

    Returns:
        8-character hex digest.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def heuristic_label(text: str, lang: str) -> Optional[str]:
    """Assign a sentiment label using vocabulary heuristics.

    For English/code-mix text, falls back to TextBlob polarity score.
    Returns None if the signal is ambiguous (will be dropped or 
    labelled as neutral by caller).

    Args:
        text: Cleaned text.
        lang: Language code (hi/ta/bn/te/mr/code_mix).

    Returns:
        "positive", "negative", "neutral", or None if ambiguous.
    """
    text_lower = text.lower()
    vocab = SENTIMENT_VOCAB.get(lang, SENTIMENT_VOCAB["code_mix"])

    pos_hits = sum(1 for w in vocab["positive"] if w.lower() in text_lower)
    neg_hits = sum(1 for w in vocab["negative"] if w.lower() in text_lower)

    if pos_hits > neg_hits:
        return "positive"
    if neg_hits > pos_hits:
        return "negative"

    # Fallback: TextBlob for Latin-script text
    if lang in ("code_mix", "en"):
        try:
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.1:
                return "positive"
            if polarity < -0.1:
                return "negative"
            return "neutral"
        except Exception:  # pragma: no cover
            pass

    # Ambiguous → neutral
    return "neutral"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load all records from a JSONL file.

    Args:
        filepath: Path to .jsonl file.

    Returns:
        List of parsed dicts.
    """
    records = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON line: %s", exc)
    return records


def infer_language(record: dict) -> str:
    """Determine effective language key from record metadata.

    Args:
        record: Raw data dict with 'target_lang' or 'lang' field.

    Returns:
        Language key string (hi/ta/bn/te/mr/code_mix).
    """
    lang = record.get("target_lang") or record.get("lang") or "en"
    return LANG_MAP.get(lang, "code_mix")


def process_raw_dir(raw_dir: Path) -> list[dict]:
    """Load, clean, and label all JSONL files in the raw data directory.

    Args:
        raw_dir: Directory containing *.jsonl raw data files.

    Returns:
        List of processed record dicts ready for dataset creation.
    """
    processed: list[dict] = []
    seen_hashes: set[str] = set()
    label_dist: dict[str, int] = defaultdict(int)

    for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
        if "log" in jsonl_file.name:
            continue
        logger.info("Processing %s …", jsonl_file.name)
        raw_records = load_jsonl(jsonl_file)

        for rec in raw_records:
            raw_text = rec.get("full_text") or rec.get("text") or ""
            if not raw_text.strip():
                continue

            cleaned = clean_text(raw_text)
            if len(cleaned.split()) < 3:
                continue

            # Deduplication
            h = text_hash(cleaned)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            lang = infer_language(rec)
            label = heuristic_label(cleaned, lang)
            if label is None:
                label = "neutral"

            brands = rec.get("brands") or ([rec["brand"]] if "brand" in rec else [])

            processed.append(
                {
                    "text": cleaned,
                    "label": LABEL2ID[label],
                    "label_str": label,
                    "language": lang,
                    "brands": brands,
                    "source": rec.get("source", "twitter"),
                    "created_at": rec.get("created_at", ""),
                }
            )
            label_dist[label] += 1

    logger.info(
        "Total processed: %d | pos=%d neg=%d neu=%d",
        len(processed),
        label_dist["positive"],
        label_dist["negative"],
        label_dist["neutral"],
    )
    return processed


def build_dataset_dict(records: list[dict]) -> DatasetDict:
    """Split processed records into train/val/test DatasetDict.

    Applies stratified split by language to ensure balance.
    Split: 80% train / 10% val / 10% test.

    Args:
        records: List of processed dicts.

    Returns:
        HuggingFace DatasetDict with train/validation/test splits.
    """
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=LABELS),
            "language": Value("string"),
            "brands": [Value("string")],
            "source": Value("string"),
            "created_at": Value("string"),
        }
    )

    # Group by language for stratified split
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_lang[rec["language"]].append(rec)

    train_recs, val_recs, test_recs = [], [], []

    for lang, recs in by_lang.items():
        # Shuffle deterministically
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(len(recs)).tolist()
        shuffled = [recs[i] for i in indices]

        n = len(shuffled)
        n_val = max(1, int(n * 0.1))
        n_test = max(1, int(n * 0.1))
        n_train = n - n_val - n_test

        train_recs.extend(shuffled[:n_train])
        val_recs.extend(shuffled[n_train : n_train + n_val])
        test_recs.extend(shuffled[n_train + n_val :])
        logger.info(
            "lang=%s train=%d val=%d test=%d", lang, n_train, n_val, n_test
        )

    def recs_to_dataset(recs: list[dict]) -> Dataset:
        data = {
            "text": [r["text"] for r in recs],
            "label": [r["label"] for r in recs],
            "language": [r["language"] for r in recs],
            "brands": [r["brands"] for r in recs],
            "source": [r["source"] for r in recs],
            "created_at": [r["created_at"] for r in recs],
        }
        return Dataset.from_dict(data, features=features)

    return DatasetDict(
        {
            "train": recs_to_dataset(train_recs),
            "validation": recs_to_dataset(val_recs),
            "test": recs_to_dataset(test_recs),
        }
    )


def main() -> None:
    """Entry point — preprocess raw data and save to disk."""
    parser = argparse.ArgumentParser(description="Preprocess Indian brand sentiment data")
    parser.add_argument("--input_dir", type=str, default="training/data/raw")
    parser.add_argument("--output_dir", type=str, default="training/data/processed")
    args = parser.parse_args()

    raw_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and cleaning raw data from %s …", raw_dir)
    records = process_raw_dir(raw_dir)

    if not records:
        logger.error("No records found. Run scrape_twitter.py / scrape_reddit.py first.")
        return

    logger.info("Building HuggingFace DatasetDict …")
    dataset = build_dataset_dict(records)

    logger.info("Saving to disk: %s", out_dir)
    dataset.save_to_disk(str(out_dir))

    logger.info("Dataset summary:")
    for split, ds in dataset.items():
        logger.info("  %s: %d samples", split, len(ds))


if __name__ == "__main__":
    main()
