"""
Twitter / X Scraper for Indian Brand Sentiment Dataset.

Collects tweets mentioning Indian brands across six languages:
Hindi (hi), Tamil (ta), Bengali (bn), Telugu (te),
Marathi (mr), and English-Hindi code-mix.

Usage:
    python scrape_twitter.py --lang hi --brand Jio --max_results 1000
    python scrape_twitter.py --all  # scrapes all brand-language combos
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator

import tweepy
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training/data/raw/scrape_twitter.log"),
    ],
)
logger = logging.getLogger("scrape_twitter")

# ─── Constants ────────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES: dict[str, str] = {
    "hi": "Hindi (हिंदी)",
    "ta": "Tamil (தமிழ்)",
    "bn": "Bengali (বাংলা)",
    "te": "Telugu (తెలుగు)",
    "mr": "Marathi (मराठी)",
}

BRANDS: list[str] = [
    "Jio",
    "Zomato",
    "Flipkart",
    "BYJU'S",
    "Paytm",
    "Ola",
    "Swiggy",
    "Tata",
    "HDFC",
    "Airtel",
]

# Brand → search queries per language
BRAND_QUERIES: dict[str, dict[str, list[str]]] = {
    "Jio": {
        "hi": ["Jio नेटवर्क lang:hi", "JioFiber lang:hi", "@JioCare lang:hi"],
        "ta": ["Jio நெட்வொர்க் lang:ta", "JioFiber lang:ta"],
        "bn": ["Jio নেটওয়ার্ক lang:bn", "@Jio lang:bn"],
        "te": ["Jio నెట్‌వర్క్ lang:te", "JioFiber lang:te"],
        "mr": ["Jio नेटवर्क lang:mr", "JioFiber lang:mr"],
        "code_mix": ["Jio ka network lang:en -is:retweet", "Jio speed problem"],
    },
    "Zomato": {
        "hi": ["Zomato डिलीवरी lang:hi", "@Zomato lang:hi", "Zomato खाना lang:hi"],
        "ta": ["Zomato டெலிவரி lang:ta", "@Zomato lang:ta"],
        "bn": ["Zomato ডেলিভারি lang:bn", "@Zomato lang:bn"],
        "te": ["Zomato డెలివరీ lang:te", "@Zomato lang:te"],
        "mr": ["Zomato डिलिव्हरी lang:mr", "@Zomato lang:mr"],
        "code_mix": ["Zomato delivery late lang:en", "Zomato order cancel"],
    },
    "Flipkart": {
        "hi": ["Flipkart सेल lang:hi", "@Flipkart lang:hi", "Flipkart डिलीवरी lang:hi"],
        "ta": ["Flipkart விற்பனை lang:ta", "@Flipkart lang:ta"],
        "bn": ["Flipkart বিক্রয় lang:bn", "@Flipkart lang:bn"],
        "te": ["Flipkart సేల్ lang:te", "@Flipkart lang:te"],
        "mr": ["Flipkart विक्री lang:mr", "@Flipkart lang:mr"],
        "code_mix": ["Flipkart sale mast hai", "Flipkart delivery problem"],
    },
    "BYJU'S": {
        "hi": ["BYJU'S लर्निंग lang:hi", "Byjus app lang:hi"],
        "ta": ["BYJU'S கற்றல் lang:ta", "Byjus app lang:ta"],
        "bn": ["BYJU'S শিক্ষা lang:bn", "Byjus app lang:bn"],
        "te": ["BYJU'S నేర్చుకోవడం lang:te"],
        "mr": ["BYJU'S शिक्षण lang:mr"],
        "code_mix": ["Byjus app useless", "BYJU'S subscription refund"],
    },
    "Paytm": {
        "hi": ["Paytm पेमेंट lang:hi", "@Paytm lang:hi", "Paytm UPI lang:hi"],
        "ta": ["Paytm பணம் lang:ta", "@Paytm lang:ta"],
        "bn": ["Paytm পেমেন্ট lang:bn", "@Paytm lang:bn"],
        "te": ["Paytm చెల్లింపు lang:te"],
        "mr": ["Paytm पेमेंट lang:mr"],
        "code_mix": ["Paytm payment failed", "Paytm se paise gaye"],
    },
    "Ola": {
        "hi": ["Ola कैब lang:hi", "@Ola_Cabs lang:hi"],
        "ta": ["Ola கேப் lang:ta", "@Ola_Cabs lang:ta"],
        "bn": ["Ola ক্যাব lang:bn"],
        "te": ["Ola క్యాబ్ lang:te"],
        "mr": ["Ola कॅब lang:mr"],
        "code_mix": ["Ola driver rude", "Ola cab cancel last minute"],
    },
    "Swiggy": {
        "hi": ["Swiggy डिलीवरी lang:hi", "@Swiggy lang:hi"],
        "ta": ["Swiggy டெலிவரி lang:ta", "@Swiggy lang:ta"],
        "bn": ["Swiggy ডেলিভারি lang:bn"],
        "te": ["Swiggy డెలివరీ lang:te"],
        "mr": ["Swiggy डिलिव्हरी lang:mr"],
        "code_mix": ["Swiggy delivery boy amazing", "Swiggy food cold"],
    },
    "Tata": {
        "hi": ["Tata Motors गाड़ी lang:hi", "TataMotors lang:hi"],
        "ta": ["Tata Motors கார் lang:ta"],
        "bn": ["Tata Motors গাড়ি lang:bn"],
        "te": ["Tata Motors కార్ lang:te"],
        "mr": ["Tata Motors गाडी lang:mr"],
        "code_mix": ["Tata Nexon best car India", "Tata electric vehicle"],
    },
    "HDFC": {
        "hi": ["HDFC बैंक lang:hi", "@HDFCBank lang:hi"],
        "ta": ["HDFC வங்கி lang:ta"],
        "bn": ["HDFC ব্যাংক lang:bn"],
        "te": ["HDFC బ్యాంక్ lang:te"],
        "mr": ["HDFC बँक lang:mr"],
        "code_mix": ["HDFC card blocked", "HDFC net banking down"],
    },
    "Airtel": {
        "hi": ["Airtel नेटवर्क lang:hi", "@Airtel_presence lang:hi"],
        "ta": ["Airtel நெட்வொர்க் lang:ta"],
        "bn": ["Airtel নেটওয়ার্ক lang:bn"],
        "te": ["Airtel నెట్‌వర్క్ lang:te"],
        "mr": ["Airtel नेटवर्क lang:mr"],
        "code_mix": ["Airtel 5G speed test", "Airtel recharge expensive"],
    },
}

RAW_DIR = Path("training/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

TWEET_FIELDS = [
    "id", "text", "lang", "created_at",
    "public_metrics", "author_id", "context_annotations",
]
USER_FIELDS = ["id", "name", "username", "public_metrics", "verified"]


def get_twitter_client() -> tweepy.Client:
    """Initialize Tweepy v4 Client with Bearer Token authentication.

    Returns:
        tweepy.Client: Authenticated Twitter API v2 client.

    Raises:
        EnvironmentError: If TWITTER_BEARER_TOKEN is not set.
    """
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise EnvironmentError(
            "TWITTER_BEARER_TOKEN not set. Add it to your .env file."
        )
    return tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)


def build_query(brand: str, lang: str, query_template: str) -> str:
    """Build a Twitter search query with safety filters.

    Args:
        brand: Brand name (e.g., "Jio").
        lang: BCP-47 language code (e.g., "hi").
        query_template: Raw query template string.

    Returns:
        Cleaned query string with spam/bot filters appended.
    """
    filters = "-is:retweet -is:reply has:lang"
    if "lang:" not in query_template:
        query_template = f"{query_template} lang:{lang}"
    return f"({query_template}) {filters}"


def stream_tweets(
    client: tweepy.Client,
    query: str,
    max_results: int = 100,
) -> Iterator[dict]:
    """Stream tweets via paginated Twitter API v2 search.

    Args:
        client: Authenticated Tweepy client.
        query: Twitter search query string.
        max_results: Maximum total tweets to fetch.

    Yields:
        Tweet dict with text, metadata, and author info.
    """
    fetched = 0
    page_size = min(100, max_results)

    for page in tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=TWEET_FIELDS,
        user_fields=USER_FIELDS,
        expansions=["author_id"],
        max_results=page_size,
        limit=max(1, max_results // page_size),
    ):
        if not page.data:
            break

        # Build user lookup by id
        users_by_id: dict[str, dict] = {}
        if page.includes and "users" in page.includes:
            for user in page.includes["users"]:
                users_by_id[str(user.id)] = {
                    "followers_count": user.public_metrics.get("followers_count", 0),
                    "verified": getattr(user, "verified", False),
                }

        for tweet in page.data:
            metrics = tweet.public_metrics or {}
            author_data = users_by_id.get(str(tweet.author_id), {})
            record = {
                "tweet_id": str(tweet.id),
                "text": tweet.text,
                "lang": tweet.lang,
                "created_at": (
                    tweet.created_at.isoformat()
                    if tweet.created_at
                    else datetime.now(timezone.utc).isoformat()
                ),
                "like_count": metrics.get("like_count", 0),
                "retweet_count": metrics.get("retweet_count", 0),
                "reply_count": metrics.get("reply_count", 0),
                "author_followers": author_data.get("followers_count", 0),
                "verified": author_data.get("verified", False),
            }
            yield record
            fetched += 1
            if fetched >= max_results:
                return


def scrape_brand_language(
    client: tweepy.Client,
    brand: str,
    lang: str,
    max_per_query: int = 500,
) -> list[dict]:
    """Scrape all query variants for a given brand-language pair.

    Args:
        client: Authenticated Tweepy client.
        brand: Brand name.
        lang: Language code or "code_mix".
        max_per_query: Max tweets per query template.

    Returns:
        Deduplicated list of tweet dicts.
    """
    queries = BRAND_QUERIES.get(brand, {}).get(lang, [])
    if not queries:
        logger.warning("No queries defined for brand=%s lang=%s", brand, lang)
        return []

    all_tweets: dict[str, dict] = {}
    out_lang = "en" if lang == "code_mix" else lang

    for template in queries:
        query = build_query(brand, out_lang, template)
        logger.info("Query: %s", query)
        try:
            for tweet in stream_tweets(client, query, max_results=max_per_query):
                tweet["brand"] = brand
                tweet["target_lang"] = lang
                all_tweets[tweet["tweet_id"]] = tweet
        except tweepy.TweepyException as exc:
            logger.error("Twitter API error for query '%s': %s", query, exc)
            time.sleep(15)

    return list(all_tweets.values())


def save_jsonl(records: list[dict], filepath: Path) -> None:
    """Append records to a JSONL file (one JSON object per line).

    Args:
        records: List of tweet dicts.
        filepath: Destination file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), filepath)


def scrape_all(max_per_query: int = 500) -> None:
    """Scrape all brand × language combinations and save as JSONL files.

    Args:
        max_per_query: Max tweets per individual search query.
    """
    client = get_twitter_client()
    all_langs = list(SUPPORTED_LANGUAGES.keys()) + ["code_mix"]

    for brand in BRANDS:
        for lang in all_langs:
            logger.info("─── Scraping brand=%s lang=%s ───", brand, lang)
            tweets = scrape_brand_language(client, brand, lang, max_per_query)
            if tweets:
                fname = f"{brand.lower().replace(\"'\", '')}_{lang}.jsonl"
                save_jsonl(tweets, RAW_DIR / fname)
                logger.info(
                    "brand=%s lang=%s collected=%d unique tweets",
                    brand, lang, len(tweets),
                )
            time.sleep(1)  # polite rate-limit buffer


def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Scrape Indian brand tweets via Twitter API v2"
    )
    parser.add_argument("--brand", type=str, help="Single brand to scrape")
    parser.add_argument(
        "--lang",
        type=str,
        choices=list(SUPPORTED_LANGUAGES.keys()) + ["code_mix"],
        help="Language code",
    )
    parser.add_argument("--max_results", type=int, default=500)
    parser.add_argument("--all", action="store_true", help="Scrape all combinations")
    args = parser.parse_args()

    if args.all:
        scrape_all(args.max_results)
    elif args.brand and args.lang:
        client = get_twitter_client()
        tweets = scrape_brand_language(client, args.brand, args.lang, args.max_results)
        fname = f"{args.brand.lower()}_{args.lang}.jsonl"
        save_jsonl(tweets, RAW_DIR / fname)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
