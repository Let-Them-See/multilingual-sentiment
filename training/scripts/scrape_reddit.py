"""
Reddit Scraper for Indian Brand Sentiment Dataset.

Collects posts and comments from Indian subreddits mentioning
target brands. Uses PRAW (Python Reddit API Wrapper).

Usage:
    python scrape_reddit.py --subreddit india --max_posts 500
    python scrape_reddit.py --all
"""

import os
import json
import time
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator

import praw
from praw.models import Submission, Comment
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training/data/raw/scrape_reddit.log"),
    ],
)
logger = logging.getLogger("scrape_reddit")

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_SUBREDDITS: list[str] = [
    "india",
    "bangalore",
    "mumbai",
    "Chennai",
    "hyderabad",
    "IndianMobilephones",
    "india_investments",
    "Zomato",
    "FlipkartOffers",
    "IndiaInvestments",
    "indiasocial",
    "AskIndia",
]

BRANDS: list[str] = [
    "Jio", "Zomato", "Flipkart", "BYJU'S", "Paytm",
    "Ola", "Swiggy", "Tata", "HDFC", "Airtel",
]

# Brand → keyword patterns for mention detection
BRAND_PATTERNS: dict[str, re.Pattern] = {
    brand: re.compile(
        r"\b" + re.escape(brand.replace("'", "")) + r"\b",
        re.IGNORECASE,
    )
    for brand in BRANDS
}

RAW_DIR = Path("training/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

COMMENT_DEPTH_LIMIT = 3
MIN_BODY_LENGTH = 20
MAX_COMMENTS_PER_POST = 20


def get_reddit_client() -> praw.Reddit:
    """Initialize PRAW Reddit client with environment credentials.

    Returns:
        praw.Reddit: Authenticated PRAW client (read-only).

    Raises:
        EnvironmentError: If Reddit credentials are not set.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "IndianSentimentScraper/1.0")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in .env"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        ratelimit_seconds=1,
    )


def detect_brands(text: str) -> list[str]:
    """Identify which target brands are mentioned in a text chunk.

    Args:
        text: Raw post or comment text.

    Returns:
        List of matched brand name strings.
    """
    return [brand for brand, pat in BRAND_PATTERNS.items() if pat.search(text)]


def post_to_record(post: Submission, brands: list[str]) -> dict:
    """Convert a PRAW Submission to a flat metadata dict.

    Args:
        post: PRAW Submission object.
        brands: List of detected brand names in this post.

    Returns:
        Serializable dict with post fields and metadata.
    """
    return {
        "post_id": post.id,
        "subreddit": str(post.subreddit),
        "title": post.title,
        "text": (post.selftext or "").strip(),
        "full_text": f"{post.title}. {post.selftext or ''}".strip(),
        "url": f"https://reddit.com{post.permalink}",
        "score": post.score,
        "upvote_ratio": post.upvote_ratio,
        "num_comments": post.num_comments,
        "created_at": datetime.fromtimestamp(
            post.created_utc, tz=timezone.utc
        ).isoformat(),
        "author": str(post.author) if post.author else "[deleted]",
        "brands": brands,
        "source": "reddit_post",
        "lang": "en",  # Reddit posts are predominantly English/code-mix
    }


def comment_to_record(
    comment: Comment,
    post_id: str,
    subreddit: str,
    brands: list[str],
) -> dict:
    """Convert a PRAW Comment to a flat metadata dict.

    Args:
        comment: PRAW Comment object.
        post_id: Parent post ID.
        subreddit: Subreddit name.
        brands: Detected brand names.

    Returns:
        Serializable dict with comment fields.
    """
    return {
        "post_id": post_id,
        "comment_id": comment.id,
        "subreddit": subreddit,
        "text": comment.body.strip(),
        "full_text": comment.body.strip(),
        "score": comment.score,
        "created_at": datetime.fromtimestamp(
            comment.created_utc, tz=timezone.utc
        ).isoformat(),
        "author": str(comment.author) if comment.author else "[deleted]",
        "brands": brands,
        "source": "reddit_comment",
        "lang": "en",
    }


def iter_subreddit_posts(
    reddit: praw.Reddit,
    subreddit_name: str,
    max_posts: int = 500,
) -> Iterator[dict]:
    """Iterate through hot + top posts from a subreddit, yielding
    post records and their top comments filtered by brand mentions.

    Args:
        reddit: Authenticated PRAW client.
        subreddit_name: Name of subreddit (without r/).
        max_posts: Maximum posts to process.

    Yields:
        Flat dicts for posts and comments containing brand mentions.
    """
    sub = reddit.subreddit(subreddit_name)
    processed = 0

    for category in ("hot", "top", "new"):
        if processed >= max_posts:
            break
        try:
            source = getattr(sub, category)(limit=max_posts // 3)
            for post in source:
                if processed >= max_posts:
                    break
                try:
                    full_text = f"{post.title} {post.selftext or ''}"
                    post_brands = detect_brands(full_text)

                    if not post_brands:
                        continue

                    if len(full_text.strip()) < MIN_BODY_LENGTH:
                        continue

                    yield post_to_record(post, post_brands)
                    processed += 1

                    # Collect top comments
                    post.comments.replace_more(limit=COMMENT_DEPTH_LIMIT)
                    comment_count = 0
                    for comment in post.comments.list():
                        if comment_count >= MAX_COMMENTS_PER_POST:
                            break
                        if (
                            isinstance(comment, Comment)
                            and len(comment.body.strip()) >= MIN_BODY_LENGTH
                            and comment.body not in ("[deleted]", "[removed]")
                        ):
                            comment_brands = detect_brands(comment.body)
                            if comment_brands:
                                yield comment_to_record(
                                    comment, post.id,
                                    subreddit_name, comment_brands,
                                )
                                comment_count += 1

                except praw.exceptions.PRAWException as exc:
                    logger.warning("Skipping post %s: %s", getattr(post, 'id', '?'), exc)

        except praw.exceptions.PRAWException as exc:
            logger.error("Error accessing r/%s (%s): %s", subreddit_name, category, exc)
            time.sleep(5)


def scrape_all(max_posts_per_subreddit: int = 500) -> None:
    """Scrape all target subreddits and save to per-subreddit JSONL files.

    Args:
        max_posts_per_subreddit: Max posts to process per subreddit.
    """
    reddit = get_reddit_client()

    for subreddit in TARGET_SUBREDDITS:
        logger.info("─── Scraping r/%s ───", subreddit)
        records: list[dict] = []

        for record in iter_subreddit_posts(reddit, subreddit, max_posts_per_subreddit):
            records.append(record)

        if records:
            filepath = RAW_DIR / f"reddit_{subreddit}.jsonl"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("r/%s → saved %d records to %s", subreddit, len(records), filepath)
        else:
            logger.warning("r/%s → no brand-mentioning posts found", subreddit)

        time.sleep(2)  # polite pause between subreddits


def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Scrape Indian brand mentions from Reddit"
    )
    parser.add_argument(
        "--subreddit",
        type=str,
        help="Single subreddit name (e.g., india)",
    )
    parser.add_argument("--max_posts", type=int, default=500)
    parser.add_argument("--all", action="store_true", help="Scrape all subreddits")
    args = parser.parse_args()

    if args.all:
        scrape_all(args.max_posts)
    elif args.subreddit:
        reddit = get_reddit_client()
        records = list(
            iter_subreddit_posts(reddit, args.subreddit, args.max_posts)
        )
        filepath = RAW_DIR / f"reddit_{args.subreddit}.jsonl"
        with open(filepath, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Saved %d records → %s", len(records), filepath)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
