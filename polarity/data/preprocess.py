"""Text preprocessing: deduplication, language filtering, normalization."""

import re
import hashlib
import logging
import unicodedata
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text for dedup."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove duplicate documents based on text hash.

    Args:
        df: DataFrame with text column.
        text_col: Name of text column.

    Returns:
        Deduplicated DataFrame.
    """
    original_len = len(df)
    df = df.copy()
    df["_text_hash"] = df[text_col].apply(compute_text_hash)
    df = df.drop_duplicates(subset="_text_hash", keep="first")
    df = df.drop(columns=["_text_hash"])
    removed = original_len - len(df)
    if removed > 0:
        logger.info("Deduplication removed %d documents (%d remaining)", removed, len(df))
    return df.reset_index(drop=True)


def detect_language(text: str) -> str:
    """Detect language of text, returning ISO code.

    Falls back to 'unknown' if detection fails.
    """
    try:
        from langdetect import detect
        # Use first 1000 chars for speed
        return detect(text[:1000])
    except Exception:
        return "unknown"


def filter_language(
    df: pd.DataFrame,
    target_lang: str = "en",
    text_col: str = "text",
) -> pd.DataFrame:
    """Keep only documents in the target language.

    Args:
        df: DataFrame with text column.
        target_lang: Target language ISO code.
        text_col: Name of text column.

    Returns:
        Filtered DataFrame.
    """
    original_len = len(df)
    df = df.copy()
    df["_lang"] = df[text_col].apply(detect_language)
    df = df[df["_lang"] == target_lang]
    df = df.drop(columns=["_lang"])
    removed = original_len - len(df)
    if removed > 0:
        logger.info(
            "Language filter removed %d non-%s documents (%d remaining)",
            removed, target_lang, len(df),
        )
    return df.reset_index(drop=True)


def normalize_text(text: str) -> str:
    """Normalize unicode, whitespace, and remove some boilerplate patterns.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text.
    """
    # Unicode normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    # Replace various dash/quote characters with ASCII equivalents
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " -- ").replace("\u2013", " - ")

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove common boilerplate patterns
    boilerplate_patterns = [
        r"(?i)subscribe to our newsletter.*",
        r"(?i)click here to read more.*",
        r"(?i)advertisement\s*",
        r"(?i)share this article.*",
        r"(?i)copyright \d{4}.*",
        r"(?i)all rights reserved.*",
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_length(
    df: pd.DataFrame,
    min_length: int = 50,
    max_length: int = 50000,
    text_col: str = "text",
) -> pd.DataFrame:
    """Filter documents by text length.

    Args:
        df: DataFrame with text column.
        min_length: Minimum character count.
        max_length: Maximum character count.
        text_col: Text column name.

    Returns:
        Filtered DataFrame.
    """
    original_len = len(df)
    lengths = df[text_col].str.len()
    mask = (lengths >= min_length) & (lengths <= max_length)
    df = df[mask].reset_index(drop=True)
    removed = original_len - len(df)
    if removed > 0:
        logger.info("Length filter removed %d documents (%d remaining)", removed, len(df))
    return df


def preprocess_pipeline(
    df: pd.DataFrame,
    config: dict,
    text_col: str = "text",
) -> pd.DataFrame:
    """Run full preprocessing pipeline.

    Steps:
        1. Length filter (pre-normalization)
        2. Language filter
        3. Text normalization
        4. Deduplication (post-normalization)
        5. Length filter (post-normalization)

    Args:
        df: Raw document DataFrame.
        config: Config dict (uses 'preprocessing' section).
        text_col: Name of text column.

    Returns:
        Preprocessed DataFrame.
    """
    prep = config.get("preprocessing", {})
    logger.info("Starting preprocessing pipeline on %d documents", len(df))

    # 1. Initial length filter
    min_len = prep.get("min_doc_length", 50)
    max_len = prep.get("max_doc_length", 50000)
    df = filter_length(df, min_length=min_len, max_length=max_len, text_col=text_col)

    # 2. Language filter
    target_lang = prep.get("language", "en")
    df = filter_language(df, target_lang=target_lang, text_col=text_col)

    # 3. Normalize text
    if prep.get("normalize_unicode", True):
        df = df.copy()
        df[text_col] = df[text_col].apply(normalize_text)
        logger.info("Text normalization complete")

    # 4. Deduplication
    if prep.get("dedup_enabled", True):
        df = deduplicate(df, text_col=text_col)

    # 5. Post-normalization length filter
    df = filter_length(df, min_length=min_len, max_length=max_len, text_col=text_col)

    logger.info("Preprocessing complete: %d documents remaining", len(df))
    return df
