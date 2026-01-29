"""Train/validation/test splitting with stratification."""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    label_col: str = "label",
    source_col: str = "source",
    stratify_by_source: bool = True,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test with stratification.

    Stratifies by label. If stratify_by_source is True, also tries to
    stratify by source to reduce outlet leakage (uses combined strata).

    Args:
        df: Labeled DataFrame (only rows with non-null label).
        test_size: Fraction for test set.
        val_size: Fraction for validation set.
        label_col: Label column name.
        source_col: Source column name.
        stratify_by_source: If True, stratify by label+source combined.
        random_seed: Random seed.

    Returns:
        (train_df, val_df, test_df)
    """
    labeled = df[df[label_col].notna()].copy()
    if len(labeled) == 0:
        raise ValueError("No labeled documents to split")

    # Build stratification key
    if stratify_by_source and source_col in labeled.columns:
        labeled["_strat"] = labeled[label_col].astype(str) + "_" + labeled[source_col].astype(str).str.lower()
        # If any stratum has < 2 members, fall back to label-only
        strat_counts = labeled["_strat"].value_counts()
        if (strat_counts < 2).any():
            logger.warning(
                "Some label+source strata have <2 members; falling back to label-only stratification"
            )
            labeled["_strat"] = labeled[label_col]
    else:
        labeled["_strat"] = labeled[label_col]

    # First split: train+val vs test
    train_val, test = train_test_split(
        labeled,
        test_size=test_size,
        stratify=labeled["_strat"],
        random_state=random_seed,
    )

    # Second split: train vs val
    val_frac = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_frac,
        stratify=train_val["_strat"],
        random_state=random_seed,
    )

    # Clean up
    for split_df in [train, val, test]:
        if "_strat" in split_df.columns:
            split_df.drop(columns=["_strat"], inplace=True)

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train), len(val), len(test),
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def temporal_split(
    df: pd.DataFrame,
    split_date: str,
    label_col: str = "label",
    date_col: str = "date",
    val_size: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train on earlier, test on later.

    Args:
        df: Labeled DataFrame with date column.
        split_date: Date string (YYYY-MM-DD) for train/test boundary.
        label_col: Label column.
        date_col: Date column.
        val_size: Fraction of training data for validation.
        random_seed: Random seed.

    Returns:
        (train_df, val_df, test_df)
    """
    labeled = df[df[label_col].notna()].copy()
    if date_col not in labeled.columns:
        raise ValueError(f"Date column '{date_col}' not found")

    labeled[date_col] = pd.to_datetime(labeled[date_col], errors="coerce")
    labeled = labeled.dropna(subset=[date_col])

    train_val = labeled[labeled[date_col] < split_date].reset_index(drop=True)
    test = labeled[labeled[date_col] >= split_date].reset_index(drop=True)

    if len(train_val) == 0 or len(test) == 0:
        raise ValueError(
            f"Temporal split at {split_date} produced empty sets. "
            f"train_val={len(train_val)}, test={len(test)}"
        )

    # Split train_val into train and val
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val[label_col],
        random_state=random_seed,
    )

    logger.info(
        "Temporal split at %s: train=%d, val=%d, test=%d",
        split_date, len(train), len(val), len(test),
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
