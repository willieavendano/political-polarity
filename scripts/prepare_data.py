"""Data preparation: ingest, preprocess, label, and split."""

import argparse
import logging
import json
from pathlib import Path

import pandas as pd

from polarity.utils.config import load_config, ensure_dirs
from polarity.utils.reproducibility import set_seed
from polarity.utils.logging_setup import setup_logging
from polarity.data.ingest import load_jsonl, load_jsonl_dir, raw_text_to_jsonl, raw_html_to_jsonl
from polarity.data.preprocess import preprocess_pipeline
from polarity.data.labeling import load_outlet_bias, load_human_labels, assign_labels
from polarity.data.splitting import stratified_split, temporal_split

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare data for polarity analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--convert-txt", type=str, default=None,
                        help="Convert .txt files from this directory to JSONL first")
    parser.add_argument("--convert-html", type=str, default=None,
                        help="Convert .html files from this directory to JSONL first")
    parser.add_argument("--skip-lang-filter", action="store_true",
                        help="Skip language detection (faster)")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("log_file"),
    )
    set_seed(config.get("random_seed", 42))
    ensure_dirs(config)

    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]

    # Step 0: Convert raw files if requested
    if args.convert_txt:
        raw_text_to_jsonl(args.convert_txt, Path(raw_dir) / "converted_txt.jsonl")

    if args.convert_html:
        raw_html_to_jsonl(args.convert_html, Path(raw_dir) / "converted_html.jsonl")

    # Step 1: Load JSONL data
    logger.info("=== Step 1: Loading data ===")
    df = load_jsonl_dir(raw_dir)
    if len(df) == 0:
        logger.error("No data found in %s. Run generate_synthetic first.", raw_dir)
        return

    # Step 2: Preprocess
    logger.info("=== Step 2: Preprocessing ===")
    if args.skip_lang_filter:
        config["preprocessing"]["language"] = None
        # Temporarily disable lang filter
        from polarity.data import preprocess as pp
        original_filter = pp.filter_language
        pp.filter_language = lambda df, **kwargs: df
        df = preprocess_pipeline(df, config)
        pp.filter_language = original_filter
    else:
        df = preprocess_pipeline(df, config)

    # Step 3: Assign labels
    logger.info("=== Step 3: Assigning labels ===")
    outlet_bias_file = config["data"].get("outlet_bias_file", "")
    human_labels_file = config["data"].get("human_labels_file", "")

    outlet_bias = load_outlet_bias(outlet_bias_file) if outlet_bias_file else {}
    human_labels = load_human_labels(human_labels_file) if human_labels_file else {}

    df = assign_labels(df, outlet_bias, human_labels)

    # Step 4: Split
    logger.info("=== Step 4: Splitting data ===")
    eval_config = config.get("evaluation", {})

    labeled_df = df[df["label"].notna()].copy()
    unlabeled_df = df[df["label"].isna()].copy()

    if len(labeled_df) == 0:
        logger.error("No labeled documents. Provide outlet_bias.csv or human_labels.csv.")
        return

    # Encode labels
    class_to_int = config["labels"]["class_to_int"]
    labeled_df["label_int"] = labeled_df["label"].map(class_to_int)

    if eval_config.get("temporal_split", False):
        split_date = eval_config.get("temporal_split_date", "2024-06-01")
        train_df, val_df, test_df = temporal_split(
            labeled_df,
            split_date=split_date,
            val_size=eval_config.get("val_size", 0.15),
            random_seed=config.get("random_seed", 42),
        )
    else:
        train_df, val_df, test_df = stratified_split(
            labeled_df,
            test_size=eval_config.get("test_size", 0.15),
            val_size=eval_config.get("val_size", 0.15),
            stratify_by_source=eval_config.get("stratify_by_source", True),
            random_seed=config.get("random_seed", 42),
        )

    # Save processed data
    processed_path = Path(processed_dir)
    train_df.to_json(processed_path / "train.jsonl", orient="records", lines=True)
    val_df.to_json(processed_path / "val.jsonl", orient="records", lines=True)
    test_df.to_json(processed_path / "test.jsonl", orient="records", lines=True)

    if len(unlabeled_df) > 0:
        unlabeled_df.to_json(
            processed_path / "unlabeled.jsonl", orient="records", lines=True
        )

    # Save full processed dataset
    df.to_json(processed_path / "all_processed.jsonl", orient="records", lines=True)

    logger.info("=== Data preparation complete ===")
    logger.info("Train: %d, Val: %d, Test: %d, Unlabeled: %d",
                len(train_df), len(val_df), len(test_df), len(unlabeled_df))
    logger.info("Saved to: %s", processed_path)


if __name__ == "__main__":
    main()
