"""Evaluate a trained model with comprehensive metrics and robustness checks."""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

from polarity.utils.config import load_config, ensure_dirs
from polarity.utils.reproducibility import set_seed, artifact_path
from polarity.utils.logging_setup import setup_logging
from polarity.models.inference import PolarityInference
from polarity.evaluation.metrics import (
    compute_metrics,
    calibration_curve_data,
    error_analysis,
    topic_leakage_check,
)
from polarity.evaluation.plots import (
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_polarity_distribution,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate polarity model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-type", type=str, default="baseline",
                        choices=["baseline", "transformer"])
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model (default: latest in artifacts)")
    parser.add_argument("--vectorizer-path", type=str, default=None,
                        help="Path to TF-IDF vectorizer (baseline only)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("log_file"),
    )
    set_seed(config.get("random_seed", 42))
    ensure_dirs(config)

    artifacts_dir = config["data"]["artifacts_dir"]
    processed_dir = config["data"]["processed_dir"]
    class_names = config["labels"]["classes"]
    class_to_int = config["labels"]["class_to_int"]

    # Determine paths
    model_path = args.model_path
    vectorizer_path = args.vectorizer_path

    if args.model_type == "baseline":
        if model_path is None:
            model_path = str(Path(artifacts_dir) / "baseline_model_latest.pkl")
        if vectorizer_path is None:
            vectorizer_path = str(Path(artifacts_dir) / "tfidf_vectorizer_latest.pkl")
    else:
        if model_path is None:
            model_path = str(Path(artifacts_dir) / "transformer_model_latest")

    # Load model
    logger.info("=== Loading model: %s ===", args.model_type)
    inference = PolarityInference(
        model_type=args.model_type,
        model_path=model_path,
        vectorizer_path=vectorizer_path if args.model_type == "baseline" else "",
        class_names=class_names,
    )

    # Load eval data
    eval_path = Path(processed_dir) / f"{args.split}.jsonl"
    eval_df = pd.read_json(eval_path, lines=True)
    if "label_int" not in eval_df.columns:
        eval_df["label_int"] = eval_df["label"].map(class_to_int)

    y_true = eval_df["label_int"].values
    texts = eval_df["text"].tolist()

    # Score
    logger.info("=== Scoring %d documents ===", len(texts))
    scored_df = inference.score_dataframe(eval_df)

    y_pred = scored_df["pred_label"].map(class_to_int).values
    y_proba = scored_df[["prob_left", "prob_center", "prob_right"]].values

    # Metrics
    logger.info("=== Computing metrics ===")
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names)

    # Save
    prefix = f"{args.model_type}_{args.split}"
    metrics_path = artifact_path(artifacts_dir, f"{prefix}_eval_metrics", ".json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Plots
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        artifact_path(artifacts_dir, f"{prefix}_confusion_matrix", ".png"),
    )
    cal_data = calibration_curve_data(y_true, y_proba)
    plot_calibration_curve(
        cal_data,
        artifact_path(artifacts_dir, f"{prefix}_calibration", ".png"),
    )
    plot_polarity_distribution(
        scored_df["polarity_score"].values,
        artifact_path(artifacts_dir, f"{prefix}_polarity_dist", ".png"),
    )

    # Error analysis
    logger.info("=== Error analysis ===")
    error_analysis(
        eval_df, y_true, y_pred, y_proba, class_names,
        max_save=config.get("evaluation", {}).get("max_misclassified_saved", 200),
        output_path=artifact_path(artifacts_dir, f"{prefix}_errors", ".csv"),
    )

    # Topic leakage check
    logger.info("=== Topic leakage check ===")
    leakage = topic_leakage_check(eval_df, y_pred, class_names)
    leakage_path = artifact_path(artifacts_dir, f"{prefix}_topic_leakage", ".json")
    with open(leakage_path, "w") as f:
        json.dump(leakage, f, indent=2)

    logger.info("=== Evaluation complete ===")
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro-F1: %.4f", metrics["macro_f1"])
    if "expected_calibration_error" in metrics:
        logger.info("ECE: %.4f", metrics["expected_calibration_error"])


if __name__ == "__main__":
    main()
