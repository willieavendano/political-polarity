"""Train transformer model for polarity classification."""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

from polarity.utils.config import load_config, ensure_dirs
from polarity.utils.reproducibility import set_seed, artifact_path
from polarity.utils.logging_setup import setup_logging
from polarity.models.transformer_model import TransformerPolarityModel
from polarity.evaluation.metrics import compute_metrics, calibration_curve_data, error_analysis
from polarity.evaluation.plots import (
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_training_history,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train transformer polarity model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("log_file"),
    )
    set_seed(config.get("random_seed", 42))
    ensure_dirs(config)

    processed_dir = config["data"]["processed_dir"]
    artifacts_dir = config["data"]["artifacts_dir"]
    tf_config = config.get("transformer", {})

    # Load splits
    logger.info("=== Loading data splits ===")
    train_df = pd.read_json(Path(processed_dir) / "train.jsonl", lines=True)
    val_df = pd.read_json(Path(processed_dir) / "val.jsonl", lines=True)
    test_df = pd.read_json(Path(processed_dir) / "test.jsonl", lines=True)

    class_names = config["labels"]["classes"]
    class_to_int = config["labels"]["class_to_int"]

    for split_df in [train_df, val_df, test_df]:
        if "label_int" not in split_df.columns:
            split_df["label_int"] = split_df["label"].map(class_to_int)

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label_int"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label_int"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label_int"].values

    # Initialize model
    logger.info("=== Initializing transformer model ===")
    model = TransformerPolarityModel(
        model_name=tf_config.get("model_name", "distilbert-base-uncased"),
        num_classes=len(class_names),
        class_names=class_names,
        max_length=tf_config.get("max_length", 512),
        device=args.device,
    )

    # Train
    logger.info("=== Training transformer ===")
    history = model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        batch_size=tf_config.get("batch_size", 16),
        learning_rate=tf_config.get("learning_rate", 2e-5),
        weight_decay=tf_config.get("weight_decay", 0.01),
        num_epochs=tf_config.get("num_epochs", 5),
        early_stopping_patience=tf_config.get("early_stopping_patience", 2),
        warmup_ratio=tf_config.get("warmup_ratio", 0.1),
        cpu_friendly=tf_config.get("cpu_friendly", True),
        cpu_batch_size=tf_config.get("cpu_batch_size", 4),
        cpu_num_epochs=tf_config.get("cpu_num_epochs", 2),
    )

    # Save model
    model_dir = Path(artifacts_dir) / "transformer_model_latest"
    model.save(model_dir)
    logger.info("Saved transformer model to %s", model_dir)

    # Plot training history
    plot_training_history(
        history,
        artifact_path(artifacts_dir, "transformer_training_history", ".png"),
    )

    # Evaluate on test set
    logger.info("=== Test Results ===")
    batch_size = tf_config.get("cpu_batch_size", 4) if model.device.type == "cpu" else tf_config.get("batch_size", 16)
    test_proba = model.predict_proba(test_texts, batch_size=batch_size)
    test_pred = np.argmax(test_proba, axis=1)
    test_metrics = compute_metrics(test_labels, test_pred, test_proba, class_names)

    # Save metrics
    metrics_path = artifact_path(artifacts_dir, "transformer_metrics", ".json")
    with open(metrics_path, "w") as f:
        json.dump({
            "test": test_metrics,
            "training_history": history,
        }, f, indent=2, default=str)
    logger.info("Saved metrics to %s", metrics_path)

    # Plots
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], class_names,
        artifact_path(artifacts_dir, "transformer_confusion_matrix", ".png"),
    )
    cal_data = calibration_curve_data(test_labels, test_proba)
    plot_calibration_curve(
        cal_data,
        artifact_path(artifacts_dir, "transformer_calibration", ".png"),
    )

    # Error analysis
    logger.info("=== Error Analysis ===")
    error_analysis(
        test_df, test_labels, test_pred, test_proba, class_names,
        max_save=config.get("evaluation", {}).get("max_misclassified_saved", 200),
        output_path=artifact_path(artifacts_dir, "transformer_errors", ".csv"),
    )

    # Interpretability example
    logger.info("=== Interpretability Example ===")
    interp_config = tf_config.get("interpretability", {})
    method = interp_config.get("method", "leave_one_out")
    top_k = interp_config.get("top_k_tokens", 15)

    if len(test_texts) > 0:
        sample_text = test_texts[0]
        importance = model.get_token_importance(sample_text, method=method, top_k=top_k)
        logger.info("Sample interpretability (%s):", method)
        for token, score in importance[:10]:
            logger.info("  %s: %.4f", token, score)

    logger.info("=== Transformer training complete ===")


if __name__ == "__main__":
    main()
