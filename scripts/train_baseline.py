"""Train baseline model (Logistic Regression / SVM on TF-IDF)."""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

from polarity.utils.config import load_config, ensure_dirs
from polarity.utils.reproducibility import set_seed, artifact_path
from polarity.utils.logging_setup import setup_logging
from polarity.features.tfidf_features import (
    build_tfidf_vectorizer,
    extract_discriminative_phrases,
    save_vectorizer,
)
from polarity.features.keyword_extraction import extract_corpus_keywords
from polarity.models.baseline import BaselineModel
from polarity.evaluation.metrics import compute_metrics, calibration_curve_data, error_analysis
from polarity.evaluation.plots import (
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_class_distribution,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train baseline polarity model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
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

    # Build TF-IDF features
    logger.info("=== Building TF-IDF features ===")
    tfidf_config = config.get("features", {}).get("tfidf", {})
    stop_words = "english" if config.get("preprocessing", {}).get(
        "stopword_removal_classic", True
    ) else None

    vectorizer, X_train = build_tfidf_vectorizer(
        train_df["text"].tolist(),
        max_features=tfidf_config.get("max_features", 30000),
        ngram_range=tuple(tfidf_config.get("ngram_range", [1, 3])),
        min_df=tfidf_config.get("min_df", 3),
        max_df=tfidf_config.get("max_df", 0.95),
        sublinear_tf=tfidf_config.get("sublinear_tf", True),
        stop_words=stop_words,
    )

    X_val = vectorizer.transform(val_df["text"].tolist())
    X_test = vectorizer.transform(test_df["text"].tolist())

    y_train = train_df["label_int"].values
    y_val = val_df["label_int"].values
    y_test = test_df["label_int"].values

    # Save vectorizer
    vec_path = artifact_path(artifacts_dir, "tfidf_vectorizer", ".pkl")
    save_vectorizer(vectorizer, vec_path)

    # Also save as a stable name for inference
    save_vectorizer(vectorizer, Path(artifacts_dir) / "tfidf_vectorizer_latest.pkl")

    # Train model
    logger.info("=== Training baseline model ===")
    bl_config = config.get("baseline", {})
    model = BaselineModel(
        model_type=bl_config.get("model_type", "logistic_regression"),
        C=bl_config.get("C", 1.0),
        max_iter=bl_config.get("max_iter", 1000),
        calibration=bl_config.get("calibration", "sigmoid"),
        class_names=class_names,
        random_seed=config.get("random_seed", 42),
    )
    model.train(X_train, y_train)

    # Save model
    model_path = artifact_path(artifacts_dir, "baseline_model", ".pkl")
    model.save(model_path)
    model.save(Path(artifacts_dir) / "baseline_model_latest.pkl")

    # Evaluate on validation set
    logger.info("=== Validation Results ===")
    val_proba = model.predict_proba(X_val)
    val_pred = np.argmax(val_proba, axis=1)
    val_metrics = compute_metrics(y_val, val_pred, val_proba, class_names)

    # Evaluate on test set
    logger.info("=== Test Results ===")
    test_proba = model.predict_proba(X_test)
    test_pred = np.argmax(test_proba, axis=1)
    test_metrics = compute_metrics(y_test, test_pred, test_proba, class_names)

    # Save metrics
    metrics_path = artifact_path(artifacts_dir, "baseline_metrics", ".json")
    with open(metrics_path, "w") as f:
        json.dump({
            "validation": val_metrics,
            "test": test_metrics,
        }, f, indent=2, default=str)
    logger.info("Saved metrics to %s", metrics_path)

    # Plots
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], class_names,
        artifact_path(artifacts_dir, "baseline_confusion_matrix", ".png"),
    )
    plot_class_distribution(
        y_train, class_names,
        artifact_path(artifacts_dir, "train_class_distribution", ".png"),
    )

    cal_data = calibration_curve_data(y_test, test_proba)
    plot_calibration_curve(
        cal_data,
        artifact_path(artifacts_dir, "baseline_calibration", ".png"),
    )

    # Extract discriminative phrases
    logger.info("=== Extracting discriminative phrases ===")
    feature_names = list(vectorizer.get_feature_names_out())
    kw_config = config.get("features", {}).get("keyword_extraction", {})

    disc_phrases = extract_discriminative_phrases(
        X_train, y_train, feature_names,
        method=kw_config.get("method", "chi2"),
        top_k=kw_config.get("top_k", 20),
        class_names=class_names,
    )

    # Model coefficient phrases
    top_features = model.get_top_features(
        feature_names,
        top_k=bl_config.get("top_features_per_class", 50),
    )

    # Save phrases
    phrases_path = artifact_path(artifacts_dir, "baseline_phrases", ".json")
    with open(phrases_path, "w") as f:
        json.dump({
            "discriminative_phrases": {
                k: [(p, float(s)) for p, s in v]
                for k, v in disc_phrases.items()
            },
            "model_top_features": {
                k: [(p, float(s)) for p, s in v]
                for k, v in top_features.items()
            },
        }, f, indent=2)
    logger.info("Saved phrases to %s", phrases_path)

    # Error analysis
    logger.info("=== Error Analysis ===")
    error_analysis(
        test_df, y_test, test_pred, test_proba, class_names,
        max_save=config.get("evaluation", {}).get("max_misclassified_saved", 200),
        output_path=artifact_path(artifacts_dir, "baseline_errors", ".csv"),
    )

    logger.info("=== Baseline training complete ===")


if __name__ == "__main__":
    main()
