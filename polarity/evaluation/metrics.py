"""Evaluation metrics: accuracy, F1, calibration, error analysis."""

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels (integers).
        y_pred: Predicted labels (integers).
        y_proba: Predicted probabilities, shape (n, 3).
        class_names: Class name mapping.

    Returns:
        Dictionary with all metrics.
    """
    class_names = class_names or ["left", "center", "right"]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names)))
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    results = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=class_names,
            labels=list(range(len(class_names))),
        ),
    }

    # Calibration metrics
    if y_proba is not None:
        ece = compute_ece(y_true, y_proba)
        results["expected_calibration_error"] = ece

    logger.info("Accuracy: %.4f, Macro-F1: %.4f", acc, macro_f1)
    logger.info("\n%s", results["classification_report"])

    return results


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        ECE value.
    """
    max_probs = np.max(y_proba, axis=1)
    preds = np.argmax(y_proba, axis=1)
    correct = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (max_probs > lo) & (max_probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = max_probs[mask].mean()
        ece += (mask.sum() / len(max_probs)) * abs(bin_acc - bin_conf)

    return float(ece)


def calibration_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Generate calibration curve data for plotting.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        List of dicts with bin info.
    """
    max_probs = np.max(y_proba, axis=1)
    preds = np.argmax(y_proba, axis=1)
    correct = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (max_probs > lo) & (max_probs <= hi)
        if mask.sum() == 0:
            continue
        bins.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "mean_confidence": float(max_probs[mask].mean()),
            "mean_accuracy": float(correct[mask].mean()),
            "count": int(mask.sum()),
        })

    return bins


def error_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    top_phrases_fn=None,
    max_save: int = 200,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Analyze misclassified examples.

    Args:
        df: Original DataFrame with text and metadata.
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        class_names: Class name mapping.
        top_phrases_fn: Optional function to get top phrases for a text.
        max_save: Maximum misclassified examples to save.
        output_path: Optional path to save analysis.

    Returns:
        DataFrame of misclassified examples with analysis.
    """
    misclassified_mask = y_true != y_pred
    misclassified_idx = np.where(misclassified_mask)[0]

    logger.info(
        "Misclassified: %d / %d (%.1f%%)",
        len(misclassified_idx), len(y_true),
        100 * len(misclassified_idx) / max(len(y_true), 1),
    )

    if len(misclassified_idx) == 0:
        return pd.DataFrame()

    # Limit
    if len(misclassified_idx) > max_save:
        misclassified_idx = misclassified_idx[:max_save]

    records = []
    for idx in misclassified_idx:
        record = {
            "index": int(idx),
            "true_label": class_names[y_true[idx]],
            "pred_label": class_names[y_pred[idx]],
            "confidence": float(np.max(y_proba[idx])),
            "prob_left": float(y_proba[idx][0]),
            "prob_center": float(y_proba[idx][1]),
            "prob_right": float(y_proba[idx][2]),
        }

        # Add text snippet
        if "text" in df.columns:
            text = str(df.iloc[idx]["text"])
            record["text_snippet"] = text[:300]

        if "doc_id" in df.columns:
            record["doc_id"] = str(df.iloc[idx]["doc_id"])

        if "source" in df.columns:
            record["source"] = str(df.iloc[idx]["source"])

        records.append(record)

    errors_df = pd.DataFrame(records)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        errors_df.to_csv(output_path, index=False)
        logger.info("Saved error analysis to %s", output_path)

    return errors_df


def topic_leakage_check(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    class_names: list[str],
    text_col: str = "text",
    top_k: int = 10,
) -> dict[str, list[tuple[str, int]]]:
    """Check for potential topic leakage.

    Identifies if certain topics dominate predictions, which could
    indicate the model is learning topic correlations rather than
    ideological signal.

    Uses simple word frequency analysis per predicted class.

    Args:
        df: DataFrame with text column.
        y_pred: Predicted labels.
        class_names: Class names.
        text_col: Text column.
        top_k: Number of top words to check.

    Returns:
        Dict mapping class -> list of (word, count) for most common words.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    results = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_pred == cls_idx
        if not mask.any():
            results[cls_name] = []
            continue

        texts = df.loc[mask, text_col].tolist()
        cv = CountVectorizer(
            max_features=500, stop_words="english",
            ngram_range=(1, 2), min_df=2,
        )
        try:
            X = cv.fit_transform(texts)
        except ValueError:
            results[cls_name] = []
            continue

        word_counts = X.sum(axis=0).A1
        feature_names = cv.get_feature_names_out()
        top_idx = np.argsort(word_counts)[::-1][:top_k]
        results[cls_name] = [
            (feature_names[i], int(word_counts[i])) for i in top_idx
        ]

    logger.warning(
        "TOPIC LEAKAGE CHECK: Review the top words per class below. "
        "If you see topic-specific words (e.g., 'abortion', 'guns') "
        "dominating, the model may be learning topic rather than ideology."
    )
    for cls_name, words in results.items():
        logger.info("  %s: %s", cls_name, [w[0] for w in words[:5]])

    return results
