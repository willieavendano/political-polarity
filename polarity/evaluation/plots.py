"""Visualization utilities for evaluation and reporting."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray | list,
    class_names: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix.

    Args:
        cm: Confusion matrix array.
        class_names: Class names for axes.
        output_path: Path to save plot.
        title: Plot title.
    """
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", output_path)


def plot_calibration_curve(
    bins_data: list[dict],
    output_path: str | Path,
    title: str = "Calibration Curve",
) -> None:
    """Plot calibration curve.

    Args:
        bins_data: List of dicts with mean_confidence, mean_accuracy, count.
        output_path: Path to save plot.
        title: Plot title.
    """
    if not bins_data:
        logger.warning("No calibration bins to plot")
        return

    confidences = [b["mean_confidence"] for b in bins_data]
    accuracies = [b["mean_accuracy"] for b in bins_data]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(confidences, accuracies, "bo-", label="Model")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration curve to %s", output_path)


def plot_polarity_distribution(
    scores: np.ndarray,
    output_path: str | Path,
    title: str = "Polarity Score Distribution",
    subset_name: str | None = None,
) -> None:
    """Plot distribution of polarity scores.

    Args:
        scores: Polarity scores, range [-1, +1].
        output_path: Path to save plot.
        title: Plot title.
        subset_name: Optional subset name for title.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if subset_name:
        title = f"{title} ({subset_name})"

    ax.hist(scores, bins=40, range=(-1, 1), color="#4a86c8", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Center")
    ax.axvline(np.mean(scores), color="red", linestyle="-", linewidth=2,
               label=f"Mean: {np.mean(scores):.3f}")
    ax.axvline(np.median(scores), color="green", linestyle="-.", linewidth=2,
               label=f"Median: {np.median(scores):.3f}")

    ax.set_xlabel("Polarity Score (Left ← → Right)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved polarity distribution plot to %s", output_path)


def plot_class_distribution(
    labels: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    title: str = "Class Distribution",
) -> None:
    """Plot bar chart of class distribution.

    Args:
        labels: Integer class labels.
        class_names: Class name mapping.
        output_path: Path to save plot.
        title: Plot title.
    """
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2196F3", "#9E9E9E", "#F44336"]  # blue, gray, red
    bars = ax.bar(
        [class_names[i] for i in unique],
        counts,
        color=[colors[i] for i in unique],
        edgecolor="black",
    )

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(count), ha="center", va="bottom", fontweight="bold",
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved class distribution plot to %s", output_path)


def plot_training_history(
    history: dict,
    output_path: str | Path,
    title: str = "Training History",
) -> None:
    """Plot training/validation loss and accuracy curves.

    Args:
        history: Dict with train_loss, val_loss, val_accuracy lists.
        output_path: Path to save plot.
        title: Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()

    if "val_accuracy" in history:
        ax2.plot(epochs, history["val_accuracy"], "g-o", label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"{title} - Validation Accuracy")
        ax2.legend()

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training history plot to %s", output_path)
