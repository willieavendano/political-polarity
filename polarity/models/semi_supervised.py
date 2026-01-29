"""Tier 3: Semi-supervised label refinement with safeguards."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class SemiSupervisedLabeler:
    """Propose additional labels using model confidence + consistency.

    Safeguards against drift:
    - Only labels documents with high confidence (above threshold)
    - Uses multiple rounds of consistency checking
    - Maintains a holdout set for calibration monitoring
    - Caps the number of new labels per round

    WARNING: Semi-supervised expansion can amplify existing biases in the
    initial labels. Use with caution and always verify a sample manually.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        consistency_rounds: int = 3,
        holdout_fraction: float = 0.1,
        max_new_labels_per_round: int = 500,
        random_seed: int = 42,
    ):
        self.confidence_threshold = confidence_threshold
        self.consistency_rounds = consistency_rounds
        self.holdout_fraction = holdout_fraction
        self.max_new_labels_per_round = max_new_labels_per_round
        self.random_seed = random_seed

    def propose_labels(
        self,
        predict_proba_fn,
        unlabeled_texts: list[str],
        unlabeled_doc_ids: list[str],
        class_names: list[str],
    ) -> pd.DataFrame:
        """Propose labels for unlabeled documents.

        Args:
            predict_proba_fn: Callable that takes texts -> probabilities array.
            unlabeled_texts: Texts without labels.
            unlabeled_doc_ids: Document IDs for unlabeled texts.
            class_names: Class name list.

        Returns:
            DataFrame with columns: doc_id, proposed_label, confidence,
            entropy, consistent (bool).
        """
        if not unlabeled_texts:
            logger.info("No unlabeled texts to process")
            return pd.DataFrame()

        logger.info("Semi-supervised: evaluating %d unlabeled documents", len(unlabeled_texts))

        # Multiple prediction rounds for consistency
        all_preds = []
        for round_num in range(self.consistency_rounds):
            probs = predict_proba_fn(unlabeled_texts)
            preds = np.argmax(probs, axis=1)
            all_preds.append(preds)

        all_preds = np.array(all_preds)  # (rounds, n_docs)

        # Final probabilities (from last round)
        final_probs = predict_proba_fn(unlabeled_texts)
        max_probs = np.max(final_probs, axis=1)
        entropies = entropy(final_probs, axis=1)
        predicted_classes = np.argmax(final_probs, axis=1)

        # Consistency: all rounds agree
        consistent = np.all(all_preds == all_preds[0:1], axis=0)

        # Build proposals
        proposals = []
        for i in range(len(unlabeled_texts)):
            proposals.append({
                "doc_id": unlabeled_doc_ids[i],
                "proposed_label": class_names[predicted_classes[i]],
                "confidence": float(max_probs[i]),
                "entropy": float(entropies[i]),
                "consistent": bool(consistent[i]),
            })

        df = pd.DataFrame(proposals)

        # Filter: high confidence AND consistent
        accepted = df[
            (df["confidence"] >= self.confidence_threshold) & df["consistent"]
        ].copy()

        # Cap the number
        if len(accepted) > self.max_new_labels_per_round:
            accepted = accepted.nlargest(self.max_new_labels_per_round, "confidence")

        logger.info(
            "Semi-supervised: %d/%d documents meet threshold (conf>=%.2f, consistent)",
            len(accepted), len(df), self.confidence_threshold,
        )

        if len(accepted) > 0:
            dist = accepted["proposed_label"].value_counts()
            logger.info("Proposed label distribution:\n%s", dist.to_string())

        logger.warning(
            "CAUTION: Semi-supervised labels may amplify biases in the "
            "initial training data. Always verify a random sample manually."
        )

        return accepted

    def calibration_check(
        self,
        predict_proba_fn,
        holdout_texts: list[str],
        holdout_labels: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """Check calibration on holdout set to detect drift.

        Args:
            predict_proba_fn: Prediction function.
            holdout_texts: Holdout texts.
            holdout_labels: Holdout true labels.
            n_bins: Number of calibration bins.

        Returns:
            Dict with ECE and per-bin calibration data.
        """
        probs = predict_proba_fn(holdout_texts)
        max_probs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        correct = (preds == holdout_labels).astype(float)

        # Expected Calibration Error
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bins_data = []

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (max_probs > lo) & (max_probs <= hi)
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = max_probs[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / len(max_probs)) * abs(bin_acc - bin_conf)
            bins_data.append({
                "bin": f"({lo:.2f}, {hi:.2f}]",
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "count": int(bin_size),
            })

        logger.info("Calibration check ECE: %.4f", ece)
        return {"ece": float(ece), "bins": bins_data}
