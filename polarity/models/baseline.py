"""Baseline model: Logistic Regression / Linear SVM on TF-IDF features."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class BaselineModel:
    """Calibrated linear model for 3-class polarity classification.

    Supports LogisticRegression (natively calibrated) or LinearSVC
    (calibrated via Platt scaling or isotonic regression).

    Attributes:
        model: The fitted sklearn estimator.
        class_names: List of class names in order.
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",
        C: float = 1.0,
        max_iter: int = 1000,
        calibration: str = "sigmoid",
        class_names: list[str] | None = None,
        random_seed: int = 42,
    ):
        """Initialize baseline model.

        Args:
            model_type: 'logistic_regression' or 'linear_svm'.
            C: Regularization parameter.
            max_iter: Maximum iterations for solver.
            calibration: Calibration method ('sigmoid' or 'isotonic').
            class_names: Ordered class names.
            random_seed: Random seed.
        """
        self.model_type = model_type
        self.calibration_method = calibration
        self.class_names = class_names or ["left", "center", "right"]
        self.model = None
        self._raw_model = None

        if model_type == "logistic_regression":
            self._raw_model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver="lbfgs",
                random_state=random_seed,
                class_weight="balanced",
            )
        elif model_type == "linear_svm":
            base_svm = LinearSVC(
                C=C,
                max_iter=max_iter,
                random_state=random_seed,
                class_weight="balanced",
            )
            self._raw_model = CalibratedClassifierCV(
                base_svm,
                method=calibration,
                cv=3,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, X_train: csr_matrix, y_train: np.ndarray) -> None:
        """Train the model.

        Args:
            X_train: TF-IDF feature matrix.
            y_train: Integer class labels.
        """
        logger.info(
            "Training %s on %d samples with %d features",
            self.model_type, X_train.shape[0], X_train.shape[1],
        )
        self._raw_model.fit(X_train, y_train)
        self.model = self._raw_model
        logger.info("Training complete")

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 3) with P(left), P(center), P(right).
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)

    def predict(self, X: csr_matrix) -> np.ndarray:
        """Predict class labels (integers).

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted class indices.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def polarity_score(self, X: csr_matrix) -> np.ndarray:
        """Compute continuous polarity score: P(right) - P(left).

        Range: [-1, +1] where -1 = fully left, +1 = fully right.

        Args:
            X: Feature matrix.

        Returns:
            Array of polarity scores.
        """
        probs = self.predict_proba(X)
        # Assuming class order: left=0, center=1, right=2
        return probs[:, 2] - probs[:, 0]

    def get_top_features(
        self,
        feature_names: list[str],
        top_k: int = 50,
    ) -> dict[str, list[tuple[str, float]]]:
        """Get top features (coefficients) driving each class.

        Only works for LogisticRegression (has .coef_ attribute).

        Args:
            feature_names: Feature names from vectorizer.
            top_k: Number of top features per class.

        Returns:
            Dictionary mapping class_name -> list of (feature, coefficient).
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        # Get coefficients
        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_
        elif hasattr(self.model, "calibrated_classifiers_"):
            # CalibratedClassifierCV wraps the base estimator
            base = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base, "coef_"):
                coefs = base.coef_
            else:
                logger.warning("Cannot extract coefficients from calibrated SVM")
                return {}
        else:
            logger.warning("Model does not expose coefficients")
            return {}

        results = {}
        for cls_idx, cls_name in enumerate(self.class_names):
            if cls_idx >= coefs.shape[0]:
                continue
            class_coefs = coefs[cls_idx]
            top_idx = np.argsort(class_coefs)[::-1][:top_k]
            results[cls_name] = [
                (feature_names[i], float(class_coefs[i]))
                for i in top_idx
                if class_coefs[i] > 0
            ]

        return results

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "model_type": self.model_type,
                "class_names": self.class_names,
            }, f)
        logger.info("Saved baseline model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.model_type = data["model_type"]
        instance.class_names = data["class_names"]
        instance._raw_model = instance.model
        logger.info("Loaded baseline model from %s", path)
        return instance
