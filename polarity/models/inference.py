"""Inference pipeline: load models and score new documents."""

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from polarity.data.preprocess import normalize_text
from polarity.features.tfidf_features import load_vectorizer
from polarity.models.baseline import BaselineModel

logger = logging.getLogger(__name__)


class PolarityInference:
    """Unified inference interface for baseline and transformer models.

    Loads a trained model and provides document-level polarity predictions.
    """

    def __init__(
        self,
        model_type: str = "baseline",
        model_path: str | Path = "",
        vectorizer_path: str | Path = "",
        class_names: list[str] | None = None,
        device: str | None = None,
    ):
        """Initialize inference pipeline.

        Args:
            model_type: 'baseline' or 'transformer'.
            model_path: Path to saved model.
            vectorizer_path: Path to TF-IDF vectorizer (baseline only).
            class_names: Class name list.
            device: Device for transformer ('cpu' or 'cuda').
        """
        self.model_type = model_type
        self.class_names = class_names or ["left", "center", "right"]
        self.model = None
        self.vectorizer = None

        if model_type == "baseline":
            self.model = BaselineModel.load(model_path)
            self.vectorizer = load_vectorizer(vectorizer_path)
        elif model_type == "transformer":
            from polarity.models.transformer_model import TransformerPolarityModel
            self.model = TransformerPolarityModel.load(model_path, device=device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        logger.info("Inference pipeline ready (model_type=%s)", model_type)

    def score_text(self, text: str) -> dict:
        """Score a single text document.

        Args:
            text: Raw text.

        Returns:
            Dictionary with probabilities, polarity score, and predicted label.
        """
        clean = normalize_text(text)

        if self.model_type == "baseline":
            X = self.vectorizer.transform([clean])
            proba = self.model.predict_proba(X)[0]
        else:
            proba = self.model.predict_proba([clean])[0]

        pred_idx = int(np.argmax(proba))
        polarity = float(proba[2] - proba[0])
        ent = float(-np.sum(proba * np.log(proba + 1e-10)))

        return {
            "predicted_label": self.class_names[pred_idx],
            "probabilities": {
                self.class_names[i]: float(proba[i])
                for i in range(len(self.class_names))
            },
            "polarity_score": polarity,
            "entropy": ent,
        }

    def score_texts(self, texts: list[str]) -> list[dict]:
        """Score multiple texts.

        Args:
            texts: List of raw texts.

        Returns:
            List of score dictionaries.
        """
        return [self.score_text(t) for t in texts]

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Score all documents in a DataFrame.

        Args:
            df: DataFrame with text column.
            text_col: Name of text column.

        Returns:
            DataFrame with prediction columns added.
        """
        df = df.copy()
        texts = df[text_col].apply(normalize_text).tolist()

        if self.model_type == "baseline":
            X = self.vectorizer.transform(texts)
            proba = self.model.predict_proba(X)
        else:
            proba = self.model.predict_proba(texts)

        df["pred_label"] = [self.class_names[i] for i in np.argmax(proba, axis=1)]
        df["prob_left"] = proba[:, 0]
        df["prob_center"] = proba[:, 1]
        df["prob_right"] = proba[:, 2]
        df["polarity_score"] = proba[:, 2] - proba[:, 0]
        df["entropy"] = [-np.sum(p * np.log(p + 1e-10)) for p in proba]

        logger.info("Scored %d documents", len(df))
        return df

    def get_interpretability(
        self,
        text: str,
        method: str = "leave_one_out",
        top_k: int = 15,
    ) -> list[tuple[str, float]]:
        """Get token/feature importance for a text.

        Args:
            text: Input text.
            method: Interpretability method.
            top_k: Number of top features.

        Returns:
            List of (token/feature, importance) tuples.
        """
        if self.model_type == "baseline":
            # For baseline, return top TF-IDF features present in text
            clean = normalize_text(text)
            X = self.vectorizer.transform([clean])
            feature_names = self.vectorizer.get_feature_names_out()

            nonzero = X.nonzero()[1]
            proba = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))

            if hasattr(self.model.model, "coef_"):
                coefs = self.model.model.coef_[pred_idx]
                feature_scores = [
                    (feature_names[j], float(coefs[j] * X[0, j]))
                    for j in nonzero
                ]
            else:
                feature_scores = [
                    (feature_names[j], float(X[0, j]))
                    for j in nonzero
                ]

            feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            return feature_scores[:top_k]
        else:
            return self.model.get_token_importance(text, method=method, top_k=top_k)
