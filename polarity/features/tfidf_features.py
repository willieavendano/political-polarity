"""TF-IDF feature extraction and discriminative phrase analysis."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif

logger = logging.getLogger(__name__)


def build_tfidf_vectorizer(
    texts: list[str],
    max_features: int = 30000,
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 3,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    stop_words: Optional[str] = "english",
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Build and fit a TF-IDF vectorizer.

    Args:
        texts: List of document texts.
        max_features: Maximum number of features.
        ngram_range: N-gram range (e.g., (1, 3) for unigrams through trigrams).
        min_df: Minimum document frequency.
        max_df: Maximum document frequency ratio.
        sublinear_tf: Use sublinear TF scaling (1 + log(tf)).
        stop_words: Stop words setting ('english' or None).

    Returns:
        (fitted vectorizer, TF-IDF matrix)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
        dtype=np.float32,
    )

    X = vectorizer.fit_transform(texts)
    logger.info(
        "TF-IDF: %d documents x %d features (ngram_range=%s)",
        X.shape[0], X.shape[1], ngram_range,
    )
    return vectorizer, X


def extract_discriminative_phrases(
    X: csr_matrix,
    y: np.ndarray,
    feature_names: list[str],
    method: str = "chi2",
    top_k: int = 50,
    class_names: list[str] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Find the most discriminative phrases per class.

    Args:
        X: TF-IDF feature matrix.
        y: Integer class labels.
        feature_names: Feature names from vectorizer.
        method: 'chi2' or 'mutual_info'.
        top_k: Number of top phrases per class.
        class_names: Optional class name mapping.

    Returns:
        Dictionary mapping class name -> list of (phrase, score).
    """
    classes = sorted(np.unique(y))
    if class_names is None:
        class_names = [str(c) for c in classes]

    results: dict[str, list[tuple[str, float]]] = {}

    if method == "chi2":
        # Chi-square per class (one-vs-rest)
        for cls, cls_name in zip(classes, class_names):
            y_binary = (y == cls).astype(int)
            scores, pvalues = chi2(X, y_binary)
            top_idx = np.argsort(scores)[::-1][:top_k]
            results[cls_name] = [
                (feature_names[i], float(scores[i]))
                for i in top_idx
                if scores[i] > 0
            ]
    elif method == "mutual_info":
        # Mutual information is computed globally, then we rank per class
        # by looking at which features have highest mean TF-IDF within each class
        mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
        top_mi_idx = np.argsort(mi_scores)[::-1][:top_k * len(classes)]

        for cls, cls_name in zip(classes, class_names):
            mask = y == cls
            class_means = np.asarray(X[mask].mean(axis=0)).flatten()
            # Among top MI features, rank by class mean TF-IDF
            scored = [
                (feature_names[i], float(class_means[i] * mi_scores[i]))
                for i in top_mi_idx
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            results[cls_name] = scored[:top_k]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'chi2' or 'mutual_info'.")

    for cls_name, phrases in results.items():
        logger.info("Top %d phrases for '%s': %s ...", top_k, cls_name,
                     [p[0] for p in phrases[:5]])

    return results


def save_vectorizer(vectorizer: TfidfVectorizer, path: str | Path) -> None:
    """Save fitted vectorizer to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved TF-IDF vectorizer to %s", path)


def load_vectorizer(path: str | Path) -> TfidfVectorizer:
    """Load fitted vectorizer from disk."""
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Loaded TF-IDF vectorizer from %s", path)
    return vectorizer
