"""Export trained baseline model to JSON for browser-side inference.

Extracts TF-IDF vocabulary + IDF weights and LogReg coefficients
so the model can run entirely in JavaScript with no server.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from polarity.utils.config import load_config
from polarity.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export model to JSON for JS inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="docs/model.json",
                        help="Output JSON path (default: docs/model.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level="INFO")
    artifacts_dir = config["data"]["artifacts_dir"]

    # Load vectorizer
    vec_path = Path(artifacts_dir) / "tfidf_vectorizer_latest.pkl"
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Load model
    model_path = Path(artifacts_dir) / "baseline_model_latest.pkl"
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]

    # Extract vocabulary: word -> index (convert numpy int64 to Python int)
    vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    # IDF weights
    idf = vectorizer.idf_.tolist()

    # Extract LogReg weights
    coef = model.coef_.tolist()       # shape (3, n_features)
    intercept = model.intercept_.tolist()  # shape (3,)

    # N-gram range and other params
    ngram_range = list(vectorizer.ngram_range)
    sublinear_tf = bool(vectorizer.sublinear_tf)

    # Stop words used during fitting
    raw_sw = getattr(vectorizer, "stop_words_", None) or set()
    stop_words = sorted(list(raw_sw))

    # Build export
    export = {
        "class_names": ["left", "center", "right"],
        "vocab": vocab,
        "idf": idf,
        "coef": coef,
        "intercept": intercept,
        "ngram_range": ngram_range,
        "sublinear_tf": sublinear_tf,
        "stop_words": stop_words,
        "n_features": len(idf),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(export, f)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Exported model to %s (%.2f MB)", out_path, size_mb)
    logger.info("Vocabulary size: %d, Features: %d", len(vocab), len(idf))


if __name__ == "__main__":
    main()
