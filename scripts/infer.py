"""Run inference on new documents."""

import argparse
import logging
import json
import sys
from pathlib import Path

import pandas as pd

from polarity.utils.config import load_config, ensure_dirs
from polarity.utils.reproducibility import set_seed
from polarity.utils.logging_setup import setup_logging
from polarity.models.inference import PolarityInference

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run polarity inference on new documents")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-type", type=str, default="baseline",
                        choices=["baseline", "transformer"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--vectorizer-path", type=str, default=None)
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL file (or '-' for stdin text)")
    parser.add_argument("--text", type=str, default=None,
                        help="Direct text input for single-document scoring")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file")
    parser.add_argument("--show-interpretability", action="store_true",
                        help="Show feature importance for each document")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("log_file"),
    )
    set_seed(config.get("random_seed", 42))

    artifacts_dir = config["data"]["artifacts_dir"]
    class_names = config["labels"]["classes"]

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
    inference = PolarityInference(
        model_type=args.model_type,
        model_path=model_path,
        vectorizer_path=vectorizer_path if args.model_type == "baseline" else "",
        class_names=class_names,
    )

    # Single text mode
    if args.text:
        result = inference.score_text(args.text)
        print("\n=== Polarity Analysis ===")
        print(f"Predicted: {result['predicted_label']}")
        print(f"Polarity:  {result['polarity_score']:+.3f} (Left ← → Right)")
        print(f"Confidence:")
        for cls, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"  {cls:>8}: {prob:.3f} {bar}")
        print(f"Entropy:   {result['entropy']:.3f}")

        if args.show_interpretability:
            importance = inference.get_interpretability(args.text)
            print("\nTop contributing features:")
            for feat, score in importance[:10]:
                direction = "→" if score > 0 else "←"
                print(f"  {direction} {feat}: {score:+.4f}")

        print("\n⚠  WARNING: This classifies TEXT CONTENT, not the author's beliefs.")
        print("   Do NOT use to infer an individual's political affiliation.")
        return

    # Batch mode
    if args.input:
        df = pd.read_json(args.input, lines=True)
        scored = inference.score_dataframe(df)

        if args.output:
            scored.to_json(args.output, orient="records", lines=True)
            logger.info("Saved %d scored documents to %s", len(scored), args.output)
        else:
            # Print summary
            print(f"\n=== Scored {len(scored)} documents ===")
            print(f"Mean polarity: {scored['polarity_score'].mean():+.3f}")
            print(f"Label distribution:")
            print(scored["pred_label"].value_counts().to_string())

        return

    # Interactive mode (stdin)
    print("=== Political Polarity Analyzer (Interactive) ===")
    print("Enter text to analyze (Ctrl+D / Ctrl+Z to quit):")
    print("WARNING: This classifies TEXT CONTENT, not individuals.\n")

    try:
        while True:
            text = input("> ").strip()
            if not text:
                continue
            result = inference.score_text(text)
            print(f"  → {result['predicted_label']} "
                  f"(polarity: {result['polarity_score']:+.3f}, "
                  f"L:{result['probabilities']['left']:.2f} "
                  f"C:{result['probabilities']['center']:.2f} "
                  f"R:{result['probabilities']['right']:.2f})")
    except (EOFError, KeyboardInterrupt):
        print("\nDone.")


if __name__ == "__main__":
    main()
