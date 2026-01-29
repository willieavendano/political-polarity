"""Generate subset aggregation reports with privacy safeguards."""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

from polarity.utils.config import load_config, ensure_dirs
from polarity.utils.reproducibility import set_seed, artifact_path
from polarity.utils.logging_setup import setup_logging
from polarity.models.inference import PolarityInference
from polarity.reporting.subset_analysis import (
    compute_subset_report,
    save_subset_report,
    format_subset_report,
)
from polarity.evaluation.plots import plot_polarity_distribution

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate aggregated polarity reports by subset"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-type", type=str, default="baseline",
                        choices=["baseline", "transformer"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--vectorizer-path", type=str, default=None)
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL (default: all_processed.jsonl)")
    parser.add_argument("--group-by", type=str, nargs="+", default=None,
                        help="Metadata fields to group by")
    parser.add_argument("--k", type=int, default=None,
                        help="k-anonymity threshold (default from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("log_file"),
    )
    set_seed(config.get("random_seed", 42))
    ensure_dirs(config)

    artifacts_dir = config["data"]["artifacts_dir"]
    processed_dir = config["data"]["processed_dir"]
    class_names = config["labels"]["classes"]
    class_to_int = config["labels"]["class_to_int"]
    subset_config = config.get("subset", {})

    k_anonymity = args.k or subset_config.get("k_anonymity", 30)
    group_by_fields = args.group_by or subset_config.get("group_by_fields", ["source"])

    # Load data
    input_path = args.input or str(Path(processed_dir) / "all_processed.jsonl")
    logger.info("Loading data from %s", input_path)
    df = pd.read_json(input_path, lines=True)

    # Load model
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

    inference = PolarityInference(
        model_type=args.model_type,
        model_path=model_path,
        vectorizer_path=vectorizer_path if args.model_type == "baseline" else "",
        class_names=class_names,
    )

    # Score all documents
    logger.info("Scoring %d documents", len(df))
    scored_df = inference.score_dataframe(df)

    polarity_scores = scored_df["polarity_score"].values
    pred_proba = scored_df[["prob_left", "prob_center", "prob_right"]].values
    pred_labels = scored_df["pred_label"].map(class_to_int).values

    # Generate reports
    print("\n" + "=" * 70)
    print("POLITICAL POLARITY SUBSET ANALYSIS")
    print("=" * 70)
    print()
    print("ETHICS NOTICE:")
    print("  - These are AGGREGATE statistics about TEXT CONTENT")
    print("  - Do NOT use to infer any individual's political beliefs")
    print("  - Do NOT use to target or discriminate against any group")
    print("  - Subsets with fewer than k={} documents are suppressed".format(k_anonymity))
    print()

    all_reports = {}
    for field in group_by_fields:
        logger.info("=== Generating report for field: %s ===", field)
        report = compute_subset_report(
            df=scored_df,
            polarity_scores=polarity_scores,
            pred_labels=pred_labels,
            pred_proba=pred_proba,
            class_names=class_names,
            group_by_field=field,
            k_anonymity=k_anonymity,
        )

        if report:
            all_reports[field] = report
            print(format_subset_report(report))
            print()

            # Save report
            report_path = artifact_path(
                artifacts_dir, f"subset_report_{field}", ".json"
            )
            save_subset_report(report, report_path)

            # Plot per-group polarity distributions
            for group_name, group_data in report.get("groups", {}).items():
                mask = None
                if field in scored_df.columns:
                    mask = scored_df[field] == group_name
                elif "subset_meta" in scored_df.columns:
                    mask = scored_df["subset_meta"].apply(
                        lambda x: x.get(field) == group_name
                        if isinstance(x, dict) else False
                    )

                if mask is not None and mask.any():
                    group_scores = polarity_scores[mask]
                    safe_name = str(group_name).replace("/", "_").replace(" ", "_")
                    plot_polarity_distribution(
                        group_scores,
                        artifact_path(
                            artifacts_dir,
                            f"polarity_dist_{field}_{safe_name}",
                            ".png",
                        ),
                        subset_name=f"{field}={group_name}",
                    )

    # Overall distribution
    plot_polarity_distribution(
        polarity_scores,
        artifact_path(artifacts_dir, "polarity_dist_overall", ".png"),
        title="Overall Polarity Distribution",
    )

    logger.info("=== Subset reporting complete ===")


if __name__ == "__main__":
    main()
