"""Aggregated subset analysis with privacy safeguards (k-anonymity)."""

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)

# ============================================================================
# PRIVACY AND ETHICS NOTICE
# ============================================================================
# This module performs AGGREGATE analysis only. It:
# - Never infers individual political beliefs
# - Never infers demographic attributes from text
# - Enforces k-anonymity minimum group sizes
# - Only uses externally provided subset metadata
# - Suppresses results for groups smaller than k
#
# WARNING: Even aggregate statistics can be misused. Do not use these
# outputs to target, stereotype, or discriminate against any group.
# ============================================================================


def compute_subset_report(
    df: pd.DataFrame,
    polarity_scores: np.ndarray,
    pred_labels: np.ndarray,
    pred_proba: np.ndarray,
    class_names: list[str],
    group_by_field: str,
    k_anonymity: int = 30,
    top_phrases_per_subset: int = 20,
    phrase_fn=None,
) -> dict:
    """Compute aggregated polarity report for a subset field.

    Args:
        df: DataFrame with subset metadata.
        polarity_scores: Polarity scores (P(right) - P(left)).
        pred_labels: Predicted class labels (integers).
        pred_proba: Predicted probabilities.
        class_names: Class name mapping.
        group_by_field: Metadata field to group by.
        k_anonymity: Minimum group size (suppress smaller groups).
        top_phrases_per_subset: Number of top phrases per subset.
        phrase_fn: Optional function(texts, labels) -> phrases dict.

    Returns:
        Report dictionary with per-group statistics.
    """
    # Extract group field from subset_meta or top-level columns
    if group_by_field in df.columns:
        groups = df[group_by_field].fillna("unknown")
    elif "subset_meta" in df.columns:
        groups = df["subset_meta"].apply(
            lambda x: x.get(group_by_field, "unknown")
            if isinstance(x, dict) else "unknown"
        )
    else:
        logger.warning("Group field '%s' not found in data", group_by_field)
        return {}

    unique_groups = groups.unique()
    report = {
        "group_by": group_by_field,
        "k_anonymity_threshold": k_anonymity,
        "groups": {},
        "suppressed_groups": [],
    }

    for group_name in sorted(unique_groups):
        mask = groups == group_name
        group_size = mask.sum()

        if group_size < k_anonymity:
            report["suppressed_groups"].append({
                "group": str(group_name),
                "size": int(group_size),
                "reason": f"Group size ({group_size}) < k-anonymity threshold ({k_anonymity})",
            })
            logger.info(
                "Suppressed group '%s' (n=%d < k=%d)",
                group_name, group_size, k_anonymity,
            )
            continue

        group_scores = polarity_scores[mask]
        group_labels = pred_labels[mask]
        group_proba = pred_proba[mask]

        # Class distribution
        class_counts = {}
        for cls_idx, cls_name in enumerate(class_names):
            class_counts[cls_name] = int((group_labels == cls_idx).sum())

        # Entropy of predictions (uncertainty measure)
        group_entropy = float(np.mean([
            entropy(p) for p in group_proba
        ]))

        group_report = {
            "size": int(group_size),
            "polarity": {
                "mean": float(np.mean(group_scores)),
                "median": float(np.median(group_scores)),
                "std": float(np.std(group_scores)),
                "min": float(np.min(group_scores)),
                "max": float(np.max(group_scores)),
            },
            "class_distribution": class_counts,
            "mean_entropy": group_entropy,
        }

        report["groups"][str(group_name)] = group_report

    # Summary
    if report["groups"]:
        all_means = [g["polarity"]["mean"] for g in report["groups"].values()]
        report["summary"] = {
            "total_groups_reported": len(report["groups"]),
            "total_groups_suppressed": len(report["suppressed_groups"]),
            "overall_polarity_range": {
                "min_group_mean": float(min(all_means)),
                "max_group_mean": float(max(all_means)),
            },
        }

    logger.info(
        "Subset report for '%s': %d groups reported, %d suppressed",
        group_by_field,
        len(report["groups"]),
        len(report["suppressed_groups"]),
    )

    return report


def save_subset_report(report: dict, output_path: str | Path) -> None:
    """Save subset report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved subset report to %s", output_path)


def format_subset_report(report: dict) -> str:
    """Format subset report as a readable string.

    Args:
        report: Report dictionary from compute_subset_report.

    Returns:
        Formatted string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"SUBSET ANALYSIS: {report.get('group_by', 'unknown')}")
    lines.append(f"k-anonymity threshold: {report.get('k_anonymity_threshold', 30)}")
    lines.append("=" * 70)

    # Ethics reminder
    lines.append("")
    lines.append("WARNING: These are AGGREGATE statistics about TEXT CONTENT.")
    lines.append("Do NOT use to infer individual beliefs or target groups.")
    lines.append("")

    for group_name, data in sorted(report.get("groups", {}).items()):
        lines.append(f"--- {group_name} (n={data['size']}) ---")
        pol = data["polarity"]
        lines.append(f"  Polarity:  mean={pol['mean']:+.3f}, "
                      f"median={pol['median']:+.3f}, std={pol['std']:.3f}")
        lines.append(f"  Classes:   {data['class_distribution']}")
        lines.append(f"  Entropy:   {data['mean_entropy']:.3f}")
        lines.append("")

    if report.get("suppressed_groups"):
        lines.append("SUPPRESSED GROUPS (below k-anonymity threshold):")
        for sg in report["suppressed_groups"]:
            lines.append(f"  - {sg['group']} (n={sg['size']})")
        lines.append("")

    if "summary" in report:
        s = report["summary"]
        lines.append("SUMMARY:")
        lines.append(f"  Groups reported: {s['total_groups_reported']}")
        lines.append(f"  Groups suppressed: {s['total_groups_suppressed']}")
        if "overall_polarity_range" in s:
            rng = s["overall_polarity_range"]
            lines.append(f"  Polarity range: [{rng['min_group_mean']:+.3f}, "
                          f"{rng['max_group_mean']:+.3f}]")

    return "\n".join(lines)
