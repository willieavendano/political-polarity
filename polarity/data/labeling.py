"""Label management: outlet-level distant supervision, human labels, merging."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Valid label values
VALID_LABELS = {"left", "center", "right"}


def load_outlet_bias(path: str | Path) -> dict[str, str]:
    """Load outlet-level bias labels from CSV.

    Expected CSV format:
        outlet_name,label
        nytimes,left
        foxnews,right
        reuters,center

    Args:
        path: Path to outlet_bias.csv.

    Returns:
        Dictionary mapping outlet name -> label.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Outlet bias file not found: %s", path)
        return {}

    df = pd.read_csv(path)
    if "outlet_name" not in df.columns or "label" not in df.columns:
        raise ValueError("outlet_bias.csv must have columns: outlet_name, label")

    # Validate labels
    invalid = set(df["label"].str.lower()) - VALID_LABELS
    if invalid:
        raise ValueError(f"Invalid labels in outlet_bias.csv: {invalid}")

    mapping = dict(zip(df["outlet_name"].str.lower(), df["label"].str.lower()))
    logger.info("Loaded %d outlet bias labels", len(mapping))
    return mapping


def load_human_labels(path: str | Path) -> dict[str, str]:
    """Load human-annotated labels from CSV.

    Expected CSV format:
        doc_id,label
        doc_001,left
        doc_002,right

    Args:
        path: Path to human_labels.csv.

    Returns:
        Dictionary mapping doc_id -> label.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Human labels file not found: %s", path)
        return {}

    df = pd.read_csv(path)
    if "doc_id" not in df.columns or "label" not in df.columns:
        raise ValueError("human_labels.csv must have columns: doc_id, label")

    invalid = set(df["label"].str.lower()) - VALID_LABELS
    if invalid:
        raise ValueError(f"Invalid labels in human_labels.csv: {invalid}")

    mapping = dict(zip(df["doc_id"], df["label"].str.lower()))
    logger.info("Loaded %d human labels", len(mapping))
    return mapping


def assign_labels(
    df: pd.DataFrame,
    outlet_bias: dict[str, str],
    human_labels: dict[str, str],
    source_col: str = "source",
    doc_id_col: str = "doc_id",
) -> pd.DataFrame:
    """Assign labels to documents using tiered strategy.

    Priority:
        1. Human labels (Tier 2) - highest priority
        2. Outlet-level distant supervision (Tier 1) - fallback

    Documents with no label get label=None and are excluded from training.

    WARNING: Outlet-level labels are weak supervision. An outlet labeled
    'left' may publish articles with centrist or even right-leaning content.
    This is a known limitation of distant supervision.

    Args:
        df: Document DataFrame.
        outlet_bias: Outlet -> label mapping.
        human_labels: doc_id -> label mapping.
        source_col: Column containing outlet/source name.
        doc_id_col: Column containing document ID.

    Returns:
        DataFrame with 'label' column added.
    """
    df = df.copy()

    # Tier 1: Outlet-level distant supervision
    if outlet_bias and source_col in df.columns:
        df["label"] = df[source_col].str.lower().map(outlet_bias)
        tier1_count = df["label"].notna().sum()
        logger.info(
            "Tier 1 (outlet labels): assigned %d labels (%.1f%%)",
            tier1_count, 100 * tier1_count / max(len(df), 1),
        )
        logger.warning(
            "BIAS WARNING: Outlet-level labels are weak supervision. "
            "An outlet's overall lean does NOT mean every article matches that lean. "
            "Use Tier 2 human labels for higher quality."
        )
    else:
        df["label"] = None

    # Tier 2: Human labels override
    if human_labels:
        human_mask = df[doc_id_col].isin(human_labels)
        for doc_id, label in human_labels.items():
            df.loc[df[doc_id_col] == doc_id, "label"] = label
        tier2_count = human_mask.sum()
        logger.info("Tier 2 (human labels): assigned/overrode %d labels", tier2_count)

    labeled = df["label"].notna().sum()
    unlabeled = df["label"].isna().sum()
    logger.info("Total: %d labeled, %d unlabeled documents", labeled, unlabeled)

    if labeled > 0:
        dist = df[df["label"].notna()]["label"].value_counts()
        logger.info("Label distribution:\n%s", dist.to_string())

    return df


def create_label_template(output_path: str | Path, doc_ids: list[str]) -> None:
    """Create a CSV template for human labeling.

    Args:
        output_path: Where to save the template.
        doc_ids: List of document IDs to label.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "doc_id": doc_ids,
        "label": "",
        "confidence": "",
        "notes": "",
    })

    header_comment = (
        "# Labeling Guidelines:\n"
        "# label: one of 'left', 'center', 'right'\n"
        "#   - 'left': content aligns with liberal/progressive/Democrat positions\n"
        "#   - 'right': content aligns with conservative/Republican positions\n"
        "#   - 'center': content is balanced, mixed, unclear, or apolitical\n"
        "# confidence: 'high', 'medium', 'low' (optional)\n"
        "# notes: free text (optional)\n"
        "#\n"
        "# IMPORTANT: Label the CONTENT of the article, not the outlet.\n"
        "# A left-leaning outlet can publish a center or right-leaning article.\n"
        "# If unsure, use 'center'.\n"
    )

    with open(output_path, "w") as f:
        f.write(header_comment)
        df.to_csv(f, index=False)

    logger.info("Created labeling template at %s with %d entries", output_path, len(doc_ids))
