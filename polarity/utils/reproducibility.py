"""Reproducibility utilities: seed setting, artifact timestamping."""

import os
import random
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info("Random seed set to %d", seed)


def artifact_path(base_dir: str, name: str, ext: str = "") -> Path:
    """Generate a timestamped artifact path.

    Args:
        base_dir: Base artifacts directory.
        name: Artifact name (e.g., 'baseline_model').
        ext: File extension (e.g., '.pkl').

    Returns:
        Path like ./artifacts/20240101_120000_baseline_model.pkl
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{name}{ext}"
    path = Path(base_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def latest_artifact(base_dir: str, pattern: str) -> Path | None:
    """Find the most recent artifact matching a glob pattern."""
    base = Path(base_dir)
    matches = sorted(base.glob(pattern), reverse=True)
    return matches[0] if matches else None
