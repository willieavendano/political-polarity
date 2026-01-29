"""Configuration loader and manager."""

import os
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file with defaults.

    Args:
        config_path: Path to config YAML. If None, uses default config.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", path)
    return config


def get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    """Safely get a nested config value.

    Example:
        get_nested(config, "transformer", "model_name")
    """
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def ensure_dirs(config: dict) -> None:
    """Create all required directories from config."""
    dirs = [
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["data"]["labels_dir"],
        config["data"]["artifacts_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Ensured all data directories exist")
