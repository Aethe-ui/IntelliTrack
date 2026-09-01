"""Configuration loading and access utilities for IntelliTrack."""

import logging
from functools import reduce
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load a YAML configuration file and return it as a nested dictionary.

    Args:
        path: Filesystem path to the YAML config file.

    Returns:
        Parsed configuration as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed as YAML.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug("Loaded config from %s", path)
    return config


def get(config: dict, dotted_key: str, default: Any = None) -> Any:
    """Retrieve a value from a nested config dict using a dot-separated key path.

    Args:
        config: The top-level configuration dictionary.
        dotted_key: A dot-delimited key string, e.g. ``"control.pid.kp"``.
        default: Value to return if any key in the path is missing.

    Returns:
        The value at the resolved path, or ``default`` if not found.

    Example::

        cfg = load_config("configs/default.yaml")
        kp = get(cfg, "control.pid.kp", default=0.05)
    """
    keys = dotted_key.split(".")
    try:
        return reduce(lambda d, k: d[k], keys, config)
    except (KeyError, TypeError):
        return default
