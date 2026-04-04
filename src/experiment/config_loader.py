"""Load YAML experiment configuration."""

from __future__ import annotations

import os
from typing import Any

import yaml


def load_config(yaml_path: str) -> dict[str, Any]:
    """Load and validate an experiment configuration YAML file.

    Args:
        yaml_path: Path to YAML config.

    Returns:
        Parsed configuration dictionary.
    """

    abs_yaml_path = os.path.abspath(yaml_path)
    with open(abs_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("YAML config root must be a mapping/dict.")

    # Не преобразуем пути — оставляем относительными.
    # Преобразование будет выполнено в runner'е или writer'е.
    return config