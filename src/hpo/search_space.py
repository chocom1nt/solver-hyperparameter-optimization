"""Search space helpers for Optuna."""

from __future__ import annotations

from typing import Any


def build_search_space(parameters: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a categorical Optuna search space from a parameter grid.

    Expected input format (coming from YAML `parameters`):
    `{"limits/time": [10, 30], "presolving/maxrounds": [0, 5]}`

    For each key we create a simple definition that Optuna will treat as
    categorical values.

    Args:
        parameters: Parameter grid dict where each value is a list (or scalar).

    Returns:
        A dict mapping parameter keys to a definition:
        `{"type": "categorical", "choices": [...]}`.
    """

    search_space: dict[str, dict[str, Any]] = {}
    for key, value in parameters.items():
        if isinstance(value, list):
            choices = value
        else:
            choices = [value]
        search_space[key] = {"type": "categorical", "choices": choices}
    return search_space

