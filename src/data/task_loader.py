"""Load MIPLIB-like tasks from a directory."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _task_name_from_path(path: str) -> str:
    """Derive task name from a `.mps` / `.mps.gz` file path."""

    base = os.path.basename(path)
    if base.lower().endswith(".gz"):
        base = os.path.splitext(base)[0]
    return os.path.splitext(base)[0]


def _guess_metadata_csv(data_dir: str) -> str | None:
    """Guess the path to `metadata.csv` for a raw tasks directory."""

    # Typical case for this coursework: `data/raw` -> sibling `data/metadata.csv`.
    parent = os.path.dirname(os.path.abspath(data_dir))
    candidate_1 = os.path.join(parent, "metadata.csv")
    if os.path.exists(candidate_1):
        return candidate_1

    candidate_2 = os.path.join(data_dir, "metadata.csv")
    if os.path.exists(candidate_2):
        return candidate_2

    return None


def load_tasks(data_dir: str, filters: dict[str, Any] | None = None) -> list[str]:
    """Scan `data_dir` and return paths to `.mps` / `.mps.gz` instances.

    If `filters` is provided, and `metadata.csv` exists, it will filter tasks
    using columns present in `metadata.csv`.

    Args:
        data_dir: Directory containing `.mps` / `.mps.gz` files.
        filters: Optional filtering criteria, e.g. `{'min_vars': 100, 'max_vars': 5000}`.

    Returns:
        List of absolute paths to tasks.
    """

    supported_exts = (".mps", ".mps.gz")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Tasks directory does not exist: {data_dir}")

    all_files: list[str] = []
    for entry in os.listdir(data_dir):
        if entry.lower().endswith(supported_exts):
            all_files.append(os.path.join(data_dir, entry))

    if not filters:
        return sorted(os.path.abspath(p) for p in all_files)

    metadata_csv = _guess_metadata_csv(data_dir)
    if metadata_csv is None:
        logger.warning(
            "filters provided but metadata.csv was not found; returning all tasks."
        )
        return sorted(os.path.abspath(p) for p in all_files)

    if not os.path.exists(metadata_csv):
        logger.warning("metadata.csv missing; returning all tasks.")
        return sorted(os.path.abspath(p) for p in all_files)

    df = pd.read_csv(metadata_csv)
    if "name" not in df.columns:
        logger.warning("metadata.csv has no `name` column; returning all tasks.")
        return sorted(os.path.abspath(p) for p in all_files)

    task_to_file = {_task_name_from_path(p): p for p in all_files}
    df = df[df["name"].isin(task_to_file.keys())]

    # Supported filter keys:
    # - min_vars, max_vars
    # - min_constraints, max_constraints
    # - min_binary/max_binary, min_integer/max_integer, min_continuous/max_continuous
    col_map = {
        "min_vars": ("num_vars", "min"),
        "max_vars": ("num_vars", "max"),
        "min_constraints": ("num_constraints", "min"),
        "max_constraints": ("num_constraints", "max"),
        "min_binary": ("num_binary", "min"),
        "max_binary": ("num_binary", "max"),
        "min_integer": ("num_integer", "min"),
        "max_integer": ("num_integer", "max"),
        "min_continuous": ("num_continuous", "min"),
        "max_continuous": ("num_continuous", "max"),
    }

    mask = pd.Series([True] * len(df), index=df.index)
    for f_key, f_val in filters.items():
        if f_key not in col_map:
            continue
        col, bound_type = col_map[f_key]
        if col not in df.columns:
            continue
        if bound_type == "min":
            mask &= df[col] >= f_val
        else:
            mask &= df[col] <= f_val

    df_filtered = df[mask]
    selected_names = df_filtered["name"].tolist()
    selected_paths = [task_to_file[n] for n in selected_names if n in task_to_file]
    return sorted(os.path.abspath(p) for p in selected_paths)

