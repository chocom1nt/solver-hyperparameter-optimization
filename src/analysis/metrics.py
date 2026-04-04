"""Additional metrics computed from experiment results."""

from __future__ import annotations

from typing import Any

import pandas as pd


def gap(obtained: float | None, best_known: float | None) -> float | None:
    """Compute relative optimality gap.

    By convention we assume the objective is a minimization problem:
    `gap = (obtained - best_known) / |best_known|`.

    Args:
        obtained: Obtained objective value.
        best_known: Reference (best known) objective value.

    Returns:
        Relative gap if values are valid; otherwise `None`.
    """

    if obtained is None or best_known is None:
        return None

    denom = abs(best_known)
    if denom < 1e-12:
        # If the best known is (near) 0, use absolute difference.
        return float(abs(obtained - best_known))

    return float((obtained - best_known) / denom)


def success_rate(df: pd.DataFrame) -> float:
    """Compute percentage of instances solved optimally.

    The function uses either `status == 'OPTIMAL'` (preferred) or
    `solved == 1` as a fallback.

    Args:
        df: DataFrame with at least `status` or `solved`.

    Returns:
        Success rate in percents (0..100).
    """

    if df.empty:
        return 0.0

    if "status" in df.columns:
        optimal_count = (df["status"].astype(str) == "OPTIMAL").sum()
        return 100.0 * float(optimal_count) / float(len(df))

    if "solved" in df.columns:
        solved_count = (df["solved"].astype(int) == 1).sum()
        return 100.0 * float(solved_count) / float(len(df))

    raise ValueError("DataFrame must contain `status` or `solved` columns.")

