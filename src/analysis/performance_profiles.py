"""Dolan–Moré performance profile utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _default_method_column(data: pd.DataFrame) -> pd.Series:
    """Create a default `method` column from `solver_name` and params."""

    if "solver_name" not in data.columns:
        raise ValueError("Expected `solver_name` column in results DataFrame.")

    param_cols = [c for c in data.columns if c.startswith("param_")]
    if not param_cols:
        return data["solver_name"].astype(str)

    # Join param values to form an identifier for method (solver+settings).
    params_part = data[param_cols].astype(str).agg(",".join, axis=1)
    return data["solver_name"].astype(str) + "|" + params_part


def performance_profile(data: pd.DataFrame, metric: str = "wall_time") -> pd.DataFrame:
    """Compute Dolan–Moré performance profile data.

    The result is suitable for plotting via `plot_performance_profile`.

    Assumptions:
    - `data` contains `task_name`, `status` or `solved`, `metric`, and method identifier.
    - If `method` column is absent, it is created from (`solver_name` + `param_*`).

    Args:
        data: Experiment results (as a DataFrame).
        metric: Column name of a runtime metric (smaller is better).

    Returns:
        DataFrame with columns:
        - `tau`: performance coefficient
        - `method`: method label
        - `fraction`: fraction of tasks solved within `tau * t_best`
    """

    if data.empty:
        return pd.DataFrame(columns=["tau", "method", "fraction"])

    required = {"task_name", metric}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df = data.copy()
    if "method" not in df.columns:
        df["method"] = _default_method_column(df)

    # Determine solved indicator.
    if "solved" in df.columns:
        solved_mask = df["solved"].astype(int) == 1
    elif "status" in df.columns:
        solved_mask = df["status"].astype(str).isin(["OPTIMAL", "FEASIBLE"])
    else:
        solved_mask = pd.Series([True] * len(df), index=df.index)

    df["_time"] = pd.to_numeric(df[metric], errors="coerce")
    df.loc[~solved_mask, "_time"] = np.inf

    # Best known time per task across methods (among solved ones only).
    t_best = (
        df.groupby("task_name")["_time"]
        .min()
        .replace(np.inf, np.nan)
        .dropna()
    )

    if t_best.empty:
        return pd.DataFrame(columns=["tau", "method", "fraction"])

    tasks = t_best.index.tolist()
    df = df[df["task_name"].isin(tasks)]

    # Compute ratios to estimate tau range.
    ratios = []
    for task in tasks:
        t0 = float(t_best.loc[task])
        if not np.isfinite(t0) or t0 <= 0:
            continue
        subset = df[df["task_name"] == task]
        for _, r in subset.iterrows():
            t = float(r["_time"])
            if not np.isfinite(t):
                ratios.append(np.inf)
            else:
                ratios.append(t / t0)

    finite_ratios = [x for x in ratios if np.isfinite(x)]
    tau_max = max(1.0, float(np.max(finite_ratios)) if finite_ratios else 1.0)

    # A reasonable tau grid for plotting.
    # For tau_max close to 1, keep a small grid.
    n_points = 30
    tau_values = np.linspace(1.0, tau_max, num=n_points)
    tau_values = np.unique(np.round(tau_values, 6))

    methods = sorted(df["method"].unique().tolist())
    out_rows: list[dict[str, Any]] = []

    for method in methods:
        d_m = df[df["method"] == method][["_time", "task_name"]].set_index(
            "task_name"
        )
        for tau in tau_values:
            count = 0
            valid = 0
            for task, t0 in t_best.items():
                valid += 1
                t = float(d_m["_time"].get(task, np.inf))
                # If the method time is inf -> not counted.
                if np.isfinite(t) and t <= float(tau) * float(t0):
                    count += 1
            fraction = float(count) / float(valid) if valid else 0.0
            out_rows.append(
                {"tau": float(tau), "method": str(method), "fraction": fraction}
            )

    return pd.DataFrame(out_rows).sort_values(["method", "tau"]).reset_index(drop=True)


def plot_performance_profile(profile_data: pd.DataFrame, ax: Any = None):
    """Plot a Dolan–Moré performance profile.

    Args:
        profile_data: Output of `performance_profile`.
        ax: Optional matplotlib axis.

    Returns:
        Matplotlib axis.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if profile_data.empty:
        ax.set_title("No data for performance profile")
        return ax

    sns.lineplot(
        data=profile_data,
        x="tau",
        y="fraction",
        hue="method",
        ax=ax,
        linewidth=2,
    )
    ax.set_xlabel("tau")
    ax.set_ylabel("Fraction of solved instances")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    return ax

