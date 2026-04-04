"""Optuna-based hyperparameter optimization for SCIP.

This module provides a lightweight integration:
it samples parameter combinations from a categorical search space and evaluates
them by running SCIP on a subset of tasks.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna

from src.solvers.base_solver import BaseSolver


def _suggest_params(trial: optuna.trial.Trial, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Suggest one parameter set for a trial.

    Args:
        trial: Optuna trial object.
        search_space: A search space definition as produced by `build_search_space`.

    Returns:
        Suggested parameters for the current trial.
    """

    params: dict[str, Any] = {}
    for key, spec in search_space.items():
        if spec.get("type") != "categorical":
            raise ValueError(f"Unsupported search space type: {spec}")
        params[key] = trial.suggest_categorical(key, spec["choices"])
    return params


def run_optuna_for_scip(
    solver: BaseSolver,
    tasks: list[str],
    search_space: dict[str, dict[str, Any]],
    n_trials: int = 20,
    metric_fn: Callable[[list[dict[str, Any]]], float] | None = None,
    seed: int | None = None,
) -> optuna.study.Study:
    """Run Optuna to find good SCIP parameters.

    Args:
        solver: Solver wrapper implementing `solve`.
        tasks: List of task paths used for evaluation.
        search_space: Output of `build_search_space()`.
        n_trials: Number of Optuna trials.
        metric_fn: Metric function that maps a list of solver outputs into a scalar.
                   Defaults to mean `wall_time` over solved runs (inf if none solved).
        seed: Optional random seed for reproducibility.

    Returns:
        Optuna study instance.
    """

    if metric_fn is None:

        def metric_fn_default(outputs: list[dict[str, Any]]) -> float:
            times = [
                float(o["wall_time"])
                for o in outputs
                if bool(o.get("solved")) and o.get("wall_time") is not None
            ]
            if not times:
                return float("inf")
            return float(np.mean(times))

        metric_fn = metric_fn_default

    def objective(trial: optuna.trial.Trial) -> float:
        """Optuna objective: minimize mean runtime over evaluation tasks."""
        params = _suggest_params(trial, search_space)

        time_limit = None
        params_for_solver = dict(params)
        if "limits/time" in params_for_solver:
            try:
                time_limit_val = float(params_for_solver.pop("limits/time"))
                if time_limit_val > 0:
                    time_limit = time_limit_val
            except Exception:
                # Keep original parameter if parsing fails.
                pass

        outputs = []
        for task_path in tasks:
            out = solver.solve(task_path, params_for_solver, time_limit=time_limit)
            outputs.append(out)
        return float(metric_fn(outputs))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study

