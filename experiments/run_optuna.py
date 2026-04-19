"""Run Optuna optimization for SCIP parameters.

This script performs Bayesian optimization to find good SCIP parameter
configurations that minimize solving time while still finding optimal
or feasible solutions.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure `coursework/` is on sys.path for `import src...`.
coursework_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if coursework_root not in sys.path:
    sys.path.insert(0, coursework_root)

import optuna
from optuna.samplers import TPESampler

from src.data.task_loader import load_tasks
from src.experiment.config_loader import load_config
from src.hpo.search_space import build_search_space
from src.hpo.optuna_optimizer import run_optuna_for_scip
from src.solvers.scip_solver import ScipSolver


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization for SCIP"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration with parameters to optimize",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials (default: 20)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=3,
        help="Number of tasks to use for evaluation (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds per trial (default: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Extract parameters to optimize
    solver_cfg = config.get("solvers", [{}])[0]
    parameters = solver_cfg.get("parameters", {})

    # Build Optuna search space
    search_space = build_search_space(parameters)
    print(f"Optimizing {len(search_space)} parameters:")
    for key, spec in search_space.items():
        print(f"  - {key}: {spec['choices']}")
    print()

    # Load tasks (use subset for faster optimization)
    tasks_source = config.get("tasks", {}).get("source")
    if not os.path.isabs(tasks_source):
        tasks_source = os.path.join(coursework_root, tasks_source)

    all_tasks = load_tasks(tasks_source, config.get("tasks", {}).get("filter"))
    if not all_tasks:
        print("No tasks found!")
        return

    # Use subset of tasks for faster optimization
    tasks_to_use = all_tasks[: min(args.tasks, len(all_tasks))]
    print(f"Using {len(tasks_to_use)} tasks for optimization")
    for t in tasks_to_use:
        print(f"  - {os.path.basename(t)}")
    print()

    # Create solver
    solver = ScipSolver()

    # Run optimization
    print(f"Running Optuna optimization ({args.trials} trials)...")
    study = run_optuna_for_scip(
        solver=solver,
        tasks=tasks_to_use,
        search_space=search_space,
        n_trials=args.trials,
        seed=args.seed,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nBest trial #{study.best_trial.number}:")
    print(f"  Value (mean wall time): {study.best_trial.value:.2f}s")
    print(f"  Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    print(f"\nBest configuration for your YAML:")
    print("-" * 40)
    yaml_lines = []
    for key, value in study.best_trial.params.items():
        yaml_lines.append(f"  {key}: [{value}]")
    print("parameters:")
    print(",\n".join(yaml_lines))
    print("-" * 40)

    print(f"\nStudy statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Best value: {study.best_value:.2f}")
    print(f"  Completed trials: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)}")


if __name__ == "__main__":
    main()
