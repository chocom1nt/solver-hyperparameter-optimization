"""Experiment runner for grid search over solver parameters."""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import Any

from src.data.task_loader import load_tasks
from src.experiment.result_writer import ResultWriter

from importlib import import_module

logger = logging.getLogger(__name__)


def _task_name_from_path(path: str) -> str:
    """Get task name without `.mps` / `.mps.gz` extension."""
    base = os.path.basename(path)
    if base.lower().endswith(".gz"):
        base = os.path.splitext(base)[0]
    return os.path.splitext(base)[0]


def _expand_param_grid(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a parameter grid specification into a list of dicts."""
    keys = []
    values_list = []
    for k, v in parameters.items():
        keys.append(k)
        if isinstance(v, list):
            values_list.append(v)
        else:
            values_list.append([v])

    combos: list[dict[str, Any]] = []
    for prod in itertools.product(*values_list):
        combos.append(dict(zip(keys, prod)))
    return combos


def _resolve_solver_class(class_name: str) -> type:
    """Resolve solver class by name from `src.solvers` package."""
    solvers_pkg = import_module("src.solvers")
    if not hasattr(solvers_pkg, class_name):
        raise ValueError(
            f"Solver class `{class_name}` not found in `src.solvers` package."
        )
    return getattr(solvers_pkg, class_name)


def _setup_logging(root_dir: str, config_name: str) -> Path:
    """Configure logging to file and return log file path.

    Args:
        root_dir: Root directory of the project.
        config_name: Name of the config file (for log naming).

    Returns:
        Path to the log file.
    """
    log_dir = Path(root_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file name based on config and timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_basename = os.path.splitext(config_name)[0]
    log_file = f"{config_basename}_{timestamp}.log"
    log_path = log_dir / log_file

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console
            logging.FileHandler(log_path, encoding="utf-8"),  # File
        ],
    )

    logger.info("=" * 60)
    logger.info("Experiment started")
    logger.info("Config: %s", config_name)
    logger.info("Log file: %s", log_path)
    logger.info("=" * 60)

    return log_path


class ExperimentRunner:
    """Run computational experiments based on YAML configuration."""

    def __init__(self, config: dict[str, Any], root_dir: str = None):
        self.config = config
        self.root_dir = root_dir if root_dir is not None else os.getcwd()

        tasks_cfg = config.get("tasks", {})
        self.tasks_source = tasks_cfg.get("source")
        self.tasks_filter = tasks_cfg.get("filter")
        if not self.tasks_source:
            raise ValueError("Config must include `tasks.source`.")

        solvers_cfg = config.get("solvers")
        if not isinstance(solvers_cfg, list) or not solvers_cfg:
            raise ValueError("Config must include non-empty `solvers` list.")
        self.solvers_cfg = solvers_cfg

        output_cfg = config.get("output", {})
        if not isinstance(output_cfg, dict) or "path" not in output_cfg:
            raise ValueError("Config must include `output.path`.")

        output_cfg = dict(output_cfg)
        output_path = output_cfg.get("path")
        if output_path and not os.path.isabs(output_path):
            output_path = os.path.join(self.root_dir, output_path)
            output_cfg["path"] = output_path
        self.output_cfg = output_cfg

        # Compute param keys to create stable columns.
        param_keys: set[str] = set()
        for solver_cfg in self.solvers_cfg:
            params = solver_cfg.get("parameters", {}) or {}
            if isinstance(params, dict):
                param_keys.update(params.keys())
        self.param_keys = sorted(param_keys)
        self.writer = ResultWriter(output_cfg, self.param_keys)

    def run(self) -> None:
        """Execute the experiment according to configuration."""
        # Setup logging before anything else
        config_name = self.config.get("name", "unknown")
        log_path = _setup_logging(self.root_dir, config_name)

        tasks_source_abs = self.tasks_source
        if not os.path.isabs(tasks_source_abs):
            tasks_source_abs = os.path.join(self.root_dir, tasks_source_abs)

        logger.info("Loading tasks from %s", tasks_source_abs)
        tasks = load_tasks(tasks_source_abs, self.tasks_filter)
        logger.info("Found %d tasks", len(tasks))

        if not tasks:
            logger.warning("No tasks matched the filter. Exiting.")
            self.writer.close()
            return

        for solver_idx, solver_cfg in enumerate(self.solvers_cfg, start=1):
            solver_name = solver_cfg.get("name", f"solver_{solver_idx}")
            solver_class_name = solver_cfg.get("class")
            if not solver_class_name:
                raise ValueError("Each solver entry must include `class`.")

            parameters = solver_cfg.get("parameters", {}) or {}
            if not isinstance(parameters, dict):
                raise ValueError("`parameters` must be a dict.")

            repetitions = int(solver_cfg.get("repetitions", 1))
            repetitions = max(1, repetitions)

            logger.info(
                "Running solver %s (%s). Repetitions=%d",
                solver_name,
                solver_class_name,
                repetitions,
            )

            solver_class = _resolve_solver_class(solver_class_name)
            solver_obj = solver_class()

            param_combos = _expand_param_grid(parameters)
            logger.info(
                "Parameter combinations: %d (keys=%d)",
                len(param_combos),
                len(parameters),
            )

            for task_path in tasks:
                task_name = _task_name_from_path(task_path)
                for combo_idx, param_combo in enumerate(param_combos, start=1):
                    logger.info(
                        "Task=%s Solver=%s Combo=%d/%d",
                        task_name,
                        solver_name,
                        combo_idx,
                        len(param_combos),
                    )

                    for rep in range(1, repetitions + 1):
                        try:
                            result = solver_obj.solve(task_path, param_combo)
                        except Exception as e:
                            result = {
                                "status": f"ERROR: {type(e).__name__}",
                                "objective": None,
                                "wall_time": None,
                                "solved": False,
                                "error": str(e),
                            }
                        self.writer.write_result(
                            task_name=task_name,
                            solver_name=str(solver_name),
                            params=param_combo,
                            result=result,
                        )

        self.writer.close()
        logger.info("Experiment completed. Log saved to: %s", log_path)
