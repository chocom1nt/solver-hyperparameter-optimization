"""Base interface for MILP solvers used in experiments."""

from __future__ import annotations

import abc
from typing import Any


class BaseSolver(abc.ABC):
    """Abstract solver wrapper.

    Implementations are expected to solve a MILP model stored in a file and
    return a normalized result dictionary.
    """

    @abc.abstractmethod
    def solve(
        self,
        problem_path: str,
        params: dict[str, Any],
        time_limit: float | None = None,
    ) -> dict[str, Any]:
        """Solve the model located at `problem_path`.

        Args:
            problem_path: Path to an instance file (e.g. `.mps` or `.mps.gz`).
            params: Solver-specific parameters.
            time_limit: Optional time limit in seconds.

        Returns:
            A dictionary with at least:
            - `status` (str): e.g. `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, ...
            - `objective` (float | None): objective value if available
            - `wall_time` (float): wall-clock seconds spent in `Solve`
            - `solved` (bool): whether an optimal/feasible solution was found
        """

