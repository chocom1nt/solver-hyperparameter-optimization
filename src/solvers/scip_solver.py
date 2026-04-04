"""SCIP solver wrapper (via PySCIPOpt).

This module provides a clean interface to SCIP solver using PySCIPOpt,
with proper parameter handling and status extraction.
"""

from __future__ import annotations

import gzip
import logging
import os
import tempfile
import time
from typing import Any

from pyscipopt import Model

from .base_solver import BaseSolver

logger = logging.getLogger(__name__)


# Mapping of user-friendly parameter names to actual SCIP parameter names.
# SCIP uses spaces between hierarchy levels, not slashes.
# Format: "user/key/subkey" -> "user / key / subkey"
_PARAM_ALIASES: dict[str, str] = {
    # Presolving parameters
    "presolving/maxrounds": "presolving / maxrounds",
    "presolving/dual/freq": "presolving / dual / freq",
    "presolving/immrestart/freq": "presolving / immrestart / freq",
    "presolving/restart/freq": "presolving / restart / freq",
    # Branching parameters
    "branching/rule": "branching / rule",
    "branching/vanillafull": "branching / vanillafull",
    "branching/chg": "branching / chg",
    # Heuristic parameters
    "heuristics/coefdiving/freq": "heuristics / coefdiving / freq",
    "heuristics/fracdiving/freq": "heuristics / fracdiving / freq",
    "heuristics/guideddiving/freq": "heuristics / guideddiving / freq",
    "heuristics/veclendiving/freq": "heuristics / veclendiving / freq",
    "heuristics/rins/freq": "heuristics / rins / freq",
    "heuristics/rens/freq": "heuristics / rens / freq",
    "heuristics/shifting/freq": "heuristics / shifting / freq",
    "heuristics/oneopt/freq": "heuristics / oneopt / freq",
    # Separation parameters
    "separating/maxcuts": "separating / maxcuts",
    "separating/maxcutsroot": "separating / maxcutsroot",
    "separating/cutpool/running": "separating / cutpool / running",
    # Propagation parameters
    "propagating/maxrounds": "propagating / maxrounds",
    "propagating/maxroundsroot": "propagating / maxroundsroot",
    # Display parameters
    "display/verblevel": "display / verblevel",
    "display/freq": "display / freq",
    "display/lpinfo": "display / lpinfo",
    # Timing parameters
    "timing/clocktype": "timing / clocktype",
}


class ScipSolver(BaseSolver):
    """MILP solver wrapper using PySCIPOpt.

    PySCIPOpt provides direct access to SCIP parameters, avoiding the
    limitations of OR-Tools' parameter passing mechanism.

    Attributes:
        _warnings_count: Number of failed parameter settings (for reporting).
    """

    def __init__(self) -> None:
        """Initialize the solver wrapper."""
        self._warnings_count = 0

    def _unpack_gz(self, problem_path: str) -> str:
        """Unpack .mps.gz file to a temporary .mps file.

        PySCIPOpt cannot read .mps.gz directly, so we need to unpack it.

        Args:
            problem_path: Path to the problem file (possibly .mps.gz).

        Returns:
            Path to the unpacked .mps file.
        """
        if not problem_path.lower().endswith(".gz"):
            return problem_path

        logger.debug("Unpacking gzipped file: %s", problem_path)
        fd, tmp_path = tempfile.mkstemp(suffix=".mps")
        os.close(fd)

        with gzip.open(problem_path, "rb") as f_in:
            with open(tmp_path, "wb") as f_out:
                f_out.write(f_in.read())

        logger.debug("Unpacked to: %s", tmp_path)
        return tmp_path

    def _cleanup_tmp_file(self, tmp_path: str, original_path: str) -> None:
        """Remove temporary file if it was created from .mps.gz.

        Args:
            tmp_path: Path to the temporary file.
            original_path: Original problem path.
        """
        if tmp_path != original_path:
            try:
                os.remove(tmp_path)
                logger.debug("Cleaned up temporary file: %s", tmp_path)
            except OSError as e:
                logger.warning("Failed to remove temp file %s: %s", tmp_path, e)

    def _resolve_param_name(self, key: str) -> str:
        """Resolve parameter name from user-friendly format to SCIP format.

        Args:
            key: Parameter key from config (e.g., 'presolving/maxrounds').

        Returns:
            Canonical SCIP parameter name with spaces.
        """
        # Check if there's a known alias
        if key in _PARAM_ALIASES:
            return _PARAM_ALIASES[key]

        # Try converting slashes to spaces as fallback
        return key.replace("/", " / ")

    def _set_params(self, model: Model, params: dict[str, Any]) -> int:
        """Set SCIP parameters from a dictionary.

        Args:
            model: PySCIPOpt Model instance.
            params: Dictionary of parameter names to values.

        Returns:
            Number of successfully set parameters.
        """
        success_count = 0
        for key, value in params.items():
            if value is None:
                continue

            # Skip time limit - handled separately
            if key == "limits/time":
                continue

            try:
                canonical_key = self._resolve_param_name(key)

                # Set parameter based on type
                if isinstance(value, bool):
                    model.setParam(canonical_key, bool(value))
                elif isinstance(value, int):
                    model.setParam(canonical_key, int(value))
                elif isinstance(value, float):
                    model.setParam(canonical_key, float(value))
                else:
                    model.setParam(canonical_key, str(value))

                logger.debug("Set SCIP param: %s = %s", canonical_key, value)
                success_count += 1

            except Exception as e:
                self._warnings_count += 1
                logger.debug("Failed to set param %s=%s: %s", key, value, e)

        return success_count

    def _get_status_string(self, model: Model) -> str:
        """Extract status string from PySCIPOpt model.

        Handles both old (enum with .name) and new (string) return types.

        Args:
            model: PySCIPOpt Model instance.

        Returns:
            Status as a string (e.g., 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE').
        """
        status = model.getStatus()
        if isinstance(status, str):
            return status
        # For older PySCIPOpt versions, status might be an enum
        if hasattr(status, "name"):
            return str(status.name)
        return str(status)

    def solve(
        self,
        problem_path: str,
        params: dict[str, Any],
        time_limit: float | None = None,
    ) -> dict[str, Any]:
        """Solve the given MILP using SCIP via PySCIPOpt.

        Args:
            problem_path: Path to `.mps` or `.mps.gz`.
            params: SCIP parameter values (as dict).
            time_limit: Optional time limit in seconds.
                Overrides `limits/time` from params if both are provided.

        Returns:
            Normalized result dictionary with keys:
            - status: Solver status string
            - objective: Objective value (if solution found)
            - wall_time: Solving time in seconds
            - solved: Whether optimal/feasible solution was found
        """
        start = time.perf_counter()
        self._warnings_count = 0

        tmp_path: str | None = None
        try:
            # Verify problem file exists
            if not os.path.exists(problem_path):
                raise FileNotFoundError(f"Problem file not found: {problem_path}")

            logger.info("Loading problem: %s", problem_path)

            # Unpack if gzipped
            tmp_path = self._unpack_gz(problem_path)
            actual_path = tmp_path

            # Create SCIP model
            model = Model()

            # Set time limit (explicit argument takes precedence)
            effective_time_limit = time_limit
            if "limits/time" in params and time_limit is None:
                try:
                    effective_time_limit = float(params["limits/time"])
                except (TypeError, ValueError):
                    pass

            if effective_time_limit is not None:
                model.setParam("limits/time", effective_time_limit)
                logger.debug("Time limit: %.2f seconds", effective_time_limit)

            # Set remaining parameters (exclude time limit)
            solve_params = {k: v for k, v in params.items() if k != "limits/time"}
            self._set_params(model, solve_params)

            # Read problem from file (must be .mps, not .mps.gz)
            model.readProblem(actual_path)

            # Solve
            model.optimize()

            wall_time = time.perf_counter() - start

            # Get solution status
            status_str = self._get_status_string(model)

            # Determine if solved optimally or feasibly
            solved = status_str in {"OPTIMAL", "FEASIBLE"}

            # Get objective value
            objective: float | None = None
            if solved:
                obj_val = model.getObjVal()
                if obj_val is not None:
                    try:
                        objective = float(obj_val)
                    except (TypeError, ValueError):
                        pass

            logger.info(
                "Solved %s: status=%s, time=%.2fs, objective=%s",
                os.path.basename(problem_path),
                status_str,
                wall_time,
                objective,
            )

            return {
                "status": status_str,
                "objective": objective,
                "wall_time": float(wall_time),
                "solved": bool(solved),
            }

        except Exception as e:
            wall_time = time.perf_counter() - start
            logger.exception("Error solving %s: %s", problem_path, e)
            return {
                "status": f"ERROR: {type(e).__name__}",
                "objective": None,
                "wall_time": float(wall_time),
                "solved": False,
                "error": str(e),
            }
        finally:
            # Clean up temporary file if created
            if tmp_path is not None:
                self._cleanup_tmp_file(tmp_path, problem_path)
