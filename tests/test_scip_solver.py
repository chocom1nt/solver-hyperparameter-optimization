"""Unit tests for `ScipSolver`.

The test uses a tiny hand-written MPS MILP with two binary variables:

minimize    x + y
subject to  x + y >= 1
with x, y in {0, 1}
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure `coursework/` is on sys.path for `import src...`.
_COURSEWORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _COURSEWORK_ROOT not in sys.path:
    sys.path.insert(0, _COURSEWORK_ROOT)

ortools = pytest.importorskip("ortools")
from ortools.linear_solver.python import model_builder  # noqa: E402

from src.solvers.scip_solver import ScipSolver  # noqa: E402


def test_scip_solver_solve_returns_expected_structure(tmp_path):
    """SCIP should solve the toy MPS and return normalized result keys."""

    solver = model_builder.Solver("SCIP")
    if not solver.solver_is_supported():
        pytest.skip("OR-Tools SCIP backend is not available.")

    mps_text = """\
NAME          TOY
ROWS
 N  OBJ
 G  C1
COLUMNS
    X         OBJ       1
    X         C1        1
    Y         OBJ       1
    Y         C1        1
RHS
    RHS1      C1        1
BOUNDS
 BV BNDX      X         0
 BV BNDY      Y         0
ENDATA
"""

    mps_path = tmp_path / "toy.mps"
    mps_path.write_text(mps_text, encoding="utf-8")

    solver = ScipSolver()
    result = solver.solve(str(mps_path), params={}, time_limit=5)

    assert isinstance(result, dict)
    assert "status" in result
    assert "objective" in result
    assert "wall_time" in result
    assert "solved" in result

    assert result["wall_time"] >= 0.0
    assert result["status"] in {"OPTIMAL", "FEASIBLE"}

    # For a trivial problem, SCIP should at least find a feasible solution
    # within a short time limit.
    assert result["solved"] is True

