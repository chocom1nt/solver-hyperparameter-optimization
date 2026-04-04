"""Utilities to extract instance metadata from MPS files.

The implementation uses OR-Tools' MPS reader. For some advanced metadata
categories it may be helpful to use PySCIPOpt, but for the scope of this
coursework we keep it lightweight.
"""

from __future__ import annotations

import gzip
import os
import re
import tempfile
from typing import Any

from ortools.linear_solver.python import model_builder


def _unpack_if_needed(mps_file: str) -> tuple[str, bool]:
    """Unpack `.mps.gz` into a temporary `.mps` file if required."""

    if not mps_file.lower().endswith(".gz"):
        return mps_file, False

    with gzip.open(mps_file, "rb") as f_in:
        fd, tmp_path = tempfile.mkstemp(suffix=".mps")
        os.close(fd)
        with open(tmp_path, "wb") as f_out:
            f_out.write(f_in.read())
    return tmp_path, True


def _safe_int(x: Any) -> int:
    """Convert OR-Tools numeric fields into an int when possible."""

    try:
        return int(x)
    except Exception:
        # Fall back to float->int if the value is given as a float.
        return int(float(x))


def extract_metadata(mps_file: str) -> dict[str, Any]:
    """Extract problem metadata from an MPS file.

    Args:
        mps_file: Path to an `.mps` or `.mps.gz` file.

    Returns:
        Dictionary with the following keys:
        - `name`: file name without extension
        - `num_vars`
        - `num_binary`
        - `num_integer`
        - `num_continuous`
        - `num_constraints`
        - `obj_sense`: 'min' or 'max'
    """

    unpacked_path: str | None = None
    tmp_to_remove = False
    try:
        unpacked_path, tmp_to_remove = _unpack_if_needed(mps_file)

        base = os.path.basename(mps_file)
        if base.lower().endswith(".gz"):
            base = os.path.splitext(base)[0]
        name = os.path.splitext(base)[0]

        model = model_builder.Model()
        ok = model.import_from_mps_file(unpacked_path)
        if not ok:
            raise RuntimeError(f"Failed to import MPS: {unpacked_path}")

        num_vars = int(model.num_variables)
        num_constraints = int(model.num_constraints)

        # Objective sense:
        # Many MPS instances encode it as `OBJSENSE MAX` / `OBJSENSE MIN`.
        # If absent, default to `min`.
        obj_sense = "min"
        try:
            with open(
                unpacked_path, "rt", encoding="utf-8", errors="ignore"
            ) as f:
                for _ in range(500):
                    line = f.readline()
                    if not line:
                        break
                    match = re.search(
                        r"\bOBJSENSE\b\s+([A-Za-z]+)",
                        line,
                        flags=re.IGNORECASE,
                    )
                    if match:
                        val = match.group(1).strip().upper()
                        if val.startswith("MAX"):
                            obj_sense = "max"
                        elif val.startswith("MIN"):
                            obj_sense = "min"
                        break
        except OSError:
            pass

        num_binary = 0
        num_integer_non_binary = 0
        num_continuous = 0

        for i in range(num_vars):
            var = model.var_from_index(i)
            is_integral = bool(var.is_integral)
            lb = float(var.lower_bound)
            ub = float(var.upper_bound)

            # Heuristic: binary variables are integral with bounds [0, 1].
            if is_integral and abs(lb - 0.0) <= 1e-9 and abs(ub - 1.0) <= 1e-9:
                num_binary += 1
            elif is_integral:
                num_integer_non_binary += 1
            else:
                num_continuous += 1

        return {
            "name": name,
            "num_vars": _safe_int(num_vars),
            "num_binary": _safe_int(num_binary),
            "num_integer": _safe_int(num_integer_non_binary),
            "num_continuous": _safe_int(num_continuous),
            "num_constraints": _safe_int(num_constraints),
            "obj_sense": obj_sense,
        }
    finally:
        if tmp_to_remove and unpacked_path is not None:
            try:
                os.remove(unpacked_path)
            except OSError:
                pass

