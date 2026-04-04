"""Persist experiment results to CSV or SQLite."""

from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _sanitize_col_name(key: str) -> str:
    """Convert parameter key into a safe SQL/CSV column name."""

    col = "".join(ch if ch.isalnum() else "_" for ch in key)
    while "__" in col:
        col = col.replace("__", "_")
    return f"param_{col}".strip("_")


@dataclass(frozen=True)
class ResultWriterConfig:
    """Writer configuration."""

    format: str
    path: str
    param_keys: list[str]


class ResultWriter:
    """Write normalized experiment results."""

    def __init__(self, output_config: dict[str, Any], param_keys: list[str]):
        format_ = str(output_config.get("format", "csv")).lower()
        path_ = str(output_config["path"])

        self._cfg = ResultWriterConfig(
            format=format_,
            path=path_,
            param_keys=list(param_keys),
        )

        self._param_cols = [_sanitize_col_name(k) for k in self._cfg.param_keys]
        self._base_cols = [
            "task_name",
            "solver_name",
            *self._param_cols,
            "status",
            "objective",
            "wall_time",
            "solved",
            "timestamp",
        ]

        if self._cfg.format not in {"csv", "sqlite"}:
            raise ValueError(f"Unsupported output format: {self._cfg.format}")

        out_dir = os.path.dirname(os.path.abspath(self._cfg.path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if self._cfg.format == "sqlite":
            self._conn = sqlite3.connect(self._cfg.path)
            self._ensure_table()
        else:
            self._conn = None

    def _ensure_table(self) -> None:
        """Create results table (if not exists)."""

        columns_sql = ", ".join(
            [
                "task_name TEXT NOT NULL",
                "solver_name TEXT NOT NULL",
                *[
                    # Params may be numeric or strings depending on SCIP.
                    f"{col} TEXT"
                    for col in self._param_cols
                ],
                "status TEXT",
                "objective REAL",
                "wall_time REAL",
                "solved INTEGER",
                "timestamp TEXT",
            ]
        )

        query = f"CREATE TABLE IF NOT EXISTS results ({columns_sql});"
        cur = self._conn.cursor()
        cur.execute(query)
        self._conn.commit()

    def close(self) -> None:
        """Close SQLite connection (no-op for CSV)."""

        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _row_from_result(
        self,
        *,
        task_name: str,
        solver_name: str,
        params: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a row matching the writer columns."""

        row: dict[str, Any] = {
            "task_name": task_name,
            "solver_name": solver_name,
            "status": result.get("status"),
            "objective": result.get("objective"),
            "wall_time": result.get("wall_time"),
            "solved": 1 if result.get("solved") else 0,
            "timestamp": _utc_timestamp(),
        }

        for key, col in zip(self._cfg.param_keys, self._param_cols):
            row[col] = params.get(key)
        return row

    def write_result(
        self,
        *,
        task_name: str,
        solver_name: str,
        params: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Append one result record.

        Args:
            task_name: Instance name.
            solver_name: Solver identifier.
            params: Parameters used (grid configuration).
            result: Normalized result dict from a solver wrapper.
        """

        row = self._row_from_result(
            task_name=task_name,
            solver_name=solver_name,
            params=params,
            result=result,
        )

        if self._cfg.format == "csv":
            self._write_csv_row(row)
        else:
            self._write_sqlite_row(row)

    def _write_csv_row(self, row: dict[str, Any]) -> None:
        """Append one row to CSV."""

        file_exists = os.path.exists(self._cfg.path)
        with open(self._cfg.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._base_cols)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in self._base_cols})

    def _write_sqlite_row(self, row: dict[str, Any]) -> None:
        """Insert one row into SQLite."""

        placeholders = ", ".join(["?"] * len(self._base_cols))
        cols_sql = ", ".join(self._base_cols)
        values = [row.get(c) for c in self._base_cols]

        cur = self._conn.cursor()
        cur.execute(
            f"INSERT INTO results ({cols_sql}) VALUES ({placeholders});",
            values,
        )
        self._conn.commit()

