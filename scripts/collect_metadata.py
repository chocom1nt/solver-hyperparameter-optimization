"""Collect metadata for all `.mps` / `.mps.gz` instances.

This script scans a directory, calls `extract_metadata()` for each instance,
and stores results into `data/metadata.csv`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import pandas as pd

# Ensure `coursework/` is on sys.path so that `import src...` works
# when executing `python coursework/scripts/collect_metadata.py`.
_COURSEWORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _COURSEWORK_ROOT not in sys.path:
    sys.path.insert(0, _COURSEWORK_ROOT)

from src.data.metadata_extractor import extract_metadata  # noqa: E402

logger = logging.getLogger(__name__)


def _task_name_from_path(path: str) -> str:
    """Derive task name from a `.mps` / `.mps.gz` file path."""

    base = os.path.basename(path)
    if base.lower().endswith(".gz"):
        base = os.path.splitext(base)[0]
    return os.path.splitext(base)[0]


def collect_metadata(data_dir: str, output_csv: str) -> None:
    """Run metadata extraction and write results to CSV."""

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            df_existing = pd.read_csv(output_csv)
            existing_names = set(df_existing["name"].astype(str).tolist()) if "name" in df_existing.columns else set()
        except Exception:
            existing_names = set()
    else:
        existing_names = set()

    supported_exts = (".mps", ".mps.gz")
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(supported_exts)
    ]
    files = sorted(files)

    rows: list[dict[str, Any]] = []
    for idx, path in enumerate(files, start=1):
        name = _task_name_from_path(path)
        if name in existing_names:
            continue
        logger.info("Extracting metadata %d/%d: %s", idx, len(files), name)
        try:
            rows.append(extract_metadata(path))
        except Exception as e:
            logger.exception("Failed to extract metadata for %s: %s", path, e)

    if not rows:
        logger.info("No new instances found. metadata.csv unchanged.")
        return

    df_new = pd.DataFrame(rows)
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        df_new.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df_new.to_csv(output_csv, mode="w", header=True, index=False)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    # Defaults are resolved relative to the `coursework/` directory so that
    # the script is runnable from different working directories.
    coursework_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_dir = os.path.join(coursework_root, "data", "raw")
    default_output = os.path.join(coursework_root, "data", "metadata.csv")

    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help="Directory with .mps/.mps.gz files (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Path to metadata.csv (default: data/metadata.csv).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    collect_metadata(args.data_dir, args.output)


if __name__ == "__main__":
    main()

