"""Run a grid search experiment based on a YAML configuration."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration (e.g. experiments/configs/default.yaml).",
    )
    args = parser.parse_args()

    # Ensure `coursework/` is on sys.path for `import src...`.
    coursework_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if coursework_root not in sys.path:
        sys.path.insert(0, coursework_root)

    # Local import after sys.path fix.
    from src.experiment.config_loader import load_config  # noqa: WPS433
    from src.experiment.runner import ExperimentRunner  # noqa: WPS433

    config = load_config(args.config)
    runner = ExperimentRunner(config, root_dir=coursework_root)
    runner.run()


if __name__ == "__main__":
    main()

