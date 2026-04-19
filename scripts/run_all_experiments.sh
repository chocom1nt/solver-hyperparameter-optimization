#!/usr/bin/env bash
set -euo pipefail

# Example sequential launcher for multiple configs.
# On Windows you can run the same commands via PowerShell.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_1="${ROOT_DIR}/experiments/configs/default.yaml"

python "${ROOT_DIR}/experiments/run_grid.py" --config "${CONFIG_1}"

