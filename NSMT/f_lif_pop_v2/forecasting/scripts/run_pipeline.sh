#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-/home/yschoi/.conda/envs/snn_recall/bin/python}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$EXPERIMENT_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$EXPERIMENT_PYTHON" "$TASK_DIR/scripts/run_pipeline.py" --finalize "$@"
