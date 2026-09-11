#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$TASK_DIR/scripts/run_parallel.sh" \
  --suite ett-population-ablation-20260911 --gpus 0 1 2 3 \
  --variants temporal no_attention --population-codes gaussian repeat \
  --seeds 7 13 21 --epochs 10 --patience 3 --batch-size 128
