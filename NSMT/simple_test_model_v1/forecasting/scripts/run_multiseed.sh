#!/usr/bin/env bash
# Extend the fixed seed-7 quick matrix with seeds 13 and 21, one suite at a time.
# Each suite uses all four GPUs, one independent process per GPU. Existing
# results are protected by the launcher's duplicate-suite checks.
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for EXPERIMENT_SEED in 13 21; do
  bash "$TASK_DIR/scripts/run_parallel.sh" \
    --suite "ett-multiseed-20260911-seed${EXPERIMENT_SEED}" \
    --gpus 0 1 2 3 --epochs 10 --patience 3 --batch-size 128 --seed "$EXPERIMENT_SEED"
done
