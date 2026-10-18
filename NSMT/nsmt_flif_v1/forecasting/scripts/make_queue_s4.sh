#!/usr/bin/env bash
# Writes scripts/queues/flif-v1-s4.txt (queue files are git-ignored; this script is the record).
# Stage 4 (MODI_PROJECT_LOG 2026-10-18): the neocortex path alone as a forecaster (--model neoOnly) and a linear baseline.
#   unit {step, patch} x neuron {lif, flif} x {on+flat, off+flat (T_s = 4), on+last} = 12 conditions, + linear
#   x ETTh2/ETTm2 x H96/H720 x seeds 42/7/2026 = 156 runs; patience 10 + best-validation weights; slowest first.
# usage: make_queue_s4.sh [SUITE [EPOCHS [CELLS...]]]   (smoke: make_queue_s4.sh smoke-s4-<date> 1 "ETTh1 96")
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; SUITE=${1:-flif-v1-s4}; EP=${2:-}; shift $(( $# > 2 ? 2 : $# ))
CELLS=("$@"); [ ${#CELLS[@]} -eq 0 ] && CELLS=("ETTm2 96" "ETTm2 720" "ETTh2 96" "ETTh2 720")
SEEDS="42 7 2026"; [ -n "$EP" ] && SEEDS="42"
Q="$W/scripts/queues/$SUITE.txt"; : > "$Q"
C="PATIENCE=10 TEST_BEST=1 SUITE=$SUITE${EP:+ EPOCHS=$EP}"
CONDS="neo_step_on_flat_flif neo_step_on_last_flif neo_step_on_flat_lif neo_step_on_last_lif neo_step_off_flat_flif neo_step_off_flat_lif
       neo_patch_on_flat_flif neo_patch_on_last_flif neo_patch_on_flat_lif neo_patch_on_last_lif neo_patch_off_flat_flif neo_patch_off_flat_lif linear"
for cell in "${CELLS[@]}"; do for c in $CONDS; do for s in $SEEDS; do
  echo "COND=$c $C bash nsmt_flif_v1/forecasting/scripts/run_ett.sh $cell $s" >> "$Q"; done; done; done
echo "$Q: $(wc -l < "$Q") lines"
