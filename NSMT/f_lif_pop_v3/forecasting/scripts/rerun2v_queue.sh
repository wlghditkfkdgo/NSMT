#!/usr/bin/env bash
# Prereg 2V: the training queue of the six new ablation cells x {q1_R, q1} x {ETTh1, ETTh2} x 8 seeds = 192 runs,
# for NSMT/scripts/gpu_pool2.sh, and its git-tracked copy results/<suite>/run_commands.txt. q1_R (R on) first.
# usage: rerun2v_queue.sh <suite>
set -u
SUITE=${1:?suite}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q=$W/scripts/queues/rerun-ett-${SUITE#etthard-}.txt; RES=$W/results/$SUITE; mkdir -p "$(dirname "$Q")" "$RES"
[ -e "$Q" ] && { echo "queue $Q exists"; exit 1; }
for B in q1_R q1; do
  for S in 7 13 21 42 123 256 512 1024; do
    for CELL in lif-frozen-hidden lif-bn-flatten lif-bn-hidden none-frozen-flatten none-frozen-hidden none-bn-flatten; do
      for DS in ETTh1 ETTh2; do
        echo "SUITE=$SUITE bash f_lif_pop_v3/forecasting/scripts/rerun2v_ett.sh $DS 96 $B-$CELL $S" | tee -a "$Q" >> "$RES/run_commands.txt"
      done
    done
  done
done
echo "[queue] $(grep -c . "$Q") runs in $Q"
