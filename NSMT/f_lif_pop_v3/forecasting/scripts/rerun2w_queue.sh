#!/usr/bin/env bash
# Prereg 2W: the selection queue -- 12 settings (rerun_common.GRID_2W) x {q1_R, q1} x {ETTh1, ETTh2} x seeds
# {7, 13, 21} = 144 runs -- and its git-tracked copy results/<suite>/run_commands.txt. Seeds outermost so a
# partial queue covers every setting.  usage: rerun2w_queue.sh <suite>
set -u
SUITE=${1:?suite}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q=$W/scripts/queues/rerun-ett-${SUITE#etthard-}-select.txt; RES=$W/results/$SUITE; mkdir -p "$(dirname "$Q")" "$RES"
[ -e "$Q" ] && { echo "queue $Q exists"; exit 1; }
for S in 7 13 21; do
  for B in q1_R q1; do
    for A in direct hmem; do for LR in 0.0005 0.001 0.002; do for WD in all xnb; do
      for DS in ETTh1 ETTh2; do
        echo "SUITE=$SUITE bash f_lif_pop_v3/forecasting/scripts/rerun2w_ett.sh $DS 96 $B-$A-$LR-$WD $S" | tee -a "$Q" >> "$RES/run_commands.txt"
      done
    done; done; done
  done
done
echo "[queue] $(grep -c . "$Q") runs in $Q"
