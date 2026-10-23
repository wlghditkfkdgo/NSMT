#!/usr/bin/env bash
# Prereg 2Y: 5 new cells (mem-96-all, mem-336-last1, mem-96-last1, nomem-336-all, nomem-336-last1) x {q1_R, q1} x
# {ETTh1, ETTh2} x seeds {7, 13, 21} = 60 runs; mem-336-all is the 2W pick (not retrained). Also the git-tracked copy
# results/<suite>/run_commands.txt.   usage: rerun2y_queue.sh <suite>
set -u
SUITE=${1:?suite}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q=$W/scripts/queues/rerun-ett-${SUITE#etthard-}.txt; RES=$W/results/$SUITE; mkdir -p "$(dirname "$Q")" "$RES"
[ -e "$Q" ] && { echo "queue $Q exists"; exit 1; }
for CELL in mem-336-last1 nomem-336-all nomem-336-last1 mem-96-all mem-96-last1; do for B in q1_R q1; do for S in 7 13 21; do
  for DS in ETTh1 ETTh2; do
    echo "SUITE=$SUITE bash f_lif_pop_v3/forecasting/scripts/rerun2y_ett.sh $DS 96 $B-$CELL $S" | tee -a "$Q" >> "$RES/run_commands.txt"
  done
done; done; done
echo "[queue] $(grep -c . "$Q") runs in $Q"
