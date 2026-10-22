#!/usr/bin/env bash
# Prereg 2X: 2 patchings (p8s4, p16s8) x {q1_R, q1} x {ETTh1, ETTh2} x seeds {7, 13, 21} = 24 runs (p8s4 first,
# it is the slow one), and the git-tracked copy results/<suite>/run_commands.txt.   usage: rerun2x_queue.sh <suite>
set -u
SUITE=${1:?suite}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q=$W/scripts/queues/rerun-ett-${SUITE#etthard-}.txt; RES=$W/results/$SUITE; mkdir -p "$(dirname "$Q")" "$RES"
[ -e "$Q" ] && { echo "queue $Q exists"; exit 1; }
for PATCH in p8s4 p16s8; do for B in q1_R q1; do for S in 7 13 21; do for DS in ETTh1 ETTh2; do
  echo "SUITE=$SUITE bash f_lif_pop_v3/forecasting/scripts/rerun2x_ett.sh $DS 96 $B-$PATCH $S" | tee -a "$Q" >> "$RES/run_commands.txt"
done; done; done; done
echo "[queue] $(grep -c . "$Q") runs in $Q"
