#!/usr/bin/env bash
# Prereg 2T: write the ETT training queue for NSMT/scripts/gpu_pool2.sh (one line per run, the worker
# appends the GPU) and its git-tracked copy results/<suite>/run_commands.txt. Order 2N -> 2O -> 2P -> 2Q (B),
# datasets and seeds interleaved. A condition whose calibration was refused (calibration_manifest.json)
# is not queued (D-CN).
# usage: rerun_ett_queue.sh <suite> <calibration suite>
set -u
SUITE=${1:?suite}; CAL=${2:?calibration suite}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q=$W/scripts/queues/rerun-ett-${SUITE#etthard-spikede-}.txt; RES=$W/results/$SUITE
mkdir -p "$(dirname "$Q")" "$RES"
[ -e "$Q" ] && { echo "queue $Q exists"; exit 1; }
refused() {   # <data> <cond> -> 0 if its calibration was refused
  python3 - "$W/results/$CAL/calibration_manifest.json" "$1" "$2" <<'PY'
import json, sys
m, data, cond = json.load(open(sys.argv[1]))['entries'], sys.argv[2], sys.argv[3]
regime = 'Roff' if '_R' not in cond else ('R_a1' if cond.endswith('_a1') else 'R')
label = f"{data}_{regime}_{'q1' if cond.startswith('q1') else 'pearson'}"
sys.exit(0 if m[label]['refused'] else 1)
PY
}
for C in q1 pearson q1_R pearson_R q1_R_a1 pearson_R_a1 q1_R_lin pearson_R_lin; do
  for S in 7 13 21 42 123 256 512 1024; do
    for DS in ETTh1 ETTh2; do
      if refused $DS $C; then echo "# not queued (calibration refused): $DS $C seed $S" >> "$RES/run_commands.txt"; continue; fi
      echo "SUITE=$SUITE bash f_lif_pop_v3/forecasting/scripts/rerun_ett.sh $DS 96 $C $S" | tee -a "$Q" >> "$RES/run_commands.txt"
    done
  done
done
echo "[queue] $(grep -c . "$Q") runs in $Q; $(grep -c '^#' "$RES/run_commands.txt") not queued"
