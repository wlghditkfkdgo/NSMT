#!/usr/bin/env bash
# Prereg 2T D-CP (2U D-CW): a run cut short by an infrastructure fault (here: CUDA out of memory because
# another job filled the shared GPU) is rerun ONCE with the same settings and seed, and that is recorded.
# A G11 stop is a model outcome and is never rerun. For every SKIP line of the suite this script
#   1. requires the run's stdout to show 'CUDA out of memory' and no G11 stop (anything else: refuse),
#   2. refuses a cell that was already requeued once (results/<suite>/infra_reruns.txt),
#   3. moves the failed run directory and stdout to log/aborted/<suite>-infra-<stamp>/ (nothing is deleted),
#      so the rerun gets the cell's only run directory and the evaluator finds exactly one stdout,
#   4. writes the queue line (same launcher, dataset, condition, seed) and an infra_reruns.txt record.
# usage: rerun_infra_requeue.sh <suite> <launcher>        e.g. etthard-2u-20261021 rerun2u_ett.sh
set -u
SUITE=${1:?suite}; LAUNCHER=${2:?launcher}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES=$W/results/$SUITE; LOG=$W/log/$SUITE; STAMP=$(date +%Y%m%dT%H%M%S)
AB=$W/log/aborted/$SUITE-infra-$STAMP; Q=$W/scripts/queues/rerun-infra-${SUITE#etthard-}-$STAMP.txt
REC=$RES/infra_reruns.txt
grep '^SKIP' "$RES/raw.txt" | while read -r line; do
  DS=$(echo "$line" | grep -oE 'ds=[A-Za-z0-9]+' | cut -d= -f2); C=$(echo "$line" | grep -oE 'cond=[A-Za-z0-9_]+' | cut -d= -f2)
  S=$(echo "$line" | grep -oE 'seed=[0-9]+' | cut -d= -f2)
  key="$DS $C $S"
  if [ -f "$REC" ] && grep -q "^REQUEUED $key " "$REC"; then
    # already rerun once: only a SKIP of the earlier attempt is expected here; a second failure stays a failure
    continue
  fi
  OUT=$(ls "$LOG"/ett_${DS}_p96_${C}_seed${S}_*.stdout 2>/dev/null)
  [ "$(echo "$OUT" | grep -c .)" = 1 ] || { echo "REFUSED $key: $(echo "$OUT" | grep -c .) stdout files"; continue; }
  if grep -aq 'G11 violated' "$OUT" || ! grep -aq 'CUDA out of memory' "$OUT"; then
    echo "REFUSED $key: not an out-of-memory failure"; continue
  fi
  RUN=$(grep -a '^save_result_path' "$OUT" | head -1 | sed -E 's/^save_result_path-*//')
  mkdir -p "$AB"
  [ -n "$RUN" ] && [ -d "$RUN" ] && mv "$RUN" "$AB/$(basename "$RUN")_${DS}"
  mv "$OUT" "$AB/"
  echo "SUITE=$SUITE bash f_lif_pop_v3/forecasting/scripts/$LAUNCHER $DS 96 $C $S" >> "$Q"
  echo "REQUEUED $key reason=CUDA_out_of_memory moved_to=log/aborted/$(basename "$AB") queue=$(basename "$Q") at=$STAMP" | tee -a "$REC"
done
[ -f "$Q" ] && echo "[requeue] $(grep -c . "$Q") runs in $Q" || echo "[requeue] nothing to requeue"
