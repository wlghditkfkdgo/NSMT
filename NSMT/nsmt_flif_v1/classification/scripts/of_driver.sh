#!/usr/bin/env bash
# MODI 2026-11-19: driver of (가) the full-signal aux test and (나) the overfitting test (docs/MODI_PROJECT_LOG.md; plan approved 11-19).
# Run from NSMT/ in tmux, one run per GPU:
#   F   make_queue_auxf.sh -> pool (60 runs); stop unless flif-v1-ett-auxf has 48 RESULT and flif-v1-mitbih-auxf 12, no SKIP
#   S   select_of.py --stage S  (aux form per neuron, validation)          -> queue A1 -> pool (24 runs, seed 42)
#   A1  select_of.py --stage A1 (learning rate)                            -> queue A2 -> pool (24)
#   A2  select_of.py --stage A2 (head dropout)                             -> queue A3 -> pool (24)
#   A3  select_of.py --stage A3 (weight decay)                             -> queue B  -> pool (<= 24; seeds 7 / 2026 of the final settings)
# select_of.py checks every row it needs (missing / duplicated / SKIP / wrong setting -> exit 1); a failed selection stops the driver.
# Every line goes to nsmt_flif_v1/forecasting/results/flif-v1-of-pool.log. gpu_pool2.sh always prints rc=0. Energy recount, summaries
# and log entries are done by hand.
set -u
GPUS=${1:-0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; FR=$PWD/nsmt_flif_v1/forecasting/results; QD=$C/scripts/queues; PL=$FR/flif-v1-of-pool.log
say() { echo "[driver $(date '+%F %T')] $*" | tee -a "$PL"; }
pool() { local q=$1; say "pool start: $(basename "$q") $(grep -c . "$q") jobs"; [ -s "$q" ] && bash scripts/gpu_pool2.sh "$q" "$GPUS" 2>&1 | tee -a "$PL"; say "pool done: $(basename "$q")"; }
bash "$C/scripts/make_queue_auxf.sh" | tee -a "$PL"
pool "$QD/auxf-20261119.txt"
e=$(grep -c '^RESULT' "$FR/flif-v1-ett-auxf/raw.txt" 2>/dev/null); m=$(grep -c '^RESULT' "$C/results/flif-v1-mitbih-auxf/raw.txt" 2>/dev/null)
k=$(cat "$FR/flif-v1-ett-auxf/raw.txt" "$C/results/flif-v1-mitbih-auxf/raw.txt" 2>/dev/null | grep -c '^SKIP')
say "phase F: ETT ${e:-0} RESULT, MIT-BIH ${m:-0} RESULT, ${k:-0} SKIP (expected 48, 12, 0)"
if [ "${e:-0}" -ne 48 ] || [ "${m:-0}" -ne 12 ] || [ "${k:-0}" -ne 0 ]; then say "STOP: phase F incomplete"; exit 1; fi
prev=S
for nxt in A1 A2 A3 B; do
  q=$QD/of-$nxt-20261119.txt; rm -f "$q"
  (cd "$C" && $PY select_of.py --stage $prev --queue_next "$q") 2>&1 | tee -a "$PL"
  if [ "${PIPESTATUS[0]}" -ne 0 ] || [ ! -e "$q" ]; then say "STOP: selection $prev failed"; exit 1; fi
  pool "$q"
  prev=$nxt
done
say "ALL DONE (B finished; run summarize_of.py after the energy recounts)"
