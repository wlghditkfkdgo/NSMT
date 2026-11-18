#!/usr/bin/env bash
# MODI 2026-11-18: driver of the long-input learning-rate test (docs/MODI_PROJECT_LOG.md; run from NSMT/ in tmux, one run per GPU).
#   1. phase A: the queue of make_queue_lrlong.sh (3 retrain runs + 72 selection runs, seed 42)
#   2. stop unless every phase-A suite has its RESULT rows and no SKIP; then select_lrlong.py (validation only) writes the selection
#      JSON and the phase-B queue (seeds 7 / 2026 at each selected lr that is not the default)
#   3. phase B: that queue (may be empty)
# Every line goes to nsmt_flif_v1/forecasting/results/flif-v1-ett-lrlong-pool.log. gpu_pool2.sh always prints rc=0; failures are the
# SKIP lines of the suites. Energy recount, the summary and the log entry are done by hand afterwards.
set -u
GPUS=${1:-0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; FR=$PWD/nsmt_flif_v1/forecasting/results; QD=$C/scripts/queues
QA=$QD/lrlong-a-20261118.txt; QB=$QD/lrlong-b-20261118.txt; PL=$FR/flif-v1-ett-lrlong-pool.log
say() { echo "[driver $(date '+%F %T')] $*" | tee -a "$PL"; }
say "phase A start: $(grep -c . "$QA") jobs"
bash scripts/gpu_pool2.sh "$QA" "$GPUS" 2>&1 | tee -a "$PL"
nr=0; ns=0
for s in $FR/flif-v1-ett-lr-repro $FR/flif-v1-ett-lr{720,336}-{5e-05,0.0001,0.00025}; do
  r=$(grep -c '^RESULT' "$s/raw.txt" 2>/dev/null); k=$(grep -c '^SKIP' "$s/raw.txt" 2>/dev/null)
  nr=$((nr + ${r:-0})); ns=$((ns + ${k:-0})); say "  $(basename $s): ${r:-0} RESULT, ${k:-0} SKIP"
done
say "phase A done: $nr RESULT, $ns SKIP (expected 75, 0)"
if [ "$nr" -ne 75 ] || [ "$ns" -ne 0 ]; then say "STOP: phase A incomplete -- no selection, no phase B"; exit 1; fi
rm -f "$QB"; (cd "$C" && $PY select_lrlong.py --queue_b "$QB") 2>&1 | tee -a "$PL"
if [ "${PIPESTATUS[0]}" -ne 0 ] || [ ! -e "$QB" ]; then say "STOP: selection failed -- no phase B"; exit 1; fi
say "phase B start: $(grep -c . "$QB") jobs"
[ -s "$QB" ] && bash scripts/gpu_pool2.sh "$QB" "$GPUS" 2>&1 | tee -a "$PL"
say "ALL DONE"
