#!/usr/bin/env bash
# ad_driver.sh (2026-11-02, exp/f-lif-pop-v3; PROJECT_LOG AD pre-registration): runs every pre-registered stage of the
# anomaly-detection tuning in order without waiting for a session, as tune_driver.sh does for the other tasks (that file is in
# use by the CinC driver and is not edited): queue (make_queue_ad.sh) -> pool -> validation-only selection
# (summarize_ad_tune.py writes selected_<stage>.json) -> next stage ... -> final retraining on the whole training file -> A1-A4.
# Fail closed: a stage whose selection file is not written (missing run, SKIP) stops the driver; nothing is selected by hand.
# Logging to docs/PROJECT_LOG.md and commits stay with the session (driver.log in the suite folder records every step).
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/ad_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SUITE=flif-v1-ad-spkmux-tune
R="$C/results/$SUITE"; mkdir -p "$R"; LOGF="$R/driver.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
wait_pool() { while ps -eo args | grep -q "[g]pu_pool2.sh $1"; do sleep 120; done; }
run_stage() {                                                 # $1 = make_queue stage, $2 = pool log name
  local qf rel
  qf=$(cd "$C/scripts" && bash make_queue_ad.sh "$1" 2>>"$LOGF" | tail -1 | cut -d: -f1)
  [ -s "$qf" ] || { log "STOP: queue for $1 missing or empty ($qf)"; exit 1; }
  rel=${qf#$N/}
  (cd "$N" && nohup bash scripts/gpu_pool2.sh "$rel" "$SLOTS" > "$R/pool_log_$2.txt" 2>&1 &)
  log "launched $(wc -l < "$qf") jobs: $rel"
  sleep 30; wait_pool "$rel"
  log "pool done: $rel (RESULT $(grep -c '^RESULT' "$R/raw.txt" 2>/dev/null), SKIP $(grep -c '^SKIP' "$R/raw.txt" 2>/dev/null))"
}
log "driver start (slots $SLOTS)"
for st in lr layers res; do
  run_stage "adtune_$st" "$st"
  (cd "$C" && $PY summarize_ad_tune.py --stage $st >> "$LOGF" 2>&1)
  [ -s "$R/selected_$st.json" ] || { log "STOP: no selection for $st (incomplete stage)"; exit 1; }
  log "selected_$st.json: $(tr -d '\n ' < "$R/selected_$st.json" | cut -c1-400)"
done
run_stage adfinal final
(cd "$C" && $PY summarize_ad_tune.py --stage final >> "$LOGF" 2>&1) && log "final summary written: $R/summary_final.md" || log "STOP: final summary failed"
log "driver end"
