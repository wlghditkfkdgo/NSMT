#!/usr/bin/env bash
# hp_driver.sh (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG head input form pre-registration): runs the head input form stages in the
# user's order (11-03: CinC -> MIT-BIH, classification only) without waiting for a session, as hm_driver.sh: queue
# (make_queue_hp.sh) -> pool -> completeness check (all RESULT lines, no SKIP) -> energy recount (MIT-BIH) -> summarize_hp.py ->
# next stage. Fail closed: a missing run, a SKIP, a failed recount or summary stops the driver. Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/hp_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
R="$C/results/flif-v1-headpool"; mkdir -p "$R"; LOGF="$R/driver.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
wait_pool() { while ps -eo args | grep -q "[g]pu_pool2.sh $1"; do sleep 120; done; }
declare -A NEED=([cinc]=24 [mitbih]=24)
log "driver start (slots $SLOTS)"
for t in cinc mitbih; do
  S="$C/results/flif-v1-headpool-$t"; mkdir -p "$S"
  qf=$(cd "$C/scripts" && bash make_queue_hp.sh "hp_$t" 2>>"$LOGF" | tail -1 | cut -d: -f1)
  [ -s "$qf" ] || { log "STOP: queue for $t missing or empty ($qf)"; exit 1; }
  rel=${qf#$N/}
  (cd "$N" && nohup bash scripts/gpu_pool2.sh "$rel" "$SLOTS" > "$S/pool_log.txt" 2>&1 &)
  log "launched $(wc -l < "$qf") jobs: $rel"
  sleep 30; wait_pool "$rel"
  nr=$(grep -c '^RESULT' "$S/raw.txt" 2>/dev/null); ns=$(grep -c '^SKIP' "$S/raw.txt" 2>/dev/null)
  log "pool done: $t (RESULT ${nr:-0}, SKIP ${ns:-0})"
  [ "${nr:-0}" -eq "${NEED[$t]}" ] && [ "${ns:-0}" -eq 0 ] || { log "STOP: $t incomplete (need ${NEED[$t]} RESULT, 0 SKIP)"; exit 1; }
  if [ "$t" = mitbih ]; then
    (cd "$C" && $PY -u energy_recount_mitbih.py --suite flif-v1-headpool-mitbih >> "$LOGF" 2>&1) || { log "STOP: MIT-BIH energy recount failed"; exit 1; }
  fi
  (cd "$C" && $PY summarize_hp.py --task $t >> "$LOGF" 2>&1) || { log "STOP: summary of $t failed"; exit 1; }
  log "summary written: $R/summary_$t.md"
done
log "driver end"
