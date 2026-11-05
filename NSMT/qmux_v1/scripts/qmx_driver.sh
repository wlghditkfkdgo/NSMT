#!/usr/bin/env bash
# qmx_driver.sh (2026-11-05, exp/f-lif-pop-v3; PROJECT_LOG bidirectional Neocortex pre-registration): runs the stages in the user's order (11-05: ETT -> MIT-BIH)
# without waiting for a session, as cinc_mixer_v1/scripts/hp_driver.sh: queue (make_queue_qmx.sh) -> pool -> completeness (all RESULT lines, no SKIP) -> ETT energy
# recount -> summarize_qmx.py -> next stage. Fail closed: a missing run, a SKIP, a failed recount or an incomplete summary stops the driver. Logging and commits stay
# with the session. usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash qmux_v1/scripts/qmx_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/qmux_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
R="$C/results/qmx-driver"; mkdir -p "$R"; LOGF="$R/driver.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
wait_pool() { while ps -eo args | grep -q "[g]pu_pool2.sh $1"; do sleep 120; done; }
declare -A NEED=([ett]=204 [mitbih]=54)
log "driver start (slots $SLOTS)"
for t in ett mitbih; do
  S="$C/results/qmx-$t"; mkdir -p "$S"
  qf=$(cd "$C/scripts" && bash make_queue_qmx.sh "qmx_$t" 2>>"$LOGF" | tail -1 | cut -d: -f1)
  [ -s "$qf" ] || { log "STOP: queue for $t missing or empty ($qf)"; exit 1; }
  rel=${qf#$N/}
  (cd "$N" && nohup bash scripts/gpu_pool2.sh "$rel" "$SLOTS" > "$S/pool_log.txt" 2>&1 &)
  log "launched $(wc -l < "$qf") jobs: $rel"
  sleep 30; wait_pool "$rel"
  nr=$(grep -c '^RESULT' "$S/raw.txt" 2>/dev/null); ns=$(grep -c '^SKIP' "$S/raw.txt" 2>/dev/null)
  log "pool done: $t (RESULT ${nr:-0}, SKIP ${ns:-0})"
  [ "${nr:-0}" -eq "${NEED[$t]}" ] && [ "${ns:-0}" -eq 0 ] || { log "STOP: $t incomplete (need ${NEED[$t]} RESULT, 0 SKIP)"; exit 1; }
  if [ "$t" = ett ]; then
    (cd "$C" && $PY -B -u energy_recount_ett.py --suite qmx-ett >> "$LOGF" 2>&1) || { log "STOP: ETT energy recount failed"; exit 1; }
  fi
  (cd "$C" && $PY -B summarize_qmx.py --task $t >> "$LOGF" 2>&1) || { log "STOP: summary of $t failed"; exit 1; }
  log "summary written: $S/summary.md"
done
log "driver end"
