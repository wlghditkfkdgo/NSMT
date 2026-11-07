#!/usr/bin/env bash
# l2_driver.sh (2026-11-07, exp/f-lif-pop-v3; PROJECT_LOG L2 pre-registration): MSE-loss test of the tuned SpkMixer on ETT, without waiting for a session:
# queue (make_queue_l2.sh l2) -> pool -> completeness (12 RESULT lines, no SKIP) -> energy_recount_ett.py -> summarize_l2.py. Fail closed: any failure stops the driver.
# Logging and commits stay with the session. usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/l2_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SUITE=flif-v1-ett-spkmux-l2
S="$C/results/$SUITE"; mkdir -p "$S"; LOGF="$S/driver.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
log "driver start (slots $SLOTS)"
qf=$(bash "$C/scripts/make_queue_l2.sh" l2 2>>"$LOGF" | tail -1 | cut -d: -f1)
[ -s "$qf" ] || { log "STOP: queue missing or empty ($qf)"; exit 1; }
log "launched $(wc -l < "$qf") jobs: ${qf#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$S/pool_log.txt" 2>&1
nr=$(grep -c '^RESULT' "$S/raw.txt" 2>/dev/null); ns=$(grep -c '^SKIP' "$S/raw.txt" 2>/dev/null)
log "pool done (RESULT ${nr:-0}, SKIP ${ns:-0})"
[ "${nr:-0}" -eq 12 ] && [ "${ns:-0}" -eq 0 ] || { log "STOP: incomplete (need 12 RESULT, 0 SKIP)"; exit 1; }
(cd "$C" && $PY -B -u energy_recount_ett.py --suite "$SUITE" >> "$LOGF" 2>&1) || { log "STOP: energy recount failed"; exit 1; }
(cd "$C" && $PY -B summarize_l2.py --suite "$SUITE" >> "$LOGF" 2>&1) || { log "STOP: summary failed"; exit 1; }
log "summary written: $S/summary.md"
log "driver end"
