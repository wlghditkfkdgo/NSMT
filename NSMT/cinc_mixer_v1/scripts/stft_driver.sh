#!/usr/bin/env bash
# stft_driver.sh (2026-11-16, exp/f-lif-pop-v3; PROJECT_LOG STFT band test pre-registration): ETT, then MIT-BIH, without waiting for a session:
# queue (make_queue_stft.sh) -> pool -> completeness (RESULT count, no SKIP) -> energy recount -> summarize_stft.py. Fail closed: any failure stops the driver.
# Logging and commits stay with the session. usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/stft_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
stage() {   # stage <make_queue stage> <suite> <runs> <energy script>
  local st=$1 suite=$2 want=$3 en=$4 S qf nr ns
  S="$C/results/$suite"; mkdir -p "$S"; local LOGF="$S/driver.log"
  echo "[$(date '+%F %T')] driver start (slots $SLOTS)" >> "$LOGF"
  qf=$(bash "$C/scripts/make_queue_stft.sh" "$st" 2>>"$LOGF" | tail -1 | cut -d: -f1)
  [ -s "$qf" ] || { echo "[$(date '+%F %T')] STOP: queue missing or empty ($qf)" >> "$LOGF"; exit 1; }
  echo "[$(date '+%F %T')] launched $(wc -l < "$qf") jobs: ${qf#$N/}" >> "$LOGF"
  (cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$S/pool_log.txt" 2>&1
  nr=$(grep -c '^RESULT' "$S/raw.txt" 2>/dev/null); ns=$(grep -c '^SKIP' "$S/raw.txt" 2>/dev/null)
  echo "[$(date '+%F %T')] pool done (RESULT ${nr:-0}, SKIP ${ns:-0})" >> "$LOGF"
  [ "${nr:-0}" -eq "$want" ] && [ "${ns:-0}" -eq 0 ] || { echo "[$(date '+%F %T')] STOP: incomplete (need $want RESULT, 0 SKIP)" >> "$LOGF"; exit 1; }
  (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$en" --suite "$suite" >> "$LOGF" 2>&1) || { echo "[$(date '+%F %T')] STOP: energy recount failed" >> "$LOGF"; exit 1; }
  (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B summarize_stft.py --task "$st" >> "$LOGF" 2>&1) || { echo "[$(date '+%F %T')] STOP: summary failed" >> "$LOGF"; exit 1; }
  echo "[$(date '+%F %T')] summary written: $S/summary.md" >> "$LOGF"
  echo "[$(date '+%F %T')] driver end" >> "$LOGF"
}
stage ett flif-v1-ett-spkmux-stft 84 energy_recount_ett.py
stage mitbih flif-v1-mitbih-spkmux-stft 21 energy_recount_mitbih.py
