#!/usr/bin/env bash
# adsf_driver.sh (2026-11-18, exp/f-lif-pop-v3; PROJECT_LOG AD SF-de pre-registration): SF-de on anomaly detection without waiting for a session.
#   lr pool (45, holdout) -> completeness (45 RESULT, no SKIP) -> summarize_adsf.py --stage select (validation only)
#   -> final pool (9, whole file, selected lr) -> completeness (54 RESULT, no SKIP) -> summarize_adsf.py --stage final
# Fail closed: any failure stops the driver. Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/adsf_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
R="$C/results/flif-v1-ad-sfde"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
pool() {   # pool <stage> <jobs wanted>
  local qf; qf=$(bash "$C/scripts/make_queue_adsf.sh" "$1" 2>>"$LOGF" | tail -1 | cut -d: -f1)
  [ -s "$qf" ] || die "queue missing or empty ($1: $qf)"
  [ "$(wc -l < "$qf")" -eq "$2" ] || die "queue $1 has $(wc -l < "$qf") lines, want $2"
  say "launched $2 jobs ($1): ${qf#$N/}"
  (cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$R/pool_log_$1.txt" 2>&1
  say "pool done ($1)"
}
count() {  # count <RESULT rows wanted>
  local nr ns; nr=$(grep -c '^RESULT' "$R/raw.txt" 2>/dev/null); ns=$(grep -c '^SKIP' "$R/raw.txt" 2>/dev/null)
  say "flif-v1-ad-sfde: RESULT ${nr:-0}, SKIP ${ns:-0} (want $1, 0)"
  [ "${nr:-0}" -eq "$1" ] && [ "${ns:-0}" -eq 0 ] || die "incomplete flif-v1-ad-sfde"
}
say "driver start (slots $SLOTS)"
pool lr 45; count 45
pyc summarize_adsf.py --stage select || die "selection failed"
say "selected: $(tr -d '\n ' < "$R/selected_lr.json" | cut -c1-200)"
pool final 9; count 54
pyc summarize_adsf.py --stage final || die "final summary failed"
say "driver end"
