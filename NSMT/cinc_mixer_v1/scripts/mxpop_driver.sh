#!/usr/bin/env bash
# mxpop_driver.sh (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG mixer-only population pre-registration; user request 11-19 "[mixer에만 population]
# 실험을 AD 이후에 진행"): MIT-BIH part without waiting for a session (the ETT part waits for the user's input-length decision).
#   pool mxmit (12) -> completeness (12 RESULT, no SKIP) -> energy_recount_mitbih.py (integer rule) -> energy_pool_mitbih.py (pooled-integer rule,
#   report) -> summarize_pn.py --stage mxmit
# Fail closed: any failure stops the driver. Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/mxpop_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-mitbih-spkmux-mxpop
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
say "driver start (slots $SLOTS)"
qf=$(bash "$C/scripts/make_queue_pn.sh" mxmit 2>>"$LOGF" | tail -1 | cut -d: -f1)
[ -s "$qf" ] || die "queue missing or empty ($qf)"
[ "$(wc -l < "$qf")" -eq 12 ] || die "queue has $(wc -l < "$qf") lines, want 12"
say "launched 12 jobs (mxmit): ${qf#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
say "pool done (mxmit)"
nr=$(grep -c '^RESULT' "$R/raw.txt" 2>/dev/null); ns=$(grep -c '^SKIP' "$R/raw.txt" 2>/dev/null)
say "$SU: RESULT ${nr:-0}, SKIP ${ns:-0} (want 12, 0)"
[ "${nr:-0}" -eq 12 ] && [ "${ns:-0}" -eq 0 ] || die "incomplete $SU"
pyc energy_recount_mitbih.py --suite $SU || die "energy recount (integer rule) failed"
pyc energy_pool_mitbih.py --suite $SU || die "energy recount (pooled-integer rule) failed"
pyc summarize_pn.py --stage mxmit || die "summary failed"
say "driver end"
