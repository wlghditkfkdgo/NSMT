#!/usr/bin/env bash
# pn_driver.sh (2026-11-18, exp/f-lif-pop-v3; PROJECT_LOG neuron implementation / placement pre-registration): without waiting for a session,
#   sf     SF-de tuning (72) -> completeness (ETT 60, MIT-BIH 12, no SKIP) -> lr selection (summarize_pn.py --stage select, validation only)
#          -> ETT energy recount of flif-v1-ett-sfde -> part 1 ETT summary
#   main   SF-de MIT-BIH final (3) + grid ETT (108) + grid MIT-BIH (27) -> completeness -> energy recounts -> part 1 MIT-BIH, parts 2 / 3 summaries
#   len336 SF-de 336 (12) + pop LIF mixer-only 336 (12) on SLOTS336 -> completeness -> energy recount -> part 4 summary
# Fail closed: any failure stops the driver. Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 SLOTS336=... nohup bash cinc_mixer_v1/scripts/pn_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}; SLOTS336=${SLOTS336:-$SLOTS}
LOGF="$C/results/flif-v1-ett-spkmux-pn/driver.log"; mkdir -p "$(dirname "$LOGF")"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
pool() {   # pool <stage> <slots> <pool log>
  local qf; qf=$(bash "$C/scripts/make_queue_pn.sh" "$1" 2>>"$LOGF" | tail -1 | cut -d: -f1)
  [ -s "$qf" ] || die "queue missing or empty ($1: $qf)"
  say "launched $(wc -l < "$qf") jobs ($1): ${qf#$N/}"
  (cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$2") > "$3" 2>&1
  say "pool done ($1)"
}
count() {  # count <suite> <RESULT rows wanted>
  local f="$C/results/$1/raw.txt" nr ns
  nr=$(grep -c '^RESULT' "$f" 2>/dev/null); ns=$(grep -c '^SKIP' "$f" 2>/dev/null)
  say "$1: RESULT ${nr:-0}, SKIP ${ns:-0} (want $2, 0)"
  [ "${nr:-0}" -eq "$2" ] && [ "${ns:-0}" -eq 0 ] || die "incomplete $1"
}
say "driver start (slots $SLOTS, slots336 $SLOTS336)"
pool sf "$SLOTS" "$C/results/flif-v1-ett-sfde-pool_log.txt"
count flif-v1-ett-sfde 60; count flif-v1-mitbih-sfde 12
pyc summarize_pn.py --stage select || die "selection failed"
pyc energy_recount_ett.py --suite flif-v1-ett-sfde || die "energy recount flif-v1-ett-sfde failed"
pyc summarize_pn.py --stage sf_ett || die "part 1 ETT summary failed"
say "part 1 ETT summary written"
pool main "$SLOTS" "$C/results/flif-v1-ett-spkmux-pn/pool_log.txt"
count flif-v1-mitbih-sfde 15; count flif-v1-ett-spkmux-pn 108; count flif-v1-mitbih-spkmux-pn 27
pyc energy_recount_mitbih.py --suite flif-v1-mitbih-sfde || die "energy recount flif-v1-mitbih-sfde failed"
pyc summarize_pn.py --stage sf_mit || die "part 1 MIT-BIH summary failed"
pyc energy_recount_ett.py --suite flif-v1-ett-spkmux-pn || die "energy recount flif-v1-ett-spkmux-pn failed"
pyc energy_recount_mitbih.py --suite flif-v1-mitbih-spkmux-pn || die "energy recount flif-v1-mitbih-spkmux-pn failed"
pyc summarize_pn.py --stage ett || die "ETT summary failed"
pyc summarize_pn.py --stage mitbih || die "MIT-BIH summary failed"
say "parts 1-3 summaries written"
pool len336 "$SLOTS336" "$C/results/flif-v1-ett-spkmux-pn336-pool_log.txt"
count flif-v1-ett-spkmux-pn336 24
pyc energy_recount_ett.py --suite flif-v1-ett-spkmux-pn336 || die "energy recount flif-v1-ett-spkmux-pn336 failed"
pyc summarize_pn.py --stage len336 || die "part 4 summary failed"
say "driver end"
