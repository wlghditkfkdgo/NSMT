#!/usr/bin/env bash
# mxpop_ett_driver.sh (2026-11-19 evening, exp/f-lif-pop-v3; PROJECT_LOG mixer-only population ETT pre-registration; user decisions 11-19:
# ETT input 96, "mixer에만 population" first, then tuning of the adopted structure): the ETT part without waiting for a session.
#   pool mxett (48) -> completeness (48 RESULT, no SKIP) -> energy_recount_ett.py (head recount + recorder body incl. population neuron ACs)
#   -> summarize_pn.py --stage mxett (verdicts, adoption -> selected_structure.json, pass criteria) -> recheck_mxett.py (independent)
# Fail closed: any failure stops the driver. Logging, commits and the tuning stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/mxpop_ett_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ett-spkmux-mxpop
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
say "driver start (slots $SLOTS)"
qf=$(bash "$C/scripts/make_queue_pn.sh" mxett 2>>"$LOGF" | tail -1 | cut -d: -f1)
[ -s "$qf" ] || die "queue missing or empty ($qf)"
[ "$(wc -l < "$qf")" -eq 48 ] || die "queue has $(wc -l < "$qf") lines, want 48"
say "launched 48 jobs (mxett): ${qf#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
say "pool done (mxett)"
nr=$(grep -c '^RESULT' "$R/raw.txt" 2>/dev/null); ns=$(grep -c '^SKIP' "$R/raw.txt" 2>/dev/null)
say "$SU: RESULT ${nr:-0}, SKIP ${ns:-0} (want 48, 0)"
[ "${nr:-0}" -eq 48 ] && [ "${ns:-0}" -eq 0 ] || die "incomplete $SU"
pyc energy_recount_ett.py --suite $SU || die "energy recount failed"
pyc summarize_pn.py --stage mxett || die "summary failed"
say "selected structure: $(tr -d '\n' < "$R/selected_structure.json" | cut -c1-200)"
pyc recheck_mxett.py || die "independent recheck failed"
say "driver end"
