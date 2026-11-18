#!/usr/bin/env bash
# sf336_driver.sh (2026-11-18, exp/f-lif-pop-v3; PROJECT_LOG 11-18 21:4x user decision): the grid of the neuron implementation / placement test was
# stopped; only SF-de at input 336 is run (12 runs: ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026, lr = the part-1 selection) to finish
# D1 (LIF 336 vs SF-de 336). pop LIF mixer-only 336 is not run.
#   pool (make_queue_pn.sh sf336) -> completeness (12 RESULT, no SKIP) -> energy recount -> summarize_pn.py --stage sf336
# Fail closed: any failure stops the driver. Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/sf336_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
LOGF="$C/results/flif-v1-ett-spkmux-pn336/driver.log"; mkdir -p "$(dirname "$LOGF")"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
say "driver start (slots $SLOTS)"
qf=$(bash "$C/scripts/make_queue_pn.sh" sf336 2>>"$LOGF" | tail -1 | cut -d: -f1)
[ -s "$qf" ] || die "queue missing or empty (sf336: $qf)"
[ "$(wc -l < "$qf")" -eq 12 ] || die "queue has $(wc -l < "$qf") lines, want 12"
say "launched $(wc -l < "$qf") jobs (sf336): ${qf#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$C/results/flif-v1-ett-spkmux-pn336-pool_log.txt" 2>&1
say "pool done (sf336)"
f="$C/results/flif-v1-ett-spkmux-pn336/raw.txt"
nr=$(grep -c '^RESULT' "$f" 2>/dev/null); ns=$(grep -c '^SKIP' "$f" 2>/dev/null)
say "flif-v1-ett-spkmux-pn336: RESULT ${nr:-0}, SKIP ${ns:-0} (want 12, 0)"
[ "${nr:-0}" -eq 12 ] && [ "${ns:-0}" -eq 0 ] || die "incomplete flif-v1-ett-spkmux-pn336"
pyc energy_recount_ett.py --suite flif-v1-ett-spkmux-pn336 || die "energy recount flif-v1-ett-spkmux-pn336 failed"
pyc summarize_pn.py --stage sf336 || die "sf336 summary failed"
say "driver end"
