#!/usr/bin/env bash
# mitmemoff_driver.sh (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG MIT-BIH memoryless-control pre-registration; user "분류 과제 통제를 먼저"):
# LIF SpkMixer muxlin ADD lr 0.02 mixer layers 2 with TMEM=off (every token-axis mixer neuron resets after each token), seeds 42 / 7 / 2026,
# suite flif-v1-mitbih-memoff: queue -> gpu_pool2.sh -> RESULT 3 / SKIP 0 -> energy_recount_mitbih.py -> cls_order_eval.py --arm muxoff
# (memoryless mixer must be order-free) -> results/check-mitmemoff-20261128/summarize_mitmemoff.py -> "driver end". Fail closed.
# usage (from NSMT/): SLOTS=1,2,3 nohup bash cinc_mixer_v1/scripts/mitmemoff_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-1,2,3}
SU=flif-v1-mitbih-memoff
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
O="$C/results/check-mitorder-20261128"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }
[ -e "$R/raw.txt" ] && die "$R/raw.txt exists (fresh suite expected)"
[ -e "$O/order_muxoff.json" ] && die "order_muxoff.json exists"
Q="$C/scripts/queues/$SU-$(date +%Y%m%d).txt"
[ -s "$Q" ] && die "queue $Q exists"
: > "$Q"
for s in 42 7 2026; do echo "SUITE=$SU COND=spkmx_lif_muxlin_add LR=0.02 TMEM=off bash cinc_mixer_v1/scripts/run_mitbih_spk.sh $s" >> "$Q"; done
[ "$(grep -c . "$Q")" -eq 3 ] || die "queue lines"
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD)): 3 jobs ${Q#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${Q#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
nr=$(cnt RESULT); ns=$(cnt SKIP)
say "pool done: RESULT $nr (want 3), SKIP $ns (want 0)"
[ "$nr" -eq 3 ] && [ "$ns" -eq 0 ] || die "incomplete"
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} OMP_NUM_THREADS=4 $PY -B -W ignore -u energy_recount_mitbih.py --suite $SU >> "$LOGF" 2>&1) || die "energy recount failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} OMP_NUM_THREADS=4 $PY -B -W ignore -u results/check-mitorder-20261128/cls_order_eval.py --arm muxoff --out "$O/order_muxoff.json" > "$O/order_muxoff.txt" 2>&1) || die "order eval failed"
(cd "$C" && OMP_NUM_THREADS=4 $PY -B -u results/check-mitmemoff-20261128/summarize_mitmemoff.py >> "$LOGF" 2>&1) || die "summary failed"
say "driver end"
