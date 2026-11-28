#!/usr/bin/env bash
# iandmem_driver.sh (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "3. 기억 없는 실험을 MIT-BIH + IAND에 대해서 진행해줘"):
# LIF SpkMixer muxlin IAND lr 0.01, 2 mixer layers, TMEM=off (every token-axis mixer neuron resets after each token), seeds 42 / 7 / 2026,
# suite flif-v1-mitbih-memoff-iand: queue -> gpu_pool2.sh -> RESULT 3 / SKIP 0 -> energy_recount_mitbih.py -> cls_order_eval.py --arm iandoff
# (memoryless model must be order-free) -> summarize_memoffi.py -> recheck_memoffi.py -> "driver end". Fail closed ("STOP: ...").
# usage (from NSMT/): SLOTS=0,2,3 nohup bash cinc_mixer_v1/scripts/iandmem_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,2,3}
SU=flif-v1-mitbih-memoff-iand
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
O="$C/results/check-iandmem-20261128"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }
[ -e "$R/raw.txt" ] && die "$R/raw.txt exists (fresh suite expected)"
for f in order_iandoff.json summary_memoffi.json recheck_memoffi.txt; do [ -e "$O/$f" ] && die "$O/$f exists"; done
Q="$C/scripts/queues/$SU-$(date +%Y%m%d).txt"
[ -s "$Q" ] && die "queue $Q exists"
: > "$Q"
for s in 42 7 2026; do echo "SUITE=$SU COND=spkmx_lif_muxlin_iand LR=0.01 TMEM=off bash cinc_mixer_v1/scripts/run_mitbih_spk.sh $s" >> "$Q"; done
[ "$(sort -u "$Q" | grep -c .)" -eq 3 ] || die "queue lines"
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD)): 3 jobs ${Q#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${Q#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
nr=$(cnt RESULT); ns=$(cnt SKIP)
say "pool done: RESULT $nr (want 3), SKIP $ns (want 0)"
[ "$nr" -eq 3 ] && [ "$ns" -eq 0 ] || die "incomplete"
G=${SLOTS%%,*}
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u energy_recount_mitbih.py --suite $SU >> "$LOGF" 2>&1) || die "energy recount failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u results/check-mitorder-20261128/cls_order_eval.py --arm iandoff \
  --out "$O/order_iandoff.json" > "$O/order_iandoff.txt" 2>&1) || die "order eval failed"
(cd "$C" && $PY -B -u results/check-iandmem-20261128/summarize_memoffi.py >> "$LOGF" 2>&1) || die "summary failed"
(cd "$C" && $PY -B -u results/check-iandmem-20261128/recheck_memoffi.py > "$O/recheck_memoffi.txt" 2>&1) || die "recheck failed"
grep -q 'ALL CHECKS PASSED' "$O/recheck_memoffi.txt" || die "recheck: not all checks passed"
say "driver end"
