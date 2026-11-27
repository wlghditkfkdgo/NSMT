#!/usr/bin/env bash
# sfde1_driver.sh (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "SF-de에 대해서 ETTh1, ETTm1의 순서 섞기 실험을
# 진행해줘"): SF-de (COND=spk_ssa NALL=lif; SSA with relative position bias, as every SF-de run) on ETTh1 (patch 8, lr 5e-4) and ETTm1
# (patch 16, lr 1e-4) x H96 / H720 x seeds 42 / 7 / 2026 = 12 runs, suite flif-v1-ett-sfde-h1m1:
#   queue -> gpu_pool2.sh -> RESULT 12 / SKIP 0 -> energy_recount_ett.py -> order_eval.py --arm sfde1 (no training)
#   -> results/check-sfde1-20261127/summarize_order3.py -> "driver end".  Fail closed: any failure stops the driver ("STOP: ...").
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/sfde1_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ett-sfde-h1m1
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
O="$C/results/check-order-20261127"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }
[ -e "$R/raw.txt" ] && die "$R/raw.txt exists (fresh suite expected)"
[ -e "$O/order_sfde1.json" ] && die "order_sfde1.json exists"
Q="$C/scripts/queues/$SU-$(date +%Y%m%d).txt"
[ -s "$Q" ] && die "queue $Q exists"
: > "$Q"
for ds in ETTm1 ETTh1; do
  case $ds in ETTh1) kn="LR=0.0005" ;; ETTm1) kn="LR=0.0001 PATCH=16" ;; esac
  for pl in 720 96; do for s in 42 7 2026; do
    echo "SUITE=$SU COND=spk_ssa NALL=lif $kn bash cinc_mixer_v1/scripts/run_ett_spk.sh $ds $pl $s" >> "$Q"
  done; done
done
[ "$(grep -c . "$Q")" -eq 12 ] && [ "$(sort -u "$Q" | grep -c .)" -eq 12 ] || die "queue: $(grep -c . "$Q") lines"
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD)): 12 jobs ${Q#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${Q#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
nr=$(cnt RESULT); ns=$(cnt SKIP)
say "pool done: RESULT $nr (want 12), SKIP $ns (want 0)"
[ "$nr" -eq 12 ] && [ "$ns" -eq 0 ] || die "incomplete"
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} OMP_NUM_THREADS=4 $PY -B -u energy_recount_ett.py --suite $SU >> "$LOGF" 2>&1) || die "energy recount failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} OMP_NUM_THREADS=4 $PY -B -W ignore -u results/check-order-20261127/order_eval.py --arm sfde1 --out "$O/order_sfde1.json" > "$O/order_sfde1.txt" 2>&1) || die "order eval failed"
(cd "$C" && OMP_NUM_THREADS=4 $PY -B -u results/check-sfde1-20261127/summarize_order3.py >> "$LOGF" 2>&1) || die "summary failed"
say "driver end"
