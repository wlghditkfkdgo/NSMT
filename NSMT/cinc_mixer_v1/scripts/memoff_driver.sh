#!/usr/bin/env bash
# memoff_driver.sh (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG memoryless-control pre-registration; user "(a)도 진행해줘"):
# muxlin IAND T4 with token_memory off (TMEM=off: every token-axis mixer neuron resets after each token),
# independent weights, LIF, one block, first-chain settings; ETTh1 / ETTm1 x H96 / H720 x seeds 42 / 7 / 2026 = 12 runs, suite flif-v1-ett-memoff:
#   queue -> gpu_pool2.sh -> RESULT 12 / SKIP 0 -> energy_recount_ett.py -> summarize_memoff.py -> results/check-memoff-20261127/recheck_memoff.py
#   -> "driver end".  Fail closed: any failure stops the driver ("STOP: ...").
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/memoff_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ett-memoff
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }
[ -e "$R/raw.txt" ] && die "$R/raw.txt exists (fresh suite expected)"
Q="$C/scripts/queues/$SU-$(date +%Y%m%d).txt"
[ -s "$Q" ] && die "queue $Q exists"
: > "$Q"
for ds in ETTm1 ETTh1; do
  case $ds in ETTh1) kn="LR=0.00025 LAYERS=1 EMB=32 MLPR=4 WD=0.1" ;; ETTm1) kn="LR=0.0005 LAYERS=2 PATCH=16 EMB=32 MLPR=0.5" ;; esac
  for pl in 720 96; do for s in 42 7 2026; do for comb in muxlin; do
    echo "SUITE=$SU COND=spkmx_lif_${comb}_iand WSHARE=0 TMEM=off $kn bash cinc_mixer_v1/scripts/run_ett_spk.sh $ds $pl $s" >> "$Q"
  done; done; done
done
[ "$(grep -c . "$Q")" -eq 12 ] && [ "$(sort -u "$Q" | grep -c .)" -eq 12 ] || die "queue: $(grep -c . "$Q") lines"
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD)): 12 jobs ${Q#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${Q#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
nr=$(cnt RESULT); ns=$(cnt SKIP)
say "pool done: RESULT $nr (want 12), SKIP $ns (want 0)"
[ "$nr" -eq 12 ] && [ "$ns" -eq 0 ] || die "incomplete"
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} OMP_NUM_THREADS=4 $PY -B -u energy_recount_ett.py --suite $SU >> "$LOGF" 2>&1) || die "energy recount failed"
(cd "$C" && OMP_NUM_THREADS=4 $PY -B -u summarize_memoff.py --suite $SU >> "$LOGF" 2>&1) || die "summary failed"
(cd "$C" && OMP_NUM_THREADS=4 $PY -B -u results/check-memoff-20261127/recheck_memoff.py --suite $SU >> "$LOGF" 2>&1) || die "independent recheck failed"
say "driver end"
