#!/usr/bin/env bash
# Prereg 2N: one ETT training run. Hard statistical selection on the fixed fractional kernel
# (pearson, shared axis per channel, q=0.5) against the plain f-LIF (q=1, bitwise full) and the
# GRU baseline. O8 budget from the config defaults: AdamW lr 1e-3 wd 1e-2, batch 128, clip 1,
# max 50 epochs, early stop 10 on val MSE, ReduceLROnPlateau(0.5, 5). The test split is NOT
# opened here (--no-test): ett_test.py opens it once, after every run is finished (2N D-BF).
# usage: run_ett.sh <DS> <PL> <cond> <seed> <gpu>
set -u
DS=${1:?}; PL=${2:?}; C=${3:?}; S=${4:?}; GPU=${5:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # torch 1.12.0+cu113: the declared topk tie rule
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-50}                                       # override to 1 for smoke tests
SUITE=${SUITE:-etthard-20260925}                           # one suite for all 48 runs of 2N
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}  # override to keep smoke runs out of the real results
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
case $C in
  q1)       MF="--model myModel --mode hard --hard_axis shared --hard_stat pearson --hard_q 1" ;;   # == plain f-LIF
  pearson)  MF="--model myModel --mode hard --hard_axis shared --hard_stat pearson --hard_q 0.5" ;;
  gru)      MF="--model GRU --mode full" ;;
  *) echo "bad cond $C"; exit 1 ;;
esac
IX="ett_${DS}_p${PL}_${C}"; OUT=$LOG/${IX}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py --task ett --data $DS \
  --pred_len $PL --suite $SUITE --seed $S -nd 0 -e $EPOCHS --no-test $MF > "$OUT" 2>&1)
RC=$?
VAL=$(grep -aoE 'Validation loss decreased \([^)]*--> [0-9.]+' "$OUT" | tail -1 | grep -oE '[0-9.]+$')
PAR=$(grep -a '\[train\] parameters' "$OUT" | grep -oE '[0-9]+' | head -1); t1=$(date +%s)
EP=$(grep -ac '^epoch ' "$OUT")
if [ $RC -ne 0 ] || [ -z "$VAL" ]; then echo "SKIP(rc=$RC, no val) ds=$DS pl=$PL cond=$C seed=$S" | tee -a $RES/raw.txt; exit 1; fi
echo "RESULT exp=ett dataset=${DS}_p${PL} cond=${C} seed=$S best_val_mse=$VAL epochs=$EP params=$PAR time_s=$((t1-t0)) sha=$SHA" | tee -a $RES/raw.txt
