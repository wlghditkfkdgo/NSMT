#!/usr/bin/env bash
# Prereg 2V: one ETT training run of the 2T -> 2U ablation (validation only; --no-test). The budget is
# run_ett.sh's. The cell name '<base>-<soma>-<norm>-<head>' sets the model:
#   base q1_R (R on) | q1 (R off), no selection (--neuron pop_flif), alpha 0.7
#   soma lif (silent members + soma, 2T) | none (members fire, 2U)
#   norm frozen (train-split standardisation x calibrated scale, 2T) | bn (per-step BN, gamma from D11, 2U)
#   head flatten (Linear(D*,32) then Linear(1344,96), 2T) | hidden (Linear + BN + SpikingJelly LIF, one Linear, 2U)
# usage: SUITE=<suite> rerun2v_ett.sh <DS> <PL> <cell> <seed> <gpu>
set -u
DS=${1:?}; PL=${2:?}; C=${3:?}; S=${4:?}; GPU=${5:?}
SUITE=${SUITE:?set SUITE}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-50}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}; mkdir -p "$RES" "$LOG"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
IFS=- read -r BASE SOMA NORM HEAD <<< "$C"
MF="--model myModel --neuron pop_flif --mode full --soma $SOMA --input_norm $NORM"
case $BASE in q1_R) MF="$MF --revin" ;; q1) ;; *) echo "bad base $BASE"; exit 1 ;; esac
case $HEAD in flatten) MF="$MF --head_mode flatten" ;; hidden) MF="$MF --hidden_dim 32 --head_mode linear" ;; *) echo "bad head $HEAD"; exit 1 ;; esac
case $SOMA in lif|none) ;; *) echo "bad soma"; exit 1 ;; esac
case $NORM in frozen|bn) ;; *) echo "bad norm"; exit 1 ;; esac
IX="ett_${DS}_p${PL}_${C}"; OUT=$LOG/${IX}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py --task ett --data $DS \
  --pred_len $PL --suite $SUITE --seed $S -nd 0 -e $EPOCHS --no-test $MF ${EXTRA:-} > "$OUT" 2>&1)
RC=$?
VAL=$(grep -aoE 'Validation loss decreased \([^)]*--> [0-9.]+' "$OUT" | tail -1 | grep -oE '[0-9.]+$')
EP=$(grep -ac '^epoch ' "$OUT"); t1=$(date +%s)
if [ $RC -ne 0 ] || [ -z "$VAL" ]; then
  echo "SKIP(rc=$RC, no val, g11=$(grep -ac 'G11 violated' "$OUT")) ds=$DS pl=$PL cond=$C seed=$S time_s=$((t1-t0)) sha=$SHA" | tee -a $RES/raw.txt; exit 1
fi
echo "RESULT exp=ett dataset=${DS}_p${PL} cond=${C} seed=$S best_val_mse=$VAL epochs=$EP time_s=$((t1-t0)) gpu=$GPU sha=$SHA" | tee -a $RES/raw.txt
