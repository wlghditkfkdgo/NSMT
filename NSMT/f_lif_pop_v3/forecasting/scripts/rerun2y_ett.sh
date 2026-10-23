#!/usr/bin/env bash
# Prereg 2Y: one ETT training run of the readout x context x state comparison on the 2W pick (population axis kept,
# per-step BN with gamma from D11, no hidden block, one output Linear, lr 0.0005, weight decay on all; validation
# only, --no-test). Cell '<base>-<mem|nomem>-<L>-<all|last1>': base q1 | q1_R (R on); mem = the neuron's state
# crosses patches, nomem = v = 0 at every patch; L = window length (336 | 96); all = the Linear reads every step,
# last1 = only the last one.
# usage: SUITE=<suite> rerun2y_ett.sh <DS> <PL> <cell> <seed> <gpu>
set -u
DS=${1:?}; PL=${2:?}; C=${3:?}; S=${4:?}; GPU=${5:?}
SUITE=${SUITE:?set SUITE}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-50}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}; mkdir -p "$RES" "$LOG"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
IFS=- read -r BASE MEM L HEAD <<< "$C"
MF="--model myModel --neuron pop_flif --mode full --soma none --input_norm bn --head_mode linear --learning_rate 0.0005"
case $BASE in q1) ;; q1_R) MF="$MF --revin" ;; *) echo "bad base $BASE"; exit 1 ;; esac
case $MEM in mem) MF="$MF --neuron_memory full" ;; nomem) MF="$MF --neuron_memory none" ;; *) echo "bad memory $MEM"; exit 1 ;; esac
case $L in 336|96) MF="$MF --seq_len $L" ;; *) echo "bad length $L"; exit 1 ;; esac
case $HEAD in all) MF="$MF --head_window 0" ;; last1) MF="$MF --head_window 1" ;; *) echo "bad head $HEAD"; exit 1 ;; esac
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
