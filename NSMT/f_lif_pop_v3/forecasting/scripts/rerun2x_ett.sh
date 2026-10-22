#!/usr/bin/env bash
# Prereg 2X: one ETT training run of the overlapping-patch comparison on the 2W pick (population axis kept,
# per-step BN with gamma from D11, no hidden block, one output Linear, lr 0.0005, weight decay on all; validation
# only, --no-test). Cell '<base>-p<P>s<S>': base q1 | q1_R (R on), patch length P, stride S (S < P overlaps).
# usage: SUITE=<suite> rerun2x_ett.sh <DS> <PL> <cell> <seed> <gpu>
set -u
DS=${1:?}; PL=${2:?}; C=${3:?}; S=${4:?}; GPU=${5:?}
SUITE=${SUITE:?set SUITE}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-50}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}; mkdir -p "$RES" "$LOG"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
IFS=- read -r BASE PATCH <<< "$C"
P=$(echo "$PATCH" | sed -E 's/^p([0-9]+)s([0-9]+)$/\1/'); ST=$(echo "$PATCH" | sed -E 's/^p([0-9]+)s([0-9]+)$/\2/')
[[ "$P" =~ ^[0-9]+$ && "$ST" =~ ^[0-9]+$ ]] || { echo "bad patch $PATCH"; exit 1; }
MF="--model myModel --neuron pop_flif --mode full --soma none --input_norm bn --head_mode linear --learning_rate 0.0005 --patch_size $P --patch_stride $ST"
case $BASE in q1) ;; q1_R) MF="$MF --revin" ;; *) echo "bad base $BASE"; exit 1 ;; esac
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
