#!/usr/bin/env bash
# Prereg 2W: one ETT training run of the tuning grid on the 2U base (population axis kept, Linear + per-step BN
# with gamma from D11 + population f-LIF; --soma none --input_norm bn). Budget as run_ett.sh except the lr of
# the cell. Cell '<base>-<arch>-<lr>-<wd>':
#   base q1 | q1_R (no selection) | pearson | pearson_R (Selective pearson shared q 0.5); _R = --revin
#   arch direct (no hidden block: one Linear on [B*C, T*128]) | hmem (Linear + BN + SpikingJelly LIF, read through
#        its membrane charge, then one Linear)
#   lr 0.0005 | 0.001 | 0.002;   wd all (AdamW decays everything) | xnb (no decay on BN parameters and biases)
# usage: SUITE=<suite> rerun2w_ett.sh <DS> <PL> <cell> <seed> <gpu>
set -u
DS=${1:?}; PL=${2:?}; C=${3:?}; S=${4:?}; GPU=${5:?}
SUITE=${SUITE:?set SUITE}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-50}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}; mkdir -p "$RES" "$LOG"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
IFS=- read -r BASE ARCH LR WD <<< "$C"
MF="--model myModel --soma none --input_norm bn --head_mode linear --learning_rate $LR"
case $BASE in
  q1)        MF="$MF --neuron pop_flif --mode full" ;;
  q1_R)      MF="$MF --neuron pop_flif --mode full --revin" ;;
  pearson)   MF="$MF --neuron sel_pop_flif --mode hard --hard_axis shared --hard_stat pearson --hard_q 0.5" ;;
  pearson_R) MF="$MF --neuron sel_pop_flif --mode hard --hard_axis shared --hard_stat pearson --hard_q 0.5 --revin" ;;
  *) echo "bad base $BASE"; exit 1 ;;
esac
case $ARCH in direct) ;; hmem) MF="$MF --hidden_dim 32 --hidden_readout membrane" ;; *) echo "bad arch"; exit 1 ;; esac
case $WD in all) ;; xnb) MF="$MF --wd_exclude_norm_bias" ;; *) echo "bad wd"; exit 1 ;; esac
case $LR in 0.0005|0.001|0.002) ;; *) echo "bad lr"; exit 1 ;; esac
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
