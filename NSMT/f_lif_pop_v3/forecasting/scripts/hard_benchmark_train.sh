#!/usr/bin/env bash
# Prereg 2M, D-AX: the two new conditions. Eight pre-registered seeds x {oracle-trained hard
# model, GRU baseline}, the 2J settings. The pearson and q=1 models are the 2J confirmation
# checkpoints and are NOT retrained. Test and confirm4 are not touched here.
#   oracle : mode=hard, hard_stat=oracle -- answer slots on recall events, every slot otherwise;
#            q is not used by the oracle and stays at its default 1.0 (named in the variant)
#   gru    : GRUBaseline, embed 32, one layer, mode flag irrelevant (full)
#
#   bash scripts/hard_benchmark_train.sh <suite>      e.g. hardbench-$(date +%H%M%S)
set -euo pipefail
cd "$(dirname "$0")/.."
SUITE=${1:?suite name}
SEEDS=(7 13 21 42 123 256 512 1024)
export LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
COMMON=(--task recall --data recall --suite "$SUITE" --n_train 2048 --n_val 256 --n_test 256 -e 12 -bs 64 --cpu --no-test)
mkdir -p "log/$SUITE"
for SEED in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} \
  nohup $PY train.py "${COMMON[@]}" --model myModel --mode hard --hard_axis shared --hard_stat oracle \
      --seed "$SEED" > "log/$SUITE/train_seed${SEED}_oracle.log" 2>&1 &
  echo "launched seed=$SEED oracle pid $!"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} \
  nohup $PY train.py "${COMMON[@]}" --model GRU --mode full --seed "$SEED" \
      > "log/$SUITE/train_seed${SEED}_gru.log" 2>&1 &
  echo "launched seed=$SEED gru pid $!"
done
wait
echo "all sixteen finished"
