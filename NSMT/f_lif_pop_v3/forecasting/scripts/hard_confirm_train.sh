#!/usr/bin/env bash
# Prereg 2J, D-AN: confirmation-stage training. Eight pre-registered seeds, each trained
# twice from the SAME initialisation (q does not touch parameter creation): the baseline
# q = 1.0 (bitwise full, gate G18) and the q recorded by hard_selection.py. Settings are the
# selection stage's. The test split is not evaluated; confirm2 is opened once, later, by
# analysis/hard_confirm.py, never here.
#
#   bash scripts/hard_confirm_train.sh <suite> <selected_q>     e.g. hardconf-$(date +%H%M%S) 0.5
set -euo pipefail
cd "$(dirname "$0")/.."
SUITE=${1:?suite name}
QSEL=${2:?selected q from hard_selection_record.json}
SEEDS=(7 13 21 42 123 256 512 1024)
export LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
mkdir -p "log/$SUITE"
for SEED in "${SEEDS[@]}"; do
  for Q in 1 "$QSEL"; do
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2} \
    nohup $PY train.py --task recall --data recall --suite "$SUITE" --mode hard \
        --hard_axis shared --hard_stat pearson --hard_q "$Q" --seed "$SEED" \
        --n_train 2048 --n_val 256 --n_test 256 -e 12 -bs 64 --cpu --no-test \
        > "log/$SUITE/train_seed${SEED}_q$Q.log" 2>&1 &
    echo "launched seed=$SEED q=$Q pid $!"
  done
done
wait
echo "all sixteen finished"
