#!/usr/bin/env bash
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT" || exit 1
data=PSM
patch_size=16
seq_len=100
c_out=25
root_path=$(dirname "$(readlink -f "$0")")/../dataset/${data}

for gating in attn
do
for lr in 0.0001
do
for model in myModel #ab1_1
do
python3 ./train.py \
    --model ${model} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 1 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 128 \
    -emb 32 \
    -nh 4 \
    -lr 0.000862736 \
    --keep_ratio 0.25 \
    --alpha 0.3996 \
    --time_layers 4 \
    --mlp_ratios 6 \
    --data ${data} \
    --pred_len 0 \
    --patch_size 8 \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_out ${c_out} \
    --anomaly_ratio 1 \
    --features M 
done
done
done