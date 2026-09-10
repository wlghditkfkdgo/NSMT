#!/usr/bin/env bash
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT" || exit 1
data=MSL
patch_size=12
seq_len=100
c_out=55
root_path=$(dirname "$(readlink -f "$0")")/../dataset/${data}

for gating in attn
do
for lr in 0.0005
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
    -nd 0 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb 32 \
    -nh 16 \
    -lr 5.5842e-05 \
    --alpha 0.3279 \
    --time_layers 1 \
    --mlp_ratios 2 \
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