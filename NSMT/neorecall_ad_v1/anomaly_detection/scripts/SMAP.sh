#!/usr/bin/env bash
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT" || exit 1
data=SMAP
patch_size=8
seq_len=100
c_out=25
root_path=$(dirname "$(readlink -f "$0")")/../dataset/${data}

for lr in 0.0005
do
for gating in attn
do
for model in myModel
do
python3 ./train.py \
    --model ${model} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 2 \
    -e 50 \
    --warm_up_epoch 0 \
    -bs 32 \
    -emb 32 \
    -nh 4 \
    -lr 0.000423568 \
    --alpha 0.1957 \
    --keep_ratio 0.10 \
    --time_layers 1 \
    --mlp_ratios 2 \
    --data ${data} \
    --pred_len 0 \
    --patch_size 12 \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_out ${c_out} \
    --anomaly_ratio 1 \
    --features M 
done
done
done