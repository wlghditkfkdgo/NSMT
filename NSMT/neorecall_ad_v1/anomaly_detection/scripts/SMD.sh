#!/usr/bin/env bash
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT" || exit 1
data=SMD
patch_size=8
seq_len=100
c_out=38
root_path=$(dirname "$(readlink -f "$0")")/../dataset/${data}

#keep_ratio=0.25
for lr in 0.0005
do
for gating in attn
do
python3 ./train.py \
    --model myModel \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 4 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 128 \
    -emb 32 \
    -nh 16 \
    -lr 0.000869989 \
    --alpha 0.1397 \
    --time_layers 2 \
    --keep_ratio 0 \
    --mlp_ratios 4 \
    --data ${data} \
    --pred_len 0 \
    --patch_size 8 \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_out ${c_out} \
    --anomaly_ratio 0.5 \
    --features M 
done
done