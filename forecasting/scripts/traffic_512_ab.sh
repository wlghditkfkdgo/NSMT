#!/bin/bash

data=traffic
patch_size=64
seq_len=512
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/${data}

for len in 720 336 192 96
do
for gating in original #ablation
do
for model in ab1 ab2
do
python3 ./train.py \
    --model ${model} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 3 \
    -e 50 \
    --warm_up_epoch 0 \
    --mlp_ratio 1 \
    --keep_ratio 0.15 \
    -bs 4 \
    -emb 128 \
    -nh 8 \
    -lr 0.0001 \
    --alpha 0.05 \
    --time_layers 1 \
    --data 'custom' \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --patience 2 \
    --label_len 48 \
    --c_in 862     
done
done
done
    # --label_len 168 \