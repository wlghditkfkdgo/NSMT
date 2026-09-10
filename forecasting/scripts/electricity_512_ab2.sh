#!/bin/bash

data=electricity
patch_size=64
seq_len=512
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/electricity

for len in 720 336 192 96
do
for gating in original #ablation
do
python3 ./train.py \
    --model ab2 \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 1 \
    -e 50 \
    --warm_up_epoch 0 \
    --keep_ratio 0.1 \
    -bs 16 \
    -emb 128 \
    --mlp_ratios 1 \
    -nh 2 \
    -lr 0.0001 \
    --alpha 0.01 \
    --time_layers 1 \
    --data 'custom' \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_in 321 \
    --patience 3
    
done
done