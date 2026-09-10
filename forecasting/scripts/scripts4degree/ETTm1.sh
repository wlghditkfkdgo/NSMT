#!/bin/bash

data=ETTm1
patch_size=16
seq_len=336
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small


keep_ratio=0.25

for layer in 1
do
for len in 720 336 192 96
do
for gating in original ablation
do
for model in degree
do
python3 ./train.py \
    --log_dir DEGREE \
    --model ${model} \
    --analysis \
    --tag ${keep_ratio}\
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 1 \
    -e 50 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb 32 \
    --mlp_ratios 1 \
    --keep_ratio ${keep_ratio} \
    -nh 8 \
    -lr 0.0001 \
    --alpha 0.1 \
    --time_layers ${layer} \
    --data ${data} \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_in 7 \
    --patience 2 \
    
done
done
done
done