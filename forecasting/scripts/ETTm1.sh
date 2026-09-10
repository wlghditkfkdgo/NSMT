#!/bin/bash

data=ETTm1
patch_size=8
seq_len=336
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small


for keep_ratio in 0.1 0.15 0.2
do
for mlp_ratio in 1 2 4
do
for len in 720 #336 192 96
do
for gating in original #ablation
do
for model in myModel ab4 #ab1 ab2 Spikformer ab1_1
do
python3 ./train.py \
    --model ${model} \
    --analysis \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 2 \
    -e 50 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb 16 \
    --mlp_ratios ${mlp_ratio} \
    --keep_ratio ${keep_ratio} \
    -nh 2 \
    -lr 0.0005 \
    --alpha 1 \
    --time_layers 1 \
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
done