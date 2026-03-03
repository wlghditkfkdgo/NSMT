#!/bin/bash

data=electricity
seq_len=96
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/electricity

patch_size=24
for len in 720 336 192 96
do
for model in myModel #ab1_1 #ab1_1 #Spikformer ab1 ab1_1
do
for gating in attn 
do
python3 ./train.py \
    --gating ${gating} \
    --model ${model} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 2 \
    -e 50 \
    --warm_up_epoch 0 \
    --keep_ratio 0.25 \
    -bs 8 \
    -emb 512 \
    --mlp_ratios 1.5 \
    -nh 16 \
    -lr 0.0005 \
    --alpha 0.1 \
    --time_layers 3 \
    --data 'custom' \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_in 321 \
    --patience 2
done
done
done