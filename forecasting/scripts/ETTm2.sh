#!/bin/bash

data=ETTm2
patch_size=16
seq_len=336
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small

for keep_ratio in 0.1
do
for gating in original
do
for len in 720 #336 192 96
do
for alpha in 1
do
for mlp_ratio in 1
do
for model in ab4 myModel
do
python3 ./train.py \
    --model ${model} \
    --keep_ratio ${keep_ratio} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 2 \
    -e 50 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb 32 \
    -nh 2 \
    -lr 0.0005 \
    --alpha ${alpha} \
    --time_layers 1 \
    --mlp_ratios ${mlp_ratio} \
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
done
    # --label_len 168 \