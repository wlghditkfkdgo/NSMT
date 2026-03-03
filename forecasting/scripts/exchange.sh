#!/bin/bash

data=exchange
patch_size=24
seq_len=96
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/${data}

for len in 720 336 192 96
do
for model in myModel #ab1_1 #ab1 ab1_1 Spikformer
do
python3 ./train.py \
    --log_dir final3 \
    --model ${model} \
    --gating attn \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 1 \
    -e 50 \
    --warm_up_epoch 0 \
    --mlp_ratio 2 \
    --keep_ratio 0.2 \
    -bs 64 \
    -emb 128 \
    -nh 8 \
    -lr 0.0005 \
    --alpha 0.9 \
    --time_layers 2 \
    --data 'custom' \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --patience 3 \
    --label_len 48 \
    
done
done
    # --label_len 168 \