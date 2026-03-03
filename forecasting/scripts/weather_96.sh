#!/bin/bash

data=weather
patch_size=8
seq_len=96
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/${data}

for len in 720 336 192 96
do
for gating in attn #ablation
do
for model in myModel #ab1 ab1_1 Spikformer
do
python3 ./train.py \
    --log_dir final3 \
    --model ${model} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 1 \
    -e 50 \
    --warm_up_epoch 0 \
    --mlp_ratio 6 \
    --keep_ratio 0.15 \
    -bs 32 \
    -emb 128 \
    -nh 8 \
    -lr 0.0005 \
    --alpha 0.1 \
    --time_layers 1 \
    --data 'custom' \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --patience 2 \
    --c_in 21 \
    
done
done
done
    # --label_len 168 \