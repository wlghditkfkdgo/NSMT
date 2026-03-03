#!/bin/bash

data=traffic
patch_size=64
seq_len=512
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/${data}

for len in 720 #336 192 96
do
for gating in original #ablation
do
for model in myModel ab4 #ab1 ab2 Spikformer ab1_1
do
python3 ./train.py \
    --model ${model} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 1 \
    -e 50 \
    --warm_up_epoch 0 \
    --mlp_ratio 4 \
    --keep_ratio 0.1 \
    -bs 8 \
    -emb 64 \
    -nh 8 \
    -lr 0.0001 \
    --alpha 0.5 \
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