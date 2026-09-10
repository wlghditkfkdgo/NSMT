#!/bin/bash

data=ETTm1
# patch_size=8
seq_len=96
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small

# keep_ratio=0.25

for seed in 42 #2026 7
do
for keep_ratio in 0.25
do
for len in 720 336 192 96
do
for patch_size in 8
do
for model in ab4 #myModel
do
python3 ./train.py \
    --log_dir rebuttal \
    --seed ${seed} \
    --model ${model} \
    --analysis \
    --gating attn \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 2 \
    -e 50 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb 64 \
    --mlp_ratios 1 \
    --max_ratio 2 \
    --keep_ratio ${keep_ratio} \
    -nh 8 \
    -lr 0.001 \
    --alpha 0.5 \
    --time_layers 2 \
    --data ${data} \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_in 7 \
    --patience 3 \
    
done
done
done
done
done