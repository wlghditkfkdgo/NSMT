#!/bin/bash

data=ETTm2
patch_size=8
seq_len=96
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small

for keep_ratio in 0.1 0.15 0.2 #0.25
do
for gating in attn #attn
do
for len in 720 #336 192 96
do
for alpha in 0.5
do
for seed in 42
do
for model in myModel
do
python3 ./train.py \
    --log_dir final3 \
    --seed ${seed} \
    --model ${model} \
    --analysis \
    --keep_ratio ${keep_ratio} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 3 \
    -e 50 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb 64 \
    -nh 16 \
    -lr 0.0005 \
    --alpha ${alpha} \
    --time_layers 2 \
    --mlp_ratios 1 \
    --data ${data} \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_in 7 \
    --patience 2
done
done
done
done
done
done
    # --label_len 168 \