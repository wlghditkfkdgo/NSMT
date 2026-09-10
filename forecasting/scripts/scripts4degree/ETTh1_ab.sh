#!/bin/bash

data=ETTh1
# patch_size=32
seq_len=336
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small

for k in 0.35 0.4
do
for gating in original
do
for len in 720 336 192 96
do
for patch_size in 16
do
for layer in 1 2 3 4
do
for emb in 16
do
for nh in 2
do
for alpha in 0.05 0.1 0.5
do
for lr in 0.0001
do
for seed in 42 #7 13 2026 #42
do
for model in degree
do
python3 ./train.py \
    --log_dir DEGREE/HYP \
    --model ${model} \
    --analysis \
    --seed ${seed} \
    --keep_ratio ${k} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 0 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 64 \
    --mlp_ratios 1 \
    -emb ${emb} \
    -nh ${nh} \
    -lr ${lr} \
    --alpha ${alpha} \
    --time_layers ${layer} \
    --data ${data} \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --patience 2 \
    --c_in 7 
done    
done
done
done
done
done
done
done
done
done
done
    # --label_len 168 \