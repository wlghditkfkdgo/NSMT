#!/bin/bash

root_path=/home/yschoi/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/classification/data
dataset_name=ECG

patch_size=16
length=187
n_classes=5
num_channels=1

for gating in attn
do
for layers in 4
do
for emb in 32
do
for lr in 0.01
do
for seed in 42
do
python3 train.py \
    --seed ${seed}\
    --log_dir ../results/ECG/${dataset_name}\
    --gating ${gating}\
    --attn SSA \
    -s --test --print_epoch 1 -nd 3 \
    -e 30 --warm_up_epoch 0 \
    --num_channels ${num_channels} \
    -bs 512 \
    -lr ${lr} \
    --max_lr 0.005 \
    -emb ${emb} \
    -nh 8 \
    --mlp_ratios 4 \
    --layers 1 \
    --keep_ratio 0.15 \
    --time_layers ${layers} \
    --alpha 0.2 \
    --patch_size ${patch_size} \
    -T ${T} \
    --no-bias \
    --scheduler cosine \
    --data UEA \
    --data_path ECG \
    --seq_len ${length} \
    --root_path ${root_path}
done
done
done
done
done
