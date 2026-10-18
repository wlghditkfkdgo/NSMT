#!/bin/bash

root_path=/home/yschoi/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/classification/data
dataset_name=ECG

patch_size=32
# patch_stride=32
# T=24
length=187
n_classes=5
num_channels=1

#keep_ratio = 0.15
for gating in attn
do
for layers in 2
do
for emb in 384
do
for lr in 0.01
do
for keep_ratio in 0
do
python3 train.py \
    --seed 42\
    --log_dir ../results/ECG/${dataset_name}\
    --gating ${gating}\
    --attn SSA \
    -s --test --print_epoch 1 -nd 2 \
    -e 30 --warm_up_epoch 0 \
    --num_channels ${num_channels} \
    -bs 512 \
    -lr ${lr} \
    --max_lr 0.005 \
    -emb ${emb} \
    -nh 8 \
    --mlp_ratios 1 \
    --layers 1 \
    --keep_ratio ${keep_ratio} \
    --time_layers ${layers} \
    --alpha 0.1 \
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
