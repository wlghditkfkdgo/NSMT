#!/bin/bash 

root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/classification/dataset
dataset_name=Heartbeat
patch_size=16
patch_stride=16
T=24
length=405
n_classes=2
num_channels=61

for gating in attn
do
for layers in 4
do
for emb in 512
do
for lr in 0.05
do
for seed in 42
do
python3 train.py \
    --model myModel \
    --seed ${seed}\
    --log_dir ${dataset_name}_seed\
    --gating ${gating}\
    --attn SSA \
    -s --test --print_epoch 1 -nd 1 \
    -e 50 --warm_up_epoch 0 \
    --num_channels ${num_channels} \
    -bs 64 \
    -lr ${lr} \
    --max_lr 0.01 \
    --keep_ratio 0.15 \
    -emb ${emb} \
    -nh 8 \
    --layers 1 \
    --time_layers ${layers} \
    --alpha 0.5 \
    --patch_size ${patch_size} \
    --patch_stride ${patch_stride} \
    -T ${T} \
    --no-bias \
    --scheduler cosine \
    --tag ${dataset_name} ${seed} \
    --num_classes ${n_classes} \
    --data_path ${dataset_name}\
    --data UEA \
    --root_path ${root_path}/${dataset_name}/\

done
done
done
done
done
