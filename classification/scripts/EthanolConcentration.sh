#!/bin/bash 

root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/classification/dataset
dataset_name=EthanolConcentration

patch_size=64
patch_stride=32
T=24
length=1751
n_classes=4
num_channels=3

for gating in attn
do
for layers in 4
do
for emb in 768
do
for lr in 0.05
do
for seed in 42
do
python3 train.py \
    --model myModel \
    --seed ${seed}\
    --log_dir ${dataset_name}\
    --gating ${gating}\
    --attn SSA \
    -s --test --print_epoch 1 -nd 0 \
    -e 15 --warm_up_epoch 0 \
    --num_channels ${num_channels} \
    -bs 16 \
    -lr ${lr} \
    --max_lr 0.005 \
    -emb ${emb} \
    -nh 8 \
    --mlp_ratios 1 \
    --layers 1 \
    --keep_ratio 0.15 \
    --time_layers ${layers} \
    --alpha 0.1 \
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
