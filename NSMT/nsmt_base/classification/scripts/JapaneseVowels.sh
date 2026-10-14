#!/bin/bash 

root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/classification/dataset
dataset_name=JapaneseVowels
patch_size=8
patch_stride=2
T=24
length=29
n_classes=9
num_channels=12

for gating in attn
do
for emb in 768
do
for lr in 0.005
do
for layers in 2
do
for seed in 42
do
python3 train.py \
    --model myModel \
    --seed ${seed}\
    --gating ${gating}\
    --log_dir ${dataset_name}\
    -s \
    --attn SSA\
    --test \
    --print_epoch 1 \
    -nd 1 \
    -e 15 \
    --keep_ratio 0.15 \
    --warm_up_epoch 0 \
    --mlp_ratios 1.5 \
    --num_channels ${num_channels} \
    -bs 16 \
    -lr ${lr} \
    --max_lr 0.01 \
    -emb ${emb} \
    -nh 16 \
    --layers 1 \
    --time_layers ${layers} \
    --alpha 0.1 \
    --patch_size ${patch_size} \
    --patch_stride ${patch_stride} \
    -T ${T} \
    --no-bias \
    --scheduler cosine \
    --num_classes ${n_classes} \
    --data_path ${dataset_name} \
    --data UEA \
    --root_path ${root_path}/${dataset_name}/ \

done
done
done
done
done
