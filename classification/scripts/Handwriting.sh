#!/bin/bash 

root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/classification/dataset
dataset_name=Handwriting
patch_size=8
patch_stride=16
T=24
length=152
n_classes=26
num_channels=3

for gating in attn
do
for emb in 512
do
for layers in 2
do
for lr in 0.005
do
python3 train.py \
    --model myModel \
    --log_dir ${dataset_name}\
    --gating ${gating}\
    --attn SSA\
    -s --test --print_epoch 1 -nd 0 \
    -e 50 --warm_up_epoch 0 \
    --num_channels ${num_channels} \
    -bs 16 \
    -lr ${lr} \
    --max_lr 0.01 \
    -emb ${emb} \
    -nh 8 \
    --layers 1 \
    --time_layers ${layers} \
    --alpha 0.1 \
    --keep_ratio 0 \
    --patch_size ${patch_size} \
    --patch_stride ${patch_stride} \
    -T ${T} \
    --no-bias \
    --seq_len ${length} \
    --num_classes ${n_classes} \
    --scheduler cosine \
    --tag ${dataset_name} ${seed} \
    --num_classes ${n_classes} \
    --data_path ${dataset_name} \
    --data UEA \
    --root_path ${root_path}/${dataset_name}/ \

done
done
done
done
