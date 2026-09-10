#!/bin/bash


data=ETTh2
# patch_size=32
seq_len=336
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small

keep_ratio=0.15
for gating in attn
do
for len in 720 #336 192 96
do
for patch_size in 8
do
for layer in 1
do
for emb in 16
do
for nh in 2
do
for alpha in 1
do
for lr in 0.0001
do
for seed in 42
do
for model in myModel
do
python3 ./train.py \
    --model ${model} \
    --seed ${seed} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 2 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb ${emb} \
    -nh ${nh} \
    -lr ${lr} \
    --alpha ${alpha} \
    --keep_ratio ${keep_ratio} \
    --mlp_ratios 2 \
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
    # --label_len 168 \