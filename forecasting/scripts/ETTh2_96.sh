#!/bin/bash


data=ETTh2
# patch_size=32
seq_len=96
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small

# keep_ratio=0.3
for keep_ratio in 0.1 0.15 0.2 0.25 #ablation
do
for len in 720
do
for patch_size in 8
do
for emb in 64
do
for nh in 8
do
for alpha in 0.8
do
for lr in 0.0005
do
for seed in 42
do
for model in myModel #Spikformer
do
python3 ./train.py \
    --log_dir final3/keep_ratio \
    --model ${model} \
    --analysis \
    --seed ${seed} \
    --gating attn \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 1 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb ${emb} \
    -nh ${nh} \
    -lr ${lr} \
    --alpha ${alpha} \
    --keep_ratio ${keep_ratio} \
    --mlp_ratios 0.5 \
    --max_ratio 2 \
    --time_layers 1 \
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
    # --label_len 168 \