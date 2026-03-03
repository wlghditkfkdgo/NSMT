#!/bin/bash

data=ETTh1
# patch_size=16
# seq_len=336
patch_size=8
seq_len=96
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small

for len in 720 336 192 96
do
for gating in attn #attn #ablation
do
for model in ab1 ab3 #ab1_1 ab1 #ab1_1 Spikformer
do
for seed in 42
do
python3 ./train.py \
    --log_dir final3 \
    --seed ${seed} \
    --model ${model} \
    --gating ${gating} \
    --keep_ratio 0.25 \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 0 \
    -e 50 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb 32 \
    -nh 2 \
    -lr 0.0005 \
    --alpha 0.8 \
    --time_layers 2 \
    --layers 1 \
    --mlp_ratios 1 \
    --max_ratio 2 \
    --patience 3 \
    --data ${data} \
    --data_path ${data}.csv \
    --pred_len ${len} \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_in 7 \
    
done
done
done
done
    # --label_len 168 \ 