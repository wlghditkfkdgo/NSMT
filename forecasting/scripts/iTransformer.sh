#!/bin/bash
# scripts/iTransformer.sh
 
MODEL=iTransformer
DATA=ETTh1
ROOT=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/forecasting/dataset/ETT-small
SEQ=96
PRED=720
C_IN=7          # ETTh1 변수 수
 
python3 train.py \
  --seed 2023 \
  --model $MODEL \
  --data  $DATA  \
  --root_path $ROOT \
  --data_path ${DATA}.csv \
  --seq_len  $SEQ  \
  --pred_len $PRED \
  --c_in     $C_IN \
  --itrans_d_model  512 \
  --itrans_n_heads  8   \
  --itrans_e_layers 2   \
  --itrans_d_ff     512 \
  --epoch 10 \
  -bs 32 \
  -lr 1e-4 \
  --scheduler reduce \
  --patience 3 \
  -s --test