data=MSL
patch_size=16
seq_len=100
c_out=55
root_path=/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/anomaly_detection/dataset/MSL

for gating in original ablation
do
for lr in 0.0005
do
python3 ./train.py \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 2 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 128 \
    -emb 128 \
    -nh 8 \
    -lr ${lr} \
    --alpha 1 \
    --time_layers 1 \
    --mlp_ratios 2 \
    --data ${data} \
    --pred_len 0 \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_out ${c_out} \
    --anomaly_ratio 1 \
    --features M     
done
done