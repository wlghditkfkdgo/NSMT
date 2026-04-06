data=SMD
patch_size=8
seq_len=100
c_out=38
root_path=/home/yschoi/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/anomaly_detection/dataset/${data}

for lr in 0.0005
do
for gating in attn
do
python3 ./train.py \
    --model myModel \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 4 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 128 \
    -emb 32 \
    -nh 8 \
    -lr ${lr} \
    --alpha 0.2 \
    --time_layers 2 \
    --keep_ratio 0 \
    --mlp_ratios 4 \
    --data ${data} \
    --pred_len 0 \
    --patch_size ${patch_size} \
    --root_path ${root_path} \
    --seq_len ${seq_len} \
    --c_out ${c_out} \
    --anomaly_ratio 0.5 \
    --features M 
done
done