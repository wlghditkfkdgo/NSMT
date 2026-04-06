data=SMAP
patch_size=8
seq_len=100
c_out=25
root_path=/home/yschoi/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/anomaly_detection/dataset/${data}

for lr in 0.0005
do
for gating in attn
do
for model in myModel
do
python3 ./train.py \
    --model ${model} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 2 \
    -e 50 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb 128 \
    -nh 8 \
    -lr ${lr} \
    --alpha 0.1 \
    --keep_ratio 0 \
    --time_layers 2 \
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
done