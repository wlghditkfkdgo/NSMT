data=MSL
patch_size=12
seq_len=100
c_out=55
root_path=/home/yschoi/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/anomaly_detection/dataset/MSL

for gating in attn
do
for lr in 0.0005
do
for model in myModel #ab1_1
do
python3 ./train.py \
    --model ${model} \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 0 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 128 \
    -emb 64 \
    -nh 8 \
    -lr ${lr} \
    --alpha 0.5 \
    --time_layers 4 \
    --mlp_ratios 4 \
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