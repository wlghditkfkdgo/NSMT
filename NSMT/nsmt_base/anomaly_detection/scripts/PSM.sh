data=PSM
patch_size=16
seq_len=100
c_out=25
root_path=/home/yschoi/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/anomaly_detection/dataset/${data}

for gating in attn
do
for lr in 0.0001
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
    -nd 1 \
    -e 100 \
    --warm_up_epoch 0 \
    -bs 256 \
    -emb 64 \
    -nh 8 \
    -lr ${lr} \
    --keep_ratio 0.25 \
    --alpha 0.9 \
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