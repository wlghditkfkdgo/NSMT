data=SWAT
patch_size=12
seq_len=100
c_out=51
root_path=/home/yschoi/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/anomaly_detection/dataset/SWaT

for lr in 0.0005
do
for gating in attn #ablation
do
python3 ./train.py \
    --model myModel \
    --gating ${gating} \
    --no-bias \
    --scheduler reduce \
    -s \
    --test \
    -nd 3 \
    -e 50 \
    --warm_up_epoch 0 \
    -bs 64 \
    -emb 128 \
    -nh 8 \
    -lr ${lr} \
    --alpha 0.8 \
    --time_layers 1 \
    --mlp_ratios 6 \
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