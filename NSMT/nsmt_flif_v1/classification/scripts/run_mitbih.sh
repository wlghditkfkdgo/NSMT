#!/usr/bin/env bash
# nsmt_flif_v1: one MIT-BIH run of the ECG model (nsmt_base/classification/ECG) with the neocortex-path neuron replaced.
# Training = the ECG framework's clean 3-seed setting (ECG/scripts/_run_mitbih_3seed.sh): 30 epochs, cosine schedule,
# bs 512, lr 0.01, emb 384, patch 32 (11 tokens), keep_ratio 0.15, aux weight 0.1, CAL-weighted CE; the last-epoch
# weights are tested on the 30% test file (per-epoch "val" columns are that test file; nothing is selected on them).
# Only the condition flags differ.
# usage: COND=<cond> run_mitbih.sh <seed> <gpu>
#   COND: base | neo_lif_both | neo_flif_both | neo_lif_mlp | neo_flif_mlp | no_neo | hippo_only   (default base)
#         stage 6: neo_pop_both | bidir_lif_both | bidir_flif_both | bidir_pop_both  (population f-LIF, bidirectional neocortex)
#   env:  ALPHA (0.5) NEO_TAU (2.0) FDE (pred) H (1.0) MEM (0) POP_TAUS (4,8,16,32) SUITE EPOCHS (30) RES LOG DATA PY THREADS
set -u
S=${1:?}; GPU=${2:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root (nsmt_flif_v1/classification)
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # Python 3.10, torch 1.12.0+cu113, spikingjelly cupy backend
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COND=${COND:-base}; ALPHA=${ALPHA:-0.5}
NEOX="--neo_tau ${NEO_TAU:-2.0} --fde_method ${FDE:-pred} --fde_step ${H:-1.0} --fde_memory ${MEM:-0}"
case $COND in
  base)          CF="" ;;
  neo_lif_both)  CF="--neo_neuron lif  --neo_where both $NEOX" ;;
  neo_flif_both) CF="--neo_neuron flif --neo_where both --neo_alpha $ALPHA $NEOX" ;;
  neo_lif_mlp)   CF="--neo_neuron lif  --neo_where mlp  $NEOX" ;;
  neo_flif_mlp)  CF="--neo_neuron flif --neo_where mlp  --neo_alpha $ALPHA $NEOX" ;;
  no_neo)        CF="--no_neo" ;;                                  # ECG framework ablation: no neocortex, no cross-attention
  hippo_only)    CF="--hippo_only" ;;                              # MODI decision 7: no neocortex, key/value = hippocampus
  neo_pop_both)  CF="--neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_taus ${POP_TAUS:-4,8,16,32} $NEOX" ;;   # stage 6
  bidir_lif_both)  CF="--neo_bidir --neo_neuron lif  --neo_where both $NEOX" ;;
  bidir_flif_both) CF="--neo_bidir --neo_neuron flif --neo_where both --neo_alpha $ALPHA $NEOX" ;;
  bidir_pop_both)  CF="--neo_bidir --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_taus ${POP_TAUS:-4,8,16,32} $NEOX" ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
SUITE=${SUITE:-flif-v1-cls-$(date +%Y%m%d)}
EPOCHS=${EPOCHS:-30}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
DATA=${DATA:-$R/nsmt_base/classification/ECG/data}          # 5class_mitbih_{train_7,test_3}_187.csv (local, not in Git)
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
OUT=$LOG/mitbih_${COND}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py \
  --model myModel --seed $S --log_dir "$LOG/MITBIH" --gating attn --attn SSA -s --test --print_epoch 1 -nd 0 \
  -e $EPOCHS --warm_up_epoch 0 --num_channels 1 -bs 512 -lr 0.01 --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 \
  --layers 1 --keep_ratio 0.15 --time_layers 2 --alpha 0.1 --patch_size 32 --no-bias --scheduler cosine \
  --data UEA --data_path ECG --seq_len 187 --root_path "$DATA" $CF --tag $COND seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/MITBIH"/*/*"+${COND}+seed${S}+model+"*/seed${S}_* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) cond=$COND seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY - "$CSV" <<'EOF'
import sys
rows = [l.rstrip('\n').split(', ') for l in open(sys.argv[1]) if l.strip()]
h, v = rows[-2], rows[-1]
d = dict(zip(h, v))
keys = ['overall_acc', 'average_acc', 'f1', 'precision', 'sensitivity', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'f1_4',
        'acc0', 'acc1', 'acc2', 'acc3', 'acc4', 'neuron_lif_uJ', 'neuron_model_fft_uJ', 'neuron_model_direct_uJ',
        'total_fft_uJ', 'total_direct_uJ', 'rho_neo', 'rho_hippo', 'n_test']
out = ' '.join(f'{k}={d[k]}' for k in keys)
out += ' syn_energy_uJ=' + d['energy'].replace(' uJ', '') + ' firing_rate=' + d['firing_rate'].replace(' %', '')
print(out)
EOF
)
[ -z "$VALS" ] && { echo "SKIP(result parse failed) cond=$COND seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; }
LRS=$(grep -aoE '^epoch +[0-9]+ \(lr=[0-9.e-]+\)' "$OUT" | grep -oE 'lr=[0-9.e-]+' | cut -d= -f2)
EP=$(echo "$LRS" | grep -c .); LR_FINAL=$(echo "$LRS" | tail -1)
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE '[0-9]+' | tail -1); t1=$(date +%s)
echo "RESULT exp=nsmt_flif_v1_cls cond=$COND dataset=MITBIH seed=$S $VALS epochs=$EPOCHS epochs_run=$EP lr_final=$LR_FINAL params=$PAR time_s=$((t1-t0)) sha=$SHA alpha=$ALPHA dir=$D" | tee -a "$RES/raw.txt"
