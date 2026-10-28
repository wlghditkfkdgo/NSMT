#!/usr/bin/env bash
# run_ppg.sh (MODI 2026-10-28): one PPG-DaLiA heart-rate regression run (Log-NCDE protocol, seed-0 split; train_ppg.py), causal
# per-token models. Pre-registration: docs/MODI_PROJECT_LOG.md 2026-10-28.
# usage: COND=<cond> [EPOCHS=100] [BS=8] [SMOKE=n] [SUITE=...] run_ppg.sh <seed> <gpu>
#   ppg_tok            Spikformer-style (T_h 4), attention window 1 token (itself; 4 s)        P0
#   ppg_ssa_w8         ... window 3 tokens (8 s = the label window)                            P1
#   ppg_ssa_w32        ... window 15 tokens (32 s)                                              P2
#   ppg_ssa_full       ... all past tokens                                                      P3
#   ppg_mix_<n>        Neo-mixer, forward T = N stacks only, n = lif | flif | pop_lif | pop_flif   P4
#   ppg_spkneo_<n>     Spikformer-style window 3 + T = N neocortex keys / values (B form), n as above   P5
#   ppg_spkneo_lif_nomem   the same with the memoryless LIF neocortex                           P6
#   ppg_gru            non-spiking GRU reference (hidden 384)
#   common: D 384, 8 heads, mlp 1, 1 block, 2 neocortex layers, tau 2, patch 256 (stride 128, 390 tokens), AdamW wd 1e-4, timm cosine,
#           lr 1e-3, batch 8, 100 epochs, MSE; B form: alpha 0 (no aux loss), --neo_grad 1.0; population LIF soma 0.5, f-LIF soma 1.0
#   keep_ratio 0 (18:2x fix): the aux low-pass target (a DCT over the whole 49,920-sample window, 9.3 GB) is not built; no loss uses it
set -u
S=${1:?}; GPU=${2:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COND=${COND:?}
SSA="--model myModel --hippo_ssa --hippo_T 4"
BF="--model myModel --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0 --neo_where both --attn_window 3"
case $COND in
  ppg_tok)       CF="$SSA --attn_window 1" ;;
  ppg_ssa_w8)    CF="$SSA --attn_window 3" ;;
  ppg_ssa_w32)   CF="$SSA --attn_window 15" ;;
  ppg_ssa_full)  CF="$SSA --attn_window -1" ;;
  ppg_mix_lif)       CF="--model neoMixer --mixer lif" ;;
  ppg_mix_flif)      CF="--model neoMixer --mixer flif --neo_alpha 0.5" ;;
  ppg_mix_pop_lif)   CF="--model neoMixer --mixer pop_lif --pop_soma_init 0.5" ;;
  ppg_mix_pop_flif)  CF="--model neoMixer --mixer pop_flif --neo_alpha 0.5 --pop_soma_init 1.0" ;;
  ppg_spkneo_lif)       CF="$BF --neo_neuron lif" ;;
  ppg_spkneo_flif)      CF="$BF --neo_neuron flif --neo_alpha 0.5" ;;
  ppg_spkneo_pop_lif)   CF="$BF --neo_neuron pop_lif --pop_soma_init 0.5" ;;
  ppg_spkneo_pop_flif)  CF="$BF --neo_neuron pop_flif --neo_alpha 0.5 --pop_soma_init 1.0" ;;
  ppg_spkneo_lif_nomem) CF="$BF --neo_neuron lif --neo_memoryless" ;;
  ppg_gru)       CF="--model gru" ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
EP=${EPOCHS:-100}; LR=${LR:-0.001}
SUITE=${SUITE:-flif-v1-ppg-$(date +%Y%m%d)}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"; RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
OUT=$LOG/PPG_${COND}_seed${S}_lr${LR}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_ppg.py \
  --data PPG --seed $S --log_dir "$LOG/PPG" --gating attn -s -nd 0 --warm_up_epoch 0 \
  -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --time_layers 2 --keep_ratio 0 --alpha 0 --tau 2.0 --no-bias \
  --scheduler cosine -bs ${BS:-8} --eval_bs ${BS:-8} -lr $LR --weight_decay 1e-4 --max_lr 0.005 --patch_size 256 --seq_len 49920 -e $EP \
  --smoke_windows ${SMOKE:-0} $CF --tag $COND lr$LR seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/PPG"/*/*"+${COND}+lr${LR}+seed${S}+"*"+epoch+${EP}+"*/seed${S}_* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"; LCSV="$D/log/last/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) data=PPG cond=$COND seed=$S lr=$LR out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY - "$CSV" "$LCSV" <<'PYEOF'
import csv, sys
b = next(csv.DictReader(open(sys.argv[1]))); l = next(csv.DictReader(open(sys.argv[2])))
print(' '.join(f'{k}={b[k]}' for k in b if k != 'weights') + ' ' + ' '.join(f'last_{k}={l[k]}' for k in ('mse', 'mae', 'mae_bpm')))
PYEOF
)
BE=$(cat "$D/log/best_epoch.txt" 2>/dev/null | tr ' ' ':')
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE 'parameters : [0-9]+' | grep -oE '[0-9]+'); t1=$(date +%s)
echo "RESULT exp=nsmt_flif_v1_ppg data=PPG cond=$COND seed=$S lr=$LR $VALS best_epoch_val=$BE params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
