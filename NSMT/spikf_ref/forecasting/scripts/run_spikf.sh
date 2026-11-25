#!/usr/bin/env bash
# run_spikf.sh (2026-11-25, exp/f-lif-pop-v3; PROJECT_LOG SpikF-LIF pre-registration; user request 11-25: "SpikF에 GeLU 대신 spiking
# neuron activation으로 교체한 실험 ... 성능과 에너지 소모량 비교 둘 다"): one ETT run of SpikF (copy of /home/yschoi/SpikF in
# spikf_ref/forecasting, PROVENANCE.md) with the settings of the released scripts (SpikF commit 041e398 scripts/run_<DS>.sh):
#   input 96, features M, batch 32, lr 5e-4, levels 1, patch_dim 32 (patch_num 48 by default), T 16, hidden 720, MAE loss (default),
#   ETTh1 5 epochs / patience 3, ETTh2 5 / 2, ETTm1 10 / 1, ETTm2 10 / 1; the released scripts never pass the seed (always 0).
# usage: ACT=<gelu|lif> run_spikf.sh <DS: ETTh1|ETTh2|ETTm1|ETTm2> <PL> <seed> <gpu>
#   env: SUITE RES LOG PY THREADS EPOCHS (overrides the per-dataset epoch count; smoke only)
#        FAST=1 -> --fast_eval (validation / final test in batches of EVBS, default 256; no in-loop test pass; anomaly mode off),
#        SYBS = syops loader batch (default 1, as released; 2026-11-26 check: 64 crashes -- syops does not reset the neurons between
#        batches, so the energy count of SpikF runs at batch 1 with the membrane state carried from window to window); both are in the
#        run folder and stdout names (_fe<EVBS>_sy<SYBS>)
# Writes one RESULT (or SKIP) line to results/<SUITE>/raw.txt; run folder log/<SUITE>/<DS>/p<PL>/<ACT>_seed<S>/ (checkpoint, test
# outputs, result.json; local), stdout log/<SUITE>/stdout/.
set -u
DS=${1:?}; PL=${2:?}; S=${3:?}; GPU=${4:?}
F="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # spikf_ref/forecasting
N="$(cd "$F/../.." && pwd)"                            # NSMT (2026-11-25: was named R and used as the worktree root -> NSMT/NSMT/... path, smoke SKIP)
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ACT=${ACT:?ACT=gelu|lif}
case $ACT in gelu|lif) ;; *) echo "bad ACT $ACT"; exit 1 ;; esac
case $DS in
  ETTh1) E=5; P=3 ;; ETTh2) E=5; P=2 ;; ETTm1) E=10; P=1 ;; ETTm2) E=10; P=1 ;;
  *) echo "bad DS $DS"; exit 1 ;;
esac
case $PL in 96|192|336|720) ;; *) echo "bad PL $PL"; exit 1 ;; esac
E=${EPOCHS:-$E}
FX=""; FT=""
case ${FAST:-0} in 0) ;; 1) FX="--fast_eval --eval_bs ${EVBS:-256} --syops_bs ${SYBS:-1}"; FT="_fe${EVBS:-256}_sy${SYBS:-1}" ;; *) echo "bad FAST"; exit 1 ;; esac
SUITE=${SUITE:?SUITE}
RES=${RES:-$F/results/$SUITE}; LOG=${LOG:-$F/log/$SUITE}
mkdir -p "$RES" "$LOG/stdout"; RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$N" && git rev-parse --short HEAD)
OUTD="$LOG/$DS/p$PL/${ACT}${FT}_seed$S"
[ -e "$OUTD" ] && { echo "SKIP(run folder exists) ds=$DS pl=$PL act=$ACT seed=$S dir=$OUTD" | tee -a "$RES/raw.txt"; exit 1; }
mkdir -p "$OUTD"
OUT=$LOG/stdout/spikf_${DS}_p${PL}_${ACT}${FT}_seed${S}_ep${E}_${SHA}.stdout; t0=$(date +%s)
(cd "$F" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY -B run_long.py -s --rank 0 --random_seed $S --activ $ACT --out "$OUTD" $FX \
  --data $DS --data_path $DS.csv --root_path "$N/forecasting/dataset/ETT-small" --features M --seq_len 96 --pred_len $PL \
  --batch_size 32 --lr 5e-4 --levels 1 --patch_dim 32 --T 16 --hidden_dim 720 --train_epoch $E --patience $P \
  --model_name SpikF_${DS}_input96_output${PL} --model SpikF > "$OUT" 2>&1)
RC=$?; t1=$(date +%s)
J="$OUTD/result.json"
if [ $RC -ne 0 ] || [ ! -s "$J" ]; then echo "SKIP(rc=$RC, no result) ds=$DS pl=$PL act=$ACT seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY -B - "$J" <<'EOF'
import json, math, sys
j = json.load(open(sys.argv[1]))
keys = ('mse', 'mae', 'val_loss_best', 'best_epoch', 'epochs_run', 'energy_uJ', 'AC_op', 'MAC_op', 'total_op', 'params', 'firing_rate')
assert all(j.get(k) is not None and math.isfinite(float(j[k])) for k in keys), 'missing or non-finite value'
print(' '.join(f'{k}={j[k]!r}' for k in keys))
EOF
)
[ -z "$VALS" ] && { echo "SKIP(result parse failed) ds=$DS pl=$PL act=$ACT seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; }
echo "RESULT exp=spikf_ref_ett ds=$DS pl=$PL act=$ACT seed=$S epochs_max=$E patience=$P fast=${FAST:-0}${FT} $VALS time_s=$((t1-t0)) sha=$SHA dir=$OUTD" | tee -a "$RES/raw.txt"
