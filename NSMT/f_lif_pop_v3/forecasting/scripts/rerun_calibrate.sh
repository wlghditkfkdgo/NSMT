#!/usr/bin/env bash
# Prereg 2T: the D11 calibrations of the spikeDE-based neurons for the reruns (train split, seed 7).
# The rule is D11 unchanged (firing rate in [0.1, 0.3] closest to 0.2, G11 bound 10 max|I|). A
# selective configuration is probed as its q = 1 class and must pass its own at-init check; each
# configuration gets its own file in results/calibration (calibration_tag), created with 'x'.
#   recall : CPU, the legacy recall calibration's settings (--n_train 1024, batch 128, 8 batches)
#            pop_flif + sel_pop_flif pearson q {0.1, 0.25, 0.5} + recent / random q 0.5
#   ett    : one dataset, GPU; {R off a0.7, R on a0.7, R on a1} x {pop_flif, sel_pop_flif pearson q0.5}
# A refused configuration is recorded (rc != 0, no 'picked') and is NOT retried with other settings.
# usage: rerun_calibrate.sh recall <suite>          rerun_calibrate.sh ett <suite> <DS> <gpu>
#        rerun_calibrate.sh recall_extra <suite> <q>   (recent / random at a newly selected q)
#        CALDIR=<folder> rerun_calibrate.sh smoke <suite> <gpu>   (smoke test: 5 recall + 2 ETTh1 files)
#        rerun_calibrate.sh ett2u <suite> <DS> <gpu>     prereg 2U: --soma none --input_norm bn; the D11 scale
#                                                        becomes BN's initial gamma (6 files per dataset)
#        CALDIR=<folder> rerun_calibrate.sh smoke2u <suite> <gpu>   (2U smoke: ETTh1 R on, q1 and pearson)
set -u
WHAT=${1:?recall|ett|recall_extra}; SUITE=${2:?suite}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # torch 1.12.0+cu113
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RES=$W/results/$SUITE; LOG=$W/log/$SUITE; mkdir -p "$RES" "$LOG"
CALDIR=${CALDIR:-calibration}                              # smoke tests write to their own folder
SHA=$(cd "$R" && git rev-parse --short HEAD)
Q1="--neuron pop_flif --soma lif --mode full"
SEL="--neuron sel_pop_flif --soma lif --mode hard --hard_axis shared"

one() {   # <label> <device env> <args...>
  local label=$1 dev=$2; shift 2
  local out=$LOG/calibrate_${label}.log
  (cd "$W" && env $dev OMP_NUM_THREADS=${THREADS:-4} $PY ./calibrate.py "$@" --calibration_dir "$CALDIR" > "$out" 2>&1)
  local rc=$?
  local file=$(grep -aoE 'written to `[^`]+`' "$out" | sed -E 's/written to `(.*)`/\1/' | xargs -r basename)
  local pick=$(grep -a 'picked input_scale' "$out" | grep -oE 'input_scale = [0-9.]+' | grep -oE '[0-9.]+$')
  echo "CALIB label=$label rc=$rc input_scale=${pick:-none} file=${file:-none} sha=$SHA" | tee -a "$RES/calibration_raw.txt"
}

case $WHAT in
  recall)
    COMMON="--task recall --data recall --n_train 1024 --input_norm frozen --seed 7 --cpu"
    one recall_q1          "CUDA_VISIBLE_DEVICES=" $COMMON $Q1
    for Q in 0.1 0.25 0.5; do
      one recall_pearson_q$Q "CUDA_VISIBLE_DEVICES=" $COMMON $SEL --hard_stat pearson --hard_q $Q
    done
    for S in recent random; do
      one recall_${S}_q0.5 "CUDA_VISIBLE_DEVICES=" $COMMON $SEL --hard_stat $S --hard_q 0.5
    done ;;
  recall_extra)
    Q=${3:?q}
    COMMON="--task recall --data recall --n_train 1024 --input_norm frozen --seed 7 --cpu"
    for S in recent random; do
      one recall_${S}_q$Q "CUDA_VISIBLE_DEVICES=" $COMMON $SEL --hard_stat $S --hard_q $Q
    done ;;
  ett)
    DS=${3:?dataset}; GPU=${4:?gpu}
    COMMON="--task ett --data $DS --input_norm frozen --seed 7 -nd 0"
    for RV in "Roff" "R" "R_a1"; do
      case $RV in Roff) EX="" ;; R) EX="--revin" ;; R_a1) EX="--revin --alpha 1" ;; esac
      one ${DS}_${RV}_q1      "CUDA_VISIBLE_DEVICES=$GPU" $COMMON $EX $Q1
      one ${DS}_${RV}_pearson "CUDA_VISIBLE_DEVICES=$GPU" $COMMON $EX $SEL --hard_stat pearson --hard_q 0.5
    done ;;
  smoke)
    GPU=${3:?gpu}
    [ "$CALDIR" != calibration ] || { echo 'smoke calibrations need their own CALDIR'; exit 1; }
    COMMON="--task recall --data recall --n_train 1024 --input_norm frozen --seed 7 --cpu"
    one recall_q1 "CUDA_VISIBLE_DEVICES=" $COMMON $Q1
    for Q in 0.1 0.25 0.5; do one recall_pearson_q$Q "CUDA_VISIBLE_DEVICES=" $COMMON $SEL --hard_stat pearson --hard_q $Q; done
    one recall_random_q0.5 "CUDA_VISIBLE_DEVICES=" $COMMON $SEL --hard_stat random --hard_q 0.5
    COMMON="--task ett --data ETTh1 --input_norm frozen --seed 7 -nd 0 --revin"
    one ETTh1_R_q1      "CUDA_VISIBLE_DEVICES=$GPU" $COMMON $Q1
    one ETTh1_R_pearson "CUDA_VISIBLE_DEVICES=$GPU" $COMMON $SEL --hard_stat pearson --hard_q 0.5 ;;
  ett2u|smoke2u)
    if [ "$WHAT" = smoke2u ]; then DS=ETTh1; GPU=${3:?gpu}; RVS="R"
      [ "$CALDIR" != calibration ] || { echo 'smoke calibrations need their own CALDIR'; exit 1; }
    else DS=${3:?dataset}; GPU=${4:?gpu}; RVS="Roff R R_a1"; fi
    COMMON="--task ett --data $DS --input_norm bn --seed 7 -nd 0"
    U1="--neuron pop_flif --soma none --mode full"
    US="--neuron sel_pop_flif --soma none --mode hard --hard_axis shared --hard_stat pearson --hard_q 0.5"
    for RV in $RVS; do
      case $RV in Roff) EX="" ;; R) EX="--revin" ;; R_a1) EX="--revin --alpha 1" ;; esac
      one ${DS}_${RV}_q1      "CUDA_VISIBLE_DEVICES=$GPU" $COMMON $EX $U1
      one ${DS}_${RV}_pearson "CUDA_VISIBLE_DEVICES=$GPU" $COMMON $EX $US
    done ;;
  ett2v)   # prereg 2V: the two new soma x norm combinations, q1 only (R on and off); usage: ett2v <suite> <DS> <gpu> <soma> <norm>
    DS=${3:?dataset}; GPU=${4:?gpu}; SOMA=${5:?soma}; NORM=${6:?norm}
    COMMON="--task ett --data $DS --input_norm $NORM --seed 7 -nd 0 --neuron pop_flif --soma $SOMA --mode full"
    one ${DS}_Roff_q1 "CUDA_VISIBLE_DEVICES=$GPU" $COMMON
    one ${DS}_R_q1    "CUDA_VISIBLE_DEVICES=$GPU" $COMMON --revin ;;
  ett2x)   # prereg 2X: the 2W pick's calibration for another patching; usage: ett2x <suite> <DS> <gpu> <patch> <stride>
    DS=${3:?dataset}; GPU=${4:?gpu}; P=${5:?patch}; ST=${6:?stride}
    COMMON="--task ett --data $DS --input_norm bn --seed 7 -nd 0 --neuron pop_flif --soma none --mode full --patch_size $P --patch_stride $ST"
    one ${DS}_Roff_q1 "CUDA_VISIBLE_DEVICES=$GPU" $COMMON
    one ${DS}_R_q1    "CUDA_VISIBLE_DEVICES=$GPU" $COMMON --revin ;;
  *) echo "bad target $WHAT"; exit 1 ;;
esac
