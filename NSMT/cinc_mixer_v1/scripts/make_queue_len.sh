#!/usr/bin/env bash
# make_queue_len.sh (2026-11-17, exp/f-lif-pop-v3; PROJECT_LOG input-length test pre-registration): queues of the ETT input-length test.
# 3 models at input length L (the input-96 rows are existing runs): LIF SpkMixer, population LIF SpkMixer, Spikformer-style.
# Settings are the tuned input-96 ones, not re-tuned:
#   SpkMixer (LIF, population LIF)  muxlin + IAND, 1 mixer layer per direction, lr 5e-4 ETTh2 / 1e-4 ETTm2
#   Spikformer-style                lr 5e-4 ETTh2 / 2.5e-4 ETTm2, layers 1 ETTh2 / 2 ETTm2
#   repro : input 96 again, default path, ETTh2 H96 seed 42, LIF SpkMixer and Spikformer-style: must match the existing rows (check by hand / summarize_len --repro)
#   smoke <L> : 2 epochs, ETTh2 H96 and ETTm2 H720, seed 42, 3 models = 6 runs (suite smoke-len<L>-ett-<date>)
#   run <L>   : 4 cells x 3 seeds x 3 models = 36 runs (suite flif-v1-ett-spkmux-len<L>); population runs interleaved with the light ones
# usage (from anywhere): bash make_queue_len.sh repro | smoke <L> | run <L> ; prints <queue file>:<jobs>
set -u
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/queues"; mkdir -p "$Q"
E="bash cinc_mixer_v1/scripts/run_ett_spk.sh"
D=20261117
lr_mx() { case $1 in ETTh2) echo 0.0005 ;; ETTm2) echo 0.0001 ;; esac; }
lr_sf() { case $1 in ETTh2) echo 0.0005 ;; ETTm2) echo 0.00025 ;; esac; }
ly_sf() { case $1 in ETTh2) echo 1 ;; ETTm2) echo 2 ;; esac; }
line() {   # <prefix> <model: pop|lif|sf> <ds> <pl> <seed>
  local pre=$1 m=$2 ds=$3 pl=$4 s=$5
  case $m in
    pop) echo "$pre COND=spkmx_pop_lif_muxlin_iand LR=$(lr_mx $ds) LAYERS=1 $E $ds $pl $s" ;;
    lif) echo "$pre COND=spkmx_lif_muxlin_iand LR=$(lr_mx $ds) LAYERS=1 $E $ds $pl $s" ;;
    sf)  echo "$pre COND=spk_ssa LR=$(lr_sf $ds) LAYERS=$(ly_sf $ds) $E $ds $pl $s" ;;
  esac
}
case ${1:?repro|smoke <L>|run <L>} in
  repro) f="$Q/len-repro-$D.txt"; : > "$f"
         line "SUITE=check-len-repro-$D" lif ETTh2 96 42 >> "$f"; line "SUITE=check-len-repro-$D" sf ETTh2 96 42 >> "$f" ;;
  smoke) L=${2:?length}; f="$Q/len$L-smoke-$D.txt"; : > "$f"
         for dp in "ETTh2 96" "ETTm2 720"; do for m in pop lif sf; do line "EPOCHS=2 SEQ=$L SUITE=smoke-len$L-ett-$D" $m $dp 42 >> "$f"; done; done ;;
  run)   L=${2:?length}; f="$Q/len$L-$D.txt"; : > "$f"
         for dp in "ETTm2 720" "ETTm2 96" "ETTh2 720" "ETTh2 96"; do
           for s in 42 7 2026; do for m in pop lif sf; do line "SEQ=$L SUITE=flif-v1-ett-spkmux-len$L" $m $dp $s >> "$f"; done; done
         done ;;
  *) echo "bad stage $1"; exit 1 ;;
esac
echo "$f:$(wc -l < "$f")"
