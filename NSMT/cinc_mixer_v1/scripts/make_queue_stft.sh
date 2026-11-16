#!/usr/bin/env bash
# make_queue_stft.sh (2026-11-16, exp/f-lif-pop-v3; PROJECT_LOG STFT band test pre-registration): queues of the STFT band test.
# 7 new forms = {LIF, A (LIF + band groups), POP (population LIF), B (population LIF + band per member)} x {rep, token} minus LIF rep
# (the tuned LIF SpkMixer runs of 11-01 / 11-02 are reused). Settings are the tuned LIF SpkMixer ones, not re-tuned:
#   ETT      muxlin + IAND, 1 mixer layer per direction, lr 5e-4 ETTh2 / 1e-4 ETTm2
#   MIT-BIH  muxlin + ADD, 2 mixer layers, lr 0.02
#   smoke : 2 epochs ETT (ETTh2 H96, ETTm2 H720) and 1 epoch MIT-BIH, seed 42, all 7 forms = 21 runs
#   ett   : 4 cells x 3 seeds x 7 forms = 84 runs (suite flif-v1-ett-spkmux-stft)
#   mitbih: 3 seeds x 7 forms = 21 runs (suite flif-v1-mitbih-spkmux-stft)
# usage (from anywhere): bash make_queue_stft.sh smoke|ett|mitbih ; prints <queue file>:<jobs>
set -u
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/queues"; mkdir -p "$Q"
E="bash cinc_mixer_v1/scripts/run_ett_spk.sh"
M="bash cinc_mixer_v1/scripts/run_mitbih_spk.sh"
D=20261116
lr_mx() { case $1 in ETTh2) echo 0.0005 ;; ETTm2) echo 0.0001 ;; esac; }
# form -> "<neuron> <BAND> <TAXIS>"; the population forms first (slowest)
FORMS=("pop_lif member token" "pop_lif none token" "pop_lif member rep" "pop_lif none rep" "lif group token" "lif none token" "lif group rep")
ett_line() {   # <prefix> <ds> <pl> <seed> <form...>
  local pre=$1 ds=$2 pl=$3 s=$4 nrn=$5 band=$6 tx=$7
  echo "$pre BAND=$band TAXIS=$tx COND=spkmx_${nrn}_muxlin_iand LR=$(lr_mx $ds) LAYERS=1 $E $ds $pl $s"
}
mit_line() {   # <prefix> <seed> <form...>
  local pre=$1 s=$2 nrn=$3 band=$4 tx=$5
  echo "$pre BAND=$band TAXIS=$tx COND=spkmx_${nrn}_muxlin_add LR=0.02 MLAYERS=2 $M $s"
}
case ${1:?smoke|ett|mitbih} in
  smoke) f="$Q/stft-smoke-$D.txt"; : > "$f"
         for fm in "${FORMS[@]}"; do
           ett_line "EPOCHS=2 SUITE=smoke-stft-ett-$D" ETTh2 96 42 $fm >> "$f"
           ett_line "EPOCHS=2 SUITE=smoke-stft-ett-$D" ETTm2 720 42 $fm >> "$f"
           mit_line "EPOCHS=1 SUITE=smoke-stft-mitbih-$D" 42 $fm >> "$f"
         done ;;
  ett)   f="$Q/stft-ett-$D.txt"; : > "$f"
         for dp in "ETTm2 720" "ETTm2 96" "ETTh2 720" "ETTh2 96"; do
           for fm in "${FORMS[@]}"; do for s in 42 7 2026; do ett_line "SUITE=flif-v1-ett-spkmux-stft" $dp $s $fm >> "$f"; done; done
         done ;;
  mitbih) f="$Q/stft-mitbih-$D.txt"; : > "$f"
         for fm in "${FORMS[@]}"; do for s in 42 7 2026; do mit_line "SUITE=flif-v1-mitbih-spkmux-stft" $s $fm >> "$f"; done; done ;;
  *) echo "bad stage $1"; exit 1 ;;
esac
echo "$f:$(wc -l < "$f")"
