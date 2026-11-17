#!/usr/bin/env bash
# make_queue_mx3.sh (2026-11-17, exp/f-lif-pop-v3; PROJECT_LOG mixer-variant pre-registration): queues of the three mixer variants, LIF stage.
# Each arm changes ONE thing in the tuned LIF SpkMixer (muxlin; ETT IAND, 1 mixer layer per direction, lr 5e-4 ETTh2 / 1e-4 ETTm2;
# MIT-BIH ADD, 2 layers, lr 0.02), not re-tuned:
#   1 pooling : MIXPOOL=max | mean        (repeats pooled before the mixer, mixer + fusion once, copied back)
#   2 shortcut: ETT LAYERS=2 (plain) and LAYERS=2 MIXMS=1;  MIT-BIH MIXMS=1 (already 2 layers)
#   3 reset   : MIXRESET=subtract | zero   (directional-stack LIF neurons only)
#   repro : the control again, seed 42 (ETTh2 H96 and MIT-BIH): must equal the tuned rows in every shared field but time, sha, dir
#   smoke : 2 epochs ETT (ETTh2 H96, ETTm2 H720), 1 epoch MIT-BIH, seed 42, every arm = 17 runs
#   ett   : 6 arms x 4 cells x 3 seeds = 72 runs (suite flif-v1-ett-spkmux-mx3)
#   mitbih: 5 arms x 3 seeds = 15 runs (suite flif-v1-mitbih-spkmux-mx3)
# usage (from anywhere): bash make_queue_mx3.sh repro|smoke|ett|mitbih ; prints <queue file>:<jobs>
set -u
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/queues"; mkdir -p "$Q"
E="bash cinc_mixer_v1/scripts/run_ett_spk.sh"
M="bash cinc_mixer_v1/scripts/run_mitbih_spk.sh"
D=20261118
lr_mx() { case $1 in ETTh2) echo 0.0005 ;; ETTm2) echo 0.0001 ;; esac; }
ETT_ARMS=("MIXPOOL=max LAYERS=1" "MIXPOOL=mean LAYERS=1" "LAYERS=2" "MIXMS=1 LAYERS=2" "MIXRESET=subtract LAYERS=1" "MIXRESET=zero LAYERS=1")
MIT_ARMS=("MIXPOOL=max" "MIXPOOL=mean" "MIXMS=1" "MIXRESET=subtract" "MIXRESET=zero")
ett_line() { echo "$1 $2 COND=spkmx_lif_muxlin_iand LR=$(lr_mx $3) $E $3 $4 $5"; }    # <prefix> <arm> <ds> <pl> <seed>
mit_line() { echo "$1 $2 COND=spkmx_lif_muxlin_add LR=0.02 MLAYERS=2 $M $3"; }          # <prefix> <arm> <seed>
case ${1:?repro|smoke|ett|mitbih} in
  repro) f="$Q/mx3-repro-$D.txt"; : > "$f"
         ett_line "SUITE=check-mx3-repro-$D-ett" "LAYERS=1" ETTh2 96 42 >> "$f"
         mit_line "SUITE=check-mx3-repro-$D" "" 42 >> "$f" ;;
  smoke) f="$Q/mx3-smoke-$D.txt"; : > "$f"
         for a in "${ETT_ARMS[@]}"; do for dp in "ETTh2 96" "ETTm2 720"; do ett_line "EPOCHS=2 SUITE=smoke-mx3-ett-$D" "$a" $dp 42 >> "$f"; done; done
         for a in "${MIT_ARMS[@]}"; do mit_line "EPOCHS=1 SUITE=smoke-mx3-mitbih-$D" "$a" 42 >> "$f"; done ;;
  ett)   f="$Q/mx3-ett-$D.txt"; : > "$f"
         for dp in "ETTm2 720" "ETTm2 96" "ETTh2 720" "ETTh2 96"; do
           for a in "${ETT_ARMS[@]}"; do for s in 42 7 2026; do ett_line "SUITE=flif-v1-ett-spkmux-mx3" "$a" $dp $s >> "$f"; done; done
         done ;;
  mitbih) f="$Q/mx3-mitbih-$D.txt"; : > "$f"
         for a in "${MIT_ARMS[@]}"; do for s in 42 7 2026; do mit_line "SUITE=flif-v1-mitbih-spkmux-mx3" "$a" $s >> "$f"; done; done ;;
  *) echo "bad stage $1"; exit 1 ;;
esac
echo "$f:$(wc -l < "$f")"
