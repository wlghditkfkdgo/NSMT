#!/usr/bin/env bash
# make_queue_l2.sh (2026-11-07, exp/f-lif-pop-v3; PROJECT_LOG L2 pre-registration): queues of the MSE-loss test of the tuned SpkMixer on ETT.
# The model settings are the tuned L1 ones of flif-v1-ett-spkmux-tune: muxlin, LIF, IAND residual, 1 mixer layer per direction, lr 5e-4 on ETTh2, 1e-4 on ETTm2.
#   repro : the L1 run of ETTh2 H96 seed 42 again, default path (must match the tuned row in every field but time, sha, dir)
#   smoke : 2 epochs, L1 and MSE, ETTh2 H96 and ETTm2 H720, seed 42
#   l2    : 4 cells x seeds 42 / 7 / 2026 = 12 runs, LOSS=mse
# usage (from anywhere): bash make_queue_l2.sh repro|smoke|l2 ; prints <queue file>:<jobs>
set -u
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/queues"; mkdir -p "$Q"
RUN="bash cinc_mixer_v1/scripts/run_ett_spk.sh"
C=spkmx_lif_muxlin_iand
D=20261107
lr() { case $1 in ETTh2) echo 0.0005 ;; ETTm2) echo 0.0001 ;; esac; }
case ${1:?repro|smoke|l2} in
  repro) f="$Q/ett-l2-repro-$D.txt"; : > "$f"
         echo "SUITE=check-ett-l2-repro-$D COND=$C LR=$(lr ETTh2) LAYERS=1 $RUN ETTh2 96 42" >> "$f" ;;
  smoke) f="$Q/ett-l2-smoke-$D.txt"; : > "$f"
         for dp in "ETTh2 96" "ETTm2 720"; do ds=${dp% *}
           echo "EPOCHS=2 SUITE=smoke-ett-l2-$D COND=$C LR=$(lr $ds) LAYERS=1 $RUN $dp 42" >> "$f"
           echo "EPOCHS=2 SUITE=smoke-ett-l2-$D COND=$C LOSS=mse LR=$(lr $ds) LAYERS=1 $RUN $dp 42" >> "$f"
         done ;;
  l2)    f="$Q/ett-l2-$D.txt"; : > "$f"
         for dp in "ETTm2 720" "ETTm2 96" "ETTh2 720" "ETTh2 96"; do ds=${dp% *}
           for s in 42 7 2026; do
             echo "SUITE=flif-v1-ett-spkmux-l2 COND=$C LOSS=mse LR=$(lr $ds) LAYERS=1 $RUN $dp $s" >> "$f"
           done
         done ;;
  *) echo "bad stage $1"; exit 1 ;;
esac
echo "$f:$(wc -l < "$f")"
