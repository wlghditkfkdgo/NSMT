#!/usr/bin/env bash
# Writes scripts/queues/flif-v1-s3.txt (queue files are git-ignored; this script is the record).
# Stage 3 Phase B (MODI_PROJECT_LOG 2026-10-17): serial "aligned spiking embedding" -- the neocortex (T = N aligned)
# output is the backbone's spike embedding. Conditions: hippo_only Tb=1, serial_lif Tb=1, serial_flif Tb=1,
# serial_lif Tb=N, serial_flif Tb=N (N = 24 = the paper repetition count) x ETTh2/ETTm2 x H96/H720 x seeds 42/7/2026
# = 60 runs, patience 10 + best-validation weights (stage-2 rule). References (same rule): suite flif-v1-s2p10.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; Q="$W/scripts/queues/flif-v1-s3.txt"; : > "$Q"
C="PATIENCE=10 TEST_BEST=1 SUITE=flif-v1-s3"
for cell in "ETTm2 96" "ETTm2 720" "ETTh2 96" "ETTh2 720"; do
  for cond in "hippo_only TB=1" "serial_lif TB=1" "serial_flif TB=1" "serial_lif TB=0" "serial_flif TB=0"; do
    set -- $cond
    for s in 42 7 2026; do echo "COND=$1 $2 $C bash nsmt_flif_v1/forecasting/scripts/run_ett.sh $cell $s" >> "$Q"; done
  done
done
echo "$Q: $(wc -l < "$Q") lines"
