#!/usr/bin/env bash
# Stage 3 smoke: 1 epoch of the five Phase B conditions on ETTh1 H96 seed 42 (pipeline check, NOT an experiment).
# usage: smoke_s3.sh   (runs through gpu_pool2 on GPUs 0-3)
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"; Q="$W/scripts/queues/smoke-s3.txt"; : > "$Q"
C="PATIENCE=10 TEST_BEST=1 EPOCHS=1 SUITE=smoke-s3-$(date +%Y%m%d)"
for cond in "hippo_only TB=1" "serial_lif TB=1" "serial_flif TB=1" "serial_lif TB=0" "serial_flif TB=0"; do
  set -- $cond; echo "COND=$1 $2 $C bash nsmt_flif_v1/forecasting/scripts/run_ett.sh ETTh1 96 42" >> "$Q"
done
cd "$R" && bash scripts/gpu_pool2.sh "$Q" 0,1,2,3 > "$W/log/smoke-s3-$(date +%Y%m%d).pool.log" 2>&1
echo "smoke done: $(grep -c '^RESULT' "$W/results/smoke-s3-$(date +%Y%m%d)/raw.txt" 2>/dev/null) RESULT"
