#!/usr/bin/env bash
# Writes scripts/queues/flif-v1-s2p10.txt (queue files are git-ignored; this script is the record).
# Stage 2 with early-stopping patience 10 and best-validation-checkpoint testing (MODI_PROJECT_LOG 2026-10-15):
#   base, hippo_only, and 6 regimes L<lead>g<neo_grad> x {neo_lif_both, neo_flif_both}
#   x ETTh2/ETTm2 x H96/H720 x seeds 42/7/2026 = 14 x 12 = 168 runs, longest cells first.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; Q="$W/scripts/queues/flif-v1-s2p10.txt"; : > "$Q"
C="PATIENCE=10 TEST_BEST=1 SUITE=flif-v1-s2p10"
for cell in "ETTm2 96" "ETTm2 720" "ETTh2 96" "ETTh2 720"; do
  for s in 42 7 2026; do
    echo "COND=base $C bash nsmt_flif_v1/forecasting/scripts/run_ett.sh $cell $s" >> "$Q"
    echo "COND=hippo_only $C bash nsmt_flif_v1/forecasting/scripts/run_ett.sh $cell $s" >> "$Q"
  done
  for reg in "0 0.05" "0 1.0" "2 0.05" "2 1.0" "4 0.05" "4 1.0"; do
    set -- $reg
    for c in neo_flif_both neo_lif_both; do for s in 42 7 2026; do
      echo "COND=$c LEAD=$1 NG=$2 $C bash nsmt_flif_v1/forecasting/scripts/run_ett.sh $cell $s" >> "$Q"
    done; done
  done
done
echo "$Q: $(wc -l < "$Q") lines"
