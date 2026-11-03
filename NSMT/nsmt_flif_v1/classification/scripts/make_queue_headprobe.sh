#!/usr/bin/env bash
# MODI 2026-11-03: regenerates the (git-ignored) queue of the head readout diagnostic (docs/MODI_PROJECT_LOG.md 2026-11-03; user
# decision "진단부터", ETT cells ETTh2 / ETTm2 x H96 / H720). The diagnostic probes frozen query-form bodies; ETTh2 checkpoints exist
# (suite flif-v1-ett-query, 10-30), ETTm2 ones do not, so this trains them with the unchanged ETTm2 settings of run_ett_dual.sh:
#   nsmt4q_pop_lif / nsmt4q_pop_flif x ETTm2 x H96 / H720 x seeds 42 / 7 / 2026 = 12 runs, suite flif-v1-ett-query-m2
# Pool (from NSMT/): bash scripts/gpu_pool2.sh nsmt_flif_v1/classification/scripts/queues/headprobe-ettm2-20261103.txt 0,1,2,3
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
ETT=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh
Q=$QD/headprobe-ettm2-20261103.txt; : > $Q
for pl in 96 720; do for c in nsmt4q_pop_lif nsmt4q_pop_flif; do for s in 42 7 2026; do
  echo "COND=$c SUITE=flif-v1-ett-query-m2 bash $ETT ETTm2 $pl $s"
done; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
