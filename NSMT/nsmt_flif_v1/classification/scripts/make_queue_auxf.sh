#!/usr/bin/env bash
# MODI 2026-11-19: regenerates the (git-ignored) queue of the full-signal aux test, phase F of of_driver.sh (docs/MODI_PROJECT_LOG.md;
# user request 11-19 "저주파 auxiliary loss가 끝나면 이어서 원 시계열을 복원하는 auxiliary loss로 교체 후 실험해줘"). The _aux conditions of
# the low-pass test with the raw patches as the reconstruction target (--keep_ratio 0), same alpha / AUX:
#   suite flif-v1-ett-auxf (48 runs): nsmt4q_pop_<lif|flif>_{msn,msn_sde}_auxf (run_ett_dual.sh), ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026
#   suite flif-v1-mitbih-auxf (12 runs): the same four conditions (run_mitbih4.sh), seeds 42 / 7 / 2026
# Order: ETTm2 (longest), MIT-BIH, ETTh2. NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu); started by of_driver.sh.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; M4=nsmt_flif_v1/classification/scripts/run_mitbih4.sh
Q=$QD/auxf-20261119.txt; : > $Q
CONDS="nsmt4q_pop_lif_msn_auxf nsmt4q_pop_flif_msn_auxf nsmt4q_pop_lif_msn_sde_auxf nsmt4q_pop_flif_msn_sde_auxf"
ett() { for pl in 720 96; do for s in $SEEDS; do for c in $CONDS; do echo "COND=$c SUITE=flif-v1-ett-auxf bash $ET $1 $pl $s"; done; done; done >> $Q; }
ett ETTm2
for s in $SEEDS; do for c in $CONDS; do echo "COND=$c SUITE=flif-v1-mitbih-auxf bash $M4 $s"; done; done >> $Q
ett ETTh2
echo "$Q: $(grep -c . $Q) jobs"
