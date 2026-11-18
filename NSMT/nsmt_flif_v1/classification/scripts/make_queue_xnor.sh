#!/usr/bin/env bash
# MODI 2026-11-18: regenerates the (git-ignored) queue of the coupling attention test (docs/MODI_PROJECT_LOG.md; user decision 11-18
# "1. 진행  2. 1번에서 개선이 없으면 진행"). The XNOR + Log-PE score (--attn_score xnor_log; Lv et al. NeurIPS 2025) in both the msn
# base (Neocortex-MS query population, cross-attention q = Neocortex, k / v = Hippocampus) and Spikformer-style (self-attention).
#   suite flif-v1-ett-xnor-repro (3 runs, report only): spk_ssa / nsmt4q_pop_<lif|flif>_msn, ETTh2 H96 seed 42 retrained at this sha
#   suite flif-v1-ett-xnor (36 runs): spk_ssa_xl (run_ett_mixer.sh), nsmt4q_pop_<lif|flif>_msn_xl (run_ett_dual.sh);
#         ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026, input 96, every other setting unchanged
# Order: the retrain-noise runs first, then ETTm2 (longest) and ETTh2.
# NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; MX=nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh
Q=$QD/xnor-20261118.txt; : > $Q
R=flif-v1-ett-xnor-repro; SU=flif-v1-ett-xnor
echo "COND=spk_ssa SUITE=$R bash $MX ETTh2 96 42" >> $Q
for n in pop_lif pop_flif; do echo "COND=nsmt4q_${n}_msn SUITE=$R bash $ET ETTh2 96 42" >> $Q; done
for ds in ETTm2 ETTh2; do for pl in 720 96; do for s in $SEEDS; do
  echo "COND=spk_ssa_xl SUITE=$SU bash $MX $ds $pl $s"
  for n in pop_lif pop_flif; do echo "COND=nsmt4q_${n}_msn_xl SUITE=$SU bash $ET $ds $pl $s"; done
done; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
