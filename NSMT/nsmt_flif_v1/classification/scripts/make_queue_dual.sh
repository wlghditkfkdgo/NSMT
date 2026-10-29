#!/usr/bin/env bash
# MODI 2026-10-29: regenerates the (git-ignored) queues of the dual-path batch (docs/MODI_PROJECT_LOG.md 2026-10-29) -- the
# [NSMT hippocampus T=4] model (nsmt4_*, keys / values replaced by the neocortex) back in the comparisons next to the B form
# (spkneo_*), B-form settings (aux loss off, --neo_grad 1.0), the neuron swapped in the neocortex only (user rule, 10-29 17:1x).
#   smoke: every new condition once, 1 epoch, seed 42 (CinC fold 0: nsmt4 5; MIT-BIH: nsmt4 5 + spkneo 5; ETT ETTh2 H96: 10)
#   main : CinC fold 0, 90 epochs -- nsmt4_<n> + nsmt4_lif_nomem (spkneo @90 exists: flif-v1-rhythm-addneo90-cinc);
#          MIT-BIH -- nsmt4 + spkneo (5 each); ETT ETTh2 / ETTm2 x H96 / H720 (user decision 12 of 10-14) -- nsmt4 + spkneo;
#          seeds 42 / 7 / 2026; long CinC runs first, then MIT-BIH, then ETT H96 cells before H720
# Pools (from NSMT/): scripts/dual_driver.sh
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
NS="lif flif pop_lif pop_flif"; SEEDS="42 7 2026"
CIN=nsmt_flif_v1/classification/scripts/run_rhythm_addneo.sh; MIT=nsmt_flif_v1/classification/scripts/run_mitbih3.sh
ETT=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh
N4="$(for n in pop_flif pop_lif flif lif; do echo -n "nsmt4_$n "; done)nsmt4_lif_nomem"
SB="$(for n in pop_flif pop_lif flif lif; do echo -n "spkneo_$n "; done)spkneo_lif_nomem"
Q=$QD/dual-smoke-20261029.txt; : > $Q
for c in $N4; do echo "COND=$c DATA=CINC2017 SUITE=smoke-dual-cinc-20261029 LR=0.001 EPOCHS=1 bash $CIN 42"; done >> $Q
for c in $N4 $SB; do echo "COND=$c SUITE=smoke-dual-mitbih-20261029 EPOCHS=1 bash $MIT 42"; done >> $Q
for c in $N4 $SB; do echo "COND=$c SUITE=smoke-dual-ett-20261029 EPOCHS=1 bash $ETT ETTh2 96 42"; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
Q=$QD/dual-20261029.txt; : > $Q
for c in $N4; do for s in $SEEDS; do
  echo "COND=$c DATA=CINC2017 SUITE=flif-v1-rhythm-dual-cinc LR=0.001 EPOCHS=90 bash $CIN $s"; done; done >> $Q
for c in $N4 $SB; do for s in $SEEDS; do echo "COND=$c SUITE=flif-v1-mitbih-dual bash $MIT $s"; done; done >> $Q
for pl in 96 720; do for ds in ETTm2 ETTh2; do                     # user decision 12 (10-14): ETTh2 + ETTm2, H96 / H720
  for c in $N4 $SB; do for s in $SEEDS; do echo "COND=$c SUITE=flif-v1-ett-dual bash $ETT $ds $pl $s"; done; done
done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
