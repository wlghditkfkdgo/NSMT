#!/usr/bin/env bash
# MODI 2026-10-30: regenerates the (git-ignored) queue of one tuning stage (docs/MODI_PROJECT_LOG.md 2026-10-30), ETT ETTh2 x H96 / H720
# (user scope for quick checks), seeds 42 / 7 / 2026, runner nsmt_flif_v1/forecasting/scripts/run_ett_tune.sh.
#   usage: make_queue_tune.sh smoke | lr | smoke_layers | layers | smoke_reset | reset        (alpha: written when pre-registered)
#   lr    : the 17 compared settings x LR in {0.00025, 0.001, 0.002} (0.0005 = the default runs that already exist) = 306 runs
#   layers: the 4 Neo-mixer settings (mixall_<n>_bidir) x --time_layers in {2, 3, 4} (1 = existing runs), each at its stage-lr
#           selection (forecasting/results/flif-v1-ett-tune/selected_lr.json) = 72 runs (MODI 2026-10-31)
#   reset : the 16 settings with spikeDE neurons x --neo_reset {subtract, zero} (population: zero only; spikede = existing runs),
#           each at its lr / layers selections (selected_lr.json, selected_layers.json) = 144 runs (MODI 2026-10-31)
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; ETT=nsmt_flif_v1/forecasting/scripts/run_ett_tune.sh
CONDS="spk_ssa $(for f in mixall nsmt4q nsmt4 spkneo; do for n in pop_flif pop_lif flif lif; do [ $f = mixall ] && echo -n "mixall_${n}_bidir " || echo -n "${f}_${n} "; done; done)"
case ${1:?stage} in
  smoke)
    Q=$QD/tune-smoke-20261030.txt; : > $Q
    echo "LR=0.001 COND=spk_ssa SUITE=smoke-tune-ett-20261030 EPOCHS=1 bash $ETT ETTh2 96 42" >> $Q
    echo "LR=0.00025 COND=mixall_pop_flif_bidir SUITE=smoke-tune-ett-20261030 EPOCHS=1 bash $ETT ETTh2 96 42" >> $Q
    echo "LR=0.002 COND=spkneo_flif SUITE=smoke-tune-ett-20261030 EPOCHS=1 bash $ETT ETTh2 720 42" >> $Q
    echo "COND=nsmt4q_pop_lif SUITE=smoke-tune-ett-20261030 EPOCHS=1 bash $ETT ETTh2 96 42" >> $Q ;;
  lr)
    Q=$QD/tune-lr-20261030.txt; : > $Q
    for pl in 96 720; do for lr in 0.002 0.001 0.00025; do for c in $CONDS; do for s in $SEEDS; do
      echo "LR=$lr COND=$c SUITE=flif-v1-ett-tune bash $ETT ETTh2 $pl $s"; done; done; done; done >> $Q ;;
  smoke_layers)
    Q=$QD/tune-smoke-layers-20261031.txt; : > $Q
    echo "LR=0.0005 LAYERS=4 COND=mixall_pop_flif_bidir SUITE=smoke-tune-ett-20261031 EPOCHS=1 bash $ETT ETTh2 96 42" >> $Q ;;
  layers)
    Q=$QD/tune-layers-20261031.txt; : > $Q
    SEL=$W/../forecasting/results/flif-v1-ett-tune/selected_lr.json
    for pl in 96 720; do for L in 4 3 2; do for n in pop_flif pop_lif flif lif; do
      c=mixall_${n}_bidir; lr=$(python3 -c "import json; print(json.load(open('$SEL'))['selected']['$c'])")
      for s in $SEEDS; do echo "LR=$lr LAYERS=$L COND=$c SUITE=flif-v1-ett-tune bash $ETT ETTh2 $pl $s"; done
    done; done; done >> $Q ;;
  smoke_reset)
    Q=$QD/tune-smoke-reset-20261031.txt; : > $Q
    echo "RESET=zero COND=mixall_lif_bidir SUITE=smoke-tune-ett-20261031 EPOCHS=1 bash $ETT ETTh2 96 42" >> $Q
    echo "RESET=subtract COND=nsmt4q_flif SUITE=smoke-tune-ett-20261031 EPOCHS=1 bash $ETT ETTh2 96 42" >> $Q
    echo "RESET=zero COND=spkneo_pop_flif SUITE=smoke-tune-ett-20261031 EPOCHS=1 bash $ETT ETTh2 96 42" >> $Q ;;
  reset)
    Q=$QD/tune-reset-20261031.txt; : > $Q
    RS=$W/../forecasting/results/flif-v1-ett-tune
    for pl in 96 720; do for c in ${CONDS#spk_ssa }; do for m in subtract zero; do
      case $c in *pop_*) [ $m = subtract ] && continue ;; esac
      lr=$(python3 -c "import json; print(json.load(open('$RS/selected_lr.json'))['selected']['$c'])")
      L=$(python3 -c "import json; print(json.load(open('$RS/selected_layers.json'))['selected']['$c'])")
      for s in $SEEDS; do echo "LR=$lr LAYERS=$L RESET=$m COND=$c SUITE=flif-v1-ett-tune bash $ETT ETTh2 $pl $s"; done
    done; done; done >> $Q ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
echo "$Q: $(grep -c . $Q) jobs"
