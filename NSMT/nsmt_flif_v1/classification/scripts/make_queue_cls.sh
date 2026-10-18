#!/usr/bin/env bash
# Writes scripts/queues/<SUITE>.txt (queue files are git-ignored; this script is the record).
# MIT-BIH suite (MODI_PROJECT_LOG 2026-10-19): ECG model, neocortex neurons {spikingjelly LIF (base), spikeDE LIF, f-LIF}
#   x placement {both, mlp} + references no_neo (ECG ablation) and hippo_only (decision 7) = 7 conditions x seeds 42/7/2026
#   = 21 runs, f-LIF first (slowest).
# usage: make_queue_cls.sh [SUITE [EPOCHS]]     (smoke: make_queue_cls.sh smoke-cls-<date> 1  -> seed 42 only)
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; SUITE=${1:-flif-v1-cls}; EP=${2:-}
SEEDS="42 7 2026"; [ -n "$EP" ] && SEEDS="42"
mkdir -p "$W/scripts/queues"; Q="$W/scripts/queues/$SUITE.txt"; : > "$Q"
C="SUITE=$SUITE${EP:+ EPOCHS=$EP}"
for c in neo_flif_both neo_flif_mlp neo_lif_both neo_lif_mlp base hippo_only no_neo; do for s in $SEEDS; do
  echo "COND=$c $C bash nsmt_flif_v1/classification/scripts/run_mitbih.sh $s" >> "$Q"; done; done
echo "$Q: $(wc -l < "$Q") lines"
