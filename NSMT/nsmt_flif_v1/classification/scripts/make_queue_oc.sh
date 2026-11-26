#!/usr/bin/env bash
# MODI 2026-11-26: queue of the order / accumulation check (80 runs; docs/MODI_PROJECT_LOG.md 2026-11-26). Run from NSMT/.
# usage: bash nsmt_flif_v1/classification/scripts/make_queue_oc.sh <queue file> [suite]
set -u
Q=${1:?queue file}; SUITE=${2:-flif-v1-oc}; R=nsmt_flif_v1/classification/scripts/run_oc.sh
mkdir -p "$(dirname "$Q")"; : > "$Q"
for f in 0 1 2 3 4; do for c in gru_orig_L16 gru_shuf_L16 tf_orig_L16 tf_shuf_L16 gru_orig_L4 gru_orig_L1; do
  echo "SUITE=$SUITE bash $R SLEEPCTX $c $f" >> "$Q"; done; done                      # heaviest first
for f in 0 1 2 3 4; do for c in gru_orig gru_shuf tf_orig tf_shuf; do echo "SUITE=$SUITE bash $R SLEEPIN $c $f" >> "$Q"; done; done
for s in 42 7 2026; do for c in gru_orig gru_shuf tf_orig tf_shuf gru_orig_k5 gru_orig_k10; do echo "SUITE=$SUITE bash $R SC $c $s" >> "$Q"; done; done
for s in 42 7 2026; do for c in gru_orig gru_shuf tf_orig tf_shuf; do echo "SUITE=$SUITE bash $R MITBIH $c $s" >> "$Q"; done; done
echo "$(grep -c . "$Q") lines -> $Q"
