#!/usr/bin/env bash
# MODI 2026-11-27: queue of the PTB order check (24 runs; docs/MODI_PROJECT_LOG.md 2026-11-27 pre-registration). Run from NSMT/.
set -u
Q=${1:?queue file}; SUITE=${2:-flif-v1-oc3}; R=nsmt_flif_v1/classification/scripts/run_oc.sh
mkdir -p "$(dirname "$Q")"; : > "$Q"
for l in PTB PTB30; do for s in 42 7 2026; do for c in gru_orig gru_shuf tf_orig tf_shuf; do echo "SUITE=$SUITE bash $R $l $c $s" >> "$Q"; done; done; done
echo "$(grep -c . "$Q") lines -> $Q"
