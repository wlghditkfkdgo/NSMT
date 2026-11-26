#!/usr/bin/env bash
# MODI 2026-11-26 (evening): queue of the second diagnostic batch (54 runs; docs/MODI_PROJECT_LOG.md 2026-11-26 pre-registration):
# SLEEPPSD = Sleep-EDF with epoch-spectrum tokens (the (나) front end, user choice ①), MBIP = MIT-BIH inter-patient with past beats ((가)).
# usage (from NSMT/): bash nsmt_flif_v1/classification/scripts/make_queue_oc2.sh <queue file> [suite]
set -u
Q=${1:?queue file}; SUITE=${2:-flif-v1-oc2}; R=nsmt_flif_v1/classification/scripts/run_oc.sh
mkdir -p "$(dirname "$Q")"; : > "$Q"
for s in 42 7 2026; do for c in gru_orig_L17 gru_shuf_L17 tf_orig_L17 tf_shuf_L17 gru_orig_L5 gru_orig_L2 gru_orig_L1 gru_orig_L1_rr; do
  echo "SUITE=$SUITE bash $R MBIP $c $s" >> "$Q"; done; done
for f in 0 1 2 3 4; do for c in gru_orig_L16 gru_shuf_L16 tf_orig_L16 tf_shuf_L16 gru_orig_L4 gru_orig_L1; do
  echo "SUITE=$SUITE bash $R SLEEPPSD $c $f" >> "$Q"; done; done
echo "$(grep -c . "$Q") lines -> $Q"
