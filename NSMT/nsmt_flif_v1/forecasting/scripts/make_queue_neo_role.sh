#!/usr/bin/env bash
# Writes scripts/queues/neo-role-20261017.txt (queue files are git-ignored; this script is the record).
# Phase A (MODI_PROJECT_LOG 2026-10-17): neo_role_analysis.py on the best-validation weights of 72 stage-2 runs:
# base, hippo_only, neo_lif_both, neo_flif_both (L0g0.05), neo_lif_both@L2g0.05, neo_flif_both@L2g0.05
# x ETTh2/ETTm2 x H96/H720 x seeds 42/7/2026 (suite flif-v1-s2p10). The pool appends the GPU index (-nd <g>).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; Q="$W/scripts/queues/neo-role-20261017.txt"; : > "$Q"
PY="cd nsmt_flif_v1/forecasting && LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib OMP_NUM_THREADS=4 /home/yschoi/.conda/envs/snn_recall/bin/python neo_role_analysis.py"
grep '^RESULT' "$W/results/flif-v1-s2p10/raw.txt" \
 | grep -E 'cond=(base|hippo_only|neo_lif_both|neo_flif_both) regime=L0g0.05 |cond=neo_(lif|flif)_both regime=L2g0.05 ' \
 | grep -oE 'dir=\S+' | cut -d= -f2 | sort \
 | while read -r d; do echo "$PY --run \"$d\" --out results/neo-role-20261017 -nd" >> "$Q"; done
echo "$Q: $(wc -l < "$Q") lines"
