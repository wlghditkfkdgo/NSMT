#!/usr/bin/env bash
# MODI 2026-11-21: launcher of the 4-2 report-only token-order evaluation (order_eval_nc.py; docs/MODI_PROJECT_LOG.md 2026-11-21).
# ETT 4 shards on GPUs 0-2 + 3, MIT-BIH 1 shard on GPU 3 before its ETT shard; logs in classification/log/flif-v1-nc-order/ (local).
# Run as: tmux new-session -d -s flif_ncord "bash nsmt_flif_v1/classification/scripts/run_order_nc.sh"  (any working directory)
set -u
CLS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$CLS"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/home/yschoi/.conda/envs/snn_recall/lib}
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}; LG=log/flif-v1-nc-order; mkdir -p $LG
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i $PY order_eval_nc.py --task ett --shard $i/4 > $LG/ett-$i.log 2>&1 & done
( CUDA_VISIBLE_DEVICES=3 $PY order_eval_nc.py --task mb --shard 0/1 > $LG/mb-0.log 2>&1; CUDA_VISIBLE_DEVICES=3 $PY order_eval_nc.py --task ett --shard 3/4 > $LG/ett-3.log 2>&1 ) &
wait
echo "ALL ORDER SHARDS FINISHED $(date)" > $LG/done.txt
