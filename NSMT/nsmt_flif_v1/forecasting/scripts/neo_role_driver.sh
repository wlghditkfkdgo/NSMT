#!/usr/bin/env bash
# Phase A driver: per-run analyses (two workers per GPU), then the summary and the energy table. Run inside tmux.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib; O=results/neo-role-20261017
cd "$R"
echo "[role $(date '+%F %T')] start"
bash "$W/scripts/make_queue_neo_role.sh"
bash scripts/gpu_pool2.sh "$W/scripts/queues/neo-role-20261017.txt" 0,1,2,3,0,1,2,3 > "$W/log/neo-role-20261017.pool.log" 2>&1
echo "[role $(date '+%F %T')] runs done: $(ls "$W/$O/runs/"*.json 2>/dev/null | wc -l) JSON, failed jobs: $(grep -c 'DONE(rc=[1-9]' "$W/log/neo-role-20261017.pool.log")"
(cd "$W" && LD_LIBRARY_PATH=$L $PY neo_role_analysis.py --summarize --out $O && LD_LIBRARY_PATH=$L $PY neo_role_analysis.py --energy flif-v1-s2p10 --out $O)
echo "[role $(date '+%F %T')] ALL DONE"
