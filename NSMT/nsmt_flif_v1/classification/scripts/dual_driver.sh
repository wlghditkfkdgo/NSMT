#!/usr/bin/env bash
# MODI 2026-10-29: dual-path batch driver (run from NSMT/ in tmux). Waits until the whole-model neuron swap pool has finished
# ("ALL DONE" in nsmt_flif_v1/classification/log/nall-pool.log), then runs the main queue (scripts/make_queue_dual.sh) on 8 slots
# (2 per GPU, next to the remaining PPG runs and other sessions). The smoke queue is run by hand first.
set -u
Q=nsmt_flif_v1/classification/scripts/queues; L=nsmt_flif_v1/classification/log
until grep -q "ALL DONE" $L/nall-pool.log 2>/dev/null; do sleep 120; done
echo "[driver $(date +%F' '%T)] nall pool done -> dual main pool"
bash scripts/gpu_pool2.sh $Q/dual-20261029.txt 0,1,2,3,0,1,2,3 > $L/dual-pool.log 2>&1
echo "[driver $(date +%F' '%T)] ALL DONE"
