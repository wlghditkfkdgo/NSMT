#!/usr/bin/env bash
# MODI 2026-10-29: whole-model neuron swap batch driver (run in tmux from anywhere). Quick pool (ETTh2 H96, 3 per GPU), then the
# main pool (2 per GPU, next to the PPG-DaLiA pool). Queues: scripts/make_queue_nall.sh. The smoke queue is run by hand first.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."                    # NSMT/
Q=nsmt_flif_v1/classification/scripts/queues; L=nsmt_flif_v1/classification/log
bash scripts/gpu_pool2.sh $Q/nall-quick-20261029.txt 0,1,2,3,0,1,2,3,0,1,2,3 > $L/nall-quick-pool.log 2>&1
echo "QUICK DONE $(date)" >> $L/nall-quick-pool.log
bash scripts/gpu_pool2.sh $Q/nall-20261029.txt 0,1,2,3,0,1,2,3 > $L/nall-pool.log 2>&1
echo "ALL DONE $(date)" >> $L/nall-pool.log
