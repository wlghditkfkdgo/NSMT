#!/usr/bin/env bash
# MODI 2026-10-30: query batch + tuning stage 2a driver (run from NSMT/ in tmux). Pool A (4 slots): CinC query 15 runs. Pool B
# (4 slots): ETT query 30 runs, then the learning-rate stage (306 runs). The smoke queues are run by hand first. Pool logs report
# rc=0 always (gpu_pool2.sh prints $? after $(date)); failures are the SKIP lines of the results.
set -u
Q=nsmt_flif_v1/classification/scripts/queues; L=nsmt_flif_v1/classification/log
bash scripts/gpu_pool2.sh $Q/query-cinc-20261030.txt 0,1,2,3 > $L/query-cinc-pool.log 2>&1 &
PA=$!
{ bash scripts/gpu_pool2.sh $Q/query-ett-20261030.txt 0,1,2,3 > $L/query-ett-pool.log 2>&1; echo "QUERY ETT DONE $(date)" >> $L/query-ett-pool.log
  bash scripts/gpu_pool2.sh $Q/tune-lr-20261030.txt 0,1,2,3 > $L/tune-lr-pool.log 2>&1; echo "ALL DONE $(date)" >> $L/tune-lr-pool.log; } &
PB=$!
wait $PA; echo "ALL DONE $(date)" >> $L/query-cinc-pool.log
wait $PB
echo "[driver $(date +%F' '%T)] ALL DONE"
